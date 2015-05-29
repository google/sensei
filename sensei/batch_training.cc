/*
*  Copyright 2015 Google Inc. All Rights Reserved.
*  
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*  
*      http://www.apache.org/licenses/LICENSE-2.0
*  
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*/
#include "sensei/batch_training.h"

#include <algorithm>
#include <string>
using std::string;
#include <utility>
using std::pair;
#include <vector>
using std::vector;

#include "sensei/base/integral_types.h"
#include "sensei/base/logging.h"
#include "sensei/file/file.h"
#include "sensei/file/recordio.h"
#include "sensei/common.h"
#include "sensei/config.h"
#include "sensei/config.pb.h"
#include "sensei/data.h"
#include "sensei/data_reader.h"
#include "sensei/feature_exploration.h"
#include "sensei/feature_map.h"
#include "sensei/feature_pruning.h"
#include "sensei/internal.pb.h"
#include "sensei/optimizers.h"
#include "sensei/range.h"
#include "sensei/proto/parse_text_proto.h"
#include "sensei/util/acmrandom.h"



namespace sensei {

using config::Validator;

BatchTraining::BatchTraining(const config::CommandList& command_list)
    : run_id_(ACMRandom(ACMRandom::HostnamePidTimeSeed()).Rand64()) {
  Init(command_list);
}

void BatchTraining::Init(const config::CommandList& command_list) {
  world_.logger()->SetRunId(run_id_);

  world_.optimizer()->SetInertiaFactor(1.0);
  world_.optimizer()->SetStepMultiplier(1.0);


  logs::Line log_line;
  *log_line.mutable_command_list_config() = command_list;
  world_.logger()->AddToLogs(log_line);

  for (const config::Command& command : command_list.command())
    command_queue_.push(command);
}

void BatchTraining::Run() {
  config::Command command;
  while (!command_queue_.empty()) {
    CHECK(command_queue_.Wait(&command));
    RunCommand(command);
  }
}

void BatchTraining::RunCommand(const config::Command& command) {
  logs::Line log_line;
  *log_line.mutable_run_command() = command;
  world_.logger()->AddToLogs(log_line);
  LOG(INFO) << "RunCommand:\n" << command.DebugString();

  Validator().ValidateOrDie(command);

  if (command.has_set()) {
    Set(command.set());
    return;
  }

  if (command.has_read_data()) {
    world_.read_data()->RunCommand(command.read_data());
    return;
  }

  if (command.has_initialize_bias()) {
    InitializeBias();
    return;
  }

  if (command.has_add_new_product_features()) {
    AddNewProductFeatures(
        command.add_new_product_features().feature_exploration());
    return;
  }

  if (command.has_prune_features()) {
    PruneFeatures(command.prune_features().feature_pruning());
    return;
  }


  if (command.has_fit_model_weights()) {
    FitModelWeights(command.fit_model_weights());
    return;
  }


  if (command.has_run_sgd()) {
    RunSgdIteration(command.run_sgd());
    return;
  }

  if (command.has_sgd()) {
    RunSgdCommand(command.sgd());
    return;
  }

  if (command.has_evaluate_stats()) {
    EvaluateStats(command.evaluate_stats());
    return;
  }

  if (command.has_store_model()) {
    world_.write_model()->RunCommand(command.store_model());
    return;
  }

  if (command.has_write_model()) {
    world_.write_model()->RunCommand(command.write_model());
    return;
  }

  if (command.has_get_model()) {
    world_.write_model()->RunCommand(command.get_model());
    return;
  }

  if (command.has_repeat()) {
    for (uint32 i : Range(command.repeat().repetitions())) {
      UNUSED(i);
      for (const config::Command& c : command.repeat().command()) RunCommand(c);
    }
    return;
  }

  if (command.has_command_list()) {
    for (const config::Command& c : command.command_list().command())
      RunCommand(c);
    return;
  }

  if (command.has_internal()) {
    RunInternalCommand(command.internal());
    return;
  }


  if (command.has_from_file()) {
    string contents;
    QCHECK_OK(file::GetContents(command.from_file().path(), &contents,
                                file::Defaults()));
    config::Command command;
    using proto_util::ParseTextOrDie;
    command.MergeFrom(ParseTextOrDie<config::Command>(contents));
    RunCommand(command);
    return;
  }


  if (command.has_score_rows()) {
    world_.score_rows()->RunCommand(command.score_rows());
    return;
  }

  LOG(FATAL) << "Unknown command: " << command.DebugString();
}


void BatchTraining::Set(const config::Set& set) {
  if (set.has_name()) name_ = set.name();

  if (set.has_logging()) {
    const config::Set::Logging& logging = set.logging();
    if (logging.has_log_timestamp())
      world_.logger()->SetLogTimestamp(logging.log_timestamp());

    if (logging.has_text_log_path()) {
      CheckCanWrite(logging.text_log_path(), logging.clear_log_files());
      world_.logger()->SetTextLogPath(logging.text_log_path());
    }

    if (logging.has_recordio_log_path()) {
      CheckCanWrite(logging.recordio_log_path(), logging.clear_log_files());
      world_.logger()->SetRecordioLogPath(logging.recordio_log_path());
    }
  }

  if (set.has_regularization()) {
    world_.regularizations()->SetRegularization(set.regularization());
  }

  if (set.has_regularization_div_sqrt_n()) {
    world_.regularizations()->SetRegularizationDivSqrtN(
        set.regularization_div_sqrt_n());
  }

  if (set.has_regularization_mul_sqrt_n()) {
    world_.regularizations()->SetRegularizationMulSqrtN(
        set.regularization_mul_sqrt_n());
  }

  if (set.has_regularization_confidence()) {
    world_.regularizations()->SetRegularizationConfidence(
        set.regularization_confidence());
  }

  if (set.has_inertia_factor())
    world_.optimizer()->SetInertiaFactor(set.inertia_factor());
  if (set.has_step_multiplier())
    world_.optimizer()->SetStepMultiplier(set.step_multiplier());
  if (set.has_allow_undo()) world_.optimizer()->SetAllowUndo(set.allow_undo());

  for (double lift : set.logged_lift_fraction()) lift_fraction_.push_back(lift);
  std::sort(lift_fraction_.begin(), lift_fraction_.end());

  if (set.has_sgd_learning_rate_schedule()) {
    world_.sgd()->SetLearningRateSchedule(set.sgd_learning_rate_schedule());
  }


  if (set.has_deterministic()) {
    world_.optimizer()->SetDeterministic(set.deterministic());
    world_.sgd()->SetDeterministic(set.deterministic());
    world_.logger()->SetRunId(set.deterministic() ? 0 : run_id_);
    if (set.deterministic()) {
      world_.logger()->SetLogTimestamp(0);
    }
  }

  if (set.has_max_shard_size()) {
    world_.data()->SetMaxShardSize(set.max_shard_size());
  }
}

void BatchTraining::InitializeBias() {
  JProduct bias = JProduct({});
  if (!world_.product_map()->HasFeature(bias)) {
    LOG(FATAL) << "Asked to initialize_bias, but there is no bias feature.";
  }
  uint32 bias_j = world_.product_map()->FeatureToJConst(bias);
  world_.model()->w[bias_j] =
      world_.data()->GetTraining().GetStats().LogOdds(bias_j);
}

void BatchTraining::AddNewProductFeatures(
    const config::FeatureExploration& ti_fe) {
  logs::Line log_line;
  world_.feature_exploration()->AddNewProductFeatures(
      ti_fe, log_line.mutable_feature_exploration());
  world_.logger()->AddToLogs(log_line);
}

void BatchTraining::PruneFeatures(const config::FeaturePruning& config) {
  logs::Line log_line;
  world_.feature_pruning()->PruneFeatures(config,
                                          log_line.mutable_feature_pruning());
  world_.logger()->AddToLogs(log_line);
}


void BatchTraining::FitModelWeights(
    const config::FitModelWeights& fit_model_weights) {
  for (uint32 i : Range(fit_model_weights.iterations())) {
    UNUSED(i);
    world_.optimizer()->MakeOnePass();
  }
}

void BatchTraining::RunSgdIteration(const config::RunSgd& run_sgd) {
  if (!world_.sgd()->IsTrainingValid()) return;
  optimizer::Sgd::TrainingMode sgd_training_mode =
      run_sgd.only_new_features() ? optimizer::Sgd::NEW_FEATURES
                                  : optimizer::Sgd::ALL_FEATURES;
  world_.sgd()->SetTrainingMode(sgd_training_mode);
  for (uint32 i : Range(run_sgd.iterations())) {
    UNUSED(i);
    world_.sgd()->MakeOnePass();
  }
}

void BatchTraining::RunSgdCommand(const config::Sgd& sgd_command) {
  if (sgd_command.has_learning_rate()) {
    world_.optimizer()->SyncModelWithWeights();
  }
  world_.sgd()->RunCommand(sgd_command);
}

static void ComputeAucAndLift(const vector<double>& lift_fraction,
                              const data::ShardSet& data,
                              Model::PerShard* per_shard, double* auc,
                              vector<double>* lift_values) {
  uint32 data_size = per_shard->wxs.size();
  CHECK_EQ(data.GetStats().RowCount(), data_size);

  *auc = 0;
  lift_values->resize(lift_fraction.size());

  // The pairs (wx, y) for faster sorting.  Sorted after each iteration.
  uint32 index = 0;
  vector<pair<Double, bool>> wx_row_ordering;
  for (const data::Shard& shard : data.GetShards()) {
    for (const uint32 i : Range(shard.RowCount())) {
      wx_row_ordering.push_back(
          std::make_pair(per_shard->wxs[index++], shard.Ys()[i] == 1));
    }
  }

  // On the same data set, sorting vector<uint32> using <wxs>-based comparator
  // takes 2.3s.  This doesn't get faster when sorting an already-sorted
  // vector.  Current sorting takes 0.9s.  This gets down to 0.4s when sorting
  // an already-sorted vector.
  // TODO(lew,witoldjarnicki): Speed it up.
  std::sort(wx_row_ordering.rbegin(), wx_row_ordering.rend());

  uint32 i = 0;  // 0..i-1 = positives
  uint64 x = 0;  // False positives.
  uint64 y = 0;  // True positives.
  uint32 lift_index = 0;
  // Invariant: x + y == i.
  while (i < data_size) {
    uint64 x_new = x;
    uint64 y_new = y;
    uint32 i_new = i;
    Double wx = wx_row_ordering[i].first;
    while (i_new < data_size && wx_row_ordering[i_new].first == wx) i_new += 1;
    for (uint32 ii : Range(i, i_new)) {
      const bool is_positive = wx_row_ordering[ii].second;
      if (is_positive) {
        y_new += 1;
      } else {
        x_new += 1;
      }
    }
    *auc += (y_new + y) * (x_new - x);
    // <lift_index> indexes <lift_fraction> and <lift_values>.
    for (; lift_index < lift_fraction.size(); ++lift_index) {
      double i_lift = data_size * lift_fraction[lift_index];
      if (i_lift <= i || i_lift > i_new) break;
      // i_lift is in (i,i_new].  Let's draw a segment between (i,y) and
      // (i_new,y_new).
      const double lambda = (i_lift - i) / (i_new - i);
      // The unnormalized value of the linear interpolation between (i,y) and
      // (i_new,y_new).
      const double y_lift = y + (y_new - y) * lambda;
      // We'll normalize the value after the while loop, once we know the
      // total y.  We can divide it by the value corresponding to the random
      // model now, though.
      (*lift_values)[lift_index] = y_lift / lift_fraction[lift_index];
    }
    i = i_new;
    x = x_new;
    y = y_new;
  }
  for (double& lift : *lift_values) lift /= y;
  *auc /= 2 * x* y;
}

void BatchTraining::FillDataStatsWithAuc(const data::ShardSet& data,
                                         Model::PerShard* per_shard,
                                         logs::DataSetStats* data_set_stats) {
  world_.optimizer()->SyncModelWithWeights();
  Double auc;
  vector<double> lift_values;
  ComputeAucAndLift(lift_fraction_, data, per_shard, &auc, &lift_values);
  data_set_stats->set_auc(auc);
  for (uint32 i : Range(lift_fraction_.size())) {
    logs::Lift* lift = data_set_stats->add_lift();
    lift->set_lift_fraction(lift_fraction_[i]);
    lift->set_lift_value(lift_values[i]);
  }
}

void BatchTraining::EvaluateStats(const config::EvaluateStats& evaluate_stats) {
  logs::Line log_line;
  logs::Iteration* iteration_log = log_line.mutable_iteration();

  if (evaluate_stats.auc()) {
    FillDataStatsWithAuc(world_.data()->GetTraining(),
                         &world_.model()->training,
                         iteration_log->mutable_training_data_stats());
    if (world_.data()->GetHoldout().GetStats().RowCount() > 0) {
      FillDataStatsWithAuc(world_.data()->GetHoldout(),
                           &world_.model()->holdout,
                           iteration_log->mutable_holdout_data_stats());
    }
  }
  world_.logger()->AddToLogs(log_line);
}


void BatchTraining::RunInternalCommand(const config::Internal& internal) {
  CHECK_EQ(1, (internal.has_get_model() +           //
               internal.has_log_detailed_stats() +  //
               internal.has_log_dependees() +       //
               internal.has_get_data() +            //
               internal.has_get_scores() +          //
               0));
  logs::Line log_line;

  if (internal.has_get_model()) {
    world_.model()->GetProto(log_line.mutable_internal_model());
  }

  if (internal.has_log_detailed_stats()) {
    *log_line.mutable_internal_detailed_stats() =
        world_.data()->BuildDetailedStats();
  }

  if (internal.has_log_dependees()) {
    *log_line.mutable_internal_dependees() = world_.data()->BuildDependees();
  }

  if (internal.has_get_data()) {
    *log_line.mutable_internal_data() = world_.data()->ToInternalProto();
  }

  if (internal.has_get_scores()) {
    *log_line.mutable_data_score() = world_.score_rows()->GetScoreProto();
  }

  world_.logger()->AddToLogs(log_line);
}

}  // namespace sensei

