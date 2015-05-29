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
#include "sensei/config.h"

#include <math.h>
#include <algorithm>
#include <set>
using std::set;

#include "sensei/base/integral_types.h"
#include "sensei/file/file.h"
#include "sensei/common.h"
#include "sensei/range.h"
#include "sensei/proto/parse_text_proto.h"
#include "sensei/proto/message_differencer.h"
#include "sensei/strings/strutil.h"
#include "sensei/util/mathlimits.h"

using ::proto_util::MessageDifferencer;


namespace sensei {
namespace config {

namespace {

template <class T>
bool IsEmpty(const T& message) {
  MessageDifferencer differencer;
  T empty;
  return differencer.Equals(message, empty);
}

bool IsGlob(const string& s) { return !s.empty(); }

}  // namespace

#define CHECK_UNVALIDATED()                               \
  {                                                       \
    CHECK(IsEmpty(unvalidated))                           \
        << "Internal Sensei validation error.\n"          \
        << "Add missing validation & translation code.\n" \
        << unvalidated.DebugString();                     \
  }

#define PROCESS_FIELD(field)                                     \
  {                                                              \
    if (config.has_##field()) {                                  \
      Process(config.field());                                   \
    } else {                                                     \
      text_logger_->AddError(StrCat("Missing field: ", #field)); \
    }                                                            \
    unvalidated.clear_##field();                                 \
  }

#define MAYBE_PROCESS_FIELD(field)                     \
  {                                                    \
    if (config.has_##field()) Process(config.field()); \
    unvalidated.clear_##field();                       \
  }

#define PROCESS_DEPRECATED(field)                                   \
  {                                                                 \
    if (config.has_##field()) {                                     \
      text_logger_->AddError(StrCat("Deprecated field: ", #field)); \
    }                                                               \
    unvalidated.clear_##field();                                    \
  }

#define PROCESS_REPEATED_FIELD(field)                \
  {                                                  \
    for (const auto& f : config.field()) Process(f); \
    unvalidated.clear_##field();                     \
  }

#define PROCESS_REPEATED_DEPRECATED(field)                          \
  {                                                                 \
    if (config.field##_size() > 0) {                                \
      text_logger_->AddError(StrCat("Deprecated field: ", #field)); \
    }                                                               \
    unvalidated.clear_##field();                                    \
  }

// Please keep the Process() methods grouped and ordered EXACTLY as in
// config.proto.

// -----------------------------------------------------------------------------
// Public
void Validator::Process(const DataFiles& config) {
  DataFiles unvalidated(config);

  // format
  if (config.has_format()) {
    // This switch forces to remember to test new formats.
    switch (config.format()) {
      case DataFiles_Format_LIBSVM:
        break;
    }
  } else {
    text_logger_->AddError("Missing format.");
  }
  unvalidated.clear_format();

  // training_set
  PROCESS_FIELD(training_set);

  // holdout_set
  MAYBE_PROCESS_FIELD(holdout_set);

  // remove duplicates
  unvalidated.clear_remove_duplicate_features_in_each_row();


  CHECK_UNVALIDATED();
}


void Validator::Process(const DataFiles::DataSet& config) {
  DataFiles::DataSet unvalidated(config);
  if (config.files_glob_size() == 0) {
    text_logger_->AddError("File set has empty file glob.");
  }
  for (const string& s : config.files_glob()) {
    if (!IsGlob(s)) text_logger_->AddError("Invalid files glob: " + s);
  }
  unvalidated.clear_files_glob();

  for (const string& f : config.filter_feature()) {
    if (f.empty())
      text_logger_->AddError(
          "Empty string cannot be an element of filter_feature.");
  }
  unvalidated.clear_filter_feature();
  CHECK_UNVALIDATED();
}


void Validator::Process(const FeatureSet& config) {
  FeatureSet unvalidated(config);

  MAYBE_PROCESS_FIELD(explicit_list);
  MAYBE_PROCESS_FIELD(from_data);

  CHECK_UNVALIDATED();
}

void Validator::Process(const ExplicitFeatureList& config) {
  ExplicitFeatureList unvalidated(config);

  // feature
  unvalidated.clear_feature();

  CHECK_UNVALIDATED();
}

void Validator::Process(const FeatureSet::FromData& config) {
  FeatureSet::FromData unvalidated(config);

  // feature_prefix
  unvalidated.clear_feature_prefix();

  CHECK_UNVALIDATED();
}


void Validator::Process(const Flag& config) {
  Flag unvalidated(config);

  MAYBE_PROCESS_FIELD(command_list);

  if (!config.has_command_list()) {
    text_logger_->AddError("command_list must be set.");
  }

  PROCESS_DEPRECATED(batch_training);

  CHECK_UNVALIDATED();
}

// -----------------------------------------------------------------------------
// Experimental.


void Validator::Process(const FeatureSpec& config) {
  FeatureSpec unvalidated(config);

  PROCESS_FIELD(product);

  CHECK_UNVALIDATED();
}

void Validator::Process(const FeatureSpec_Product& config) {
  FeatureSpec_Product unvalidated(config);

  // prefix
  unvalidated.clear_prefix();

  CHECK_UNVALIDATED();
}

void Validator::Process(const ReadModel& config) {
  ReadModel unvalidated(config);

  if (!config.has_model_input_path()) {
    text_logger_->AddError("Filename for read_model not specified.");
  }
  unvalidated.clear_model_input_path();

  if (!config.has_format()) {
    text_logger_->AddError("Input model format not specified.");
  } else {
    // This switch forces us to test new formats
    switch (config.format()) {
      case ModelFormat::TEXT:
      case ModelFormat::SERIALIZED:
        break;
    }
  }
  unvalidated.clear_format();

  CHECK_UNVALIDATED();
}

void Validator::Process(const DataReader& config) {
  DataReader unvalidated(config);

  // format
  if (config.has_format()) {
    // This switch forces to remember to test new formats.
    switch (config.format()) {
      case DataReader_Format_LIBSVM:
        break;
    }
  } else {
    text_logger_->AddError("Missing format.");
  }
  unvalidated.clear_format();


  // Remove duplicated features
  unvalidated.clear_remove_duplicate_features_in_each_row();

  // training_set
  PROCESS_FIELD(training_set);

  // holdout_set
  MAYBE_PROCESS_FIELD(holdout_set);

  // read_model
  if (config.has_read_model() + (config.feature_spec_size() > 0) != 1) {
    text_logger_->AddError(
        "Exactly one of (read_model, feature_spec) must be "
        "specified.");
  }
  MAYBE_PROCESS_FIELD(read_model);

  // feature_spec
  if (config.feature_spec_size() > 0) {
    bool product_size_1_found = false;
    for (const config::FeatureSpec& fs : config.feature_spec()) {
      Process(fs);
      const config::FeatureSpec::Product& product = fs.product();
      product_size_1_found |= product.prefix_size() == 1;
      for (uint32 i : Range(product.prefix_size())) {
        for (uint32 j : Range(product.prefix_size())) {
          string pi = product.prefix(i);
          string pj = product.prefix(j);
          if (i != j && HasPrefixString(pi, pj)) {
            text_logger_->AddWarning(StrCat(AddQuotes(pj), " is a prefix of ",
                                            AddQuotes(pi), " in FeatureSpec:\n",
                                            fs.DebugString()));
          }
        }
      }
    }
    if (!product_size_1_found) {
      text_logger_->AddWarning(StrCat(
          "No explicitly added product of size 1.\n", config.DebugString()));
    }
  }
  unvalidated.clear_feature_spec();

  // thread_count
  if (config.thread_count() <= 0)
    text_logger_->AddError("thread_count must be positive");
  unvalidated.clear_thread_count();


  // max_product_size
  if (config.has_max_product_size() && config.max_product_size() < 0)
    text_logger_->AddError("max_product_size must be non-negative");
  unvalidated.clear_max_product_size();

  // add_sub_features
  unvalidated.clear_add_sub_features();

  // Looking for a duplicately defined feature, we may assume its factors are
  // sorted.  Furthermore, we may assume that the FeatureSpec factors matching
  // such a feature are sorted, too.  Therefore, we just go over all pairs of
  // FestureSpec records corresponding to the same number of factors, sort them
  // and check whether the prefixes at corresponding positions match (one of
  // them is a prefix of the other one).
  for (uint32 i : Range(config.feature_spec_size())) {
    vector<string> v_i;
    for (const string& s : config.feature_spec(i).product().prefix()) {
      v_i.push_back(s);
    }
    std::sort(v_i.begin(), v_i.end());
    for (uint32 j : Range(i)) {
      vector<string> v_j;
      for (const string& s : config.feature_spec(j).product().prefix()) {
        v_j.push_back(s);
      }
      std::sort(v_j.begin(), v_j.end());
      if (v_i.size() == v_j.size()) {
        bool matches = true;
        for (uint32 k : Range(v_i.size())) {
          if (!HasPrefixString(v_i[k], v_j[k]) &&
              !HasPrefixString(v_j[k], v_i[k])) {
            matches = false;
          }
        }
        if (matches) {
          string example_feature;
          for (uint32 k : Range(v_i.size()))
            example_feature += max(v_i[k], v_j[k]) + ", ";
          string error = StrCat(
              "Two FeaturesSpec records potentially define the same product ",
              "feature:\n", example_feature, "\n",
              config.feature_spec(i).DebugString(), "\n",
              config.feature_spec(j).DebugString());
          text_logger_->AddWarning(error);
        }
      }
    }
  }

  if (config.has_user_id_feature_name()) {
    if (config.user_id_feature_name().empty()) {
      text_logger_->AddError("user_id_feature_name must be non-empty");
    }
  }
  unvalidated.clear_user_id_feature_name();

  PROCESS_DEPRECATED(filter_feature);
  PROCESS_REPEATED_DEPRECATED(training_glob);
  PROCESS_REPEATED_DEPRECATED(holdout_glob);

  CHECK_UNVALIDATED();
}


void Validator::Process(const FeatureScoring::Bonus& config) {
  FeatureScoring::Bonus unvalidated(config);
  PROCESS_FIELD(feature_set);

  // factor
  if (config.has_factor()) {
    if (config.factor() < 0) {
      text_logger_->AddError("Bonus factor can't be negative.");
    }
  } else {
    text_logger_->AddError("Bonus factor missing.");
  }
  unvalidated.clear_factor();

  CHECK_UNVALIDATED();
}

void Validator::Process(const FeatureScoring& config) {
  FeatureScoring unvalidated(config);

  // feature_ordering
  if (config.has_feature_ordering()) {
    switch (config.feature_ordering()) {
      case FeatureScoring::WEIGHT_ABSOLUTE_VALUE:
      case FeatureScoring::WEIGHT_ABSOLUTE_VALUE_TIMES_ROW_COUNT:
      case FeatureScoring::FEATURE_OUTPUT_MUTUAL_INFORMATION:
      case FeatureScoring::FEATURE_OUTPUT_CORRELATION:
        break;
    }
  } else {
    text_logger_->AddError("Missing feature_ordering.");
  }
  unvalidated.clear_feature_ordering();

  unvalidated.clear_take_logarithm();

  MAYBE_PROCESS_FIELD(bonus);

  unvalidated.clear_logging();

  CHECK_UNVALIDATED();
}

void Validator::Process(const FeatureExploration& config) {
  FeatureExploration unvalidated(config);

  // maximum_features_added
  if (config.has_maximum_features_added() &&
      config.maximum_features_added() <= 0) {
    text_logger_->AddError("maximum_features_added must be positive");
  }
  unvalidated.clear_maximum_features_added();

  // feature_scoring
  PROCESS_FIELD(feature_scoring);

  // max_product_size
  if (config.has_max_product_size() && config.max_product_size() < 0)
    text_logger_->AddError("max_product_size must be non-negative");
  unvalidated.clear_max_product_size();

  // expected_xjbools_added
  if (config.has_expected_xjbools_added() &&
      config.expected_xjbools_added() <= 0) {
    text_logger_->AddError("expected_xjbools_added should be positive");
  }
  unvalidated.clear_expected_xjbools_added();


  if (!config.has_expected_xjbools_added() &&
      !config.has_maximum_features_added()) {
    text_logger_->AddError(
        "Either has_expected_xjbools_added or has_maximum_features_added "
        "must be set.");
  }

  PROCESS_DEPRECATED(materialize_product_features);
  PROCESS_DEPRECATED(feature_ordering);
  PROCESS_DEPRECATED(feature_ordering_multiplicative);

  CHECK_UNVALIDATED();
}

void Validator::Process(const FeaturePruning& config) {
  FeaturePruning unvalidated(config);

  PROCESS_FIELD(feature_scoring);

  // score_threshold
  unvalidated.clear_score_threshold();

  // top_count
  if (config.has_top_count() && config.top_count() < 0)
    text_logger_->AddError("top_count must be non-negative");
  unvalidated.clear_top_count();

  // top_fraction
  if (config.has_top_fraction() &&
      (config.top_fraction() < 0.0 || config.top_fraction() > 1.0)) {
    text_logger_->AddError("top_fraction must be in [0, 1]");
  }
  unvalidated.clear_top_fraction();

  if (!config.has_score_threshold() &&  //
      !config.has_top_count() &&        //
      !config.has_top_fraction()) {
    text_logger_->AddError(
        "At least one of {score_threshold, top_count, top_fraction} must be "
        "set.");
  }

  // DEPRECATED.
  PROCESS_DEPRECATED(feature_ordering);

  CHECK_UNVALIDATED();
}

// -----------------------------------------------------------------------------

void Validator::Process(const Set& config) {
  Set unvalidated(config);

  unvalidated.clear_name();
  MAYBE_PROCESS_FIELD(logging);
  MAYBE_PROCESS_FIELD(regularization);
  MAYBE_PROCESS_FIELD(regularization_div_sqrt_n);
  MAYBE_PROCESS_FIELD(regularization_mul_sqrt_n);
  MAYBE_PROCESS_FIELD(regularization_confidence);

  // inertia_factor
  if (config.has_inertia_factor() && config.inertia_factor() < 0)
    text_logger_->AddError("inertia_factor must be non-negative");
  unvalidated.clear_inertia_factor();

  // step_multiplier
  if (config.has_step_multiplier() && config.step_multiplier() < 1)
    text_logger_->AddError("step_multiplier must be at least 1");
  unvalidated.clear_step_multiplier();

  // lift_fraction
  for (double l : config.logged_lift_fraction()) {
    if (l <= 0) text_logger_->AddError(StrCat("lift must be positive: ", l));
    if (l > 1) text_logger_->AddError(StrCat("list must not exceed 1: ", l));
  }
  unvalidated.clear_logged_lift_fraction();


  MAYBE_PROCESS_FIELD(sgd_learning_rate_schedule);
  unvalidated.clear_allow_undo();
  unvalidated.clear_deterministic();
  unvalidated.clear_max_shard_size();

  PROCESS_DEPRECATED(thread_count);

  CHECK_UNVALIDATED();
}

void Validator::Process(const Set::Logging& config) {
  Set::Logging unvalidated(config);

  // log_timestamp
  unvalidated.clear_log_timestamp();

  // text_log_path
  if (config.has_text_log_path()) {
    if (!IsGlob(config.text_log_path()))
      text_logger_->AddError("Invalid path: " + config.text_log_path());
  }
  unvalidated.clear_text_log_path();

  // recordio_log_path
  if (config.has_recordio_log_path()) {
    if (!IsGlob(config.recordio_log_path()))
      text_logger_->AddError("Invalid path: " + config.recordio_log_path());
  }
  unvalidated.clear_recordio_log_path();

  unvalidated.clear_clear_log_files();

  CHECK_UNVALIDATED();
}

void Validator::Process(const Set::Regularization& config) {
  Set::Regularization unvalidated(config);

  // l1
  if (config.has_l1() && config.l1() < 0)
    text_logger_->AddError("l1 must be non-negative");
  unvalidated.clear_l1();

  // l2
  if (config.has_l2() && config.l2() < 0)
    text_logger_->AddError("l2 must be non-negative");
  unvalidated.clear_l2();

  // l1_at_weight_zero
  if (config.has_l1_at_weight_zero() && config.l1_at_weight_zero() < 0)
    text_logger_->AddError("l1_at_weight_zero must be non-negative");
  unvalidated.clear_l1_at_weight_zero();

  CHECK_UNVALIDATED();
}

void Validator::Process(const Set::SgdLearningRateSchedule& config) {
  Set::SgdLearningRateSchedule unvalidated(config);

  // start_learning_rate
  if (config.has_start_learning_rate() && config.start_learning_rate() < 0)
    text_logger_->AddError("start_learning_rate must be non-negative");
  unvalidated.clear_start_learning_rate();

  // decay_speed
  if (config.has_decay_speed() && config.decay_speed() < 0)
    text_logger_->AddError("decay_speed must be non-negative");
  unvalidated.clear_decay_speed();

  CHECK_UNVALIDATED();
}

void Validator::Process(const ReadData& config) {
  ReadData unvalidated(config);

  MAYBE_PROCESS_FIELD(data_reader);
  MAYBE_PROCESS_FIELD(set);

  if ((config.has_data_reader() +  //
       config.has_set() +          //
       0) != 1) {
    text_logger_->AddError(
        "Exactly one of the fields of ReadData must be set.\n" +
        config.DebugString());
  }

  CHECK_UNVALIDATED();
}

void Validator::Process(const ReadData::Set& config) {
  ReadData::Set unvalidated(config);

  // output_feature
  unvalidated.clear_output_feature();

  CHECK_UNVALIDATED();
}

void Validator::Process(const InitializeBias& config) {
  InitializeBias unvalidated(config);

  CHECK_UNVALIDATED();
}

void Validator::Process(const AddNewProductFeatures& config) {
  AddNewProductFeatures unvalidated(config);

  // feature_exploration
  PROCESS_FIELD(feature_exploration);

  CHECK_UNVALIDATED();
}

void Validator::Process(const PruneFeatures& config) {
  PruneFeatures unvalidated(config);

  // feature_pruning
  PROCESS_FIELD(feature_pruning);

  CHECK_UNVALIDATED();
}


void Validator::Process(const FitModelWeights& config) {
  FitModelWeights unvalidated(config);

  // iterations
  if (!config.has_iterations()) {
    text_logger_->AddError("Missing iterations.");
  } else if (config.iterations() < 0) {
    text_logger_->AddError("iterations must be non-negative");
  }
  unvalidated.clear_iterations();

  PROCESS_DEPRECATED(iterations_between_data_set_stats_computation);
  PROCESS_DEPRECATED(eval_auc);

  CHECK_UNVALIDATED();
}


void Validator::Process(const RunSgd& config) {
  RunSgd unvalidated(config);

  // iterations
  if (!config.has_iterations()) {
    text_logger_->AddError("Missing iterations.");
  } else if (config.iterations() < 0) {
    text_logger_->AddError("iterations must be non-negative");
  }
  unvalidated.clear_iterations();

  // only_new_features
  unvalidated.clear_only_new_features();

  CHECK_UNVALIDATED();
}

void Validator::Process(const Sgd::LearningRate::StoreTotalLoss& config) {
  Sgd::LearningRate::StoreTotalLoss unvalidated(config);

  CHECK_UNVALIDATED();
}

void Validator::Process(const Sgd::LearningRate::MaybeReduce& config) {
  Sgd::LearningRate::MaybeReduce unvalidated(config);

  // factor
  if (config.has_factor() && (config.factor() >= 0 || config.factor() <= 1))
    text_logger_->AddError("factor must be between 0 and 1");
  unvalidated.clear_factor();

  CHECK_UNVALIDATED();
}

void Validator::Process(const Sgd::LearningRate& config) {
  Sgd::LearningRate unvalidated(config);

  MAYBE_PROCESS_FIELD(store_total_loss);
  MAYBE_PROCESS_FIELD(maybe_reduce);

  if ((config.has_store_total_loss() +  //
       config.has_maybe_reduce() +      //
       0) != 1) {
    text_logger_->AddError(
        "Exactly one sub-command of LearningRate must be set.\n" +
        config.DebugString());
  }

  CHECK_UNVALIDATED();
}

void Validator::Process(const Sgd& config) {
  Sgd unvalidated(config);

  MAYBE_PROCESS_FIELD(learning_rate);

  CHECK_UNVALIDATED();
}

void Validator::Process(const EvaluateStats& config) {
  EvaluateStats unvalidated(config);

  // auc
  unvalidated.clear_auc();

  CHECK_UNVALIDATED();
}

void Validator::Process(const StoreModel& config) {
  StoreModel unvalidated(config);
  CHECK_UNVALIDATED();
}

void Validator::Process(const WriteModel::Set& config) {
  WriteModel::Set unvalidated(config);

  unvalidated.clear_select_best_stored();
  if (config.has_regularization_l0() && config.regularization_l0() < 0)
    text_logger_->AddError("regularization_l0 must be non-negative");
  unvalidated.clear_regularization_l0();
  unvalidated.clear_output_model_path();
  unvalidated.clear_format();

  CHECK_UNVALIDATED();
}

void Validator::Process(const WriteModel::Write& config) {
  WriteModel::Write unvalidated(config);

  CHECK_UNVALIDATED();
}

void Validator::Process(const WriteModel& config) {
  WriteModel unvalidated(config);

  if (config.has_set() + config.has_write() != 1) {
    text_logger_->AddError("Exactly one of {set, write} should be given");
  }

  MAYBE_PROCESS_FIELD(set);

  MAYBE_PROCESS_FIELD(write);

  CHECK_UNVALIDATED();
}

void Validator::Process(const GetModel& config) {
  GetModel unvalidated(config);

  CHECK_UNVALIDATED();
}

void Validator::Process(const ScoreRows::Set& config) {
  ScoreRows::Set unvalidated(config);

  if (config.output_fname().empty()) {
    text_logger_->AddError(
        "Path for scoring output should be specified and "
        "non-empty.");
  }
  unvalidated.clear_output_fname();
  if (!config.has_format() || config.format() == ScoreRows::UNKNOWN_FORMAT) {
    text_logger_->AddError("Format for scoring should be specified.");
  }
  unvalidated.clear_format();

  CHECK_UNVALIDATED();
}

void Validator::Process(const ScoreRows::WriteScores& config) {
  ScoreRows::WriteScores unvalidated(config);

  CHECK_UNVALIDATED();
}

void Validator::Process(const ScoreRows& config) {
  ScoreRows unvalidated(config);

  if (config.has_set() + config.has_write_scores() != 1) {
    text_logger_->AddError(
        "Exactly one of {set, write_scores} should be present");
  }

  MAYBE_PROCESS_FIELD(set);

  MAYBE_PROCESS_FIELD(write_scores);

  CHECK_UNVALIDATED();
}

void Validator::Process(const Quit& config) {
  Quit unvalidated(config);
  CHECK_UNVALIDATED();
}

void Validator::Process(const Repeat& config) {
  Repeat unvalidated(config);

  // repetitions
  if (config.has_repetitions()) {
    if (config.repetitions() < 0)
      text_logger_->AddError("Number of repetitions must be non-negative.");
  } else {
    text_logger_->AddError("Missing number of repetitions.");
  }
  unvalidated.clear_repetitions();

  // command
  for (const Command& command : config.command()) Process(command);
  unvalidated.clear_command();

  CHECK_UNVALIDATED();
}

void Validator::Process(const FromFile& config) {
  FromFile unvalidated(config);

  // path
  if (!config.has_path()) {
    text_logger_->AddError("Missing field: path");
  } else {
    if (!IsGlob(config.path())) {
      text_logger_->AddError("Invalid path: " + config.path());
    }
  }
  unvalidated.clear_path();

  CHECK_UNVALIDATED();
}


void Validator::Process(const Internal& config) {
  Internal unvalidated(config);

  MAYBE_PROCESS_FIELD(get_model);
  MAYBE_PROCESS_FIELD(log_detailed_stats);
  MAYBE_PROCESS_FIELD(log_dependees);
  MAYBE_PROCESS_FIELD(get_data);
  MAYBE_PROCESS_FIELD(get_scores);

  if ((config.has_get_model() +           //
       config.has_log_detailed_stats() +  //
       config.has_log_dependees() +       //
       config.has_get_data() +            //
       config.has_get_scores() +          //
       0) != 1) {
    text_logger_->AddError(
        "Exactly one of the fields of Internal must be set.\n" +
        config.DebugString());
  }

  CHECK_UNVALIDATED();
}

void Validator::Process(const Internal::GetModel& config) {
  Internal::GetModel unvalidated(config);
  CHECK_UNVALIDATED();
}

void Validator::Process(const Internal::LogDetailedStats& config) {
  Internal::LogDetailedStats unvalidated(config);
  CHECK_UNVALIDATED();
}

void Validator::Process(const Internal::LogDependees& config) {
  Internal::LogDependees unvalidated(config);
  CHECK_UNVALIDATED();
}

void Validator::Process(const Internal::GetData& config) {
  Internal::GetData unvalidated(config);
  CHECK_UNVALIDATED();
}

void Validator::Process(const Internal::GetScores& config) {
  Internal::GetScores unvalidated(config);
  CHECK_UNVALIDATED();
}

void Validator::Process(const Command& config) {
  Command unvalidated(config);

  MAYBE_PROCESS_FIELD(set);
  MAYBE_PROCESS_FIELD(read_data);
  MAYBE_PROCESS_FIELD(initialize_bias);
  MAYBE_PROCESS_FIELD(add_new_product_features);
  MAYBE_PROCESS_FIELD(prune_features);
  MAYBE_PROCESS_FIELD(fit_model_weights);
  MAYBE_PROCESS_FIELD(run_sgd);
  MAYBE_PROCESS_FIELD(sgd);
  MAYBE_PROCESS_FIELD(evaluate_stats);
  MAYBE_PROCESS_FIELD(store_model);
  MAYBE_PROCESS_FIELD(write_model);
  MAYBE_PROCESS_FIELD(get_model);
  PROCESS_DEPRECATED(quit);
  MAYBE_PROCESS_FIELD(repeat);
  MAYBE_PROCESS_FIELD(internal);
  MAYBE_PROCESS_FIELD(from_file);
  MAYBE_PROCESS_FIELD(command_list);
  MAYBE_PROCESS_FIELD(score_rows);

  if ((config.has_set() +                       //
       config.has_read_data() +                 //
       config.has_initialize_bias() +           //
       config.has_add_new_product_features() +  //
       config.has_prune_features() +            //
       config.has_fit_model_weights() +  //
       config.has_run_sgd() +            //
       config.has_sgd() +                //
       config.has_evaluate_stats() +     //
       config.has_store_model() +        //
       config.has_write_model() +        //
       config.has_get_model() +          //
       config.has_repeat() +             //
       config.has_internal() +           //
       config.has_from_file() +          //
       config.has_command_list() +       //
       config.has_score_rows() +  //
       0) != 1) {
    text_logger_->AddError(
        "Exactly one of the fields of Command must be set.\n" +
        config.DebugString());
  }

  CHECK_UNVALIDATED();
}

void Validator::Process(const CommandList& config) {
  CommandList unvalidated(config);

  PROCESS_REPEATED_FIELD(command);

  CHECK_UNVALIDATED();
}

#undef PROCESS_FIELD
#undef CHECK_UNVALIDATED

void TextLogger::AddWarning(const string& message) {
  messages_.push_back("Warning: " + message);
}

void TextLogger::AddError(const string& message) {
  messages_.push_back("Error: " + message);
  is_valid_ = false;
}


CommandList CommandListFromFlags(const string& flag_text_files,
                                 const string& flag_text) {
  using proto_util::ParseTextOrDie;
  Flag flag;
  for (const StringPiece& file :
       strings::Split(flag_text_files, ",", strings::SkipEmpty())) {
    string contents;
    QCHECK_OK(file::GetContents(file, &contents, file::Defaults()));
    flag.MergeFrom(ParseTextOrDie<Flag>(contents));
  }
  flag.MergeFrom(ParseTextOrDie<Flag>(flag_text));

  Validator::ValidateOrDie(flag);
  return flag.command_list();
}

}  // namespace config
}  // namespace sensei

