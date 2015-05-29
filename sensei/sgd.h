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
#ifndef SENSEI_SGD_H_
#define SENSEI_SGD_H_

#include <math.h>
#include <atomic>
#include <memory>
#include <string>
using std::string;
#include <vector>
using std::vector;

#include "sensei/base/integral_types.h"
#include "sensei/base/logging.h"
#include "sensei/common.h"
#include "sensei/config.pb.h"
#include "sensei/data.h"
#include "sensei/feature_map.h"
#include "sensei/log.pb.h"
#include "sensei/logger.h"
#include "sensei/model.h"
#include "sensei/optimizers.h"
#include "sensei/range.h"
#include "sensei/row_extender.h"
#include "sensei/util/fixed_size_object_pool.h"


namespace sensei {
namespace optimizer {

class Sgd {
 public:
  struct PerThread {
   public:
    typedef FixedSizeObjectPool<PerThread> Pool;
  };

  enum TrainingMode { ALL_FEATURES, NEW_FEATURES };

  Sgd(const ProductMap& product_map, Model* model,
      const Regularizations& regularizations, Logger* logger)
      : logger_(CHECK_NOTNULL(logger)),
        per_thread_pool_(concurrency::kThreadCount, 0),
        weights_owner_(nullptr),
        data_(nullptr),
        product_map_(product_map),
        model_(CHECK_NOTNULL(model)),
        prev_total_loss_(kInfinity),
        regularizations_(regularizations),
        training_mode_(ALL_FEATURES),
        deterministic_(false) {
    learning_rate_counter_.store(0, std::memory_order_relaxed);
  }

  void SetData(const data::Data& data) { data_ = &data; }

  void SetTrainingMode(TrainingMode new_training_mode) {
    training_mode_ = new_training_mode;
  }

  void SetDeterministic(bool deterministic) { deterministic_ = deterministic; }

  void SetLearningRateSchedule(config::Set::SgdLearningRateSchedule schedule) {
    if (schedule.has_start_learning_rate()) {
      learning_rate_.set_start_learning_rate(schedule.start_learning_rate());
      learning_rate_counter_.store(0, std::memory_order_relaxed);
    }
    if (schedule.has_decay_speed()) {
      learning_rate_.set_decay_speed(schedule.decay_speed());
    }
  }

  void RunCommand(const config::Sgd& command) {
    if (command.has_learning_rate()) {
      CHECK(model_->synced_with_weights);
      if (command.learning_rate().has_store_total_loss()) {
        prev_total_loss_ = model_->total_loss;
        return;
      }

      if (command.learning_rate().has_maybe_reduce()) {
        CHECK_NE(prev_total_loss_, kInfinity);

        logs::Line log_line;
        logs::Sgd::LearningRate* learning_rate_log =
            log_line.mutable_sgd()->mutable_learning_rate();
        logs::Sgd::LearningRate::MaybeReduce* maybe_reduce_log =
            learning_rate_log->mutable_maybe_reduce();
        maybe_reduce_log->set_previous_total_loss(prev_total_loss_);
        maybe_reduce_log->set_current_total_loss(model_->total_loss);

        if (prev_total_loss_ < model_->total_loss) {
          DecreaseLearningRate(command.learning_rate().maybe_reduce().factor(),
                               maybe_reduce_log);
        }
        logger_->AddToLogs(log_line);
        return;
      }
      LOG(FATAL) << "Unknown Sgd command: " << command.DebugString();
    }
  }

  void DecreaseLearningRate(Double factor,
                            logs::Sgd::LearningRate::MaybeReduce* log) {
    Double previous_learning_rate = learning_rate_.start_learning_rate();
    Double new_learning_rate = previous_learning_rate * factor;
    learning_rate_.set_start_learning_rate(new_learning_rate);

    log->set_previous_learning_rate(previous_learning_rate);
    log->set_current_learning_rate(new_learning_rate);
  }

  void SetSize(uint32 size) {
    weights_owner_.reset(new std::atomic<Double>[size]);
    weights_ = AtomicDoubleSlice(weights_owner_.get(), size);
  }

  bool IsTrainingValid() const {
    if (!learning_rate_.has_start_learning_rate()) {
      LOG(ERROR) << "start_learning_rate has not been set";
      return false;
    }
    if (data_->GetTraining().GetStats().RowCount() == 0) {
      LOG(ERROR) << "Missing data";
      return false;
    }
    return true;
  }

  void MakeOnePass() {
    CHECK(!regularizations_.IsNonStandard())
        << "SGD does not support adaptive OR zero regularization yet.";
    CHECK(model_->w.size() == weights_.length());

    for (uint32 i : Range(weights_.length())) {
      weights_[i].store(model_->w[i], std::memory_order_relaxed);
    }

    CHECK_EQ(0, per_thread_pool_.NumGrabbed());
    concurrency::ParFor(
        data_->GetTraining().GetShards(), deterministic_,
        [this](const data::Shard& shard) { ProcessShard(shard); });
    CHECK_EQ(0, per_thread_pool_.NumGrabbed());
    uint32 processed_rows_count =
        learning_rate_counter_.load(std::memory_order_relaxed);
    CHECK_EQ(processed_rows_count % data_->GetTraining().GetStats().RowCount(),
             0);
    UpdateModel();
    model_->synced_with_weights = false;
    model_->iteration_no++;
  }

 private:
  void ProcessShard(const data::Shard& shard) {
    PerThread* per_thread = per_thread_pool_.Get();
    RowExtender row_extender(&data_->GetDependees());
    for (uint32 i : Range(shard.RowCount())) {
      shard.ResetExtender(i, &row_extender);
      ProcessOneRow(row_extender);
      learning_rate_counter_.fetch_add(1, std::memory_order_relaxed);
    }
    ApplyRegularization();
    per_thread_pool_.Release(per_thread);
  }

  static void AddToAtomic(Double delta, std::atomic<Double>* atomic) {
    Double old_value = 0;
    Double new_value = 0;
    do {
      old_value = atomic->load(std::memory_order_relaxed);
      new_value = old_value + delta;
    } while (!atomic->compare_exchange_weak(old_value, new_value,
                                            std::memory_order_relaxed));
  }

  void ProcessOneRow(const RowExtender& row_extender) {
    double learning_rate = GetLearningRate();
    double y = row_extender.GetY();
    double sigmoid = 1 / (1 + exp(row_extender.Dot(weights_) * y));
    double delta = learning_rate * y * sigmoid;

    const JsSlice row = row_extender.SparseBool();
    for (const uint32& j : row) {
      if (training_mode_ == NEW_FEATURES && !model_->IsFeatureNew(j)) continue;
      AddToAtomic(delta, &weights_[j]);
    }
  }

  void ApplyRegularization() {
    Double learning_rate =
        GetLearningRate() / data_->GetTraining().GetShards().size();
    const double l1 = regularizations_.regularization().l1();
    const double l2 = regularizations_.regularization().l2();
    for (uint32 j : Range(model_->w.size())) {
      if (training_mode_ == NEW_FEATURES && !model_->IsFeatureNew(j)) continue;

      double new_value = 0;
      double old_value = 0;
      do {
        old_value = weights_[j].load(std::memory_order_relaxed);
        double delta =
            (l1 * Sign(old_value) + 2 * l2 * old_value) * learning_rate;
        new_value = old_value - delta;
        if (Sign(old_value) != Sign(new_value)) new_value = 0;
      } while (!weights_[j].compare_exchange_weak(old_value, new_value,
                                                  std::memory_order_relaxed));
    }
  }

  void UpdateModel() {
    for (uint32 j : Range(model_->w.size())) {
      model_->w[j] = weights_[j].load(std::memory_order_relaxed);
    }
  }

  double GetLearningRate() const {
    double start_learning_rate = learning_rate_.start_learning_rate();
    double decay_speed = learning_rate_.decay_speed();
    double current_count =
        learning_rate_counter_.load(std::memory_order_relaxed);
    double iteration_progress =
        current_count / data_->GetTraining().GetStats().RowCount();
    double learning_rate =
        start_learning_rate / (1 + decay_speed * iteration_progress);
    return learning_rate;
  }

  // Private member variables.
  Logger* logger_;
  PerThread::Pool per_thread_pool_;
  AtomicDoubleSlice weights_;
  std::unique_ptr<std::atomic<Double> []> weights_owner_;
  const data::Data* data_;
  const ProductMap& product_map_;
  Model* model_;
  Double prev_total_loss_;
  const Regularizations& regularizations_;
  config::Set::SgdLearningRateSchedule learning_rate_;
  // Number of ProcessOneRow calls, changing learning_rate resets it.
  std::atomic<uint32> learning_rate_counter_;
  TrainingMode training_mode_;
  bool deterministic_;
};

}  // namespace optimizer
}  // namespace sensei


#endif  // SENSEI_SGD_H_
