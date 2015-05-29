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
#ifndef SENSEI_OPTIMIZERS_H_
#define SENSEI_OPTIMIZERS_H_

#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
using std::vector;

#include "sensei/base/integral_types.h"
#include "sensei/base/logging.h"
#include "sensei/base/macros.h"
#include "sensei/common.h"
#include "sensei/config.pb.h"
#include "sensei/csr_matrix.h"
#include "sensei/data.h"
#include "sensei/feature_map.h"
#include "sensei/j_renumbering.h"
#include "sensei/log.pb.h"
#include "sensei/logger.h"
#include "sensei/model.h"
#include "sensei/range.h"
#include "sensei/row_extender.h"
#include "sensei/util/fixed_size_object_pool.h"
#include "sensei/util/to_callback.h"

// Notation consistent with notation used in papers is used within the
// algorithm implementations.
// i - index of data row
// j - index of variable
// x - data element
// w - weight


namespace sensei {

// -----------------------------------------------------------------------------
// Simple container for regularization parameters.
class Regularizations {
 public:
  const double L1(const uint64 rows_with_j, Double weight,
                  Double majorizer_a) const {
    double ret_l1 = regularization_.l1();
    double sqrt_of_rows_with_j = sqrt(rows_with_j + 1);
    double sqrt_of_majorizer_a_plus_epsilon = sqrt(majorizer_a) + kEpsilon;
    ret_l1 += regularization_div_sqrt_n_.l1() / sqrt_of_rows_with_j;
    ret_l1 += regularization_mul_sqrt_n_.l1() * sqrt_of_rows_with_j;
    ret_l1 +=
        regularization_confidence_.l1() / sqrt_of_majorizer_a_plus_epsilon;
    if (weight == 0.0) {
      ret_l1 += regularization_.l1_at_weight_zero();
      ret_l1 +=
          regularization_div_sqrt_n_.l1_at_weight_zero() / sqrt_of_rows_with_j;
      ret_l1 +=
          regularization_mul_sqrt_n_.l1_at_weight_zero() * sqrt_of_rows_with_j;
      ret_l1 += regularization_confidence_.l1_at_weight_zero() /
                sqrt_of_majorizer_a_plus_epsilon;
    }
    return ret_l1;
  }

  const double L2(const uint64 rows_with_j, Double majorizer_a) const {
    double ret_l2 = regularization_.l2();
    double sqrt_of_rows_with_j = sqrt(rows_with_j + 1);
    ret_l2 += regularization_div_sqrt_n_.l2() / sqrt_of_rows_with_j;
    ret_l2 += regularization_mul_sqrt_n_.l2() * sqrt_of_rows_with_j;
    ret_l2 += regularization_confidence_.l2() / (sqrt(majorizer_a) + kEpsilon);
    return ret_l2;
  }

  void SetRegularizationDivSqrtN(
      const config::Set::Regularization& regularization_div_sqrt_n) {
    SetFromProto(regularization_div_sqrt_n, &regularization_div_sqrt_n_);
  }

  void SetRegularization(const config::Set::Regularization& regularization) {
    SetFromProto(regularization, &regularization_);
  }

  void SetRegularizationMulSqrtN(
      const config::Set::Regularization& regularization_mul_sqrt_n) {
    SetFromProto(regularization_mul_sqrt_n, &regularization_mul_sqrt_n_);
  }

  void SetRegularizationConfidence(
      const config::Set::Regularization& regularization_confidence) {
    SetFromProto(regularization_confidence, &regularization_confidence_);
  }

  const bool IsNonStandard() const {
    return regularization_.l1_at_weight_zero() != 0 ||
           regularization_div_sqrt_n_.l1() != 0 ||
           regularization_div_sqrt_n_.l2() != 0 ||
           regularization_div_sqrt_n_.l1_at_weight_zero() != 0 ||
           regularization_mul_sqrt_n_.l1() != 0 ||
           regularization_mul_sqrt_n_.l2() != 0 ||
           regularization_mul_sqrt_n_.l1_at_weight_zero() != 0 ||
           regularization_confidence_.l1() != 0 ||
           regularization_confidence_.l2() != 0 ||
           regularization_confidence_.l1_at_weight_zero() != 0;
  }

  const config::Set::Regularization& regularization() const {
    return regularization_;
  }

 private:
  void SetFromProto(const config::Set::Regularization& pattern,
                    config::Set::Regularization* regularization) {
    if (pattern.has_l1()) regularization->set_l1(pattern.l1());
    if (pattern.has_l2()) regularization->set_l2(pattern.l2());
    if (pattern.has_l1_at_weight_zero())
      regularization->set_l1_at_weight_zero(pattern.l1_at_weight_zero());
  }

  // Private member variables.
  config::Set::Regularization regularization_;
  config::Set::Regularization regularization_div_sqrt_n_;
  config::Set::Regularization regularization_mul_sqrt_n_;
  config::Set::Regularization regularization_confidence_;
};

namespace optimizer {

// -----------------------------------------------------------------------------
// Represents one-dimensional quadratic majorizer function:
// f(w) = (a_/4 + L2) * w^2 + (b_ - a_*w0)/2 * w + L1*|w| + c.
// L1, L2, and w0 are stored externally and passed to relevant methods.
// c is ommited from dim 1 computations and storage.
class Dim1Majorizer {
 public:
  Dim1Majorizer() : a_(0), b_(0) {}

  Dim1Majorizer(Double a, Double b) : a_(a), b_(b) {}

  void SetZero() { a_ = b_ = 0; }

  void Add(const Dim1Majorizer& that) {
    a_ += that.a_;
    b_ += that.b_;
  }

  // See class comment about w0.
  // Inertia is added before l1 regularization.
  Double GetMinimum(const Regularizations& regularizations, Double w0,
                    Double inertia, Double step_multiplier,
                    const uint64 rows_with_j) const {
    // TODO(lew): Make this code branchless.
    Double aj = a_ + regularizations.L2(rows_with_j, a_) * 4.0;
    // We make a simplyfing assumption here that aj in inertia is the same.
    // TODO(lew): Try using inertia carrying both a and b.
    Double bj = a_ * w0 - step_multiplier * b_ + inertia * aj;
    if (bj > 0.0) {
      bj = std::max<Double>(0.0,
                            bj - regularizations.L1(rows_with_j, w0, a_) * 2.0);
    } else {
      bj = std::min<Double>(0.0,
                            bj + regularizations.L1(rows_with_j, w0, a_) * 2.0);
    }
    if (aj == 0.0) {
      DCHECK_EQ(bj, 0.0);
      return 0;
    }
    return bj / aj;
  }

  Double GetPrecision(const Regularizations& regularizations,
                      const uint64 rows_with_j, Double weight) const {
    return a_ / 2.0 + regularizations.L2(rows_with_j, a_) * 2.0;
  }

  // Returns f'(w0).
  Double DerivativeAtW0() const { return b_ / 2.0; }


  Double a() const { return a_; }

 private:
  Double a_;
  Double b_;  // See class comment.

  // log_loss is stored in Majorizer and it can be used to calculate c.
};
static_assert(sizeof(Dim1Majorizer) == 16, "");

// -----------------------------------------------------------------------------
// Sum of Dim1Majorizer for each coordinate. constant c is ignored.
class Majorizer {
 public:
  typedef FixedSizeObjectPool<Majorizer> Pool;

  Majorizer() : log_loss_(0), cpu_operation_count_flat_materialization_(0) {}

  explicit Majorizer(uint32 feature_count)
      : of_j_(feature_count),
        log_loss_(0),
        cpu_operation_count_flat_materialization_(0) {}

  void AddToCpuOperationCountFlatMaterialization(const uint64 delta) {
    cpu_operation_count_flat_materialization_ += delta;
  }

  uint64 GetCpuOperationCountFlatMaterialization() const {
    return cpu_operation_count_flat_materialization_;
  }

  void SetSize(uint32 size) { MildResize(size, &of_j_); }

  void SetZero() {
    for (Dim1Majorizer& d1m : of_j_) d1m.SetZero();
    log_loss_ = 0;
    cpu_operation_count_flat_materialization_ = 0;
  }

  Double GetRegularizationLoss(const Regularizations& regularizations,
                               const Model& model,
                               const data::ShardSet& shard) const {
    Double loss = 0.0;
    for (uint32 j : Range(model.w.size())) {
      Double w = model.w[j];
      uint64 rows_with_j = shard.GetStats().GetRowCountWithJPresent(j);
      loss += regularizations.L1(rows_with_j, w, of_j_[j].a()) * fabs(w);
      loss += regularizations.L2(rows_with_j, of_j_[j].a()) * w * w;
    }
    return loss;
  }

  void UpdateMinimum(const Regularizations& regularizations,
                     Double inertia_factor, Double step_multiplier,
                     const data::ShardSet& shard_set, bool allow_undo,
                     Model* model, Logger* logger) const {
    CHECK_EQ(of_j_.size(), model->GetSize());
    CHECK(model->synced_with_weights);
    model->synced_with_weights = false;
    model->iteration_no++;

    logs::Line log_line;
    logs::GradBoostUpdateMinimum* log =
        log_line.mutable_grad_boost_update_minimum();

    Double total_loss =
        log_loss_ + GetRegularizationLoss(regularizations, *model, shard_set);
    // CHECK_EQ(total_loss, model_->total_loss);

    if (allow_undo && total_loss > model->prev_total_loss) {
      LOG(INFO) << "Undo because of LogLoss increase:" << model->prev_total_loss
                << " -> " << total_loss;
      log->set_undo_iteration(true);
      model->prev_total_loss = kInfinity;
      for (uint32 j : Range(of_j_.size())) {
        model->w[j] -= model->delta_w[j];
        model->delta_w[j] = 0;
        // TODO(lew): Other fields of model do not get reset.
        //            This might be a problem for any algorithm using it.
      }
      logger->AddToLogs(log_line);
      return;
    }
    CHECK_EQ(of_j_.size(), model->GetSize());
    // The scalar product of (model->loss_derivative) and (new_w-model->w),
    // where new_w is the result of coordinate-wise GetMinimum() calls.
    Double dot_loss_derivative_vs_delta_weight = 0;
    Double log_loss_derivative_squared = 0;
    Double delta_w_squared = 0;
    for (uint32 j : Range(of_j_.size())) {
      // Majorizer has been created using model->w[j], which hasn't changed
      // since then.
      uint64 rows_with_j = shard_set.GetStats().GetRowCountWithJPresent(j);
      Double w0 = model->w[j];
      Double new_w_without_inertia = of_j_[j].GetMinimum(
          regularizations, model->w[j], 0, step_multiplier, rows_with_j);
      Double new_w = of_j_[j].GetMinimum(regularizations, model->w[j],
                                         inertia_factor * model->delta_w[j],
                                         step_multiplier, rows_with_j);
      model->precision[j] =
          of_j_[j].GetPrecision(regularizations, rows_with_j, model->w[j]);
      model->delta_w[j] = new_w - model->w[j];
      model->w[j] = new_w;
      dot_loss_derivative_vs_delta_weight +=
          model->loss_derivative[j] * model->delta_w[j];
      log_loss_derivative_squared +=
          model->loss_derivative[j] * model->loss_derivative[j];
      delta_w_squared += model->delta_w[j] * model->delta_w[j];
    }
    // TODO(lew): Add product (angle), L2(delta_w) and L2(dw) to logging.
    // If inertia was too big, undo the step.
    // TODO(lew,witoldjarnicki): A step without intertia instead of revert.
    log->set_dot_loss_derivative_vs_delta_weight(
        dot_loss_derivative_vs_delta_weight);
    double denominator = sqrt(log_loss_derivative_squared * delta_w_squared);
    if (denominator != 0.0) {
      log->set_cos_angle_loss_derivative_vs_delta_weight(
          dot_loss_derivative_vs_delta_weight / denominator);
    }
    // The gradient of logloss computed at the old w is guaranteed to point in
    // the oposite direction to the w vector change, assuming there is no
    // inertia.  If the product happens to be positive, this is a good
    // heuristic indication that inertia is doing more harm than good and we
    // should get rid of it.
    if (dot_loss_derivative_vs_delta_weight > 0) {
      // TODO(lew): Accept points with smaller log loss.
      log->set_restart_iteration(true);
      LOG(INFO) << "Restart: " << dot_loss_derivative_vs_delta_weight;
      for (uint32 j : Range(of_j_.size())) {
        Double w0 = model->w[j] - model->delta_w[j];  // TODO(lew): Numerics.
        model->w[j] = w0;
        model->delta_w[j] = 0;
        // We don't undo feature_removal_log_likelihood_loss, because it was
        // computed without inertia.
      }
    }
    logger->AddToLogs(log_line);
  }

  void Add(uint32 j, const Dim1Majorizer& dim_1_majorizer) {
    CHECK_LT(j, of_j_.size());
    of_j_[j].Add(dim_1_majorizer);
  }

  void SumAndAssign(const Range& range, const vector<Majorizer*>* majorizers) {
    // TODO(lew, witoldjarnicki): Test loop order.
    for (uint32 j : range) {
      of_j_[j].SetZero();
      for (const Majorizer* majorizer : *majorizers)
        of_j_[j].Add(majorizer->of_j_[j]);
    }
  }

  void SumStatsAndAssign(const vector<Majorizer*>* majorizers) {
    log_loss_ = 0;
    cpu_operation_count_flat_materialization_ = 0;
    for (const Majorizer* majorizer : *majorizers) {
      log_loss_ += majorizer->log_loss_;
      cpu_operation_count_flat_materialization_ +=
          majorizer->cpu_operation_count_flat_materialization_;
    }
  }

  void AddLogLoss(Double delta_log_loss) { log_loss_ += delta_log_loss; }

  Double GetLogLoss() const { return log_loss_; }

  // Returns all partial derivatives of LogLoss at w.
  // Regularization loss is included.
  vector<Double> GetLogLossDerivativeAt(const Model& model,
                                        const Regularizations& regularizations,
                                        const data::ShardSet& shard_set) const {
    CHECK_EQ(of_j_.size(), model.w.size());
    vector<Double> log_loss_derivative(of_j_.size());
    for (uint32 j : Range(model.w.size())) {
      Double lld_j = of_j_[j].DerivativeAtW0();
      const uint64 rows_with_j =
          shard_set.GetStats().GetRowCountWithJPresent(j);
      lld_j += 2 * regularizations.L2(rows_with_j, of_j_[j].a()) * model.w[j];
      if (model.w[j] == 0 &&
          lld_j - regularizations.L1(rows_with_j, model.w[j], of_j_[j].a()) <=
              0 &&
          lld_j + regularizations.L1(rows_with_j, model.w[j], of_j_[j].a()) >=
              0) {
        lld_j = 0;
      } else {
        lld_j += regularizations.L1(rows_with_j, model.w[j], of_j_[j].a()) *
                 Sign(model.w[j]);
      }
      log_loss_derivative[j] = lld_j;
    }
    return log_loss_derivative;
  }

  void RemoveAndRenumberJs(const JRenumbering& j_renumbering) {
    j_renumbering.RenumberIndicies(&of_j_);
  }

 private:
  vector<Dim1Majorizer> of_j_;
  Double log_loss_;
  uint64 cpu_operation_count_flat_materialization_;
};

// -----------------------------------------------------------------------------
// GradBoost optimizer implementation.
// The class it not thread safe.
class GradBoost {
 public:
  // TODO(lew): Find a better place for defaults.
  // Model must live longer than GradBoost.
  GradBoost(const ProductMap& product_map, Model* model,
            const Regularizations& regularizations, Logger* logger)
      : logger_(logger),
        data_(nullptr),
        model_(CHECK_NOTNULL(model)),
        regularizations_(regularizations),
        inertia_factor_(-1.0),
        step_multiplier_(-1.0),
        allow_undo_(false),
        product_map_(product_map),
        majorizer_pool_(concurrency::kThreadCount, 0,
                        util::functional::ToPermanentCallback([this] {
                          return new Majorizer(model_->GetSize());
                        })),
        deterministic_(false) {}

  ~GradBoost() {}

  void SetDeterministic(bool deterministic) { deterministic_ = deterministic; }

  void SetInertiaFactor(double inertia) { inertia_factor_ = inertia; }

  void SetStepMultiplier(double multiplier) { step_multiplier_ = multiplier; }

  void SetAllowUndo(bool allow_undo) { allow_undo_ = allow_undo; }

  void SetData(const data::Data& data) { data_ = &data; }

  // TODO(lew): Rename this function.
  void SetSize(uint32 feature_count) {
    training_majorizer_.SetSize(feature_count);
    holdout_majorizer_.SetSize(feature_count);
  }

  const logs::Iteration& GetLastIterationLog() const { return iteration_log_; }

  void SyncModelWithWeights() {
    CHECK_EQ(0, majorizer_pool_.NumGrabbed());
    if (model_->synced_with_weights) return;

    ProcessShardSet(data_->GetTraining(), &model_->training);
    ParallelProcessColumns(&training_majorizer_);
    ProcessShardSet(data_->GetHoldout(), &model_->holdout);
    ParallelProcessColumns(&holdout_majorizer_);

    Double total_loss = training_majorizer_.GetLogLoss() +
                        training_majorizer_.GetRegularizationLoss(
                            regularizations_, *model_, data_->GetTraining());
    model_->SetTotalLoss(total_loss);
    model_->synced_with_weights = true;

    AddIterationLog();
    CHECK_EQ(0, majorizer_pool_.NumGrabbed());
  }

  void MakeOnePass() {
    CHECK_NOTNULL(data_);
    CHECK_GE(inertia_factor_, 0.0) << "inertia_factor not set.";
    CHECK_GE(step_multiplier_, 0.0) << "step_multiplier not set.";

    SyncModelWithWeights();
    training_majorizer_.UpdateMinimum(regularizations_, inertia_factor_,
                                      step_multiplier_, data_->GetTraining(),
                                      allow_undo_, model_, logger_);
    SyncModelWithWeights();
  }

  void ProcessShardSet(const data::ShardSet& shard_set,
                       Model::PerShard* per_shard) {
    uint row_offset = 0;
    vector<uint32> row_offsets;
    for (const data::Shard& s : shard_set.GetShards()) {
      row_offsets.push_back(row_offset);
      row_offset += s.RowCount();
    }

    concurrency::ParFor(Range(row_offsets.size()), deterministic_,
                        [&shard_set, &per_shard, &row_offsets, this](uint32 i) {
                          ProcessRows(&shard_set.GetShards()[i],
                                      &majorizer_pool_,
                                      per_shard->wxs.begin() + row_offsets[i]);
                        });
  }

  void RemoveAndRenumberJs(const JRenumbering& j_renumbering) {
    CHECK_EQ(0, majorizer_pool_.NumGrabbed());
    if (j_renumbering.j_to_new_j().empty()) return;
    training_majorizer_.RemoveAndRenumberJs(j_renumbering);
    holdout_majorizer_.RemoveAndRenumberJs(j_renumbering);
    model_->RemoveAndRenumberJs(j_renumbering);
  }

 private:
  // We use const* instead of const&, because callbacks copy const references.
  void ProcessRows(const data::Shard* shard, Majorizer::Pool* pool,
                   vector<Double>::iterator wxs) {
    Majorizer* majorizer = pool->Get();

    RowExtender row_extender(&data_->GetDependees());
    for (uint32 i : Range(shard->RowCount())) {
      shard->ResetExtender(i, &row_extender);
      majorizer->AddToCpuOperationCountFlatMaterialization(
          row_extender.CpuOperationCountFlatMaterialization());
      ProcessOneRow(row_extender, majorizer, &(*(wxs + i)));
    }

    pool->Release(majorizer);
  }

  void ProcessOneRow(const RowExtender& row_extender,
                     Majorizer* thread_majorizer, Double* wx) const {
    Double row_l2_squared = row_extender.L2SquaredNorm();
    *wx = row_extender.Dot(model_->w);
    Double wxy = *wx * row_extender.GetY();
    Double exp_wxy = exp(wxy);
    Double A =
        (fabs(wxy) < 1e-10) ? 0.5 : (exp_wxy - 1) / ((exp_wxy + 1) * wxy);
    Double A_wxy_1 = (A * wxy - 1) * row_extender.GetY();
    thread_majorizer->AddLogLoss(log(1 + exp_wxy) - wxy);
    Double a = A * row_l2_squared;
    for (uint32 j : row_extender.SparseBool())
      thread_majorizer->Add(j, Dim1Majorizer(a, A_wxy_1));
  }

  void ParallelProcessColumns(Majorizer* majorizer) {
    vector<Majorizer*> thread_majorizers;
    while (majorizer_pool_.NumAvailable() > 0)
      thread_majorizers.push_back(majorizer_pool_.Get());

    concurrency::ParFor(
        Range(model_->w.size()).SplitEvenly(concurrency::kThreadCount),
        [majorizer, &thread_majorizers](const Range& range) {
          majorizer->SumAndAssign(range, &thread_majorizers);
        });
    majorizer->SumStatsAndAssign(&thread_majorizers);

    for (Majorizer* m : thread_majorizers) majorizer_pool_.Retire(m);
  }

  void AddIterationLog() {
    CHECK(model_->synced_with_weights);
    iteration_log_.Clear();
    iteration_log_.set_index(model_->iteration_no);
    iteration_log_.set_cpu_operation_count_flat_materialization(
        training_majorizer_.GetCpuOperationCountFlatMaterialization() +
        holdout_majorizer_.GetCpuOperationCountFlatMaterialization());
    FillVectorStats(model_->w, iteration_log_.mutable_weight_stats());
    FillVectorStats(model_->delta_w,
                    iteration_log_.mutable_delta_weight_stats());
    FillDataSetStats(
        data_->GetTraining(), training_majorizer_, &model_->training,
        iteration_log_.mutable_training_data_stats(), &model_->loss_derivative);
    if (data_->GetHoldout().GetStats().RowCount() > 0) {
      FillDataSetStats(data_->GetHoldout(), holdout_majorizer_,
                       &model_->holdout,
                       iteration_log_.mutable_holdout_data_stats(), nullptr);
    }
    // TODO(lew): More regularization stats.
    iteration_log_.mutable_regularization_stats()->set_loss(
        training_majorizer_.GetRegularizationLoss(regularizations_, *model_,
                                                  data_->GetTraining()));
    iteration_log_.set_cpu_operation_count_deep_materialization(
        CpuOperationCountDeepMaterialization());
    iteration_log_.set_prev_total_loss(model_->prev_total_loss);
    iteration_log_.set_total_loss(model_->total_loss);

    logs::Line log_line;
    *log_line.mutable_iteration() = iteration_log_;
    logger_->AddToLogs(log_line);
  }

  uint64 CpuOperationCountDeepMaterialization() const {
    vector<uint64> transitive_children(model_->GetSize(), 0);
    for (uint32 j = model_->GetSize() - 1; j != -1; --j) {
      for (uint32 j_child : data_->GetDependees().GetRow(j)) {
        CHECK_GT(j_child, j);
        transitive_children[j] += 1 + transitive_children[j_child];
      }
    }

    uint64 cpu_operation_count_deep_materialization = 0;
    for (uint32 j : Range(model_->GetSize())) {
      cpu_operation_count_deep_materialization +=
          transitive_children[j] * data_->XjBoolCountOfJ(j);
    }
    return cpu_operation_count_deep_materialization;
  }

  void FillDataSetStats(const data::ShardSet& data, const Majorizer& majorizer,
                        Model::PerShard* per_shard,
                        logs::DataSetStats* data_set_stats,
                        vector<Double>* ret_dloss) const {
    data_set_stats->set_size(data.GetStats().RowCount());
    data_set_stats->set_loss(majorizer.GetLogLoss());
    // TODO(lew): Remove regularization_ from here.
    vector<Double> dloss =
        majorizer.GetLogLossDerivativeAt(*model_, regularizations_, data);
    FillVectorStats(dloss, data_set_stats->mutable_dloss());
    if (ret_dloss != nullptr) *ret_dloss = std::move(dloss);
  }

  static void FillVectorStats(const vector<Double>& v,
                              logs::VectorStats* stats) {
    stats->set_size(v.size());
    stats->set_l1(L1Norm(v));
    stats->set_l2(L2NormSquared(v));
    stats->set_nonzero_count(NonZeroCount(v));
  }

  // Private member variables.
  Logger* logger_;

  const data::Data* data_;

  Model* model_;

  const Regularizations& regularizations_;

  Double inertia_factor_;
  Double step_multiplier_;
  bool allow_undo_;

  const ProductMap& product_map_;

  Majorizer::Pool majorizer_pool_;

  // Majorizer's are tangent at w to logloss of subsets of data.
  // In the batch variant they are computed at the same w so the final
  // majorizer (sum of in-thread majorizers)
  // is tangent at w to the summed log loss function of the whole data.
  // Sum of row majorizers.
  Majorizer training_majorizer_;
  Majorizer holdout_majorizer_;

  logs::Iteration iteration_log_;
  bool deterministic_;

  DISALLOW_COPY_AND_ASSIGN(GradBoost);
};

}  // namespace optimizer
}  // namespace sensei


#endif  // SENSEI_OPTIMIZERS_H_
