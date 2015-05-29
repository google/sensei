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
#ifndef SENSEI_DATA_H_
#define SENSEI_DATA_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
using std::string;
#include <vector>
using std::vector;

#include "sensei/base/integral_types.h"
#include "sensei/base/logging.h"
#include "sensei/base/macros.h"
#include "sensei/base/mutex.h"
#include "sensei/base/thread_annotations.h"
#include "sensei/common.h"
#include "sensei/csr_matrix.h"
#include "sensei/feature_map.h"
#include "sensei/internal.pb.h"
#include "sensei/j_renumbering.h"
#include "sensei/range.h"
#include "sensei/row_extender.h"
#include "sensei/strings/strutil.h"
#include "sensei/util/fixed_size_object_pool.h"
#include "sensei/util/to_callback.h"

// 'J' (or 'j' as variable name) is Sensei-wide name of uint32 that is
// an entry in data matrix. It is an index of data matrix's column.
// There are multiple 'maps' from J to X that are implemented as vectors.
// This is possible since J span over dense interval of uint32.


namespace sensei {
namespace data {

// -----------------------------------------------------------------------------
class Stats {
 public:
  typedef FixedSizeObjectPool<Stats> Pool;

  explicit Stats(uint32 size) { Reset(size); }

  uint32 Size() const {
    CHECK_EQ(positive_.size(), hash_.size());
    CHECK_EQ(negative_.size(), hash_.size());
    return hash_.size();
  }

  void Reset(uint32 new_size) {
    positive_row_count_ = 0;
    negative_row_count_ = 0;
    positive_ = vector<uint64>(new_size, 0);
    negative_ = vector<uint64>(new_size, 0);
    hash_ = vector<uint64>(new_size, 0);
    positive_.shrink_to_fit();
    negative_.shrink_to_fit();
    hash_.shrink_to_fit();
  }

  void Add(const Stats& src) {
    CHECK_EQ(src.Size(), Size());
    positive_row_count_ += src.positive_row_count_;
    negative_row_count_ += src.negative_row_count_;
    for (uint32 j : Range(src.Size())) {
      positive_[j] += src.positive_[j];
      negative_[j] += src.negative_[j];
      hash_[j] ^= src.hash_[j];
    }
  }

  // Methods with logic actually implemented here.
  struct CorrelationTable {
    uint64 n;
    uint64 n_y[2];
    uint64 n_x[2];
    uint64 n_xy[2][2];

    double MutualInformation() {
      DCHECK_GT(n, 0);
      double ret = 0;
      for (uint32 x : Range(2)) {
        for (uint32 y : Range(2)) {
          double p_xy = static_cast<double>(n_xy[x][y]) / n;
          double p_x = static_cast<double>(n_x[x]) / n;
          double p_y = static_cast<double>(n_y[y]) / n;
          if (p_xy != 0) ret += p_xy * log(p_xy / (p_x * p_y));
        }
      }
      return ret;
    }

    double PhiCoefficient() {
      double den = n_x[0] * n_x[1] * n_y[0] * n_y[1];
      if (den == 0) return 0;
      double nom = n_xy[1][1] * n_xy[0][0] - n_xy[1][0] * n_xy[0][1];
      return nom / sqrt(den);
    }
  };

  CorrelationTable GetCorrelationTable(const uint32 j) const {
    DCHECK_LT(j, negative_.size());
    CorrelationTable ret;
    const uint64 neg = negative_[j];  // Y != 1, feature present.
    const uint64 pos = positive_[j];  // Y == 1, feature present.
    ret.n_xy[0][0] = negative_row_count_ - neg;
    ret.n_xy[0][1] = positive_row_count_ - pos;
    ret.n_xy[1][0] = neg;
    ret.n_xy[1][1] = pos;
    for (uint32 x : Range(2)) ret.n_x[x] = ret.n_xy[x][0] + ret.n_xy[x][1];
    for (uint32 y : Range(2)) ret.n_y[y] = ret.n_xy[0][y] + ret.n_xy[1][y];
    ret.n = ret.n_x[0] + ret.n_x[1];
    return ret;
  }

  const uint64 GetRowCountWithJPresent(const uint32 j) const {
    return Negative(j) + Positive(j);
  }

  double LogOdds(const uint32 j) const {
    double pos = Positive(j);
    double neg = Negative(j);
    if (pos == 0.0) pos = 1.0;
    if (neg == 0.0) neg = 1.0;
    return log(pos / neg);
  }

  uint64 MaterializedXjBoolCount() const {
    uint64 ret = 0;
    for (const uint64 count : positive_) ret += count;
    for (const uint64 count : negative_) ret += count;
    return ret;
  }

  uint64 Negative(const uint32 j) const {
    CHECK_LT(j, negative_.size());
    return negative_[j];
  }

  uint64 Positive(const uint32 j) const {
    CHECK_LT(j, positive_.size());
    return positive_[j];
  }

  uint64 RowCount(const uint32 j) const { return Positive(j) + Negative(j); }

  uint64 Hash(const uint32 j) const {
    CHECK_LT(j, hash_.size());
    return hash_[j];
  }

  void RemoveAndRenumberJs(const JRenumbering& j_renumbering) {
    j_renumbering.RenumberIndicies(&positive_);
    j_renumbering.RenumberIndicies(&negative_);
    j_renumbering.RenumberIndicies(&hash_);
  }

  uint64 RowCount() const { return positive_row_count_ + negative_row_count_; }

  void AddRow(const RowExtender& row) {
    vector<uint64>* stats;
    if (row.GetY() == 1) {
      stats = &positive_;
      ++positive_row_count_;
    } else {
      stats = &negative_;
      ++negative_row_count_;
    }
    for (uint32 j : row.SparseBool()) {
      CHECK_LT(j, stats->size());
      ++(*stats)[j];
      hash_[j] ^= Hash64NumWithSeed(row.GetId(), 0);
    }
  }

  uint64 XjBoolCountOfJ(const uint32 j) const {
    return Negative(j) + Positive(j);
  }

  internal::Data::Stats ToInternalProto() const {
    internal::Data::Stats ret;
    ret.set_positive_row_count(positive_row_count_);
    ret.set_negative_row_count(negative_row_count_);
    for (uint32 j : Range(Size())) *ret.add_j_stat() = ToInternalProtoOfJ(j);
    return ret;
  }

  internal::Data::Stats::JStat ToInternalProtoOfJ(uint32 j) const {
    internal::Data::Stats::JStat j_stat;
    j_stat.set_j(j);
    j_stat.set_positive(positive_[j]);
    j_stat.set_negative(negative_[j]);
    j_stat.set_hash(hash_[j]);
    return j_stat;
  }

 private:
  uint64 positive_row_count_;
  uint64 negative_row_count_;
  vector<uint64> positive_;
  vector<uint64> negative_;
  vector<uint64> hash_;
};

// -----------------------------------------------------------------------------
class Shard {
 public:
  class Builder {
   public:
    typedef FixedSizeObjectPool<Builder> Pool;

    explicit Builder(uint32 max_size) : max_size_(max_size) {}

    void AddRow(const CsrMatrix::Row& js, const Double y, const uint32 id) {
      CHECK_NE(id, kInvalidId)
          << "Sensei currently supports at most 2^32-1 data rows.";
      if (shards_.empty() ||
          (shards_.back().XjBoolCount() > 0 &&
           shards_.back().XjBoolCount() + js.size() > max_size_)) {
        shards_.push_back(data::Shard());
      }
      shards_.back().AddRow(js, y, id);
    }

    void AddUserId(const uint64 user_id) { shards_.back().AddUserId(user_id); }

    vector<Shard>* GetMutableShards() { return &shards_; }

   private:
    uint32 max_size_;
    vector<Shard> shards_;
  };

  Shard() = default;

  Shard(Shard&&) = default;

  void Swap(Shard* rhs) {
    rows_.Swap(&rhs->rows_);
    ys_.swap(rhs->ys_);
    ids_.swap(rhs->ids_);
    user_ids_.swap(rhs->user_ids_);
  }

  void AddRow(const CsrMatrix::Row& js, Double y, uint32 id) {
    rows_.AddRow(js);
    ys_.push_back(y);
    ids_.push_back(id);
  }

  void AddUserId(const uint64 user_id) {
    // Make sure this is called right after AddRow
    CHECK_EQ(user_ids_.size(), RowCount() - 1);
    user_ids_.push_back(user_id);
  }

  void RemoveAndRenumberJs(const JRenumbering& j_renumbering) {
    rows_.RemoveAndRenumberJs(j_renumbering);
  }

  uint32 RowCount() const {
    CHECK_EQ(ys_.size(), rows_.RowCount());
    return ys_.size();
  }

  uint64 SizeBytes() const {
    uint64 size = 0;
    size += rows_.SizeBytes();
    size += ys_.capacity() * sizeof(Double);
    size += ids_.capacity() * sizeof(uint32);
    size += user_ids_.capacity() * sizeof(uint64);
    return size + sizeof(*this);
  }

  uint64 XjBoolCount() const { return rows_.NonZerosCount(); }

  void ResetExtender(uint32 i, RowExtender* row_extender) const {
    row_extender->ResetRow(Row(i), ys_[i], ids_[i]);
  }

  CsrMatrix::Row Row(const uint32 i) const { return rows_.GetRow(i); }

  // Unused, but don't remove.
  string ToLibsvmString(const FeatureMap& feature_map,
                        const ProductMap& product_map,
                        const CsrMatrix* dependees) const {
    RowExtender row_extender(dependees);
    vector<string> all;
    for (uint32 i : Range(RowCount())) {
      ResetExtender(i, &row_extender);
      all.push_back(row_extender.ToLibsvmString(feature_map, product_map));
    }
    return strings::Join(all, "\n");
  }

  const vector<Double>& Ys() const { return ys_; }

  const vector<uint64>& UserIds() const {
    // Make sure wer actually have user_ids if we want to use them
    CHECK_EQ(user_ids_.size(), RowCount());
    return user_ids_;
  }

  internal::Data::Shard ToInternalProto() const {
    internal::Data::Shard ret;
    for (uint32 i : Range(RowCount())) {
      internal::Data::Shard::Row* row = ret.add_row();
      for (uint32 j : rows_.GetRow(i)) row->add_j(j);
      row->set_y(ys_[i]);
      row->set_id(ids_[i]);
      if (!user_ids_.empty()) {
        row->set_user_id(user_ids_[i]);
      }
    }
    return ret;
  }

 private:
  CsrMatrix rows_;
  vector<Double> ys_;
  vector<uint32> ids_;
  vector<uint64> user_ids_;

  DISALLOW_COPY_AND_ASSIGN(Shard);
};

// -----------------------------------------------------------------------------
class ShardSet {
 public:
  ShardSet()
      : data_files_bool_feature_count_(0),
        max_shard_size_(1 << 20),
        stats_(0) {}

  const vector<Shard>& GetShards() const { return shards_; }
  vector<Shard>* GetMutableShards() { return &shards_; }

  bool IsEmpty() const { return shards_.size() == 0; }


  uint64 SizeBytes() const {
    uint64 ret = 0;
    for (const Shard& shard : shards_) ret += shard.SizeBytes();
    return ret;
  }

  void RecalcStats(uint32 new_size) {
    using util::functional::ToPermanentCallback;
    Stats::Pool pool(
        concurrency::kThreadCount, concurrency::kThreadCount,
        ToPermanentCallback([new_size, this] { return new Stats(new_size); }));

    concurrency::ParFor(shards_, [&pool, this](const Shard& shard) {
      Stats* stats = pool.Get();
      RowExtender row_extender(dependees_);
      for (uint32 i : Range(shard.RowCount())) {
        shard.ResetExtender(i, &row_extender);
        stats->AddRow(row_extender);
      }
      pool.Release(stats);
    });

    stats_.Reset(new_size);
    for (uint32 i : Range(concurrency::kThreadCount)) {
      UNUSED(i);
      Stats* stats = pool.Get();
      stats_.Add(*stats);
      pool.Retire(stats);
    }
  }

  vector<vector<uint32>> GetCoincidenceMatrix(uint32 size) const;

  void AddToDataFilesBoolFeatureCount(uint64 data_files_bool_feature_count) {
    data_files_bool_feature_count_ += data_files_bool_feature_count;
  }

  const Stats& GetStats() const { return stats_; }

  void LogStats() const {
    LOG(INFO) << "Size = " << stats_.RowCount() << "; "
              << "SizeBytes = " << SizeBytes() << "; "
              << "DataFilesBoolFeatureCount = "
              << data_files_bool_feature_count_ << "; "
              << "XjBoolCount = " << XjBoolCount();
  }

  void RemoveAndRenumberJs(const JRenumbering& j_renumbering) {
    // TODO(lew): Use smaller ranges.
    concurrency::ParFor(&shards_, [&j_renumbering, this](Shard* shard) {
      shard->RemoveAndRenumberJs(j_renumbering);
    });
    stats_.RemoveAndRenumberJs(j_renumbering);
  }

  void SetDependees(const CsrMatrix& new_dependees) {
    dependees_ = &new_dependees;
  }

  uint64 XjBoolCount() const {
    uint64 ret = 0;
    for (const Shard& shard : shards_) ret += shard.XjBoolCount();
    return ret;
  }

  uint64 XjBoolCountOfJ(const uint32 j) const {
    return stats_.XjBoolCountOfJ(j);
  }

  internal::Data::ShardSet ToInternalProto() const {
    internal::Data::ShardSet ret;
    for (const Shard& s : shards_) *ret.add_shard() = s.ToInternalProto();
    *ret.mutable_stats() = stats_.ToInternalProto();
    return ret;
  }

  void SetMaxShardSize(uint32 max_shard_size) {
    max_shard_size_ = max_shard_size;
  }

  uint32 GetMaxShardSize() const { return max_shard_size_; }

 private:
  vector<Shard> shards_;

  uint64 data_files_bool_feature_count_;
  uint32 max_shard_size_;
  Stats stats_;

  // This if for convenience.
  const CsrMatrix* dependees_;
};

// -----------------------------------------------------------------------------
class Data {
 public:
  Data() {
    // Both data shards will save a pointer to dependees.
    training_.SetDependees(dependees_);
    holdout_.SetDependees(dependees_);
  }

  void SetMaxShardSize(uint32 max_shard_size) {
    training_.SetMaxShardSize(max_shard_size);
    holdout_.SetMaxShardSize(max_shard_size);
  }

  internal::Dependees BuildDependees() const {
    internal::Dependees ret;
    for (const uint32 j : Range(dependees_.RowCount())) {
      for (const uint32 j_child : dependees_.GetRow(j)) {
        internal::Dependees::Dependee* dependee = ret.add_dependee();
        dependee->set_j(j);
        dependee->set_j_child(j_child);
      }
    }
    return ret;
  }

  internal::DetailedStats BuildDetailedStats() const {
    internal::DetailedStats stats;
    for (const uint32 j : Range(dependees_.RowCount()))
      stats.add_xjbools_count(XjBoolCountOfJ(j));
    return stats;
  }

  void LogStats() const {
    training_.LogStats();
    holdout_.LogStats();
  }


  void RemoveAndRenumberJs(const JRenumbering& j_renumbering) {
    training_.RemoveAndRenumberJs(j_renumbering);
    holdout_.RemoveAndRenumberJs(j_renumbering);
    RemoveAndRenumberDependees(j_renumbering);
  }

  void RecalcStats(uint32 j_size) {
    training_.RecalcStats(j_size);
    holdout_.RecalcStats(j_size);
  }

  uint64 XjBoolCountOfJ(const uint32 j) const {
    return training_.XjBoolCountOfJ(j) + holdout_.XjBoolCountOfJ(j);
  }

  uint64 MaterializedXjBoolCount() const {
    return training_.GetStats().MaterializedXjBoolCount() +
           holdout_.GetStats().MaterializedXjBoolCount();
  }

  internal::Data ToInternalProto() const {
    internal::Data ret;
    *ret.mutable_training() = training_.ToInternalProto();
    if (!holdout_.IsEmpty())
      *ret.mutable_holdout() = holdout_.ToInternalProto();
    *ret.mutable_dependees() = BuildDependees();
    return ret;
  }

  ShardSet* GetMutableTraining() { return &training_; }

  ShardSet* GetMutableHoldout() { return &holdout_; }

  CsrMatrix* GetMutableDependees() { return &dependees_; }

  const ShardSet& GetTraining() const { return training_; }

  const ShardSet& GetHoldout() const { return holdout_; }

  const CsrMatrix& GetDependees() const { return dependees_; }

 private:

  void RemoveAndRenumberDependees(const JRenumbering& j_renumbering) {
    if (j_renumbering.j_to_new_j().empty()) return;
    // Remove, renumber content, change indicies
    dependees_.RemoveAndRenumberJs(j_renumbering);
    dependees_.RemoveAndRenumberRows(j_renumbering);
  }

  // Private member variables.
  ShardSet training_;
  ShardSet holdout_;
  CsrMatrix dependees_;
};

// -----------------------------------------------------------------------------
// Thread-safe class that adds data shards to ShardSet
class Adder {
 public:
  explicit Adder(data::ShardSet* data) : data_(CHECK_NOTNULL(data)) {}

  // Contents of <shards> are destroyed.
  void Add(const uint64 data_files_bool_feature_count, Shard::Builder* shards) {
    MutexLock lock(&mutex_);
    for (Shard& shard : *shards->GetMutableShards()) {
      data_->GetMutableShards()->push_back(std::move(shard));
    }
    data_->AddToDataFilesBoolFeatureCount(data_files_bool_feature_count);
  }

 private:
  data::ShardSet* data_ PT_GUARDED_BY(mutex_);
  Mutex mutex_;

  DISALLOW_COPY_AND_ASSIGN(Adder);
};

// -----------------------------------------------------------------------------
// Functions for testing.

namespace test {

inline void NewSmallData(uint32 size, const vector<vector<uint32>>& rows,
                         Data* data) {
  CHECK_GE(size, 2) << "SmallData uses 2 js.";
  Shard shard;
  Double label = 1;
  for (const auto& row : rows) {
    shard.AddRow(row, label, 0);
    label = -label;
  }
  CHECK(data->GetMutableTraining()->GetMutableShards()->empty());
  data->GetMutableTraining()->GetMutableShards()->push_back(std::move(shard));
  data->RecalcStats(size);
}

}  // namespace test

}  // namespace data
}  // namespace sensei


#endif  // SENSEI_DATA_H_
