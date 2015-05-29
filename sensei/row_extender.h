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
#ifndef SENSEI_ROW_EXTENDER_H_
#define SENSEI_ROW_EXTENDER_H_

#include <string>
using std::string;
#include <vector>
using std::vector;

#include "sensei/base/integral_types.h"
#include "sensei/base/logging.h"
#include "sensei/common.h"
#include "sensei/feature_map.h"
#include "sensei/csr_matrix.h"
#include "sensei/strings/join.h"
#include "sensei/strings/strcat.h"
#include "sensei/util/array_slice.h"
#include "sensei/util/dense_hash_map.h"


namespace sensei {

class FeatureMap;
class ProductMap;

class RowExtender {
 public:
  explicit RowExtender(const CsrMatrix* dependees)
      : dependees_(dependees), cpu_operation_count_flat_materialization_(-1) {
    dependencies_.set_empty_key(kInvalidJ);
  }

  // <sparse_bool> must live until the next call to ResetRow().
  void ResetRow(const JsSlice& sparse_bool, Double y, uint64 id) {
    cpu_operation_count_flat_materialization_ = 0;
    y_ = y;
    id_ = id;
    if (dependees_ != nullptr && dependees_->NonZerosCount() != 0) {
      rewritten_row_.assign(sparse_bool.begin(), sparse_bool.end());
      ExtendSparseBool();
      sparse_bool_ = rewritten_row_;
    } else {
      sparse_bool_ = sparse_bool;
    }
  }

  const JsSlice& SparseBool() const { return sparse_bool_; }

  // Dot product. w is dense.
  Double Dot(const vector<Double>& w) const {
    return SparseDot(SparseBool(), w);
  }

  Double Dot(const AtomicDoubleSlice& w) const {
    return SparseDot(SparseBool(), w);
  }

  // TODO(lew): Try to speed up Imm and Bmm by precomputing L2SquaredNorm.
  Double L2SquaredNorm() const {
    Double norm = 0;
    return norm + SparseBool().size();
  }

  string ToLibsvmString(const FeatureMap& feature_map,
                        const ProductMap& product_map) const {
    vector<string> all;
    all.push_back(StrCat(GetY()));
    for (uint32 j : SparseBool()) {
      const string feature_name =
          product_map.JToFeature(j).ToLibsvmString(feature_map, "_X_");
      all.push_back(StrCat(feature_name, ":", 1.0));
    }
    return strings::Join(all, " ");
  }

  // TODO(witoldjarnicki) We should not reallocate new vector for each row.
  // We should have some object per thread that has <dependencies> and
  // <materialized_row>.  This class should have its own pointer to dependees.
  void ExtendSparseBool() {
    CHECK(dependees_ != nullptr);
    // Observe that the newly added feature will be used later in the loop.
    // I.e. sparse_bool_.size() is changing during the loop hence we
    // cannot use range-based 'for' here.
    for (uint32 i = 0; i < rewritten_row_.size(); ++i) {
      uint32 j = rewritten_row_[i];
      DCHECK_LT(j, dependees_->RowCount());
      // TODO(lew) : Build/put dependees from/in ProductMap.
      for (uint32 child_j : dependees_->GetRow(j)) {
        ++cpu_operation_count_flat_materialization_;
        uint32& d = dependencies_[child_j];
        DCHECK(d == 0 || d == 1);
        d += 1;
        if (d == 2) rewritten_row_.push_back(child_j);
      }
    }
    dependencies_.clear();
  }

  uint64 CpuOperationCountFlatMaterialization() const {
    return cpu_operation_count_flat_materialization_;
  }

  Double GetY() const { return y_; }

  uint64 GetId() const { return id_; }

 private:
  Double y_;
  uint64 id_;
  vector<uint32> rewritten_row_;
  const CsrMatrix* dependees_;
  uint64 cpu_operation_count_flat_materialization_;
  dense_hash_map<uint32, uint32> dependencies_;
  JsSlice sparse_bool_;
};

}  // namespace sensei


#endif  // SENSEI_ROW_EXTENDER_H_
