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
#ifndef SENSEI_J_RENUMBERING_H_
#define SENSEI_J_RENUMBERING_H_

#include <algorithm>
#include <vector>
using std::vector;

#include "sensei/base/integral_types.h"
#include "sensei/base/logging.h"
#include "sensei/common.h"
#include "sensei/range.h"
#include "sensei/util/array_slice.h"


namespace sensei {

// It stores the data needed for renumbering, the next J what gives information
// about how many Js has been recycled and mapping j_to_new_j
// TODO(lew): Move this struct to feature_prunning.h
class JRenumbering {
 public:
  // Named constructor.
  static JRenumbering RemoveJs(const vector<bool>& js_to_remove) {
    JRenumbering j_renumbering{vector<uint32>(js_to_remove.size(), 0), 0};
    for (const uint32 j : Range(js_to_remove.size())) {
      if (js_to_remove[j]) j_renumbering.SetInvalid(j);
    }
    j_renumbering.FillJToNewJ();
    return j_renumbering;
  }

  JRenumbering(vector<uint32> j_to_new_j, uint32 next_j)
      : j_to_new_j_(j_to_new_j), next_j_(next_j) {}

  // It removes all Js from the vector which are scheduled for removal, renumber
  // others and change capacity to fit the size
  void RemoveAndRenumberJs(vector<uint32>* js) const {
    js->resize(RemoveAndRenumberJsTo(*js, js->begin()));
    vector<uint32> new_js(js->size());
    new_js.assign(js->begin(), js->end());
    js->swap(new_js);
  }

  // Caller has to guarantee that there is enough space at <out_js>.
  // <out_js> must not be contained in <in_js>, except for the case where it is
  // exactly where <in_js> starts.
  uint32 RemoveAndRenumberJsTo(const util::gtl::ArraySlice<uint32>& in_js,
                               vector<uint32>::iterator out_js) const {
    DCHECK(!j_to_new_j_.empty());
    uint32 k_out = 0;
    for (uint32 j : in_js) {
      DCHECK_LT(j, j_to_new_j_.size());
      *(out_js + k_out) = j_to_new_j_[j];
      if (j_to_new_j_[j] != kInvalidJ) ++k_out;
    }
    return k_out;
  }

  // T of v at index j will be at index j_to_new_j_[j], ones which should be
  // removed will not be in vector anymore
  template <class T>
  void RenumberIndicies(vector<T>* v) const {
    if (j_to_new_j_.empty()) return;
    vector<T> new_v(next_j_);
    // In some cases (positive_stats) vectors are not at the full size (synced)
    // TODO(user): Check whether it still need to be LE (instead of EQ)
    DCHECK_LE(v->size(), j_to_new_j_.size());
    for (uint32 j : Range(v->size())) {
      if (j_to_new_j_[j] != kInvalidJ) {
        DCHECK_LT(j_to_new_j_[j], new_v.size());
        DCHECK_LT(j, j_to_new_j_.size());
        new_v[j_to_new_j_[j]] = std::move((*v)[j]);
      }
    }
    v->swap(new_v);
  }

  // Fills all entries set to 0 with consecutive numbers.
  void FillJToNewJ() {
    for (uint32& new_j : j_to_new_j_) {
      CHECK(new_j == 0 || new_j == kInvalidJ);
      if (new_j == 0) {
        new_j = next_j_;
        next_j_++;
      }
    }
  }

  bool IsNoOp() const {
    for (uint32 j : Range(j_to_new_j_.size())) {
      if (j_to_new_j_[j] != j) return false;
    }
    return true;
  }

  // Sets J to kInvalidJ (it means it was removed in FP)
  void SetInvalid(uint32 j) {
    CHECK_LT(j, j_to_new_j_.size());
    j_to_new_j_[j] = kInvalidJ;
  }

  vector<uint32> NewJToOldJ() const {
    vector<uint32> ret(next_j(), kInvalidJ);
    for (const uint32 old_j : Range(j_to_new_j_.size())) {
      if (j_to_new_j_[old_j] != kInvalidJ) ret[j_to_new_j_[old_j]] = old_j;
    }
    return ret;
  }

  const vector<uint32>& j_to_new_j() const { return j_to_new_j_; }
  const uint32 next_j() const { return next_j_; }

 private:
  vector<uint32> j_to_new_j_;
  uint32 next_j_;  // next_j_ = max_j + 1;
};

}  // namespace sensei


#endif  // SENSEI_J_RENUMBERING_H_
