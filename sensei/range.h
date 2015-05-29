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
#ifndef SENSEI_RANGE_H_
#define SENSEI_RANGE_H_

#include <algorithm>
#include <queue>
using std::priority_queue;
#include <string>
using std::string;
#include <utility>
using std::pair;
#include <vector>
using std::vector;

#include "sensei/base/integral_types.h"
#include "sensei/base/logging.h"
#include "sensei/base/macros.h"
#include "sensei/util/array_slice.h"

// -----------------------------------------------------------------------------
class UintIterator {
 public:
  explicit UintIterator(uint32 i) : i_(i) {}
  uint32 operator*() const { return i_; }
  void operator++() { ++i_; }
  bool operator!=(const UintIterator& that) const { return i_ != that.i_; }

 private:
  uint32 i_;
};

// -----------------------------------------------------------------------------
// Class similar to Python's 'range' used for 'for' loops. E.g.
// for (uint32 i : Range(100)) { ... }
class Range {
 public:
  typedef uint32 value_type;

  explicit Range(const uint32 to) : from_(0), to_(to) {}

  Range(const uint32 from, const uint32 to)
      : from_(from), to_(std::max(from, to)) {}

  // These are the iterators you are looking for.
  UintIterator begin() const { return UintIterator(from_); }
  UintIterator end() const { return UintIterator(to_); }

  uint32 size() const { return to_ - from_; }

  template <typename T>
  ::util::gtl::ArraySlice<T> SliceOfVector(const vector<T>& v) const {
    return ::util::gtl::ArraySlice<T>(v, from_, to_ - from_);
  }

  vector<Range> SplitEvenly(uint32 count) const {
    CHECK_EQ(0, from_) << "Split of such ranges is not supported.";
    vector<Range> ret;
    for (uint32 i32 : Range(count)) {
      uint64 i = i32;  // Force 64 bit computations below to avoid overflow.
      ret.push_back(Range(i * to_ / count, (i + 1) * to_ / count));
    }
    return ret;
  }

  // Returns Range of strings starting of given prefix.
  static Range WithPrefix(const vector<string>& v, const string& prefix) {
    if (prefix.empty()) return Range(v.size());
    string prefix_end = prefix;
    auto c = prefix_end.back();
    prefix_end.back()++;
    CHECK_LT(c, prefix_end.back())
        << "Support for ASCII == 255 not implemented. File a feature request.";
    auto start_iter = std::lower_bound(v.begin(), v.end(), prefix);
    auto end_iter = std::lower_bound(start_iter, v.end(), prefix_end);
    return Range(start_iter - v.begin(), end_iter - v.begin());
  }

 private:
  const uint32 from_;
  const uint32 to_;
};

// -----------------------------------------------------------------------------
template <class T>
class PrioritySumIterator {
 public:
  // v is vector of Ts with priorities.
  // It must be sorted from largest to smallest.
  explicit PrioritySumIterator(const vector<pair<double, T>>& v) : v_(v) {
    CHECK(!v_.empty());
    for (uint32 j : Range(v_.size())) InsertNext(std::make_pair(j, j));
  }

  bool HasNext() const { return !pq_.empty(); }

  // Returns next pair of Ts with highest sum of priorities.
  // Avoids duplicates.
  pair<T, T> Next() {
    DCHECK(HasNext());
    pair<double, pair<uint32, uint32>> e = pq_.top();  // sum, row, column
    pq_.pop();
    InsertNext(e.second);
    return std::make_pair(v_[e.second.first].second,
                          v_[e.second.second].second);
  }

 private:
  void InsertNext(const pair<uint32, uint32>& p) {
    DCHECK_LE(p.first, p.second);
    pair<uint32, uint32> np = p;
    np.second += 1;
    if (np.second >= v_.size()) return;
    pq_.push(std::make_pair(v_[np.first].first + v_[np.second].first, np));
  }

  // Private member variables.
  const vector<pair<double, T>>& v_;
  priority_queue<pair<double, pair<uint32, uint32>>> pq_;  // (sum, (j1, j2))
};

// -----------------------------------------------------------------------------

template <class T>
class ProductIterator {
 public:
  // <factors> must outlive ProductIterator.
  explicit ProductIterator(const vector<vector<T>>& factors)
      : factors_(factors), cursor_(factors_.size(), 0), empty_(false) {
    product_.reserve(factors_.size());
    for (uint32 i : Range(factors_.size())) {
      if (factors_[i].size() == 0) {  // one of the factors is empty
        empty_ = true;
        return;
      }
      product_.push_back(factors_[i][0]);
    }
  }

  bool IsEmpty() const { return empty_; }

  void Next() {
    for (uint32 i : Range(factors_.size())) {
      if (cursor_[i] + 1 < factors_[i].size()) {
        SetCursor(i, cursor_[i] + 1);
        return;
      } else {
        SetCursor(i, 0);
      }
    }
    empty_ = true;  // Full loop over the iterator.
  }

  const vector<T>& Get() const { return product_; }

 private:
  void SetCursor(uint32 i, uint32 value) {
    cursor_[i] = value;
    product_[i] = factors_[i][value];
  }

  // Private member variables.
  const vector<vector<T>>& factors_;
  vector<uint32> cursor_;
  vector<T> product_;
  bool empty_;

  DISALLOW_COPY_AND_ASSIGN(ProductIterator);
};

#endif  // SENSEI_RANGE_H_
