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
#ifndef SENSEI_CSR_MATRIX_H_
#define SENSEI_CSR_MATRIX_H_

#include <stddef.h>
#include <sys/types.h>
#include <algorithm>
#include <utility>
using std::pair;
#include <vector>
using std::vector;

#include "sensei/base/integral_types.h"
#include "sensei/base/logging.h"
#include "sensei/common.h"
#include "sensei/j_renumbering.h"
#include "sensei/range.h"
#include "sensei/util/array_slice.h"


namespace sensei {

// -----------------------------------------------------------------------------
// http://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_.28COO.29
class CooMatrix {
 public:
  CooMatrix() : row_count_(0) {}

  void Reserve(const size_t size) { contents_.reserve(size); }

  bool Equals(const CooMatrix& rhs) const {
    return row_count_ == rhs.row_count_ && contents_ == rhs.contents_;
  }

  uint32 RowCount() const { return row_count_; }
  void SetRowCount(const uint32 row_count) { row_count_ = row_count; }
  void SetTrue(uint32 row, uint32 column) {
    contents_.push_back(std::make_pair(row, column));
  }
  void Sort() { std::sort(contents_.begin(), contents_.end()); }
  bool IsSorted() const {
    return std::is_sorted(contents_.begin(), contents_.end());
  }
  const vector<pair<uint32, uint32> >& AllCoords() const { return contents_; }

 private:
  vector<pair<uint32, uint32> > contents_;
  uint32 row_count_;
};

// -----------------------------------------------------------------------------
class CsrMatrix {
 public:
  CsrMatrix() : boundaries_(1, 0) {}

  typedef util::gtl::ArraySlice<uint32> Row;

  void AddRow(const Row& js) {
    // TODO(lew): User can check Row capacity. Use it in DataReader.
    for (const uint32 j : js) contents_.push_back(j);
    CHECK_LE(contents_.size(), static_cast<uint32>(-1));
    boundaries_.push_back(contents_.size());
  }

  void Swap(CsrMatrix* from) {
    boundaries_.swap(from->boundaries_);
    contents_.swap(from->contents_);
  }

  void FromCooMatrix(const CooMatrix& coo_matrix) {
    DCHECK(coo_matrix.IsSorted());
    boundaries_.clear();
    contents_.clear();
    for (const pair<uint32, uint32>& ij : coo_matrix.AllCoords()) {
      const uint32 i = ij.first;
      const uint32 j = ij.second;
      while (i >= boundaries_.size()) boundaries_.push_back(contents_.size());
      contents_.push_back(j);
    }
    while (coo_matrix.RowCount() >= boundaries_.size())
      boundaries_.push_back(contents_.size());
  }

  // Firstly it removes all Js in j_renumbering.j_to_new_j[j] == kInvalidJ.
  // Then renumber all Js j -> j_renumbering.j_to_new_j.
  // j_to_new_j can be empty and then no renumbering/removing is done.
  void RemoveAndRenumberJs(const JRenumbering& j_renumbering) {
    if (j_renumbering.j_to_new_j().empty()) return;
    uint32 input_offset = 0;
    for (const uint32 i : Range(RowCount())) {
      const uint32 old_length = boundaries_[i + 1] - input_offset;
      const Row js(contents_, input_offset, old_length);
      const uint32 new_length = j_renumbering.RemoveAndRenumberJsTo(
          js, contents_.begin() + boundaries_[i]);
      input_offset = boundaries_[i + 1];
      boundaries_[i + 1] = boundaries_[i] + new_length;
    }
    contents_.resize(boundaries_[RowCount()]);
  }

  void RemoveAndRenumberRows(const JRenumbering& j_renumbering) {
    const vector<uint32>& j_to_new_j = j_renumbering.j_to_new_j();
    if (j_to_new_j.empty()) return;
    CsrMatrix temp;
    for (const uint32 old_j : j_renumbering.NewJToOldJ()) {
      CHECK_NE(old_j, kInvalidJ);
      temp.AddRow(GetRow(old_j));
    }
    Swap(&temp);
  }

  Row GetRow(const uint32 i) const {
    return Row(contents_, boundaries_[i], boundaries_[i + 1] - boundaries_[i]);
  }

  uint32 RowCount() const { return boundaries_.size() - 1; }

  void SetRowCount(const uint32 size) {
    CHECK_GE(size, RowCount());
    boundaries_.resize(size + 1, contents_.size());
  }

  uint64 SizeBytes() const {
    uint64 size = 0;
    size += contents_.capacity() * sizeof(uint32);
    size += boundaries_.capacity() * sizeof(uint32);
    return size + sizeof(*this);
  }

  CooMatrix ToCooMatrix() const {
    CooMatrix coo_matrix;
    coo_matrix.Reserve(contents_.size());
    coo_matrix.SetRowCount(RowCount());
    for (const uint32 i : Range(RowCount()))
      for (const uint32 j : GetRow(i)) coo_matrix.SetTrue(i, j);
    return coo_matrix;
  }

  uint64 NonZerosCount() const { return contents_.size(); }

 private:
  vector<uint32> boundaries_;
  vector<uint32> contents_;
};

}  // namespace sensei


#endif  // SENSEI_CSR_MATRIX_H_
