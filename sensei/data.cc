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
#include <vector>
using std::vector;

#include "sensei/data.h"


namespace sensei {
namespace data {

vector<vector<uint32>> ShardSet::GetCoincidenceMatrix(uint32 size) const {
  // TODO(lew): Ideas for improvement:
  //   - Alloc/Copy/Dealloc one row at a time for memory savings.
  //   - Use vector<vector<Atomic32>> and google3/base/atomic_refcount.h
  //     Copying at the end won't be needed.
  //   - Have coincidence incremental.
  std::unique_ptr<std::unique_ptr<std::atomic<uint32>[]>[]> coincidence(
      new std::unique_ptr<std::atomic<uint32>[]>[size]);
  for (uint32 j1 : Range(size)) {
    coincidence[j1].reset(new std::atomic<uint32>[size]);
    for (uint32 j2 : Range(size)) coincidence[j1][j2] = 0;
  }

  concurrency::ParFor(shards_, [this, size, &coincidence](const Shard& shard) {
    RowExtender row(dependees_);
    for (uint32 i : Range(shard.RowCount())) {
      shard.ResetExtender(i, &row);
      for (uint32 j1 : row.SparseBool()) {
        CHECK_LT(j1, size);
        for (uint32 j2 : row.SparseBool()) {
          coincidence[j1][j2] += 1;
        }
      }
    }
  });
  vector<vector<uint32>> coincidence_ret(size, vector<uint32>(size, 0));
  for (uint32 j1 : Range(size)) {
    for (uint32 j2 : Range(size)) {
      coincidence_ret[j1][j2] = coincidence[j1][j2];
    }
  }
  LOG(INFO) << "GetCoincidenceMatrix: End";
  return coincidence_ret;
}

}  // namespace data
}  // namespace sensei

