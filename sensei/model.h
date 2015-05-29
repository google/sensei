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
#ifndef SENSEI_MODEL_H_
#define SENSEI_MODEL_H_

#include <vector>
using std::vector;

#include "sensei/base/integral_types.h"
#include "sensei/common.h"
#include "sensei/j_renumbering.h"
#include "sensei/internal.pb.h"
#include "sensei/range.h"


namespace sensei {

class JRenumbering;

namespace internal {
class Model;
}  // namespace internal

struct Model {
  Model()
      : current_creation_time(0),
        prev_total_loss(kInfinity),
        total_loss(kInfinity),
        synced_with_weights(false),
        iteration_no(0) {}

  bool Empty() const { return w.size() == 0; }

  void SetSize(uint32 size) {
    uint32 oldsize = w.size();
    current_creation_time++;

    MildResize(size, &precision);
    MildResize(size, &w);
    MildResize(size, &delta_w);
    MildResize(size, &loss_derivative);
    MildResize(size, &creation_time);
    for (uint32 i : Range(oldsize, size))
      creation_time[i] = current_creation_time;
  }

  void SetTotalLoss(Double new_total_loss) {
    prev_total_loss = total_loss;
    total_loss = new_total_loss;
  }

  void InitPerShards(uint32 training_size, uint32 holdout_size) {
    training.Init(training_size);
    holdout.Init(holdout_size);
  }

  // TODO(lew): Use it everywhere.
  uint32 GetSize() const { return w.size(); }

  bool IsFeatureNew(uint32 j) const {
    return creation_time[j] == current_creation_time;
  }

  void MergeJToJ(uint32 from, uint32 to) {
    w[to] += w[from];
    w[from] = 0;
    delta_w[to] += delta_w[from];
    delta_w[from] = 0;
    synced_with_weights = false;
  }

  void RemoveAndRenumberJs(const JRenumbering& j_renumbering) {
    j_renumbering.RenumberIndicies(&precision);
    j_renumbering.RenumberIndicies(&w);
    j_renumbering.RenumberIndicies(&delta_w);
    j_renumbering.RenumberIndicies(&loss_derivative);
    j_renumbering.RenumberIndicies(&creation_time);
  }

  void GetProto(internal::Model* model) const {
    uint32 size = GetSize();
    model->mutable_precision()->Reserve(size);
    model->mutable_w()->Reserve(size);
    model->mutable_delta_w()->Reserve(size);
    model->mutable_loss_derivative()->Reserve(size);
    model->mutable_creation_time()->Reserve(size);
    for (uint32 i : Range(size)) {
      model->mutable_precision()->Add(precision[i]);
      model->mutable_w()->Add(w[i]);
      model->mutable_delta_w()->Add(delta_w[i]);
      model->mutable_loss_derivative()->Add(loss_derivative[i]);
      model->mutable_creation_time()->Add(creation_time[i]);
    }
    model->set_current_creation_time(current_creation_time);
    model->set_prev_total_loss(prev_total_loss);
    model->set_total_loss(total_loss);
    model->set_synced_with_weights(synced_with_weights);
  }

  vector<Double> precision;
  vector<Double> w;
  vector<Double> delta_w;
  vector<Double> loss_derivative;
  vector<uint32> creation_time;
  uint32 current_creation_time;
  // LogLoss that was achieved on previous weights i.e. w - delta_w.
  Double prev_total_loss;
  Double total_loss;
  bool synced_with_weights;
  // Number of passes over the dataset by any algorithm.
  // Includes iterations for which restart or undo were performed.
  uint32 iteration_no;

  // Mutable structures essential to the optimizer per data::Shard.
  struct PerShard {
    void Init(uint32 size) { MildResize(size, &wxs); }

    vector<Double> wxs;
  };

  PerShard training;
  PerShard holdout;
};

}  // namespace sensei


#endif  // SENSEI_MODEL_H_
