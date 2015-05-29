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
#ifndef SENSEI_FEATURE_PRUNING_H_
#define SENSEI_FEATURE_PRUNING_H_

#include "sensei/config.pb.h"
#include "sensei/data.h"
#include "sensei/feature_map.h"
#include "sensei/feature_scoring.h"
#include "sensei/log.pb.h"
#include "sensei/model.h"
#include "sensei/optimizers.h"


namespace sensei {

class World;

class FeaturePruning {
 public:
  explicit FeaturePruning(World* world);

  void PruneFeatures(const config::FeaturePruning& config,
                     logs::FeaturePruning* log);

  // Uses the following fields in <config> for stopping:
  // score_threshold, top_count, top_fraction.
  vector<bool> ComputePruning(const config::FeaturePruning& config,
                              const vector<Double>& j_to_score,
                              logs::FeaturePruning* log) const;

  // Removes all the Js set to true in <removed_js>.
  // Affects <data_>, <optimizer_>, and <product_map_>.
  void RemoveJs(const vector<bool>& removed_js);

 private:
  World* world_;
};

}  // namespace sensei


#endif  // SENSEI_FEATURE_PRUNING_H_
