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
#ifndef FEATURE_EXPLORATION_H__
#define FEATURE_EXPLORATION_H__

#include "sensei/common.h"


namespace sensei {

namespace config {
class FeatureExploration;
}
namespace data {
struct Data;
}
namespace logs {
class FeatureExploration;
}
class World;
class FeatureMap;
class FeatureScoring;
class Model;
class ProductMap;

class FeatureExploration {
 public:
  FeatureExploration(World* world, Model* model, FeatureMap* feature_map,
                     ProductMap* product_map, data::Data* data,
                     FeatureScoring* feature_scoring);
  void AddNewProductFeatures(const config::FeatureExploration& config,
                             logs::FeatureExploration* log);

 private:
  World* world_;
  Model* model_;
  FeatureMap* feature_map_;
  ProductMap* product_map_;
  FeatureScoring* feature_scoring_;
  data::Data* data_;

  Double xjbools_per_candidate_feature_estimate_;
};

}  // namespace sensei


#endif  // FEATURE_EXPLORATION_H__
