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
#ifndef SENSEI_FEATURE_SCORING_H_
#define SENSEI_FEATURE_SCORING_H_

#include <vector>
using std::vector;

#include "sensei/common.h"
#include "sensei/config.pb.h"
#include "sensei/feature_map.h"
#include "sensei/logger.h"


namespace sensei {

namespace data {
struct Data;
}
class Model;

class FeatureScoring {
 public:
  explicit FeatureScoring(const data::Data& data, const FeatureMap& feature_map,
                          const ProductMap& product_map, Logger* logger)
      : data_(data),
        feature_map_(feature_map),
        product_map_(product_map),
        logger_(logger) {}

  void ScoreFeatures(const Model& model, const config::FeatureScoring& config,
                     vector<Double>* j_to_score);

 private:
  vector<bool> FeatureSetToJs(const config::FeatureSet& feature_set) const;

 private:
  const data::Data& data_;
  const FeatureMap& feature_map_;
  const ProductMap& product_map_;
  Logger* logger_;
};

}  // namespace sensei


#endif  // SENSEI_FEATURE_SCORING_H_
