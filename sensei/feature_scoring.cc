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
#include "sensei/feature_scoring.h"

#include <math.h>

#include "sensei/base/integral_types.h"
#include "sensei/config.pb.h"
#include "sensei/data.h"
#include "sensei/model.h"
#include "sensei/range.h"
#include "sensei/world.h"


namespace sensei {

void FeatureScoring::ScoreFeatures(const Model& model,
                                   const config::FeatureScoring& config,
                                   vector<Double>* j_to_score) {
  uint32 size = model.GetSize();
  j_to_score->resize(size);

  // TODO(lew): Implelemt better heuristics for sorting features.
  //  - |w| * support_w
  //  - majoryzer(0) - majoryzer(w0)
  //  - calculate accurate w0 impact
  // Scores need to be additive.
  switch (config.feature_ordering()) {
    case config::FeatureScoring::WEIGHT_ABSOLUTE_VALUE:
      for (uint32 j : Range(size)) (*j_to_score)[j] = fabs(model.w[j]);
      break;
    case config::FeatureScoring::WEIGHT_ABSOLUTE_VALUE_TIMES_ROW_COUNT:
      for (uint32 j : Range(size))
        (*j_to_score)[j] =
            fabs(model.w[j]) * data_.GetTraining().XjBoolCountOfJ(j);
      break;
    case config::FeatureScoring::FEATURE_OUTPUT_MUTUAL_INFORMATION:
      for (uint32 j : Range(size)) {
        (*j_to_score)[j] = data_.GetTraining()
                               .GetStats()
                               .GetCorrelationTable(j)
                               .MutualInformation();
      }
      break;
    case config::FeatureScoring::FEATURE_OUTPUT_CORRELATION:
      for (uint32 j : Range(size)) {
        double phi = data_.GetTraining()
                         .GetStats()
                         .GetCorrelationTable(j)
                         .PhiCoefficient();
        (*j_to_score)[j] = fabs(phi);
      }
      break;
  }
  if (config.has_bonus()) {
    vector<bool> is_bonused = FeatureSetToJs(config.bonus().feature_set());
    CHECK_EQ(size, is_bonused.size());
    for (uint32 j : Range(size)) {
      if (is_bonused[j]) (*j_to_score)[j] *= config.bonus().factor();
    }
  }
  if (config.take_logarithm()) {
    for (Double& score : *j_to_score) score = ::log(score + 1e-10);
  }
  if (config.logging()) {
    logs::Line log_line;
    internal::FeatureScoring* log = log_line.mutable_internal_feature_scoring();
    for (uint32 j : Range(size)) {
      if (model.w[j] == 0.0) continue;
      internal::ProductMap::JProduct* j_product_log = log->add_j_product();
      *j_product_log = JToJProduct(feature_map_, product_map_, j);
      j_product_log->set_score((*j_to_score)[j]);
    }
    logger_->AddToLogs(log_line);
  }
}

vector<bool> FeatureScoring::FeatureSetToJs(
    const config::FeatureSet& feature_set) const {
  vector<bool> feature_js = feature_map_.FeatureSetToJs(feature_set);
  vector<bool> product_js = product_map_.HaveAtLeastOneFeatureJ(feature_js);
  return product_js;
}

}  // namespace sensei

