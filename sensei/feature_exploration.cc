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
#include "sensei/feature_exploration.h"

#include <math.h>
#include <algorithm>
#include <string>
using std::string;
#include <utility>
using std::pair;
#include <vector>
using std::vector;

#include "sensei/base/integral_types.h"
#include "sensei/base/logging.h"
#include "sensei/config.pb.h"
#include "sensei/csr_matrix.h"
#include "sensei/data.h"
#include "sensei/feature_map.h"
#include "sensei/feature_scoring.h"
#include "sensei/log.pb.h"
#include "sensei/model.h"
#include "sensei/range.h"
#include "sensei/world.h"
#include "sensei/strings/strutil.h"


namespace sensei {

FeatureExploration::FeatureExploration(World* world, Model* model,
                                       FeatureMap* feature_map,
                                       ProductMap* product_map,
                                       data::Data* data,
                                       FeatureScoring* feature_scoring)
    : world_(CHECK_NOTNULL(world)),
      model_(CHECK_NOTNULL(model)),
      feature_map_(CHECK_NOTNULL(feature_map)),
      product_map_(CHECK_NOTNULL(product_map)),
      feature_scoring_(CHECK_NOTNULL(feature_scoring)),
      data_(CHECK_NOTNULL(data)),
      xjbools_per_candidate_feature_estimate_(0) {}


void FeatureExploration::AddNewProductFeatures(
    const config::FeatureExploration& config, logs::FeatureExploration* log) {
  CHECK_EQ(model_->GetSize(), product_map_->Size());
  uint32 first_j_added = product_map_->Size();
  vector<Double> j_to_score(model_->GetSize());
  feature_scoring_->ScoreFeatures(*model_, config.feature_scoring(),
                                  &j_to_score);

  vector<pair<Double, uint32>> best_js;
  for (const ProductMap::FeatureAndJ* f : product_map_->GetAll()) {
    uint32 j = f->GetJ();
    best_js.push_back(std::make_pair(j_to_score[j], j));
  }
  std::sort(best_js.rbegin(), best_js.rend());

  PrioritySumIterator<uint32> psi(best_js);

  if (xjbools_per_candidate_feature_estimate_ == 0) {
    // TODO(lew,witoldjarnicki): Maybe we should start with an estimate of
    //                           average feature size instead of max feature
    //                           size ?
    xjbools_per_candidate_feature_estimate_ =
        data_->GetTraining().GetStats().RowCount() +
        data_->GetHoldout().GetStats().RowCount();
  }
  log->set_xjbools_per_candidate_feature_estimate(
      xjbools_per_candidate_feature_estimate_);
  uint32 new_feature_count = 0;
  uint64 present_features_skipped = 0;
  // TODO(lew): This is going through all the data and very slow.
  uint64 previous_xjbool_count =
      data_->GetTraining().GetStats().MaterializedXjBoolCount() +
      data_->GetHoldout().GetStats().MaterializedXjBoolCount();
  // TODO(lew): Consider calculating memory needs and using them instead of
  //            feature_exploration_size.
  {
    CooMatrix coo_dependees = data_->GetDependees().ToCooMatrix();
    Double score_sum_threshold = INFINITY;
    while (true) {
      if (config.has_expected_xjbools_added() &&
          new_feature_count * xjbools_per_candidate_feature_estimate_ >=
              config.expected_xjbools_added())
        break;
      if (config.has_maximum_features_added() &&
          new_feature_count >= config.maximum_features_added())
        break;
      if (!psi.HasNext()) break;
      pair<uint32, uint32> j_pair = psi.Next();
      DCHECK_NE(j_pair.first, j_pair.second);
      if (j_pair.first > j_pair.second) std::swap(j_pair.first, j_pair.second);
      // JToFeature method in UNSAFE mode doesn't check whether product map
      // is synced.
      const JProduct& f1 = product_map_->JToFeatureUnsafe(j_pair.first);
      const JProduct& f2 = product_map_->JToFeatureUnsafe(j_pair.second);
      JProduct new_f = JProduct::And(f1, f2);
      // TODO(witoldjarnicki): Count features skipped becuase of the next line.
      // TODO(lew): Double searching in feature map.
      if (product_map_->HasFeature(new_f)) {
        ++present_features_skipped;
        continue;
      }
      if (config.has_max_product_size() &&
          new_f.GetJs().size() > config.max_product_size()) {
        continue;
      }


      uint32 new_f_j = product_map_->FeatureToJ(new_f);
      coo_dependees.SetTrue(j_pair.first, new_f_j);
      coo_dependees.SetTrue(j_pair.second, new_f_j);

      Double score_sum = j_to_score[j_pair.first] + j_to_score[j_pair.second];
      DCHECK_LE(score_sum, score_sum_threshold);
      score_sum_threshold = score_sum;
      ++new_feature_count;
    }
    coo_dependees.Sort();
    // TODO(lew): Consider moving Dependees calculation to World::AddFeature.
    data_->GetMutableDependees()->FromCooMatrix(coo_dependees);
    world_->AddFeatures(first_j_added, product_map_->Size());
  }

  // TODO(lew,witoldjarnicki): Explicitly account for empty features and xjbools
  //                           per non-empty feature in the estimation.
  uint64 empty_features_skipped = 0;
  uint64 xjbools_count = data_->MaterializedXjBoolCount();
  uint64 xjbools_added = xjbools_count - previous_xjbool_count;
  // TODO(lew,witoldjarnicki): Consider taking longer than one iteration into
  //                           account.
  // TODO(lew,witoldjarnicki): Consider a "more continuous" condition.
  if (xjbools_added == 0) {
    xjbools_per_candidate_feature_estimate_ /= 2;
  } else {
    xjbools_per_candidate_feature_estimate_ =
        static_cast<double>(xjbools_added) / new_feature_count;
  }

  log->set_empty_features_skipped(empty_features_skipped);
  log->set_present_features_skipped(present_features_skipped);
  log->set_features_added(new_feature_count);
  log->set_xjbools_added(xjbools_added);
  log->set_xjbools_count(xjbools_count);
  LOG(INFO) << log->DebugString();
}

}  // namespace sensei

