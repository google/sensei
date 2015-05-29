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
#include "sensei/feature_pruning.h"

#include <unordered_map>
using std::unordered_map;
template<typename K, typename V>
using hash_map = unordered_map<K,V>;
#include <memory>
#include <queue>
using std::priority_queue;
#include <utility>
using std::pair;
#include <vector>
using std::vector;

#include "sensei/base/integral_types.h"
#include "sensei/base/logging.h"
#include "sensei/common.h"
#include "sensei/config.pb.h"
#include "sensei/csr_matrix.h"
#include "sensei/data.h"
#include "sensei/j_renumbering.h"
#include "sensei/world.h"
#include "sensei/util/array_slice.h"

using ::util::gtl::ArraySlice;


namespace sensei {

FeaturePruning::FeaturePruning(World* world) : world_(world) {}

void FeaturePruning::PruneFeatures(const config::FeaturePruning& config,
                                   logs::FeaturePruning* log) {
  // TODO(lew): Give an option of sorting according to j_score * feature_size.

  // We are scoring the features again, because the feature ordering may be
  // different than the ones used in feature exploration.  This is not terribly
  // costly - roughly O(model size).
  vector<Double> j_to_score;
  world_->feature_scoring()->ScoreFeatures(
      *world_->model(), config.feature_scoring(), &j_to_score);
  vector<bool> removed_js = ComputePruning(config, j_to_score, log);
  RemoveJs(removed_js);
}

vector<bool> FeaturePruning::ComputePruning(
    const config::FeaturePruning& config, const vector<Double>& j_to_score,
    logs::FeaturePruning* log) const {
  // We schedule things for removal and partially modify
  // world_->data()->dependees.
  struct ScoreComparator {
    bool operator()(const pair<Double, uint32>& lhs,
                    const pair<Double, uint32>& rhs) const {
      return lhs.first > rhs.first;
    }
  };
  priority_queue<pair<Double, uint32>, vector<pair<Double, uint32>>,
                 ScoreComparator> js_queue;
  for (const ProductMap::FeatureAndJ* f : world_->product_map()->GetAll()) {
    uint32 j = f->GetJ();
    js_queue.push(std::make_pair(j_to_score[j], j));
  }

  // Used for triggering the rechecking of blocked js.
  hash_map<uint32, vector<uint32>> triggered_by;

  uint64 blocked_features = 0;
  uint64 blocked_xjbools = 0;
  uint64 features_removed = 0;
  uint64 xjbools_count = world_->data()->MaterializedXjBoolCount();
  uint64 xjbools_removed = 0;
  uint64 feature_count = js_queue.size();

  // Invariant: first num_removed_children[j] children of j are known to have
  //            been removed.
  hash_map<uint32, uint32> num_removed_children;

  // Used for RemoveJs.
  vector<bool> removed_js(world_->model()->GetSize(), false);

  while (true) {
    if (js_queue.empty()) break;
    Double score = js_queue.top().first;
    if (score == kInfinity) break;
    if (config.has_score_threshold() && score > config.score_threshold()) break;
    if (config.has_top_count() && js_queue.size() <= config.top_count()) break;
    if (config.has_top_fraction()) {
      double top_feature_count = feature_count * config.top_fraction();
      if (js_queue.size() <= std::lround(top_feature_count)) break;
    }

    uint32 j = js_queue.top().second;
    js_queue.pop();

    // Check all js scheduled for removal(kInvalidJ) from the head of
    // dependees[j].  They will be actually removed below (see RemoveFeatures
    // below).
    const CsrMatrix::Row& j_dependees =
        world_->data()->GetDependees().GetRow(j);

    uint32* num_removed_children_at_j = &num_removed_children[j];
    while (true) {
      if (*num_removed_children_at_j >= j_dependees.size()) break;
      uint32 j_dependee = j_dependees[*num_removed_children_at_j];
      if (!removed_js[j_dependee]) break;
      ++*num_removed_children_at_j;
    }

    // Is <j> ready to be removed?
    if (*num_removed_children_at_j < j_dependees.size()) {
      // Declare <j> as blocked by one of its dependees.
      uint32 j_dependee = j_dependees[*num_removed_children_at_j];
      triggered_by[j_dependee].push_back(j);
    } else {
      features_removed += 1;
      removed_js[j] = true;
      xjbools_removed += world_->data()->XjBoolCountOfJ(j);
      // Enqeueue all previously blocked.  They could have been unblocked by
      // the removal of j.
      for (uint32 triggered : triggered_by[j])
        js_queue.push(std::make_pair(j_to_score[triggered], triggered));
      triggered_by.erase(j);
    }
  }

  // TODO(lew): Check that rows, dependees, and stats are consistent.
  for (const pair<uint32, vector<uint32>>& j_and_js_blocked : triggered_by) {
    const vector<uint32>& js_blocked = j_and_js_blocked.second;
    blocked_features += js_blocked.size();
    for (const uint32 j : js_blocked)
      blocked_xjbools += world_->data()->XjBoolCountOfJ(j);
  }

  // Since now Js are renumbered so all the datastructures above are invalid.

  xjbools_count -= xjbools_removed;
  log->set_blocked_features(blocked_features);
  log->set_blocked_xjbools(blocked_xjbools);
  log->set_features_removed(features_removed);
  log->set_xjbools_count(xjbools_count);
  log->set_xjbools_removed(xjbools_removed);

  return removed_js;
}

void FeaturePruning::RemoveJs(const vector<bool>& removed_js) {
  world_->RemoveAndRenumber(JRenumbering::RemoveJs(removed_js));
}

}  // namespace sensei

