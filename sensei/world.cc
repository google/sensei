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
#include "sensei/world.h"


namespace sensei {

internal::ProductMap::JProduct JToJProduct(const FeatureMap& feature_map,
                                           const ProductMap& product_map,
                                           uint32 j) {
  internal::ProductMap::JProduct j_product;
  j_product.set_j(j);
  for (uint32 feature_j : product_map.JToFeature(j).GetJs()) {
    j_product.add_and_j(feature_j);
    j_product.add_feature(feature_map.JToFeature(feature_j).GetBase());
  }
  return j_product;
}

World::World() {
  logger_.reset(new Logger());
  data_.reset(new data::Data());
  feature_map_.reset(new FeatureMap());
  product_map_.reset(new ProductMap());
  model_.reset(new Model());
  feature_scoring_.reset(
      new FeatureScoring(*data_, *feature_map_, *product_map_, logger_.get()));
  feature_exploration_.reset(new FeatureExploration(
      this, model_.get(), feature_map_.get(), product_map_.get(), data_.get(),
      feature_scoring_.get()));
  regularizations_.reset(new Regularizations());
  optimizer_.reset(new optimizer::GradBoost(*product_map_, model_.get(),
                                            *regularizations_, logger_.get()));
  sgd_.reset(new optimizer::Sgd(*product_map_, model_.get(), *regularizations_,
                                logger_.get()));
  feature_pruning_.reset(new FeaturePruning(this));
  read_data_.reset(new ReadData(this));
  score_rows_.reset(new ScoreRows(this));
  write_model_.reset(new WriteModel(this));
}

// TODO(lew): Make it private and inline.
void World::SetJSize(const uint32 size) {
  j_size_ = size;
  model_->SetSize(size);
  optimizer_->SetSize(size);
  sgd_->SetSize(size);
}

uint32 World::GetJSize() const { return j_size_; }

void World::AddFeatures(uint32 from_j, uint32 to_j) {
  CHECK_EQ(product_map_->Size(), to_j);
  product_map_->SyncJToFeatureMap();
  data_->GetMutableDependees()->SetRowCount(to_j);
  // TODO(lew): This is expensive data pass.
  data_->RecalcStats(to_j);
  SetJSize(to_j);
}

void World::RemoveAndRenumber(const JRenumbering& j_renumbering) {
  if (j_renumbering.IsNoOp()) return;
  j_size_ = j_renumbering.next_j();
  data_->RemoveAndRenumberJs(j_renumbering);
  optimizer_->RemoveAndRenumberJs(j_renumbering);
  product_map_->RemoveAndRenumberJs(j_renumbering);
}

internal::ProductMap::JProduct World::JToJProduct(uint32 j) const {
  return ::sensei::JToJProduct(*feature_map_, *product_map_, j);
}

}  // namespace sensei

