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
#ifndef SENSEI_WORLD_H_
#define SENSEI_WORLD_H_

#include <memory>

#include "sensei/common.h"
#include "sensei/feature_exploration.h"
#include "sensei/feature_map.h"
#include "sensei/feature_pruning.h"
#include "sensei/optimizers.h"
#include "sensei/read_data.h"
#include "sensei/score_rows.h"
#include "sensei/sgd.h"
#include "sensei/write_model.h"


namespace sensei {

class World {
 public:
  World();

  void SetJSize(const uint32 size);
  uint32 GetJSize() const;

  // Inform this World that a range of features [from_j, to_j) was just added.
  // All modules must call this methods when they add, remove or renumber
  // features.
  // TODO(lew): It would be nice if we could fit all world-wide effects into
  //            this method
  void AddFeatures(uint32 from_j, uint32 to_j);
  // Ask World to remove and renumber features in all modules.
  void RemoveAndRenumber(const JRenumbering& j_renumbering);

  // TODO(lew): Remove this one when we move to "world everywhere".
  // Legacy, do not use.

  internal::ProductMap::JProduct JToJProduct(uint32 j) const;

  data::Data* data() { return data_.get(); }
  FeatureMap* feature_map() { return feature_map_.get(); }
  ProductMap* product_map() { return product_map_.get(); }
  optimizer::GradBoost* optimizer() { return optimizer_.get(); }
  optimizer::Sgd* sgd() { return sgd_.get(); }
  FeatureExploration* feature_exploration() {
    return feature_exploration_.get();
  }
  FeaturePruning* feature_pruning() { return feature_pruning_.get(); }
  FeatureScoring* feature_scoring() { return feature_scoring_.get(); }
  Model* model() { return model_.get(); }
  Regularizations* regularizations() { return regularizations_.get(); }
  Logger* logger() { return logger_.get(); }
  ReadData* read_data() { return read_data_.get(); }
  ScoreRows* score_rows() { return score_rows_.get(); }
  WriteModel* write_model() { return write_model_.get(); }

  const data::Data& data() const { return *data_; }
  const FeatureMap& feature_map() const { return *feature_map_; }
  const ProductMap& product_map() const { return *product_map_; }
  const optimizer::GradBoost& optimizer() const { return *optimizer_; }
  const optimizer::Sgd& sgd() const { return *sgd_; }
  const FeatureExploration& feature_exploration() const {
    return *feature_exploration_;
  }
  const FeaturePruning& feature_pruning() const { return *feature_pruning_; }
  const FeatureScoring& feature_scoring() const { return *feature_scoring_; }
  const Model& model() const { return *model_; }
  const Regularizations& regularizations() const { return *regularizations_; }
  const Logger& logger() const { return *logger_; }
  const ReadData& read_data() const { return *read_data_; }
  const ScoreRows& score_rows() const { return *score_rows_; }
  const WriteModel& write_model() const { return *write_model_; }

 private:
  std::unique_ptr<data::Data> data_;
  // TODO(witoldjarnicki): Simplify the two maps below.
  // Contains only the j<->string mapping.
  std::unique_ptr<FeatureMap> feature_map_;
  // Contains only the j<->{j...} mapping.
  std::unique_ptr<ProductMap> product_map_;
  std::unique_ptr<optimizer::GradBoost> optimizer_;
  std::unique_ptr<optimizer::Sgd> sgd_;
  std::unique_ptr<FeatureExploration> feature_exploration_;
  std::unique_ptr<FeaturePruning> feature_pruning_;
  std::unique_ptr<FeatureScoring> feature_scoring_;
  std::unique_ptr<Model> model_;
  std::unique_ptr<Regularizations> regularizations_;
  std::unique_ptr<Logger> logger_;
  std::unique_ptr<ReadData> read_data_;
  std::unique_ptr<ScoreRows> score_rows_;
  std::unique_ptr<WriteModel> write_model_;
  uint32 j_size_;
};

// TODO(lew): Remove this when FeatureScoring moves to 'world'.
internal::ProductMap::JProduct JToJProduct(const FeatureMap& feature_map,
                                           const ProductMap& product_map,
                                           uint32 j);
}  // namespace sensei


#endif  // SENSEI_WORLD_H_
