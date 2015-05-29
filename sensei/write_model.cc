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
#include "sensei/write_model.h"

#include "sensei/file/recordio.h"
#include "sensei/proto/text_format.h"
#include "sensei/world.h"


namespace sensei {

WriteModel::WriteModel(World* world) : world_(world) {}

vector<ModelWeight> WriteModel::GetModelWeights() const {
  vector<ModelWeight> model_weights;

  for (const ProductMap::FeatureAndJ* feature_and_j :
       world_->product_map()->GetAll()) {
    ModelWeight model_weight;
    uint32 j = feature_and_j->GetJ();
    if (world_->model()->w[j] == 0.0) continue;
    model_weight.set_weight(world_->model()->w[j]);
    for (const string& feature :
         feature_and_j->GetKey().GetFactorNames(*world_->feature_map())) {
      model_weight.add_feature(feature);
    }
    model_weights.push_back(model_weight);
  }

  std::sort(model_weights.begin(), model_weights.end(),
            [](const ModelWeight& i, const ModelWeight& j) {
              return i.weight() < j.weight();
            });
  return model_weights;
}

void WriteModel::StoreModel() { stored_models_.push_back(BuildModel()); }

logs::Model WriteModel::BuildModel() {
  if (!world_->model()->Empty()) {
    logs::Model model;
    world_->optimizer()->SyncModelWithWeights();
    *model.mutable_last_iteration() =
        world_->optimizer()->GetLastIterationLog();
    for (const ModelWeight& model_weight : GetModelWeights())
      *model.add_weight() = model_weight;
    return model;
  }
  LOG(FATAL) << "This shouldn't happen.";
}

namespace {
double StoredModelEval(const logs::Model& model,
                       const double regularization_l0) {
  const logs::Iteration& iteration = model.last_iteration();
  double total_loss = iteration.total_loss();
  double rule_count = iteration.weight_stats().nonzero_count();
  return -(total_loss + rule_count * regularization_l0);
}
}  // namespace

logs::Model WriteModel::SelectModel() {
  if (set_.select_best_stored()) {
    // Find best (logloss) model.
    CHECK(!stored_models_.empty());
    const logs::Model* best_model = &stored_models_[0];
    double regularization_l0 = set_.regularization_l0();
    for (const logs::Model& model : stored_models_) {
      if (StoredModelEval(model, regularization_l0) >
          StoredModelEval(*best_model, regularization_l0)) {
        best_model = &model;
      }
    }
    return *best_model;
  } else {
    return BuildModel();
  }
}

void WriteModel::GetModel() {
  logs::Line log_line;
  *log_line.mutable_model() = BuildModel();
  world_->logger()->AddToLogs(log_line);
}

namespace {
template <typename OutputType>
void WriteTransformedModel(
    const logs::Model& model, string output_model_path,
    std::function<OutputType(const ModelWeight&)> transformation) {
  File* file;
  CHECK_OK(file::Open(output_model_path, "w", &file, file::Defaults()));
  RecordWriter writer(file);
  for (const ModelWeight& model_weight : model.weight())
    CHECK(writer.WriteProtocolMessage(transformation(model_weight)));
  CHECK(writer.Close());
}
}  // namespace

void WriteModel::WriteTextModel(const logs::Model& model) {
  File* file = file::OpenOrDie(set_.output_model_path(), "w", file::Defaults());
  string text;
  google::protobuf::TextFormat::PrintToString(model, &text);
  CHECK_OK(file::WriteString(file, text, file::Defaults()));
  CHECK(file->Close());
}

void WriteModel::WriteSerializedModel(const logs::Model& model) {
  WriteTransformedModel<ModelWeight>(model, set_.output_model_path(),
                                     [](const ModelWeight& w) { return w; });
}

void WriteModel::RunCommand(const config::WriteModel& config) {
  if (config.has_set()) {
    if (config.set().has_output_model_path()) {
      CheckCanWrite(config.set().output_model_path(), true);
    }
    set_.MergeFrom(config.set());
  } else if (config.has_write()) {
    Write();
  } else {
    LOG(FATAL) << "Malformed WriteModel command.";
  }
}

void WriteModel::Write() {
  logs::Model model = SelectModel();

  logs::Line log_line;
  *log_line.mutable_write_model()->mutable_last_iteration() =
      model.last_iteration();
  world_->logger()->AddToLogs(log_line);

  if (set_.output_model_path() == "") {
    output_model_.reset(new logs::Model);
    *output_model_ = model;
    return;
  }

  CHECK(set_.has_format());

  switch (set_.format()) {
    case config::ModelFormat::TEXT:
      WriteTextModel(model);
      break;
    case config::ModelFormat::SERIALIZED:
      WriteSerializedModel(model);
      break;
  }
}

void WriteModel::RunCommand(const config::StoreModel& config) { StoreModel(); }

void WriteModel::RunCommand(const config::GetModel& config) { GetModel(); }


}  // namespace sensei

