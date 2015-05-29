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
#include "sensei/data_reader.h"

#include <algorithm>
#include <unordered_set>
using std::unordered_set;
template<typename K>
using hash_set = unordered_set<K>;
#include <iterator>
#include <memory>
#include <string>
using std::string;
#include <vector>
using std::vector;

#include "sensei/base/integral_types.h"
#include "sensei/base/logging.h"
#include "sensei/file/file.h"
#include "sensei/file/recordio.h"
#include "sensei/common.h"
#include "sensei/config.pb.h"
#include "sensei/csr_matrix.h"
#include "sensei/data.h"
#include "sensei/feature_map.h"
#include "sensei/range.h"
#include "sensei/proto/text_format.h"
#include "sensei/strings/strutil.h"
#include "sensei/util/dense_hash_map.h"



namespace sensei {

// -----------------------------------------------------------------------------
// class FeatureSpec

FeatureSpec::FeatureSpec(const config::FeatureSpec& config,
                         FeatureMap* feature_map, ProductMap* product_map)
    : config_(config), feature_map_(feature_map), product_map_(product_map) {}

void FeatureSpec::AddFeatures(const vector<string>& bool_features,
                              vector<uint32>* row) {
  vector<vector<uint32>> product_factors;
  CHECK(config_.has_product());
  for (const string& prefix : config_.product().prefix())
    product_factors.push_back(PrefixToJs(bool_features, prefix));

  // TODO(lew): Make unique representation of pi.Get() to filter duplicates.
  typedef ProductIterator<uint32> PI;
  for (PI pi(product_factors); !pi.IsEmpty(); pi.Next())
    row->push_back(product_map_->FeatureToJ(JProduct(pi.Get())));
}

vector<uint32> FeatureSpec::PrefixToJs(const vector<string>& bool_features,
                                       const string& prefix) {
  vector<uint32> js;
  Range range = Range::WithPrefix(bool_features, prefix);
  // TODO(lew): Do not waste "j". Feature::Base does not need one.
  for (const string& f : range.SliceOfVector(bool_features))
    js.push_back(feature_map_->FeatureToJ(FeatureName(f)));
  return js;
}

// -----------------------------------------------------------------------------
// class ReadModel

ReadModel::ReadModel(const config::ReadModel& config, World* world)
    : max_model_j_(0), world_(world) {
  ProcessCommand(config);
}

void ReadModel::AddFeatures(const vector<string>& bool_features,
                            vector<uint32>* row) {
  for (const uint32 i : Range(GetModelSize())) {
    if (FeaturePresent(bool_features, model_features_[i])) {
      row->push_back(model_js_[i]);
    }
  }
}

void ReadModel::ProcessCommand(const config::ReadModel& config) {
  switch (config.format()) {
    case config::ModelFormat::TEXT:
      ReadTextModel(config.model_input_path());
      break;
    case config::ModelFormat::SERIALIZED:
      ReadSerializedModel<ModelWeight>(config.model_input_path());
      break;
  }
}

void ReadModel::UpdateWorld() {
  world_->model()->SetSize(max_model_j_ + 1);
  for (const uint32 i : Range(GetModelSize())) {
    world_->model()->w[model_js_[i]] = weights_[i];
  }
}

template <typename WeightType>
void ReadModel::ProcessModelWeight(const WeightType& weight) {
  vector<string> weight_features;
  vector<uint32> weight_js;
  for (const string& feature : weight.feature()) {
    weight_features.push_back(feature);
    weight_js.push_back(
        world_->feature_map()->FeatureToJ(FeatureName(feature)));
  }
  uint32 j = world_->product_map()->FeatureToJ(JProduct(weight_js));
  model_js_.push_back(j);
  max_model_j_ = std::max(max_model_j_, j);
  model_features_.push_back(weight_features);
  weights_.push_back(weight.weight());
}

void ReadModel::ReadTextModel(const string& model_input_path) {
  string contents;
  CHECK_OK(file::GetContents(model_input_path, &contents, file::Defaults()));
  logs::Model model;
  CHECK(google::protobuf::TextFormat::ParseFromString(contents, &model));
  for (const ModelWeight& weight : model.weight()) {
    ProcessModelWeight<ModelWeight>(weight);
  }
  UpdateWorld();
}

template <typename WeightType>
void ReadModel::ReadSerializedModel(const string& model_input_path) {
  RecordReader reader(file::OpenOrDie(model_input_path, "r", file::Defaults()));
  WeightType weight;
  while (reader.ReadProtocolMessage(&weight)) {
    ProcessModelWeight<WeightType>(weight);
  }
  UpdateWorld();
}

bool ReadModel::FeaturePresent(const vector<string>& bool_features,
                               const vector<string>& product_feature) {
  for (const string& feature : product_feature) {
    if (!std::binary_search(bool_features.begin(), bool_features.end(),
                            feature)) {
      return false;
    }
  }
  return true;
}

// -----------------------------------------------------------------------------
// class DataReader

DataReader::DataReader(const config::DataReader& data_reader_config,
                       const config::ReadData::Set& read_data_set_config,
                       const config::DataFiles::DataSet& data_set,
                       data::ShardSet* data, World* world,
                       WeightedSemaphore* semaphore,
                       concurrency::AtomicCounter* row_id_sequence)
    : data_reader_config_(data_reader_config),
      read_data_set_config_(read_data_set_config),
      feature_map_(world->feature_map()),
      product_map_(world->product_map()),
      data_(data),
      data_adder_(CHECK_NOTNULL(data)),
      semaphore_(semaphore),
      row_id_sequence_(row_id_sequence),
      filter_feature_(data_set.filter_feature().begin(),
                      data_set.filter_feature().end(), "" /* empty key */),
      world_(world) {
  CHECK_EQ(0, CHECK_NOTNULL(feature_map_)->Size());
  CHECK_EQ(0, CHECK_NOTNULL(product_map_)->Size());
  for (const string& glob : data_set.files_glob()) {
    vector<string> file_names;
    CHECK(File::Match(glob, &file_names)) << "error trying to match file_glob: "
                                          << AddQuotes(glob);
    QCHECK(!file_names.empty()) << "file_glob: " << AddQuotes(glob)
                                << " did not match any files.";
    for (const string& file_name : file_names) files_.push_back(file_name);
  }
  for (const string& filter_feature : data_set.filter_feature()) {
    CHECK(!filter_feature.empty());
  }
}

void DataReader::Run() {
  if (data_reader_config_.has_read_model()) {
    read_model_.reset(new ReadModel(data_reader_config_.read_model(), world_));
  }
  concurrency::ParFor(files_, [this](const string& fname) { ReadFile(fname); });
}

void DataReader::ReadFile(const string& fname) {
  concurrency::SemaphoreLock lock(semaphore_);
  LOG(INFO) << "Read: " << fname;
  FlushLogFiles(INFO);
  switch (data_reader_config_.format()) {
    case config::DataReader::LIBSVM:
      ReadLibsvmFile(fname);
      break;
  }
  uint32 file_no = file_read_counter_.GetNext() + 1;
  // TODO(lew): We also use twice the memory we should (after reading).
  LOG(INFO) << "Done " << file_no << "/" << files_.size() << "): " << fname;
}


void DataReader::ReadLibsvmFile(const string& fname) {
  uint64 data_files_bool_feature_count = 0;
  data::Shard::Builder data_shards(data_->GetMaxShardSize());
  string contents;
  QCHECK_OK(file::GetContents(fname, &contents, file::Defaults()));
  vector<StringPiece> lines =
      strings::Split(contents, "\n", strings::SkipEmpty());
  vector<uint32> buffer_row;
  for (const StringPiece& line : lines) {
    bool has_user_id = false;
    uint64 user_id;
    vector<string> tokens = strings::Split(
        line, strings::delimiter::AnyOf("\t \n"), strings::SkipEmpty());
    CHECK_GE(tokens.size(), 1);
    double y;
    CHECK(safe_strtod(tokens[0].c_str(), &y));
    CHECK(y == -1 || y == 1);
    vector<string> bool_features;
    for (uint32 k : Range(tokens.size())) {
      if (k == 0) continue;
      vector<string> pair =
          strings::Split(tokens[k], ":", strings::SkipEmpty());
      QCHECK_EQ(2, pair.size()) << "This: " << k << " " << tokens[k]
                                << std::endl
                                << "'" << line << "'";
      string feature_name = pair[0];
      if (feature_name == data_reader_config_.user_id_feature_name()) {
        CHECK(!has_user_id);
        has_user_id = true;
        CHECK(safe_strtou64(pair[1].c_str(), &user_id));
      } else {
      double x;
      CHECK(safe_strtod(pair[1].c_str(), &x));
      CHECK_EQ(1, x) << "We don't support continuous features in Sensei yet.";
      // TODO(lew,witoldjarnicki): Support of continuous features.
      // row->AddXj(x, feature_map_->FeatureToJ(Feature::Base(feature_name)));
      bool_features.push_back(feature_name);
      }
    }
    AddRow(y, &bool_features, &buffer_row, &data_shards,
           &data_files_bool_feature_count);
    if (has_user_id) {
      data_shards.AddUserId(user_id);
    }
  }
  data_adder_.Add(data_files_bool_feature_count, &data_shards);
}

void DataReader::AddRow(Double y, vector<string>* bool_features,
                        vector<uint32>* buffer_row,
                        data::Shard::Builder* shards,
                        uint64* data_files_bool_feature_count) const {
  if (!filter_feature_.empty()) {
    uint32 found = 0;
    for (const string& f : *bool_features) found += filter_feature_.count(f);
    if (found != filter_feature_.size()) return;
  }

  // TODO(lew): Takes 50% of the reading time. Consider prefix tree.
  std::sort(bool_features->begin(), bool_features->end());

  if (read_data_set_config_.has_output_feature()) {
    y = -1;
    if (std::binary_search(bool_features->begin(), bool_features->end(),
                           read_data_set_config_.output_feature())) {
      y = 1;
    }
  }

  if (data_reader_config_.has_remove_duplicate_features_in_each_row() &&
      data_reader_config_.remove_duplicate_features_in_each_row()) {
    // De-duplicate
    bool_features->resize(std::distance(
        bool_features->begin(),
        std::unique(bool_features->begin(), bool_features->end())));
  }
  CHECK(buffer_row->empty());
  // TODO(lew): Fix a situation when the same feature is observed twice.
  //            It breaks non-materialized feature specs in FE.
  if (read_model_ != nullptr) {
    read_model_->AddFeatures(*bool_features, buffer_row);
  } else {
    for (const config::FeatureSpec& fs_config :
         data_reader_config_.feature_spec()) {
      FeatureSpec feature_spec(fs_config, feature_map_, product_map_);
      feature_spec.AddFeatures(*bool_features, buffer_row);
    }
  }
  shards->AddRow(*buffer_row, y, row_id_sequence_->GetNext());
  *data_files_bool_feature_count += bool_features->size();
  buffer_row->clear();
  // TODO(lew): Remove/combine repeated features.
}

// -----------------------------------------------------------------------------
// class MultiDataReader

MultiDataReader::MultiDataReader(
    const config::DataReader& data_reader_config,
    const config::ReadData::Set& read_data_set_config, World* world)
    : semaphore_(data_reader_config.thread_count()),
      world_(world),
      training_data_reader_(data_reader_config, read_data_set_config,
                            data_reader_config.training_set(),
                            world->data()->GetMutableTraining(), world,
                            &semaphore_, &row_id_sequence_),
      holdout_data_reader_(data_reader_config, read_data_set_config,
                           data_reader_config.holdout_set(),
                           world->data()->GetMutableHoldout(), world,
                           &semaphore_, &row_id_sequence_) {}

void MultiDataReader::Run() {
  CHECK_EQ(0, world_->product_map()->Size());
  training_data_reader_.Run();
  holdout_data_reader_.Run();

  world_->feature_map()->SyncJToFeatureMap();
  world_->AddFeatures(0, world_->product_map()->Size());
}

}  // namespace sensei

