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
#ifndef SENSEI_DATA_READER_H_
#define SENSEI_DATA_READER_H_

#include <string>
using std::string;
#include <vector>
using std::vector;

#include "sensei/base/integral_types.h"
#include "sensei/base/macros.h"
#include "sensei/common.h"
#include "sensei/config.pb.h"
#include "sensei/data.h"
#include "sensei/feature_map.h"
#include "sensei/world.h"
#include "sensei/thread/weighted-semaphore.h"
#include "sensei/util/dense_hash_set.h"


namespace sensei {

class FeatureSpec {
 public:
  // config must outlive this.
  FeatureSpec(const config::FeatureSpec& config, FeatureMap* feature_map,
              ProductMap* product_map);

  void AddFeatures(const vector<string>& bool_features, vector<uint32>* row);

 private:
  vector<uint32> PrefixToJs(const vector<string>& bool_features,
                            const string& prefix);

  // Private member variables.
  const config::FeatureSpec& config_;
  FeatureMap* feature_map_;
  ProductMap* product_map_;

  DISALLOW_COPY_AND_ASSIGN(FeatureSpec);
};

// -----------------------------------------------------------------------------
class ReadModel {
 public:
  ReadModel(const config::ReadModel& config, World* world);

  void AddFeatures(const vector<string>& bool_features, vector<uint32>* row);

 private:
  void ProcessCommand(const config::ReadModel& config);

  void UpdateWorld();

  template <typename WeightType>
  void ProcessModelWeight(const WeightType& weight);

  void ReadTextModel(const string& model_input_path);

  template <typename WeightType>
  void ReadSerializedModel(const string& model_input_path);

  bool FeaturePresent(const vector<string>& bool_features,
                      const vector<string>& product_feature);

  uint32 GetModelSize() { return model_js_.size(); }

  // Private member variables.
  uint32 max_model_j_;

  // The three vectors below will have the same size equal to the value returned
  // by GetModelSize
  vector<uint32> model_js_;
  vector<vector<string>> model_features_;
  vector<Double> weights_;

  World* world_;

  DISALLOW_COPY_AND_ASSIGN(ReadModel);
};

// -----------------------------------------------------------------------------
class DataReader {
 public:
  DataReader(const config::DataReader& data_reader_config,
             const config::ReadData::Set& read_data_set_config,
             const config::DataFiles::DataSet& data_set, data::ShardSet* data,
             World* world, WeightedSemaphore* semaphore,
             concurrency::AtomicCounter* row_id_sequence);

  void Run();

 private:
  void ReadFile(const string& fname);


  void ReadLibsvmFile(const string& fname);


  void AddRow(Double y, vector<string>* bool_features,
              vector<uint32>* buffer_row, data::Shard::Builder* shards,
              uint64* data_files_bool_feature_count) const;

  // Private member variables.
  config::DataReader data_reader_config_;
  config::ReadData::Set read_data_set_config_;

  vector<string> files_;
  FeatureMap* feature_map_;
  ProductMap* product_map_;
  data::ShardSet* data_;
  data::Adder data_adder_;  // TODO(lew): Merge this class with Shard::Builder.
  WeightedSemaphore* semaphore_;
  concurrency::AtomicCounter* row_id_sequence_;
  dense_hash_set<string> filter_feature_;
  World* world_;
  std::unique_ptr<ReadModel> read_model_;

  concurrency::AtomicCounter file_read_counter_;

  DISALLOW_COPY_AND_ASSIGN(DataReader);
};

// -----------------------------------------------------------------------------
class MultiDataReader {
 public:
  MultiDataReader(const config::DataReader& data_reader_config,
                  const config::ReadData::Set& read_data_set_config,
                  World* world);

  void Run();

 private:
  WeightedSemaphore semaphore_;
  concurrency::AtomicCounter row_id_sequence_;
  World* world_;
  DataReader training_data_reader_;
  DataReader holdout_data_reader_;
};

}  // namespace sensei


#endif  // SENSEI_DATA_READER_H_
