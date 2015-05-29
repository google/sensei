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
#ifndef SENSEI_WRITE_MODEL_H_
#define SENSEI_WRITE_MODEL_H_

#include <memory>
#include <vector>
using std::vector;

#include "sensei/common.pb.h"
#include "sensei/config.pb.h"
#include "sensei/log.pb.h"


namespace sensei {

class World;

class WriteModel {
 public:
  explicit WriteModel(World* world);

  void RunCommand(const config::WriteModel& config);

  void RunCommand(const config::StoreModel& config);

  void RunCommand(const config::GetModel& config);


 private:
  void Write();

  logs::Model BuildModel();

  void StoreModel();

  vector<ModelWeight> GetModelWeights() const;

  logs::Model SelectModel();

  void GetModel();

  void WriteTextModel(const logs::Model& model);

  void WriteSerializedModel(const logs::Model& model);


  config::WriteModel::Set set_;
  vector<logs::Model> stored_models_;
  std::unique_ptr<logs::Model> output_model_;
  World* world_;
};

}  // namespace sensei


#endif  // SENSEI_WRITE_MODEL_H_
