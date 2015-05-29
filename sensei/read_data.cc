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
#include "sensei/read_data.h"

#include "sensei/config.h"
#include "sensei/config.pb.h"
#include "sensei/data_reader.h"
#include "sensei/model.h"
#include "sensei/world.h"


namespace sensei {

ReadData::ReadData(World* world) : world_(world) {}

void ReadData::RunCommand(const config::ReadData& config) {
  if (config.has_data_reader()) {
    DataReader(config.data_reader());
    return;
  }

  if (config.has_set()) {
    Set(config.set());
    return;
  }

  LOG(FATAL) << "Unknown ReadData subcommand: " << config.DebugString();
}

void ReadData::DataReader(const config::DataReader& config) {
  config::DataReader copy(config);
  // Read data and fill world_->feature_map()
  MultiDataReader(copy, set_, world_).Run();
  world_->feature_map()->LogStats();
  world_->product_map()->LogStats();

  world_->data()->LogStats();

  CHECK_GE(world_->product_map()->Size(), world_->model()->GetSize());
  world_->model()->InitPerShards(
      world_->data()->GetTraining().GetStats().RowCount(),
      world_->data()->GetHoldout().GetStats().RowCount());

  world_->optimizer()->SetData(*world_->data());
  world_->sgd()->SetData(*world_->data());
}

void ReadData::Set(const config::ReadData::Set& config) {
  set_.MergeFrom(config);
}

}  // namespace sensei

