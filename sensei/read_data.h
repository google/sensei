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
#ifndef SENSEI_READ_DATA_H_
#define SENSEI_READ_DATA_H_

#include "sensei/config.pb.h"


namespace sensei {

class World;

class ReadData {
 public:
  explicit ReadData(World* world);

  void RunCommand(const config::ReadData& config);

 private:
  void DataReader(const config::DataReader& config);
  void Set(const config::ReadData::Set& config);

  // Private member variables.
  config::ReadData::Set set_;
  World* world_;
};

}  // namespace sensei


#endif  // SENSEI_READ_DATA_H_
