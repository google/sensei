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
#ifndef SENSEI_SCORE_ROWS_H_
#define SENSEI_SCORE_ROWS_H_

#include "sensei/config.pb.h"
#include "sensei/data.h"
#include "sensei/log.pb.h"
#include "sensei/model.h"


namespace sensei {

class World;

class ScoreRows {
 public:
  explicit ScoreRows(World* world);

  void RunCommand(const config::ScoreRows& config);
  logs::DataScore GetScoreProto();

 private:
  void WriteScores();
  void WriteTextScores();
  void ProcessShardSet(logs::DataScore* data_score,
                       const data::ShardSet& shard_set,
                       const Model::PerShard& per_shard);
  void ProcessShard(logs::DataScore* data_score, const data::Shard& shard,
                    const Model::PerShard& per_shard, uint64* per_shard_offset);

  // Private member variables.
  config::ScoreRows::Set set_;
  World* world_;
};

}  // namespace sensei


#endif  // SENSEI_SCORE_ROWS_H_
