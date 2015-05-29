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
#include "sensei/score_rows.h"

#include "sensei/file/recordio.h"
#include "sensei/world.h"
#include "sensei/proto/text_format.h"


namespace sensei {

ScoreRows::ScoreRows(World* world) : world_(world) {}

void ScoreRows::RunCommand(const config::ScoreRows& config) {
  if (config.has_write_scores()) {
    switch (set_.format()) {
      case config::ScoreRows::UNKNOWN_FORMAT:
        LOG(FATAL) << "Output format was not specified.";
        break;
      case config::ScoreRows::SERIALIZED:
        WriteScores();
        break;
      case config::ScoreRows::TEXT:
        WriteTextScores();
        break;
    }
  } else if (config.has_set()) {
    set_.MergeFrom(config.set());
  } else {
    LOG(FATAL) << "Malformed score_rows command.";
  }
}

void ScoreRows::WriteScores() {
  RecordWriter writer(
      file::OpenOrDie(set_.output_fname(), "w", file::Defaults()));
  logs::DataScore data_score = GetScoreProto();
  for (const config::RowScore& row_score : data_score.row_score()) {
    CHECK(writer.WriteProtocolMessage(row_score));
  }
  CHECK(writer.Close());
}

void ScoreRows::WriteTextScores() {
  File* file = file::OpenOrDie(set_.output_fname(), "w", file::Defaults());
  string text;
  google::protobuf::TextFormat::PrintToString(GetScoreProto(), &text);
  CHECK_OK(file::WriteString(file, text, file::Defaults()));
  CHECK(file->Close());
}

logs::DataScore ScoreRows::GetScoreProto() {
  if (!world_->model()->synced_with_weights) {
    world_->optimizer()->SyncModelWithWeights();
  }
  logs::DataScore data_score;
  ProcessShardSet(&data_score, world_->data()->GetTraining(),
                  world_->model()->training);
  ProcessShardSet(&data_score, world_->data()->GetHoldout(),
                  world_->model()->holdout);
  return data_score;
}

void ScoreRows::ProcessShardSet(logs::DataScore* data_score,
                                const data::ShardSet& shard_set,
                                const Model::PerShard& per_shard) {
  uint64 per_shard_offset = 0;
  for (const data::Shard& shard : shard_set.GetShards()) {
    ProcessShard(data_score, shard, per_shard, &per_shard_offset);
  }
}

void ScoreRows::ProcessShard(logs::DataScore* data_score,
                             const data::Shard& shard,
                             const Model::PerShard& per_shard,
                             uint64* per_shard_offset) {
  for (const uint64 i : Range(shard.RowCount())) {
    config::RowScore* row_score = data_score->add_row_score();
    row_score->set_row_id(shard.UserIds()[i]);
    row_score->set_wx(per_shard.wxs[*per_shard_offset]);
    ++*per_shard_offset;
  }
}

}  // namespace sensei

