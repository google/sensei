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
#ifndef BATCH_TRAINING_H__
#define BATCH_TRAINING_H__

#include <set>
using std::set;
#include <string>
using std::string;
#include <vector>
using std::vector;

#include "sensei/base/integral_types.h"
#include "sensei/base/macros.h"
#include "sensei/config.pb.h"
#include "sensei/data.h"
#include "sensei/feature_exploration.h"
#include "sensei/log.pb.h"
#include "sensei/world.h"
#include "sensei/thread/threadsafequeue.h"



namespace sensei {

//-----------------------------------------------------------------------------
class BatchTraining {
 public:
  explicit BatchTraining(const config::CommandList& command_list);

  void Run();
  void RunCommand(const config::Command& command);



 protected:  // For testing.
  void Init(const config::CommandList& command_list);

  void Set(const config::Set& set);
  void InitializeBias();
  void AddNewProductFeatures(const config::FeatureExploration& ti_fe);
  void PruneFeatures(const config::FeaturePruning& ti_fp);
  void FitModelWeights(const config::FitModelWeights& fit_model_weights);
  void RunSgdIteration(const config::RunSgd& run_sgd);
  void RunSgdCommand(const config::Sgd& sgd_command);
  void FillDataStatsWithAuc(const data::ShardSet& data,
                            Model::PerShard* per_shard,
                            logs::DataSetStats* data_set_stats);
  void EvaluateStats(const config::EvaluateStats& evaluate_stats);

 private:
  void RunInternalCommand(const config::Internal& internal);

  // Private member variables.
  World world_;

  WaitQueue<config::Command> command_queue_;

  vector<double> lift_fraction_;
  string name_;

  uint64 run_id_;

  DISALLOW_COPY_AND_ASSIGN(BatchTraining);
};

// Exported here for testing;
vector<set<string>> AllSubsetsOfSize(const vector<string>& elts, uint32 n);

}  // namespace sensei


#endif  // BATCH_TRAINING_H__
