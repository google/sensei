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
#ifndef SENSEI_CONFIG_H_
#define SENSEI_CONFIG_H_

#include <memory>
#include <string>
using std::string;
#include <vector>
using std::vector;

#include "sensei/base/logging.h"
#include "sensei/config.pb.h"


namespace sensei {
namespace config {

class TextLogger {
 public:
  TextLogger() : is_valid_(true) {}

  bool IsValid() const { return is_valid_; }
  const vector<string>& GetMessages() const { return messages_; }

  void AddWarning(const string& message);
  void AddError(const string& message);

  void DieOnError() const {
    if (IsValid()) return;
    for (const string& s : GetMessages()) LOG(ERROR) << s;
    LOG(FATAL) << "Invalid config.";
  }

  void DieOnWarning() const {
    if (messages_.empty()) return;
    for (const string& s : messages_) LOG(ERROR) << s;
    LOG(FATAL) << "Config with warnings treated as invalid.";
  }

 private:
  bool is_valid_;
  vector<string> messages_;
};

class Validator {
 public:
  Validator()
      : text_logger_owned_(new TextLogger()),
        text_logger_(text_logger_owned_.get()) {}

  const vector<string>& GetMessages() const {
    return text_logger_->GetMessages();
  }

  bool IsValid() const { return text_logger_->IsValid(); }

  // Please keep the Process() methods grouped and ordered EXACTLY as in
  // config.proto.

  // ---------------------------------------------------------------------------
  // Public
  void Process(const DataFiles& config);
  void Process(const DataFiles::DataSet& config);
  void Process(const Flag& flag);

  // ---------------------------------------------------------------------------
  // Experimental.

  void Process(const FeatureSpec& config);
  void Process(const FeatureSpec_Product& config);
  void Process(const ReadModel& config);
  void Process(const DataReader& config);
  void Process(const FeatureScoring::Bonus& config);
  void Process(const FeatureScoring& config);
  void Process(const FeatureExploration& config);
  void Process(const FeaturePruning& config);

  // ---------------------------------------------------------------------------
  // Experimental.
  void Process(const Set& config);
  void Process(const Set::Logging& config);
  void Process(const Set::Regularization& config);
  void Process(const Set::SgdLearningRateSchedule& config);
  void Process(const ReadData& config);
  void Process(const ReadData::Set& config);
  void Process(const InitializeBias& config);
  void Process(const AddNewProductFeatures& config);
  void Process(const PruneFeatures& config);
  void Process(const FitModelWeights& config);
  void Process(const RunSgd& config);
  void Process(const Sgd::LearningRate::StoreTotalLoss& config);
  void Process(const Sgd::LearningRate::MaybeReduce& config);
  void Process(const Sgd::LearningRate& config);
  void Process(const Sgd& config);
  void Process(const EvaluateStats& config);
  void Process(const StoreModel& config);
  void Process(const WriteModel::Set& config);
  void Process(const WriteModel::Write& config);
  void Process(const WriteModel& config);
  void Process(const GetModel& config);
  void Process(const ScoreRows::Set& config);
  void Process(const ScoreRows::WriteScores& config);
  void Process(const ScoreRows& config);
  void Process(const Quit& config);
  void Process(const Repeat& config);
  void Process(const FromFile& config);
  void Process(const FeatureSet& config);
  void Process(const ExplicitFeatureList& config);
  void Process(const FeatureSet::FromData& config);
  void Process(const Internal& config);
  void Process(const Internal::GetModel& config);
  void Process(const Internal::LogDetailedStats& config);
  void Process(const Internal::LogDependees& config);
  void Process(const Internal::GetData& config);
  void Process(const Internal::GetScores& config);

  void Process(const Command& config);
  void Process(const CommandList& config);

  template <class T>
  static void ValidateOrDie(const T& message) {
    Validator validator;
    validator.Process(message);
    validator.text_logger_->DieOnError();
  }

  template <class T>
  static void ValidateOrDieOnWarning(const T& message) {
    Validator validator;
    validator.Process(message);
    validator.text_logger_->DieOnWarning();
  }

 private:
  ::std::unique_ptr<TextLogger> text_logger_owned_;
  TextLogger* text_logger_;
};


// -----------------------------------------------------------------------------

CommandList CommandListFromFlags(const string& flag_text_files,
                                 const string& flag_text);

}  // namespace config
}  // namespace sensei


#endif  // SENSEI_CONFIG_H_
