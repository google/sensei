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
#include <memory>
#include <string>
using std::string;

#include "sensei/base/commandlineflags.h"
#include "sensei/base/commandlineflags_completions.h"
#include "sensei/base/init_google.h"
#include "sensei/base/logging.h"
#include "sensei/batch_training.h"
#include "sensei/common.h"
#include "sensei/config.h"
#include "sensei/config.pb.h"

DEFINE_bool(validate_config_only, false,
            "Do not do any training, just test config.");


// TODO(witoldjarnicki): Deprecate direct sensei.config.BatchTraining support.
DEFINE_string(config_files, "",
              "Comma separated list of paths to files with protobuf "
              "sensei.Flag (in text format).  "
              "All protos need to be of the same type and "
              "will be merged (from left to right) and "
              "then merged with the one given in '--config' flag "
              "to create the final proto.");

DEFINE_string(config, "",
              "Protobuf sensei.Flag in text format.  "
              "Also PTAL at '--config_files' flag.");



namespace sensei {

void RunBatchTraining(const sensei::config::CommandList& command_list) {
  sensei::BatchTraining batch_training(command_list);


  batch_training.Run();

}

void Main() {

  config::CommandList command_list =
      config::CommandListFromFlags(FLAGS_config_files, FLAGS_config);
  if (FLAGS_validate_config_only) {
    LOG(INFO) << "\n" << command_list.DebugString();
    return;
  }
  concurrency::InitConcurrency(
      std::bind(RunBatchTraining, std::cref(command_list)));
}

}  // namespace sensei


int main(int argc, char** argv) {
  InitGoogle(argv[0], &argc, &argv, true);
  HandleCommandLineCompletions();
  sensei::Main();
  return 0;
}
