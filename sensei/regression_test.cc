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
#include <algorithm>
#include <memory>
#include <string>
using std::string;
#include <vector>
using std::vector;

#include "sensei/base/commandlineflags.h"
#include "sensei/base/integral_types.h"
#include "sensei/base/logging.h"
#include "sensei/file/file.h"
#include "sensei/file/path.h"
#include "sensei/file/temp_file.h"
#include "sensei/batch_training.h"
#include "sensei/common.h"
#include "sensei/config.h"
#include "sensei/config.pb.h"
#include "sensei/log.pb.h"
#include "sensei/range.h"
#include "sensei/proto/parse_text_proto.h"
#include "sensei/proto/text_format.h"
#include "sensei/proto/message_differencer.h"
#include "sensei/strings/strcat.h"
#define GTEST_HAS_TR1_TUPLE 0
#include "gtest/gtest.h"


DEFINE_string(client_dir, "",
              "If not empty, it must be a path to local client directory. "
              "The binary will overwrite logs in:"
              "sensei/testdata/regression/...");

DEFINE_string(test_srcdir, ".", "When building out of source provide the path "
                                "to sensei top-level directory as a value of "
                                "this flag.");



namespace sensei {


using proto_util::ParseTextOrDie;

struct Paths {
  struct PathPair {
    string actual;
    string expected;
  };
  PathPair output_model;
  PathPair text_log;
  PathPair scores;
};

void UpdateOutputPaths(Paths::PathPair* to_update, const string& base) {
  if (FLAGS_client_dir == "") {
    to_update->actual = TempFile::TempFilename(nullptr);
  } else {
    to_update->actual = file::JoinPath(FLAGS_client_dir, base);
  }
  to_update->expected = file::JoinPath(FLAGS_test_srcdir, base);
}

void UpdatePaths(config::Set* config, Paths* paths);
void UpdatePaths(config::DataReader* config, Paths* paths);
void UpdatePaths(config::ReadData* config, Paths* paths);
void UpdatePaths(config::WriteModel::Set* config, Paths* paths);
void UpdatePaths(config::WriteModel* config, Paths* paths);
void UpdatePaths(config::Command* config, Paths* paths);
void UpdatePaths(config::CommandList* config, Paths* paths);

void UpdatePaths(config::Set* config, Paths* paths) {
  if (config->has_logging() && config->logging().has_text_log_path()) {
    CHECK(!config->logging().has_recordio_log_path());
    UpdateOutputPaths(&paths->text_log, config->logging().text_log_path());
    config->mutable_logging()->set_text_log_path(paths->text_log.actual);
  }
}

void UpdatePaths(config::DataReader* config, Paths* paths) {
  for (uint32 i : Range(config->training_set().files_glob_size())) {
    config->mutable_training_set()->set_files_glob(
        i, file::JoinPath(FLAGS_test_srcdir,
                          config->mutable_training_set()->files_glob(i)));
  }
  for (uint32 i : Range(config->holdout_set().files_glob_size())) {
    config->mutable_holdout_set()->set_files_glob(
        i, file::JoinPath(FLAGS_test_srcdir,
                          config->mutable_holdout_set()->files_glob(i)));
  }
  if (config->has_read_model()) {
    config->mutable_read_model()->set_model_input_path(file::JoinPath(
        FLAGS_test_srcdir, config->read_model().model_input_path()));
  }
}

void UpdatePaths(config::ReadData* config, Paths* paths) {
  if (config->has_data_reader())
    UpdatePaths(config->mutable_data_reader(), paths);
}


void UpdatePaths(config::ScoreRows::Set* config, Paths* paths) {
  CHECK(config->has_output_fname());
  UpdateOutputPaths(&paths->scores, config->output_fname());
  config->set_output_fname(paths->scores.actual);
}

void UpdatePaths(config::ScoreRows* config, Paths* paths) {
  if (config->has_set()) UpdatePaths(config->mutable_set(), paths);
}

void UpdatePaths(config::WriteModel::Set* config, Paths* paths) {
  if (config->has_output_model_path()) {
    UpdateOutputPaths(&paths->output_model, config->output_model_path());
    config->set_output_model_path(paths->output_model.actual);
  }
}

void UpdatePaths(config::WriteModel* config, Paths* paths) {
  if (config->has_set()) UpdatePaths(config->mutable_set(), paths);
}


void UpdatePaths(config::Command* config, Paths* paths) {
  if (config->has_read_data()) UpdatePaths(config->mutable_read_data(), paths);
  if (config->has_set()) UpdatePaths(config->mutable_set(), paths);
  if (config->has_score_rows())
    UpdatePaths(config->mutable_score_rows(), paths);
  if (config->has_write_model())
    UpdatePaths(config->mutable_write_model(), paths);
}

void UpdatePaths(config::CommandList* config, Paths* paths) {
  for (config::Command& command : *config->mutable_command())
    UpdatePaths(&command, paths);
}

const string kRegressionTestPath = "/sensei/testdata/regression/";

void TestOneRun(const string& config_name) {
  const string config_path = StrCat(FLAGS_test_srcdir, kRegressionTestPath,
                                    config_name, ".config.Flag");

  LOG(INFO);
  LOG(INFO);
  LOG(INFO) << "TestOneRun(" << AddQuotes(config_path) << ")";
  string contents;
  QCHECK_OK(file::GetContents(config_path, &contents, file::Defaults()));
  config::Flag flag = ParseTextOrDie<config::Flag>(contents);
  config::CommandList command_list = flag.command_list();
  config::Validator::ValidateOrDie(command_list);

  Paths paths;

  // Update paths.
  CHECK_GE(command_list.command().size(), 1);
  UpdatePaths(&command_list, &paths);

  CHECK_NE("", paths.text_log.actual);
  CHECK_NE("", paths.text_log.expected);

  // Do the training.
  {
    BatchTraining batch_training(command_list);
    batch_training.Run();
    // Flush logs on batch_training destruction.
  }


  // Compare Log.
  proto_util::MessageDifferencer message_differencer;
  string message_differencer_report;
  message_differencer.ReportDifferencesToString(&message_differencer_report);
  message_differencer.IgnoreField(
      config::DataReader::descriptor()->FindFieldByName("training_set"));
  message_differencer.IgnoreField(
      config::DataReader::descriptor()->FindFieldByName("holdout_set"));
  message_differencer.IgnoreField(
      config::ReadModel::descriptor()->FindFieldByName("model_input_path"));
  message_differencer.IgnoreField(
      logs::Line::descriptor()->FindFieldByName("timestamp"));
  message_differencer.IgnoreField(
      logs::Line::descriptor()->FindFieldByName("run_id"));
  message_differencer.IgnoreField(
      config::Set::Logging::descriptor()->FindFieldByName("text_log_path"));
  message_differencer.IgnoreField(
      config::ScoreRows::Set::descriptor()->FindFieldByName("output_fname"));
  message_differencer.IgnoreField(
      config::WriteModel::Set::descriptor()->FindFieldByName(
          "output_model_path"));
  message_differencer.IgnoreField(internal::Data::Stats::JStat::descriptor()->
      FindFieldByName("hash"));


  string expected_text_log;
  CHECK_OK(file::GetContents(paths.text_log.expected, &expected_text_log,
                             file::Defaults()));
  logs::Lines expected_log_lines;
  google::protobuf::TextFormat::ParseFromString(expected_text_log, &expected_log_lines);

  string text_log;
  CHECK_OK(
      file::GetContents(paths.text_log.actual, &text_log, file::Defaults()));
  logs::Lines log_lines;
  google::protobuf::TextFormat::ParseFromString(text_log, &log_lines);

  ASSERT_TRUE(message_differencer.Compare(expected_log_lines, log_lines))
      << message_differencer_report << "\n"
      << "Expected log: " << paths.text_log.expected << "\n"
      << "Log:          " << paths.text_log.actual;
}

TEST(RegressionTest, b17267972) { TestOneRun("b17267972"); }
TEST(RegressionTest, T1) { TestOneRun("t1"); }
TEST(RegressionTest, T1_sgd) { TestOneRun("t1_sgd"); }
TEST(RegressionTest, T4) { TestOneRun("t4"); }
TEST(RegressionTest, T4_L0) { TestOneRun("t4_L0"); }
TEST(RegressionTest, T4_output) { TestOneRun("t4_output"); }
TEST(RegressionTest, T5) { TestOneRun("t5"); }
TEST(RegressionTest, T5_sgd) { TestOneRun("t5_sgd"); }
TEST(RegressionTest, TopPercentPruning) { TestOneRun("top_fraction_pruning"); }
TEST(RegressionTest, MultiShard) { TestOneRun("multi_shard"); }
TEST(RegressionTest, FeWithBonus) { TestOneRun("fe_with_bonus"); }
TEST(RegressionTest, UnequalTrainingAndHoldout) { TestOneRun("unequal"); }
TEST(RegressionTest, LibsvmScoring) { TestOneRun("libsvm_scoring"); }

}  // namespace sensei


#include "sensei/base/init_google.h"
int main(int argc, char** argv) {
  InitGoogle(argv[0], &argc, &argv, true);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

