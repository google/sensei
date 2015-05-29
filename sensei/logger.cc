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
#include "sensei/logger.h"

#include <stddef.h>

#include "sensei/base/logging.h"
#include "sensei/base/time.h"
#include "sensei/file/file.h"
#include "sensei/file/recordio.h"
#include "sensei/common.h"
#include "sensei/proto/text_format.h"
#include "sensei/strings/strutil.h"


namespace sensei {

namespace {

// Formats IterationLog in one line log with constant field widths.
// Repeated printing produces easily readable column format.
string IterationLogToString(const logs::Iteration& log) {
  string ret;
  if (log.has_index()) ret += StringPrintf("I %5d: ", log.index());

  Double t_size = log.training_data_stats().size();
  Double t_loss = log.training_data_stats().loss();
  Double t_dloss_l1 = log.training_data_stats().dloss().l1();
  Double t_auc = log.training_data_stats().auc();

  Double h_size = log.holdout_data_stats().size();
  Double h_loss = log.holdout_data_stats().loss();
  Double h_auc = log.holdout_data_stats().auc();

  Double r_loss = log.regularization_stats().loss();

  Double w_nonzero_count = log.weight_stats().nonzero_count();
  Double w_l1 = log.weight_stats().l1();

  if (log.training_data_stats().has_loss() &&
      log.training_data_stats().has_size()) {
    ret += "L(t)/#t = " + ToString(t_loss / t_size) + " ";
  }

  if (log.training_data_stats().has_auc())
    ret += "Auc(t) = " + ToString(t_auc) + " ";

  if (log.holdout_data_stats().has_loss() &&
      log.holdout_data_stats().has_size()) {
    ret += "L(h)/#h = " + ToString(h_loss / h_size) + " ";
  }

  if (log.holdout_data_stats().has_auc())
    ret += "Auc(h) = " + ToString(h_auc) + " ";

  if (log.training_data_stats().has_loss() &&
      log.holdout_data_stats().has_loss()) {
    ret += "L(t)+L(r) = " + ToString(t_loss + r_loss) + " ";
  }

  if (log.training_data_stats().dloss().has_l1())
    ret += "L1(dL(t)+dL(r)) = " + ToString(t_dloss_l1) + " ";

  if (log.weight_stats().has_nonzero_count())
    ret += "sum(w != 0) = " + StringPrintf("%7f", w_nonzero_count) + " ";

  if (log.weight_stats().has_l1()) ret += "L1(w) = " + ToString(w_l1) + " ";

  return ret;
}

}  // namespace

Logger::Logger() : log_timestamp_(true), run_id_(0) {}

Logger::~Logger() {
  if (!text_log_path_.empty()) {
    text_queue_.StopWaiters();
    text_fiber_->Join();
  }
  if (!recordio_log_path_.empty()) {
    recordio_queue_.StopWaiters();
    recordio_fiber_->Join();
  }
}

void Logger::SetRunId(const uint64 run_id) { run_id_ = run_id; }


void Logger::SetLogTimestamp(bool log_timestamp) {
  log_timestamp_ = log_timestamp;
}

void Logger::SetTextLogPath(string path) {
  CHECK(text_log_path_.empty());
  CHECK(!path.empty());
  CHECK(text_fiber_ == nullptr);
  text_log_path_ = path;
  text_fiber_.reset(new concurrency::Thread([this] { LogText(); }));
}

void Logger::SetRecordioLogPath(string path) {
  CHECK(recordio_log_path_.empty());
  CHECK(!path.empty());
  CHECK(recordio_fiber_ == nullptr);
  recordio_log_path_ = path;
  recordio_fiber_.reset(new concurrency::Thread([this] { LogRecordio(); }));
}

void Logger::AddToLogs(const logs::Line& log_line) {
  CHECK_EQ(1, (log_line.has_batch_training_config() +  //
               log_line.has_feature_exploration() +    //
               log_line.has_feature_pruning() +        //
               log_line.has_write_model() +                //
               log_line.has_iteration() +                  //
               log_line.has_grad_boost_update_minimum() +  //
               log_line.has_sgd() +                        //
               log_line.has_command_list_config() +        //
               log_line.has_run_command() +                //
               log_line.has_model() +                      //
               log_line.has_internal_model() +            //
               log_line.has_internal_detailed_stats() +   //
               log_line.has_internal_dependees() +        //
               log_line.has_internal_data() +             //
               log_line.has_internal_feature_scoring() +  //
               log_line.has_data_score() +  //
               0));
  logs::Line log_line_copy(log_line);
  if (log_timestamp_) log_line_copy.set_timestamp(ToUnixNanos(base::Now()));
  if (run_id_ > 0) log_line_copy.set_run_id(run_id_);
  if (log_line_copy.has_iteration())
    LOG(INFO) << IterationLogToString(log_line_copy.iteration());
  recordio_queue_.push(log_line_copy);
  text_queue_.push(log_line_copy);
}

void Logger::LogRecordio() {
  CHECK(!recordio_log_path_.empty());
  logs::Line log_line;
  while (recordio_queue_.Wait(&log_line)) {
    RecordWriter writer(
        file::OpenOrDie(recordio_log_path_, "a", file::Defaults()));
    CHECK(writer.WriteProtocolMessage(log_line));
    while (recordio_queue_.Pop(&log_line))
      CHECK(writer.WriteProtocolMessage(log_line));
    CHECK(writer.Close());
  }
}

void Logger::LogText() {
  CHECK(!text_log_path_.empty());
  logs::Line log_line;
  while (text_queue_.Wait(&log_line)) {
    logs::Lines log_lines;
    *log_lines.add_line() = log_line;
    while (text_queue_.Pop(&log_line)) *log_lines.add_line() = log_line;
    string s;
    google::protobuf::TextFormat::PrintToString(log_lines, &s);
    CHECK_OK(file::AppendStringToFile(text_log_path_, s, file::Defaults()));
  }
}

}  // namespace sensei

