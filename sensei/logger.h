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
#ifndef SENSEI_LOGGER_H_
#define SENSEI_LOGGER_H_

#include <memory>
#include <string>
using std::string;

#include "sensei/base/integral_types.h"
#include "sensei/concurrency.h"
#include "sensei/log.pb.h"
#include "sensei/thread/threadsafequeue.h"


namespace sensei {

class Logger {
 public:
  Logger();
  ~Logger();
  void SetLogTimestamp(bool log_timestamp);
  void SetTextLogPath(string path);
  void SetRecordioLogPath(string path);
  void AddToLogs(const logs::Line& log_line);
  void SetRunId(uint64 run_id);

 private:
  void LogRecordio();
  void LogText();

  // Private member variables.
  std::unique_ptr<concurrency::Thread> recordio_fiber_;
  std::unique_ptr<concurrency::Thread> text_fiber_;
  WaitQueue<logs::Line> recordio_queue_;
  WaitQueue<logs::Line> text_queue_;
  string text_log_path_;
  string recordio_log_path_;
  bool log_timestamp_;
  uint64 run_id_;
};

}  // namespace sensei


#endif  // SENSEI_LOGGER_H_
