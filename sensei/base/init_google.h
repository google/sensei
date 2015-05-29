#ifndef SENSEI_BASE_INIT_GOOGLE_H_
#define SENSEI_BASE_INIT_GOOGLE_H_

#include <gflags/gflags.h>
#include <glog/logging.h>

void InitGoogle(const char* usage, int* argc, char*** argv, bool remove_flags) {
  google::ParseCommandLineFlags(argc, argv, remove_flags);
  google::InitGoogleLogging(**argv);
}

#endif  // SENSEI_BASE_INIT_GOOGLE_H_
