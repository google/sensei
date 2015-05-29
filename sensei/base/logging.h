#ifndef SENSEI_BASE_LOGGING_H_
#define SENSEI_BASE_LOGGING_H_

#include <glog/logging.h>
#define QCHECK CHECK
#define CHECK_OK CHECK
#define QCHECK_OK CHECK
#define QCHECK_EQ CHECK_EQ
using google::INFO;
using google::FlushLogFiles;

#endif  // SENSEI_BASE_LOGGING_H_
