#ifndef SENSEI_UTIL_ACMRANDOM_H_
#define SENSEI_UTIL_ACMRANDOM_H_

#include <chrono>  // NOLINT
#include <random>

#include "sensei/base/port.h"

class ACMRandom {
 public:
  explicit ACMRandom(int32 seed) : engine_(seed) {}
  uint64 Rand64() {
    return static_cast<uint64>(engine_() - 1) * engine_.max() + engine_();
  }
  static int32 HostnamePidTimeSeed() {
    return getpid() ^
           std::chrono::system_clock::now().time_since_epoch().count();
  }

  std::minstd_rand0 engine_;
};

#endif  // SENSEI_UTIL_ACMRANDOM_H_
