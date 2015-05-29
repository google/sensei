#ifndef SENSEI_BASE_TIME_H_
#define SENSEI_BASE_TIME_H_

#include <chrono>  // NOLINT

namespace base {
std::chrono::time_point<std::chrono::system_clock> Now() {
  return std::chrono::system_clock::now();
}
};

int64 ToUnixNanos(std::chrono::time_point<std::chrono::system_clock> t) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             t.time_since_epoch())
      .count();
}

#endif  // SENSEI_BASE_TIME_H_
