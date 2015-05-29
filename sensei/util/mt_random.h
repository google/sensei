#ifndef SENSEI_UTIL_MT_RANDOM_H_
#define SENSEI_UTIL_MT_RANDOM_H_

#include <random>
using std::mt19937;

#include "sensei/util/acmrandom.h"

class MTRandom {
 public:
  explicit MTRandom(uint32 seed) : engine_(seed) {}

  static uint32 WeakSeed32() { return ACMRandom::HostnamePidTimeSeed(); }

  double RandDouble() {
    std::uniform_real_distribution<double> distr(0.0, 1.0);
    return distr(engine_);
  }

 private:
  mt19937 engine_;
};

#endif  // SENSEI_UTIL_MT_RANDOM_H_
