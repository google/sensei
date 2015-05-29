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
#ifndef SENSEI_CONCURRENCY_H_
#define SENSEI_CONCURRENCY_H_

#include <atomic>
#include <thread>  // NOLINT

#include "sensei/base/logging.h"
#include "sensei/thread/threadpool.h"
#include "sensei/thread/weighted-semaphore.h"


namespace sensei {
namespace concurrency {

const uint32 kThreadCount = 4;

class Thread {
 public:
  explicit Thread(std::function<void()> f) : impl_(f) {}

  void Join() { impl_.join(); }

 private:
  std::thread impl_;
};

// -----------------------------------------------------------------------------
class SemaphoreLock {
 public:
  explicit SemaphoreLock(WeightedSemaphore* semaphore) : semaphore_(semaphore) {
    // Note: if Acquire() fails, the subsequent Release may also crash; so
    // adding a CHECK here just moves the location of a CHECK failure closer to
    // the problematic code.
    CHECK(semaphore_->Acquire(1));
  }

  ~SemaphoreLock() { semaphore_->Release(1); }

 private:
  WeightedSemaphore* semaphore_;
};

inline void InitConcurrency(std::function<void()> closure) { closure(); }

template <typename F, typename Collection>
void ParFor(const Collection& items, bool deterministic, const F& f) {
  if (deterministic) {
    for (const auto& item : items) f(item);
  } else {
    FixedSizeThreadPool pool(kThreadCount);
    for (auto i = items.begin(); i != items.end(); ++i) {
      pool.Execute([&f, i] { f(*i); });
    }
  }
}

template <typename F, typename Collection>
void ParFor(Collection* items, bool deterministic, const F& f) {
  if (deterministic) {
    for (auto& item : *items) f(&item);
  } else {
    FixedSizeThreadPool pool(kThreadCount);
    for (auto i = items->begin(); i != items->end(); ++i) {
      pool.Execute([&f, i] { f(&*i); });
    }
  }
}

template <typename F, typename Collection>
void ParFor(const Collection& items, const F& f) {
  ParFor(items, false, f);
}

template <typename F, typename Collection>
void ParFor(Collection* items, const F& f) {
  ParFor(items, false, f);
}

class AtomicCounter {
 public:
  AtomicCounter() : counter_(0) {}

  int64 Value() const { return counter_.load(); }
  void SetNext(int64 num) { counter_.store(num); }
  int64 GetNext() { return counter_.fetch_add(1); }

 private:
  std::atomic<uint64> counter_;
};

}  // namespace concurrency
}  // namespace sensei


#endif  // SENSEI_CONCURRENCY_H_
