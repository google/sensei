#ifndef SENSEI_THREAD_WEIGHTED_SEMAPHORE_H_
#define SENSEI_THREAD_WEIGHTED_SEMAPHORE_H_

#include <condition_variable>  // NOLINT
using std::condition_variable;
#include <mutex>  // NOLINT
using std::mutex;
using std::lock_guard;
using std::unique_lock;

class WeightedSemaphore {
 public:
  explicit WeightedSemaphore(uint64 start) : value_(start) {}

  bool Acquire(uint64 cost) {
    unique_lock<mutex> lock(mutex_);
    available_.wait(lock, [this, cost]() { return cost <= value_; });
    value_ -= cost;
    return true;
  }

  void Release(uint64 cost) {
    lock_guard<mutex> lock(mutex_);
    value_ += cost;
    available_.notify_all();
  }

 private:
  uint64 value_;
  mutex mutex_;
  condition_variable available_;
};

#endif  // SENSEI_THREAD_WEIGHTED_SEMAPHORE_H_
