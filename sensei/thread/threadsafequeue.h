#ifndef SENSEI_THREAD_THREADSAFEQUEUE_H_
#define SENSEI_THREAD_THREADSAFEQUEUE_H_

#include <condition_variable>  // NOLINT
using std::condition_variable;
#include <mutex>  // NOLINT
using std::mutex;
using std::lock_guard;
using std::unique_lock;
#include <queue>
using std::priority_queue;
using std::queue;

template <typename T>
class WaitQueue {
 public:
  WaitQueue() : is_stopped_(false) {}

  bool empty() const {
    lock_guard<mutex> lock(mutex_);
    return contents_.empty();
  }

  void push(const T& x) {
    lock_guard<mutex> lock(mutex_);
    available_.notify_all();
    contents_.push(x);
  }

  bool Wait(T* p) {
    unique_lock<mutex> lock(mutex_);
    available_.wait(lock,
                    [this]() { return !contents_.empty() || is_stopped_; });
    if (!contents_.empty()) {
      *p = contents_.front();
      contents_.pop();
      return true;
    }
    // StopWaiters() was called
    return false;
  }

  void StopWaiters() {
    lock_guard<mutex> lock(mutex_);
    is_stopped_ = true;
    available_.notify_all();
  }

  bool Pop(T* p) {
    lock_guard<mutex> lock(mutex_);
    if (!contents_.empty()) {
      *p = contents_.front();
      contents_.pop();
      return true;
    }
    return false;
  }

 private:
  queue<T> contents_;
  mutable mutex mutex_;
  condition_variable available_;
  bool is_stopped_;
};

#endif  // SENSEI_THREAD_THREADSAFEQUEUE_H_
