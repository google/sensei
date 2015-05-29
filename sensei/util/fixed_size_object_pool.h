#ifndef SENSEI_UTIL_FIXED_SIZE_OBJECT_POOL_H_
#define SENSEI_UTIL_FIXED_SIZE_OBJECT_POOL_H_

#include <condition_variable>  // NOLINT
using std::condition_variable;
#include <functional>
using std::function;
#include <mutex>  // NOLINT
using std::mutex;
using std::lock_guard;
using std::unique_lock;
#include <vector>
using std::vector;
using std::vector;

#include "sensei/base/logging.h"

template <typename T>
class FixedSizeObjectPool {
 public:
  FixedSizeObjectPool(size_t maximum_size, size_t initial_num_created)
      : FixedSizeObjectPool(maximum_size, initial_num_created,
                            []() { return new T(); }) {}
  FixedSizeObjectPool(size_t maximum_size, size_t initial_num_created,
                      std::function<T*()> factory)
      : max_size_(maximum_size), num_grabbed_(0), factory_(factory) {
    Init(initial_num_created);
  }

  T* Get() {
    unique_lock<mutex> lock(mutex_);
    available_.wait(lock, [this]() { return num_grabbed_ < max_size_; });
    num_grabbed_++;
    T* result;
    if (!contents_.empty()) {
      result = contents_.back();
      contents_.pop_back();
    } else {
      result = factory_();
    }
    return result;
  }

  void Release(T* object) {
    lock_guard<mutex> lock(mutex_);
    CHECK_GT(num_grabbed_, 0);
    CHECK_NOTNULL(object);
    CHECK_GT(max_size_, contents_.size());
    contents_.push_back(object);
    num_grabbed_--;
    available_.notify_all();
  }

  void Retire(T* object) {
    lock_guard<mutex> lock(mutex_);
    CHECK_GT(num_grabbed_, 0);
    CHECK_NOTNULL(object);
    delete object;
    num_grabbed_--;
    available_.notify_all();
  }

  size_t NumGrabbed() const {
    lock_guard<mutex> lock(mutex_);
    return num_grabbed_;
  }

  size_t NumAvailable() const {
    lock_guard<mutex> lock(mutex_);
    return max_size_ - num_grabbed_;
  }

 private:
  void Init(size_t initial_num_created) {
    lock_guard<mutex> lock(mutex_);
    contents_.reserve(initial_num_created);
    for (size_t i = 0; i < initial_num_created; ++i) {
      contents_.push_back(factory_());
    }
  }

  vector<T*> contents_;
  size_t max_size_;
  size_t num_grabbed_;
  function<T*()> factory_;
  mutable mutex mutex_;
  condition_variable available_;
};

#endif  // SENSEI_UTIL_FIXED_SIZE_OBJECT_POOL_H_
