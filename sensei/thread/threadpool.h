#ifndef SENSEI_THREAD_THREADPOOL_H_
#define SENSEI_THREAD_THREADPOOL_H_

#include <functional>
using std::function;
#include <vector>
using std::vector;
using std::vector;
#include <thread>  // NOLINT
using std::thread;

#include "sensei/thread/threadsafequeue.h"

class FixedSizeThreadPool {
 public:
  explicit FixedSizeThreadPool(size_t size) { InitWorkers(size); }

  void Execute(function<void()> task) { tasks_.push(task); }

  ~FixedSizeThreadPool() {
    tasks_.StopWaiters();
    for (thread& t : threads_) {
      t.join();
    }
  }

 private:
  void InitWorkers(size_t size) {
    for (size_t i = 0; i < size; ++i) {
      threads_.emplace_back([this]() {
        function<void()> task;
        while (tasks_.Wait(&task)) task();
      });
    }
  }

  WaitQueue<function<void()>> tasks_;
  vector<thread> threads_;
};

#endif  // SENSEI_THREAD_THREADPOOL_H_
