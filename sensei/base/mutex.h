#ifndef SENSEI_BASE_MUTEX_H_
#define SENSEI_BASE_MUTEX_H_

#include <mutex>  // NOLINT

#include "sensei/base/thread_annotations.h"

class LOCKABLE Mutex {
 public:
  Mutex() {}
  ~Mutex() {}

  // Block if necessary until this Mutex is free, then acquire it exclusively.
  void Lock() EXCLUSIVE_LOCK_FUNCTION() {
    impl_.lock();
  }

  // Release this Mutex. Caller must hold it exclusively.
  void Unlock() UNLOCK_FUNCTION() {
    impl_.unlock();
  }

 private:
  std::mutex impl_;

  DISALLOW_COPY_AND_ASSIGN(Mutex);
};



// -----------------------------------------------------------------------------
// MutexLock(mu) acquires mu when constructed and releases it when destroyed.
class SCOPED_LOCKABLE MutexLock {
 public:
  explicit MutexLock(Mutex *mu) EXCLUSIVE_LOCK_FUNCTION(mu) : mu_(mu) {
    this->mu_->Lock();
  }
  ~MutexLock() UNLOCK_FUNCTION() { this->mu_->Unlock(); }
 private:
  Mutex *const mu_;
  DISALLOW_COPY_AND_ASSIGN(MutexLock);
};

#endif  // SENSEI_BASE_MUTEX_H_
