#ifndef SENSEI_CONTAINERS_CONCURRENT_HASH_MAP_H_
#define SENSEI_CONTAINERS_CONCURRENT_HASH_MAP_H_

#include <mutex>  //NOLINT
using std::mutex;
using std::lock_guard;
#include <unordered_map>
using std::unordered_map;
#include <utility>
using std::pair;
using std::make_pair;

#include "sensei/base/logging.h"

namespace concurrent {

class AtomicSizer {};

template <typename Key, typename Element, typename Hash, typename Equal,
          typename Sizer, typename Lock, int32 begin_level>
class ConcurrentHashMap {
 private:
  typedef unordered_map<Key, Element*, Hash, Equal> Map;

 public:
  explicit ConcurrentHashMap(size_t capacity) {}

  ~ConcurrentHashMap() {
    for (auto& i : contents_) {
      if (i.second != nullptr) {
        delete i.second;
      }
    }
  }

  struct Reservation {
    explicit Reservation(ConcurrentHashMap* h)
        : map_(&h->contents_), lock_(h->mutex_), iter_(h->contents_.begin()) {}

    Reservation(ConcurrentHashMap* h, const Key& key)
        : map_(&h->contents_), lock_(h->mutex_) {
      if (h->contents_.count(key) != 0) {
        iter_ = h->contents_.find(key);
      } else {
        iter_ = h->contents_.insert(make_pair(key, nullptr)).first;
      }
    }

    bool empty() { return iter_ == map_->end() || iter_->second == nullptr; }

    bool Set(Element* e) {
      CHECK(Equal()(e->GetKey(), iter_->first));
      bool result = false;
      if (iter_->second != nullptr) {
        delete iter_->second;
        result = true;
      }
      iter_->second = e;
      return result;
    }

    Element* operator->() const { return iter_->second; }

    Element* get() const { return iter_->second; }

    void operator++() { iter_++; }

    Element* Release() {
      Element* result = iter_->second;
      iter_->second = nullptr;
      return result;
    }

    Map* map_;
    typename Map::iterator iter_;
    lock_guard<mutex> lock_;
  };

  size_t Size() {
    lock_guard<mutex> lock(mutex_);
    return contents_.size();
  }

  bool Set(Element* e) { return Reservation(this, e->GetKey()).Set(e); }

 private:
  Map contents_;
  mutex mutex_;
};

}  // namespace concurrent

#endif  // SENSEI_CONTAINERS_CONCURRENT_HASH_MAP_H_
