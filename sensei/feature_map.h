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
// FeatureMap and ProductMap implementation.
//
// Most of the code is shared and present in FeatureMapBase<T>.
// FeatureMap is exactly FeatureMapBase<FeatureName>.
// ProductMap is FeatureMapBase<JProduct> + some auxiliary functions.
//
// For examples of use PTAL at feature_map_test.cc
//
// No method is thread-safe unless noted otherwise.
//
// J is a synonym to uint32 used to index features all over Sensei.

#ifndef SENSEI_FEATURE_MAP_H_
#define SENSEI_FEATURE_MAP_H_

#include <sys/types.h>
#include <algorithm>
#include <functional>
#include <memory>
#include <string>
using std::string;
#include <vector>
using std::vector;

#include "sensei/base/integral_types.h"
#include "sensei/base/logging.h"
#include "sensei/base/macros.h"
#include "sensei/base/mutex.h"
#include "sensei/containers/concurrent_hash_map.h"
#include "sensei/common.h"
#include "sensei/config.pb.h"
#include "sensei/internal.pb.h"
#include "sensei/j_renumbering.h"
#include "sensei/strings/util.h"


namespace sensei {

// -----------------------------------------------------------------------------
// Common code for FeatureMap and ProductMap.
// Implements parallel map Feature <-> uint32 (J).
// Most important methods are FeatureToJ, JToFeature and SyncJToFeatureMap.
template <class Feature>
class FeatureMapBase {
 public:
  // Container for map's key, value and 'link' i.e. Feature, J and void*.
  // 'link' and the whole container is needed for ConcurrentHashMap
  // implementation.
  class FeatureAndJ {
   public:
    FeatureAndJ(const Feature& feature, uint32 j) : feature_(feature), j_(j) {}

    const Feature& GetFeature() const { return feature_; }
    uint32 GetJ() const { return j_; }

    void RenumberJs(const JRenumbering& j_renumbering) {
      DCHECK_LT(j_, j_renumbering.j_to_new_j().size());
      j_ = j_renumbering.j_to_new_j()[j_];
    }

    // Internal API for ConcurrentHashMap.
    const Feature& GetKey() const { return feature_; }
    void** GetLink() { return &link_; }

   private:
    // Feature kept by FeatureAndJ.
    Feature feature_;
    // J kept by FeatureAndJ.
    uint32 j_;
    // Needed for ConcurrentHashMap
    void* link_;

    DISALLOW_COPY_AND_ASSIGN(FeatureAndJ);
  };

  FeatureMapBase();

  // Will create new J if feature is new.
  // Thread-safe.
  uint32 FeatureToJ(const Feature& feature);

  // Will return kInvalidJ if feature is not in the map.
  // Thread-safe.
  uint32 FeatureToJConst(const Feature& feature) const;

  // Thread-safe.
  bool HasFeature(const Feature& feature) const;

  // Number of entries in this map.
  uint32 Size() const;

  // Takes into account hash map size and keys(strings) on heap.
  uint64 SizeInBytes() const;

  // Print statistics to LOG(INFO).
  void LogStats() const;

  // Returned reference is valid as long as the map remains unmodified.
  const Feature& JToFeature(uint32 j) const;
  // Same as above but does not check whether map is synchronized.
  const Feature& JToFeatureUnsafe(uint32 j) const;

  // Returned pointers are valid as long as the map remains unmodified.
  const vector<const FeatureAndJ*>& GetAll() const {
    CHECK(j_to_feature_synced_);
    return j_to_feature_;
  }

  // Synchronizes J->Feature map to Feature->J map.
  void SyncJToFeatureMap();

  // Applies renumbering to a whole map and its elements.
  void RemoveAndRenumberJs(const JRenumbering& j_renumbering);

 private:
  // TODO(lew): Try LockFreeHashTable
  // TODO(lew): Try SpinLock instead of Mutex, and 4 or 5 instead of 0 which
  //            controls initial number of Mutexes / Spinlocks.
  typedef concurrent::ConcurrentHashMap<
      Feature, FeatureAndJ, typename Feature::HashFunctor,
      std::equal_to<Feature>, concurrent::AtomicSizer, Mutex, 0> Map;

  // This map's initial reserved size (number of entries).
  const uint32 kMapInitialSize = 1 << 18;

  // Workaround for not having Map::ConstReservation.
  // Be sure you know what you're doing.
  Map* MutableMap() const;

  // Private member variables.
  // Provides J for new Feature stored.
  concurrency::AtomicCounter j_sequence_;

  // Feature -> J map.
  std::unique_ptr<Map> feature_to_j_;

  // True iff j_to_feature_ synced to feature_to_j_.
  std::atomic<bool> j_to_feature_synced_;

  // J -> Feature map.
  vector<const FeatureAndJ*> j_to_feature_;

  DISALLOW_COPY_AND_ASSIGN(FeatureMapBase);
};

// -----------------------------------------------------------------------------
// Feature's name (string) container compatible with FeatureMapBase.
class FeatureMap;
class FeatureName {
 public:
  struct HashFunctor {
    uint32 operator()(const FeatureName& f) const { return f.Hash(); }
  };

  FeatureName(const FeatureName&) = default;
  FeatureName(FeatureName&&) = default;
  FeatureName& operator=(FeatureName&& other) = default;
  explicit FeatureName(const string& base) : base_(base) {}

  // Gets feature name.
  const string& GetBase() const { return base_; }

  // Returns size of memory allocated on the heap.
  uint32 HeapSizeInBytes() const;

  // Returns 32bit hash.
  uint32 Hash() const;

  bool operator==(const FeatureName& that) const;

 private:
  // Feature's name.
  string base_;
};

// -----------------------------------------------------------------------------
// Feature <-> J bimap.
class FeatureMap : public FeatureMapBase<FeatureName> {
 public:
  // Returns vector indicating for each J whether it matches the given
  // <feature_set>.
  vector<bool> FeatureSetToJs(const config::FeatureSet& feature_set) const;
};

// -----------------------------------------------------------------------------
// Container for set of Js representing product-feature.
// Compatible with FeatureMapBase.
class JProduct {
 public:
  struct HashFunctor {
    uint32 operator()(const JProduct& f) const { return f.Hash(); }
  };

  JProduct(const JProduct&) = default;
  JProduct(JProduct&&) = default;
  JProduct& operator=(JProduct&& other) = default;

  explicit JProduct(const vector<uint32>& and_js) : and_js_(and_js) {
    std::sort(and_js_.begin(), and_js_.end());
    uint32 unique_js =
        std::unique(and_js_.begin(), and_js_.end()) - and_js_.begin();
    and_js_.reserve(unique_js);
    and_js_.resize(unique_js);
  }

  // Combines 2 product features into one.
  static JProduct And(const JProduct& f1, const JProduct& f2) {
    vector<uint32> new_and_js(f1.GetJs().size() + f2.GetJs().size());
    std::merge(f1.GetJs().begin(), f1.GetJs().end(), f2.GetJs().begin(),
               f2.GetJs().end(), new_and_js.begin());
    return JProduct(new_and_js);  // TODO(lew): There is another sorting inside.
  }

  // Returns all Js in this JProduct.
  const vector<uint32>& GetJs() const { return and_js_; }

  // Returns size of memory allocated on the heap.
  uint32 HeapSizeInBytes() const;

  vector<string> GetFactorNames(const FeatureMap& feature_map) const;

  // Returns a whole JProduct in Libsvm format.
  string ToLibsvmString(const FeatureMap& feature_map,
                        const string& separator) const;

  // Returns 32bit hash.
  uint32 Hash() const;

  bool operator==(const JProduct& that) const;

 private:
  vector<uint32> and_js_;  // Sorted set of Js making this JProduct.
};

// -----------------------------------------------------------------------------
// JProduct <-> J bimap.
class ProductMap : public FeatureMapBase<JProduct> {
 public:
  // Returns vector indicating for every product J whether it matches
  // ProductJ containing at least one j such that <feature_js>[j] == true.
  vector<bool> HaveAtLeastOneFeatureJ(const vector<bool>& feature_js) const;
};

// -----------------------------------------------------------------------------
// class FeatureMapBase private implementation

template <class Feature>
FeatureMapBase<Feature>::FeatureMapBase()
    : j_to_feature_synced_(true) {
  feature_to_j_ = std::unique_ptr<Map>(new Map(kMapInitialSize));
}

template <class Feature>
uint32 FeatureMapBase<Feature>::FeatureToJConst(const Feature& feature) const {
  // TODO(lew): Implement feature use counter.
  typename Map::Reservation r(feature_to_j_.get(), feature);
  if (r.empty()) {
    return kInvalidJ;
  } else {
    return r->GetJ();
  }
}

template <class Feature>
uint32 FeatureMapBase<Feature>::FeatureToJ(const Feature& feature) {
  // TODO(lew): Implement feature use counter.
  typename Map::Reservation r(feature_to_j_.get(), feature);
  if (r.empty()) {
    uint32 j = j_sequence_.GetNext();
    CHECK_NE(kInvalidJ, j) << "Too many features.";
    FeatureAndJ* f = new FeatureAndJ(feature, j);
    r.Set(f);
    j_to_feature_synced_ = false;
    return j;
  } else {
    return r->GetJ();
  }
}

template <class Feature>
bool FeatureMapBase<Feature>::HasFeature(const Feature& feature) const {
  typename Map::Reservation r(feature_to_j_.get(), feature);
  return !r.empty();
}

template <class Feature>
const Feature& FeatureMapBase<Feature>::JToFeature(uint32 j) const {
  CHECK(j_to_feature_synced_);
  return JToFeatureUnsafe(j);
}

template <class Feature>
const Feature& FeatureMapBase<Feature>::JToFeatureUnsafe(uint32 j) const {
  CHECK_LT(j, j_to_feature_.size())
      << "Probably a missing call to SyncJToFeatureMap().";
  return j_to_feature_[j]->GetFeature();
}

template <class Feature>
void FeatureMapBase<Feature>::SyncJToFeatureMap() {
  // Notice that we're overwriting the old values, too.
  // TODO(lew,witoldjarnicki): Consider optimizing this.
  if (j_to_feature_synced_) return;

  j_to_feature_ = vector<const FeatureAndJ*>();
  uint32 next_j = j_sequence_.Value();
  // Since we remember at hashmap feature_to_j_ all entries which had been ever
  // made and we are not removing them at phase of RemoveAndRenumberJs in
  // FeaturePrunning::PruneFeatures() - next_j can be smaller than size of the
  // hashmap.
  CHECK_GE(MutableMap()->Size(), next_j) << "Something went horribly wrong.";

  MildResize(next_j, &j_to_feature_);
  uint32 valid_j_count = 0;
  for (typename Map::Reservation r(MutableMap()); !r.empty(); ++r) {
    if (r->GetJ() != kInvalidJ) {
      LOG_IF(FATAL, j_to_feature_[r->GetJ()] != nullptr)
          << "Something went horribly wrong.";
      j_to_feature_[r->GetJ()] = r.get();
      valid_j_count++;
    }
  }
  CHECK_EQ(valid_j_count, next_j);
  j_to_feature_synced_ = true;
}

template <class Feature>
uint32 FeatureMapBase<Feature>::Size() const {
  return j_sequence_.Value();
}

template <class Feature>
uint64 FeatureMapBase<Feature>::SizeInBytes() const {
  uint64 bytes = Size() * sizeof(FeatureAndJ);
  for (typename Map::Reservation r(MutableMap()); !r.empty(); ++r)
    bytes += r->GetFeature().HeapSizeInBytes();
  return bytes;
}

template <class Feature>
void FeatureMapBase<Feature>::LogStats() const {
  LOG(INFO) << "FeatureMap size = " << Size() << "("
            << SizeInBytes() / 1024. / 1024. << "MiB)";
}

template <class Feature>
typename FeatureMapBase<Feature>::Map* FeatureMapBase<Feature>::MutableMap()
    const {
  return const_cast<FeatureMapBase*>(this)->feature_to_j_.get();
}

template <class Feature>
void FeatureMapBase<Feature>::RemoveAndRenumberJs(
    const JRenumbering& j_renumbering) {
  if (j_renumbering.j_to_new_j().empty()) return;
  const uint32 next_j = j_renumbering.next_j();
  j_sequence_.SetNext(next_j);
  CHECK_NE(kInvalidJ, next_j) << "Too many features.";
  std::unique_ptr<Map> new_map(new Map(feature_to_j_->Size()));
  uint32 valid_j_count = 0;
  for (typename Map::Reservation r(feature_to_j_.get()); !r.empty(); ++r) {
    FeatureAndJ* feature_and_j = r.Release();
    if (feature_and_j->GetJ() != kInvalidJ)
      feature_and_j->RenumberJs(j_renumbering);
    if (feature_and_j->GetJ() != kInvalidJ) valid_j_count++;
    new_map->Set(feature_and_j);
  }

  feature_to_j_ = std::move(new_map);
  j_to_feature_synced_ = false;
  CHECK_EQ(valid_j_count, next_j);
  SyncJToFeatureMap();
}

}  // namespace sensei


#endif  // SENSEI_FEATURE_MAP_H_
