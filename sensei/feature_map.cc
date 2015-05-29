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
#include "sensei/feature_map.h"

#include "sensei/base/integral_types.h"
#include "sensei/common.h"
#include "sensei/strings/strutil.h"
#include "sensei/util/hash.h"


namespace sensei {

// -----------------------------------------------------------------------------
// class FeatureName

uint32 FeatureName::HeapSizeInBytes() const {
  uint32 size = 0;
  // Current string implementation does not allocate heap memory in case when
  // string is smaller than kMaxSmallStringSize.
  // PTAL http://cs/file:string$%20_S_local_capacity
  const uint32 kMaxSmallStringSize = 15;
  if (base_.size() > kMaxSmallStringSize) size += base_.size();
  return size;
}

uint32 FeatureName::Hash() const { return GoodFastHash<string>()(base_); }

bool FeatureName::operator==(const FeatureName& that) const {
  return base_ == that.base_;
}

// -----------------------------------------------------------------------------
// class JProduct

uint32 JProduct::HeapSizeInBytes() const {
  uint32 size = 0;
  size += and_js_.size() * sizeof(uint32);
  return size;
}

vector<string> JProduct::GetFactorNames(const FeatureMap& feature_map) const {
  vector<string> ret;
  for (uint32 j : GetJs()) {
    ret.push_back(feature_map.JToFeature(j).GetBase());
  }
  return ret;
}

string JProduct::ToLibsvmString(const FeatureMap& feature_map,
                                const string& separator) const {
  return strings::Join(GetFactorNames(feature_map), separator);
}

uint32 JProduct::Hash() const { return HashVector(and_js_); }

bool JProduct::operator==(const JProduct& that) const {
  return and_js_ == that.and_js_;
}

// -----------------------------------------------------------------------------
// FeatureMap

vector<bool> FeatureMap::FeatureSetToJs(
    const config::FeatureSet& feature_set) const {
  vector<bool> j_in_set(Size(), false);

  if (feature_set.has_explicit_list()) {
    for (const string& name : feature_set.explicit_list().feature()) {
      uint32 j = FeatureToJConst(FeatureName(name));
      if (j != kInvalidJ) j_in_set[j] = true;
    }
  }

  if (feature_set.has_from_data()) {
    for (const FeatureMap::FeatureAndJ* feature_and_j : GetAll()) {
      const string& name = feature_and_j->GetFeature().GetBase();
      for (const string& prefix : feature_set.from_data().feature_prefix()) {
        if (HasPrefixString(name, prefix)) {
          j_in_set[feature_and_j->GetJ()] = true;
        }
      }
    }
  }
  return j_in_set;
}

//-----------------------------------------------------------------------------
// ProductMap

vector<bool> ProductMap::HaveAtLeastOneFeatureJ(
    const vector<bool>& feature_js_given) const {
  vector<bool> ret(Size(), false);
  for (const ProductMap::FeatureAndJ* feature_and_j : GetAll()) {
    uint32 product_j = feature_and_j->GetJ();
    const vector<uint32>& feature_js_in_product =
        feature_and_j->GetFeature().GetJs();

    // Iterate over all feature_j in current feature_and_j.
    bool one_of_feature_j_is_in_set = false;
    for (uint32 feature_j : feature_js_in_product) {
      DCHECK_LT(feature_j, feature_js_given.size());
      one_of_feature_j_is_in_set |= feature_js_given[feature_j];
    }
    ret[product_j] = one_of_feature_j_is_in_set;
  }
  return ret;
}

}  // namespace sensei

