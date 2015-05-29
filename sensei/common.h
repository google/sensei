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
#ifndef SENSEI_COMMON_H_
#define SENSEI_COMMON_H_

#include <math.h>
#include <stddef.h>
#include <atomic>
#include <limits>
#include <memory>
#include <set>
using std::set;
#include <string>
using std::string;
#include <vector>
using std::vector;

#include "sensei/base/integral_types.h"
#include "sensei/base/logging.h"
#include "sensei/base/stringprintf.h"
#include "sensei/file/file.h"
#include "sensei/concurrency.h"
#include "sensei/range.h"
#include "sensei/strings/strcat.h"
#include "sensei/util/array_slice.h"

// Use LL only during debugging.
#define LL(a) (LOG(INFO) << StrCat(#a, " = ", (a), "; "), (a))

#define UNUSED(x) static_cast<void>(x)


namespace sensei {

typedef double Double;
typedef ::util::gtl::ArraySlice<uint32> JsSlice;

static_assert(sizeof(Double) == 8, "");

static const Double kInfinity = std::numeric_limits<Double>::infinity();
static const Double kEpsilon = std::numeric_limits<Double>::epsilon();

// -1 is reserved as j, it will never be used as feature index.
static const uint32 kInvalidJ = static_cast<uint32>(-1);

// -1 is reserved, it will never be used as row index.
static const uint32 kInvalidId = static_cast<uint32>(-1);

// Unused, but don't remove.
inline Double BetaStdDev(const Double a, const Double b) {
  const Double nominator = (a + 1) * (b + 1);
  const Double denominator = (a + b + 2) * (a + b + 2) * (a + b + 3);
  return sqrt(nominator / denominator);
}

inline string ToString(Double d) { return StringPrintf("% 1.17f", d); }

inline Double Sign(Double x) { return (x > 0) - (x < 0); }

// Unused, but don't remove.
// Returns ceil(a / b), asserts a > 0 and b > 0.
inline uint32 CeilDiv(uint32 a, uint32 b) {
  DCHECK_GT(a, 0);
  return (a - 1) / b + 1;
}

inline string AddQuotes(const string& s) { return "'" + s + "'"; }

inline uint32 HashVector(const vector<uint32>& v) {
  uint32 h = MIX32;
  for (uint32 e : v) h = Hash32NumWithSeed(e, h);
  return h;
}

inline Double L1Norm(const vector<Double>& v) {
  Double ret = 0;
  for (const Double d : v) ret += fabs(d);
  return ret;
}

inline Double L2NormSquared(const vector<Double>& v) {
  Double ret = 0;
  for (const Double d : v) ret += d * d;
  return ret;
}

inline uint32 NonZeroCount(const vector<Double>& v) {
  uint32 ret = 0;
  for (const Double d : v) ret += (d != 0);
  return ret;
}

inline string Quote(const string& s, const string& quote) {
  return StrCat(quote, s, quote);
}

inline string Quote(const string& s) { return Quote(s, "'"); }

// Smallest capacity bigger than size and bigger then previous 'GoodCapacity' by
// overallocation_ratio.
// If result is bigger than 2^13,  rounds it up so that 13 last bits are zero.
inline uint64 GoodCapacity(uint64 x, double overallocation_ratio) {
  if (x == 0) return 0;
  uint64 ret =
      pow(overallocation_ratio, ceil(log(x) / log(overallocation_ratio)));
  const uint64 kZ = 1 << 13;
  if (ret > kZ) {
    ret += kZ - 1;
    ret &= ~(kZ - 1);
  }
  return ret;
}

// Resizes vector and sets its capacity up to 33/32 of size.
template <class T>
inline void MildResize(uint32 size, vector<T>* v) {
  const double kMildOverallocation = 33.0 / 32.0;
  v->reserve(GoodCapacity(size, kMildOverallocation));
  v->resize(size);
}


inline void CheckCanWrite(const string& path, bool clear_file) {
  LOG(INFO) << "Trying to open " << AddQuotes(path) << " file before training "
            << "to check path correctness and rights.";
  QCHECK(file::OpenOrDie(path, (clear_file ? "w" : "a"), file::Defaults())
             ->Close());
}

inline Double SparseDot(const JsSlice& js, const vector<Double>& w) {
  Double dot = 0;
  for (uint32 j : js) {
    DCHECK_LT(j, w.size());
    dot += w[j];
  }
  return dot;
}

typedef ::util::gtl::MutableArraySlice<std::atomic<Double>>
    AtomicDoubleSlice;

inline Double SparseDot(const JsSlice& js, const AtomicDoubleSlice& w) {
  Double dot = 0;
  for (uint32 j : js) {
    DCHECK_LT(j, w.length());
    dot += w[j].load(std::memory_order_relaxed);
  }
  return dot;
}

inline vector<set<string>> AllSubsetsOfSize(const vector<string>& elts,
                                            uint32 n) {
  if (n == 0) return vector<set<string>>(1);  // Just empty subset.
  vector<set<string>> ret;
  vector<set<string>> n1_subsets = AllSubsetsOfSize(elts, n - 1);
  for (const set<string>& subset : n1_subsets) {
    for (const string& elt : elts) {
      if (subset.empty() || elt < *subset.begin()) {
        ret.push_back(subset);
        ret.back().insert(elt);
      }
    }
  }
  return ret;
}

inline vector<set<string>> AllSubsets(const vector<string>& elts) {
  vector<set<string>> ret;
  for (uint32 i : Range(elts.size() + 1)) {
    vector<set<string>> subset = AllSubsetsOfSize(elts, i);
    ret.insert(ret.end(), subset.begin(), subset.end());
  }
  return ret;
}

}  // namespace sensei


#endif  // SENSEI_COMMON_H_
