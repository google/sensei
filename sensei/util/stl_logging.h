#ifndef SENSEI_UTIL_STL_LOGGING_H_
#define SENSEI_UTIL_STL_LOGGING_H_

#include <utility>
using std::pair;

#include "sensei/base/logging.h"
#include "sensei/base/port.h"

// We are adding things into the std namespace.
// Note that this is technically undefined behavior!
namespace std {

// Pair
template <typename First, typename Second>
ostream& operator<<(ostream& out, const pair<First, Second>& p) {
  return out << '(' << p.first << ", " << p.second << ')';
}

}  // namespace std

#endif  // SENSEI_UTIL_STL_LOGGING_H_
