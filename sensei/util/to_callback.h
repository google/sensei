#ifndef SENSEI_UTIL_TO_CALLBACK_H_
#define SENSEI_UTIL_TO_CALLBACK_H_

namespace util {
namespace functional {

// This function is here just to provide compatibility with internal Google
// codebase
template <typename T>
T ToPermanentCallback(T function) {
  return function;
}

}  // namespace functional
}  // namespace util

#endif  // SENSEI_UTIL_TO_CALLBACK_H_
