#ifndef SENSEI_PROTO_PARSE_TEXT_PROTO_H_
#define SENSEI_PROTO_PARSE_TEXT_PROTO_H_

#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "sensei/base/logging.h"

namespace proto_util {

template <typename T>
T ParseTextOrDie(const string& input) {
  T result;
  CHECK(google::protobuf::TextFormat::ParseFromString(input, &result));
  return result;
}

}  // namespace proto_util

#endif  // SENSEI_PROTO_PARSE_TEXT_PROTO_H_
