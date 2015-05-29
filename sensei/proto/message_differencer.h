#ifndef SENSEI_PROTO_MESSAGE_DIFFERENCER_H_
#define SENSEI_PROTO_MESSAGE_DIFFERENCER_H_

#include <google/protobuf/descriptor.h>
using google::protobuf::Descriptor;
using google::protobuf::FieldDescriptor;
#include <google/protobuf/message.h>
using google::protobuf::Message;
using google::protobuf::Reflection;

#include <unordered_set>

#include "sensei/util/mathlimits.h"

namespace proto_util {

class MessageDifferencer {
 public:
  bool Equals(const Message& message1, const Message& message2) {
    CHECK(message1.GetDescriptor() == message2.GetDescriptor());
    const Descriptor* desc = message1.GetDescriptor();
    for (int i = 0; i < desc->field_count(); ++i) {
      if (!FieldEquals(message1, message2, desc->field(i))) {
        report_->append("Messages differ on field " +
                        desc->field(i)->full_name() + "\n");
        return false;
      }
    }
    return true;
  }

  bool Compare(const Message& message1, const Message& message2) {
    return Equals(message1, message2);
  }

  void IgnoreField(const FieldDescriptor* field_desc) {
    ignored_.insert(field_desc);
  }

  void ReportDifferencesToString(string* report) { report_ = report; }

 private:
  template <typename T>
  bool AlmostEquals(const T x, const T y) {
    if (x == y) {
      return true;
    }
    if (!MathLimits<T>::IsFinite(x) || !MathLimits<T>::IsFinite(y)) {
      return false;
    }
    T abs_x = fabsl(x);
    T abs_y = fabsl(y);
    if (abs_x <= MathLimits<T>::kStdError &&
        abs_y <= MathLimits<T>::kStdError) {
      return true;
    }
    T diff = fabsl(x - y);
    return diff < MathLimits<T>::kStdError ||
           diff < MathLimits<T>::kStdError * std::max(abs_x, abs_y);
  }

  bool RepeatedFieldEquals(const Message& message1, const Message& message2,
                           const FieldDescriptor* field_desc) {
    const Reflection* refl1 = message1.GetReflection();
    const Reflection* refl2 = message2.GetReflection();
    int field_size = refl1->FieldSize(message1, field_desc);
    if (field_size != refl2->FieldSize(message2, field_desc)) {
      return false;
    }
    for (int i = 0; i < field_size; ++i) {
      switch (field_desc->cpp_type()) {
        case FieldDescriptor::CppType::CPPTYPE_INT32:
          if (refl1->GetRepeatedInt32(message1, field_desc, i) !=
              refl2->GetRepeatedInt32(message2, field_desc, i)) {
            return false;
          }
          break;
        case FieldDescriptor::CppType::CPPTYPE_INT64:
          if (refl1->GetRepeatedInt64(message1, field_desc, i) !=
              refl2->GetRepeatedInt64(message2, field_desc, i)) {
            return false;
          }
          break;
        case FieldDescriptor::CppType::CPPTYPE_UINT32:
          if (refl1->GetRepeatedUInt32(message1, field_desc, i) !=
              refl2->GetRepeatedUInt32(message2, field_desc, i)) {
            return false;
          }
          break;
        case FieldDescriptor::CppType::CPPTYPE_UINT64:
          if (refl1->GetRepeatedUInt64(message1, field_desc, i) !=
              refl2->GetRepeatedUInt64(message2, field_desc, i)) {
            return false;
          }
          break;
        case FieldDescriptor::CppType::CPPTYPE_DOUBLE:
          if (!AlmostEquals(
                  refl1->GetRepeatedDouble(message1, field_desc, i),
                  refl2->GetRepeatedDouble(message2, field_desc, i))) {
            return false;
          }
          break;
        case FieldDescriptor::CppType::CPPTYPE_FLOAT:
          if (!AlmostEquals(refl1->GetRepeatedFloat(message1, field_desc, i),
                            refl2->GetRepeatedFloat(message2, field_desc, i))) {
            return false;
          }
          break;
        case FieldDescriptor::CppType::CPPTYPE_BOOL:
          if (refl1->GetRepeatedBool(message1, field_desc, i) !=
              refl2->GetRepeatedBool(message2, field_desc, i)) {
            return false;
          }
          break;
        case FieldDescriptor::CppType::CPPTYPE_ENUM:
          if (refl1->GetRepeatedEnum(message1, field_desc, i)->number() !=
              refl2->GetRepeatedEnum(message2, field_desc, i)->number()) {
            return false;
          }
          break;
        case FieldDescriptor::CppType::CPPTYPE_STRING:
          if (refl1->GetRepeatedString(message1, field_desc, i) !=
              refl2->GetRepeatedString(message2, field_desc, i)) {
            return false;
          }
          break;
        case FieldDescriptor::CppType::CPPTYPE_MESSAGE:
          if (!Equals(refl1->GetRepeatedMessage(message1, field_desc, i),
                      refl2->GetRepeatedMessage(message2, field_desc, i))) {
            return false;
          }
          break;
      }
    }
    return true;
  }

  bool FieldEquals(const Message& message1, const Message& message2,
                   const FieldDescriptor* field_desc) {
    if (ShouldIgnore(field_desc)) {
      return true;
    }
    if (field_desc->is_repeated()) {
      return RepeatedFieldEquals(message1, message2, field_desc);
    }

    const Reflection* refl1 = message1.GetReflection();
    const Reflection* refl2 = message2.GetReflection();
    switch (field_desc->cpp_type()) {
      case FieldDescriptor::CppType::CPPTYPE_INT32:
        return refl1->GetInt32(message1, field_desc) ==
               refl2->GetInt32(message2, field_desc);
      case FieldDescriptor::CppType::CPPTYPE_INT64:
        return refl1->GetInt64(message1, field_desc) ==
               refl2->GetInt64(message2, field_desc);
      case FieldDescriptor::CppType::CPPTYPE_UINT32:
        return refl1->GetUInt32(message1, field_desc) ==
               refl2->GetUInt32(message2, field_desc);
      case FieldDescriptor::CppType::CPPTYPE_UINT64:
        return refl1->GetUInt64(message1, field_desc) ==
               refl2->GetUInt64(message2, field_desc);
      case FieldDescriptor::CppType::CPPTYPE_DOUBLE:
        return AlmostEquals(refl1->GetDouble(message1, field_desc),
                            refl2->GetDouble(message2, field_desc));
      case FieldDescriptor::CppType::CPPTYPE_FLOAT:
        return AlmostEquals(refl1->GetFloat(message1, field_desc),
                            refl2->GetFloat(message2, field_desc));
      case FieldDescriptor::CppType::CPPTYPE_BOOL:
        return refl1->GetBool(message1, field_desc) ==
               refl2->GetBool(message2, field_desc);
      case FieldDescriptor::CppType::CPPTYPE_ENUM:
        return refl1->GetEnum(message1, field_desc)->number() ==
               refl2->GetEnum(message2, field_desc)->number();
      case FieldDescriptor::CppType::CPPTYPE_STRING:
        return refl1->GetString(message1, field_desc) ==
               refl2->GetString(message2, field_desc);
      case FieldDescriptor::CppType::CPPTYPE_MESSAGE:
        return Equals(refl1->GetMessage(message1, field_desc),
                      refl2->GetMessage(message2, field_desc));
    }
  }

  bool ShouldIgnore(const FieldDescriptor* field_desc) {
    return ignored_.count(field_desc) > 0;
  }

  unordered_set<const FieldDescriptor*> ignored_;
  string* report_;
};

}  // namespace proto_util

#endif  // SENSEI_PROTO_MESSAGE_DIFFERENCER_H_
