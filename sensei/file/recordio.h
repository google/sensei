#ifndef SENSEI_FILE_RECORDIO_H_
#define SENSEI_FILE_RECORDIO_H_

#include <memory>

#include <google/protobuf/message.h>          // NOLINT
#include <google/protobuf/io/coded_stream.h>  // NOLINT
using google::protobuf::io::CodedInputStream;

#include "sensei/file/file.h"

class RecordWriter {
 public:
  explicit RecordWriter(File* f) : f_(f) {}
  template <typename T>
  bool WriteProtocolMessage(T msg) {
    string serialized;
    msg.SerializeToString(&serialized);
    return file::WriteString(f_, serialized, file::Defaults());
  }
  bool Close() { return f_->Close(); }

 private:
  File* f_;
};

class RecordReader {
 public:
  explicit RecordReader(File* f) : f_(f) {
    CHECK(file::ReadFileToString(f, &contents_, file::Defaults()));
    stream_.reset(new CodedInputStream(
        reinterpret_cast<const uint8*>(contents_.data()), contents_.size()));
  }

  template <typename T>
  bool ReadProtocolMessage(T* msg) {
    CHECK_NOTNULL(msg);
    return msg->ParseFromCodedStream(stream_.get());
  }

 private:
  File* f_;
  string contents_;
  std::unique_ptr<CodedInputStream> stream_;
};

#endif  // SENSEI_FILE_RECORDIO_H_
