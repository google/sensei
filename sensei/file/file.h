// Copyright 2012 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: tomasz.kaftal@gmail.com (Tomasz Kaftal)
//
// The file provides simple file functionalities.
// TODO(user): Tests needed.
#ifndef SENSEI_FILE_FILE_H_
#define SENSEI_FILE_FILE_H_

#include <glob.h>

#include <string>
using std::string;

#include "sensei/base/integral_types.h"
#include "sensei/base/logging.h"
#include "sensei/base/macros.h"
#include "sensei/strings/join.h"

// Use this file mode value, if you want the file system default behaviour when
// creating a file. The exact behaviour depends on the file system.
static const mode_t DEFAULT_FILE_MODE = static_cast<mode_t>(0x7FFFFFFFU);

// Wrapper class for system functions which handle basic file operations.
// The operations are virtual to enable subclassing, if there is a need for
// different filesystem/file-abstraction support.
class File {
 public:
  // Do *not* call the destructor directly (with the "delete" keyword)
  // nor use scoped_ptr; instead use Close().
  virtual ~File();

  // Factory method to create a new file object. Call Open on the
  // resulting object to open the file.  Using the appropriate flags
  // (+) will result in the file being created if it does not already
  // exist
  static File* Create(const std::string& file_name,
                      const std::string& mode);

  // Tries to open the file under file_name in the given mode. Will fail if
  // there are any errors on the way.
  static File* OpenOrDie(const std::string& file_name, const std::string& mode);

  // Utility static method for checking file existence.
  static bool Exists(const string& file);

  // Join two path components, adding a slash if necessary.  If basename is an
  // absolute path then JoinPath ignores dirname and simply returns basename.
  static string JoinPath(const string& dirname, const string& basename);

  // Return true if file exists.  Returns false if file does not exist or if an
  // error is encountered.
  virtual bool Exists() const ABSTRACT;

  // Open a file that has already been created
  virtual bool Open() ABSTRACT;

  // Deletes the file returning true iff successful.
  virtual bool Delete() ABSTRACT;

  // Flush and Close access to a file handle and delete this File
  // object. Returns true on success.
  virtual bool Close() ABSTRACT;

  // Reads data and returns it in OUTPUT. Returns a value < 0 on error,
  // or the of bytes read otherwise. Returns zero on end-of-file.
  virtual int64 Read(void* OUTPUT, uint64 length) ABSTRACT;

  // Reads one line, or max_length characters if the line is longer, into
  // the buffer.
  virtual char* ReadLine(char* buffer, uint64 max_length) ABSTRACT;

  // Try to write 'length' bytes from 'buffer', returning
  // the number of bytes that were actually written.
  // Return <= 0 on error.
  virtual int64 Write(const void* buffer, uint64 length) ABSTRACT;

  // Traditional seek + read/write interface.
  // We do not support seeking beyond the end of the file and writing to
  // extend the file. Use Append() to extend the file.
  virtual bool Seek(int64 position) ABSTRACT;

  // If we're currently at eof.
  virtual bool eof() ABSTRACT;

  // Returns the file name given during File::Create(...) call.
  virtual const string& CreateFileName() { return create_file_name_; }

  static bool Match(StringPiece pattern, vector<string>* output) {
    CHECK_NOTNULL(output);
    glob_t result;
    int return_value = glob(pattern.as_string().c_str(),
                            GLOB_NOSORT | GLOB_TILDE, nullptr, &result);
    if (return_value == GLOB_NOMATCH) {
      return true;
    }
    if (return_value != 0) {
      return false;
    }
    for (size_t i = 0; i < result.gl_pathc; ++i) {
      output->emplace_back(result.gl_pathv[i]);
    }
    globfree(&result);
    return true;
  }

 protected:
  explicit File(const string& create_file_name);

  // Name of the created file.
  const string create_file_name_;
};

namespace file {

// Unused in opensource version
typedef bool Options;

inline Options Defaults() { return false; }

inline File* OpenOrDie(const StringPiece file_name, const StringPiece mode,
                       const file::Options& options) {
  return File::OpenOrDie(file_name.as_string(), mode.as_string());
}

inline bool Open(const StringPiece file_name, const StringPiece mode,
                 File** file, const file::Options& options) {
  *file = File::Create(file_name.as_string(), mode.as_string());
  return (*file)->Open();
}

static const size_t kGetContentsBufferSize = 1 << 20;  // 1 MB

inline bool ReadFileToString(File* file, string* output,
                             const file::Options& options) {
  CHECK_NOTNULL(output);
  output->clear();
  char buffer[kGetContentsBufferSize];
  int64 bytes_read;
  do {
    bytes_read = file->Read(buffer, kGetContentsBufferSize);
    if (bytes_read < -1) {
      return false;
    }
    output->append(buffer, bytes_read);
  } while (bytes_read > 0);
  return true;
}

inline bool GetContents(const StringPiece file_name, string* output,
                        const file::Options& options) {
  File* f = OpenOrDie(file_name, "r", file::Defaults());
  bool result = ReadFileToString(f, output, options);
  f->Close();
  return result;
}

inline bool WriteString(File* file, StringPiece contents,
                        const file::Options& options) {
  while (!contents.empty()) {
    int64 bytes_written = file->Write(contents.data(), contents.size());
    if (bytes_written < 0) {
      return false;
    }
    contents.remove_prefix(bytes_written);
  }
  return true;
}

inline bool AppendStringToFile(const StringPiece file_name,
                               StringPiece contents,
                               const file::Options& options) {
  File* f = OpenOrDie(file_name, "a", options);
  bool result = WriteString(f, contents, options);
  f->Close();
  return result;
}

}  // namespace file

#endif  // SENSEI_FILE_FILE_H_
