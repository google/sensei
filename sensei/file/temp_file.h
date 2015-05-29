#ifndef SENSEI_FILE_TEMP_FILE_H_
#define SENSEI_FILE_TEMP_FILE_H_

#include <cstdio>

class TempFile {
 public:
  static string TempFilename(void* ignored) {
    const char* ret = tmpnam(nullptr);
    CHECK_NOTNULL(ret);
    return string(ret);
  }
};

#endif  // SENSEI_FILE_TEMP_FILE_H_
