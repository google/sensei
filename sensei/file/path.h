#ifndef SENSEI_FILE_PATH_H_
#define SENSEI_FILE_PATH_H_

#include "sensei/strings/strcat.h"

namespace file {
string JoinPath(const string& path1, const string& path2) {
  return StrCat(path1, "/", path2);
}
}

#endif  // SENSEI_FILE_PATH_H_
