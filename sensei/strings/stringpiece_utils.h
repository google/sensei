#ifndef SENSEI_STRINGS_STRINGPIECE_UTILS_H_
#define SENSEI_STRINGS_STRINGPIECE_UTILS_H_

namespace strings {

inline bool EqualIgnoreCase(const StringPiece a, const StringPiece b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); i++) {
    if (!(a[i] == b[i] ||
          (isalpha(a[i]) && isalpha(b[i]) && tolower(a[i]) == tolower(b[i])))) {
      return false;
    }
  }
  return true;
}

inline size_t RemoveLeadingWhitespace(StringPiece* a) {
  size_t result = 0;
  const char* ptr = a->data();
  while (result < a->size() && isspace(*ptr)) {
    result++;
    ptr++;
  }
  a->remove_prefix(result);
  return result;
}

}  // namespace strings

#endif  // SENSEI_STRINGS_STRINGPIECE_UTILS_H_
