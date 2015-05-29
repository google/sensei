// Copyright 2009 Google Inc. All Rights Reserved.
//
// Based on ideas suggested by Sanjay Ghemawat (sanjay@google.com)

#ifndef SENSEI_UTIL_ARRAY_SLICE_H_
#define SENSEI_UTIL_ARRAY_SLICE_H_

#include <initializer_list>
#include <type_traits>
#include <vector>
using std::vector;

namespace util {
namespace gtl {

template <typename T>
class ArraySlice {
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef pointer iterator;
  typedef const_pointer const_iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;

 public:
  static const size_type npos = -1;

  ArraySlice() : ptr_(nullptr), len_(0) {}
  ArraySlice(const_pointer array, size_type length)
      : ptr_(array), len_(length) {}

  // Implicit conversion constructors
  ArraySlice(const vector<value_type>& v)  // NOLINT(runtime/explicit)
      : ptr_(v.data()),
        len_(v.size()) {}

  // Substring of another ArraySlice.
  // pos must be non-negative and <= x.length().
  // len must be non-negative and will be pinned to at most x.length() - pos.
  // If len==npos, the substring continues till the end of x.
  ArraySlice(const ArraySlice& x, size_type pos, size_type len)
      : ptr_(x.ptr_ + pos), len_(len) {}

  const_pointer data() const { return ptr_; }
  size_type size() const { return len_; }
  size_type length() const { return size(); }
  bool empty() const { return size() == 0; }

  void clear() {
    ptr_ = nullptr;
    len_ = 0;
  }

  const_reference operator[](size_type i) const { return ptr_[i]; }
  const_reference at(size_type i) const { return ptr_[i]; }
  const_reference front() const { return ptr_[0]; }
  const_reference back() const { return ptr_[len_ - 1]; }

  const_iterator begin() const { return ptr_; }
  const_iterator end() const { return ptr_ + len_; }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

  void remove_prefix(size_type n) {
    ptr_ += n;
    len_ -= n;
  }
  void remove_suffix(size_type n) { len_ -= n; }
  void pop_back() { remove_suffix(1); }
  void pop_front() { remove_prefix(1); }

 private:
  const_pointer ptr_;
  size_type len_;
};

template <typename T>
class MutableArraySlice {
 public:
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef pointer iterator;
  typedef const_pointer const_iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;

  static const size_type npos = -1;

  MutableArraySlice() : ptr_(nullptr), len_(0) {}
  MutableArraySlice(pointer array, size_type length)
      : ptr_(array), len_(length) {}

  // Implicit conversion constructors
  MutableArraySlice(vector<value_type>* v)  // NOLINT(runtime/explicit)
      : ptr_(v->data()),
        len_(v->size()) {}

  // Substring of another MutableArraySlice.
  // pos must be non-negative and <= x.length().
  // len must be non-negative and will be pinned to at most x.length() - pos.
  // If len==npos, the substring continues till the end of x.
  MutableArraySlice(const MutableArraySlice& x, size_type pos, size_type len)
      : ptr_(x.ptr_ + pos), len_(len) {}

  // Accessors.
  pointer data() const { return ptr_; }
  size_type size() const { return len_; }
  size_type length() const { return size(); }
  bool empty() const { return size() == 0; }

  void clear() {
    ptr_ = nullptr;
    len_ = 0;
  }

  reference operator[](size_type i) const { return ptr_[i]; }
  reference at(size_type i) const { return ptr_[i]; }
  reference front() const { return ptr_[0]; }
  reference back() const { return ptr_[len_ - 1]; }

  iterator begin() const { return ptr_; }
  iterator end() const { return ptr_ + len_; }
  reverse_iterator rbegin() const { return reverse_iterator(end()); }
  reverse_iterator rend() const { return reverse_iterator(begin()); }

  void remove_prefix(size_type n) {
    ptr_ += n;
    len_ -= n;
  }
  void remove_suffix(size_type n) { len_ -= n; }
  void pop_back() { remove_suffix(1); }
  void pop_front() { remove_prefix(1); }

 private:
  pointer ptr_;
  size_type len_;
};

template <typename T>
const typename ArraySlice<T>::size_type ArraySlice<T>::npos;
template <typename T>
const typename MutableArraySlice<T>::size_type MutableArraySlice<T>::npos;

}  // namespace gtl
}  // namespace util
#endif  // SENSEI_UTIL_ARRAY_SLICE_H_
