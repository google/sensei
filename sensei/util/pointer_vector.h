// DEPRECATED(hwright,titus,billydonahue): Prefer vector<std::unique_ptr<>>
// unless you have code portability requirements that forbid use of
// unique_ptr (in which case, see go/pg3).
//
// Maintainers: This file is part of portable google3 (go/pg3).

#ifndef SENSEI_UTIL_GTL_POINTER_VECTOR_H_
#define SENSEI_UTIL_GTL_POINTER_VECTOR_H_

#include <stddef.h>

#include <algorithm>
#include <iterator>
#include <vector>
using std::vector;

#include "sensei/base/logging.h"
#include "sensei/base/macros.h"
#include "sensei/base/scoped_ptr.h"
#include "sensei/base/template_util.h"
#include "sensei/util/container_logging.h"

namespace util {
namespace gtl {

// Container class PointerVector<T> is modeled after the interface
// of vector<scoped_ptr<T> >, with the important distinction that
// it compiles.  It has the same performance characteristics:
// O(1) access to the nth element, O(N) insertion in the middle, and
// amortized O(1) insertion at the end. See clause 23 of the C++ standard.
// And base/scoped_ptr.h, of course.
//
// Exceptions are:
//
// 1) You can not construct a PointerVector of multiple elements unless they're
//    all NULL, nor can you insert() multiple elements.
// 1a) This means the iteration constructor and copy consructor are not
//     supported.
//
// 2) assignment is not supported.
//
// 3) The iterator form of insert is not supported.
//
// 4) resize() only ever fills with NULL.
//
// 5) You can't use relational operators to compare 2 PointerVectors
//
template <typename T>
class PointerVector {
 private:
  // A type similar to scoped_ptr, but which can be created,
  // copied, or destroyed only by PointerVector.
  class Element;  // only accessible through the public typedefs.
  class Holder;  // Private interface for moving Element objects.
  typedef std::vector<Holder> BaseVector;
  template <bool IsConst = false> class Iterator;

 public:
  typedef Element value_type;
  typedef value_type &reference;
  typedef const value_type &const_reference;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef value_type *pointer;
  typedef const value_type *const_pointer;

  typedef Iterator<> iterator;
  typedef Iterator<true> const_iterator;

  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

  PointerVector() : data_() { }
  explicit PointerVector(size_t n) : data_(n) { }

  ~PointerVector();

  iterator begin() { return iterator(data_.begin()); }
  iterator end() { return iterator(data_.end()); }
  const_iterator begin() const { return const_iterator(data_.begin()); }
  const_iterator end() const { return const_iterator(data_.end()); }

  reverse_iterator rbegin() { return reverse_iterator(end()); }
  reverse_iterator rend() { return reverse_iterator(begin()); }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

  size_t size() const { return data_.size(); }
  size_t max_size() const { return data_.max_size(); }

  void resize(size_t n) {
    if (n < size()) {
      erase(begin() + n, end());
    }
    data_.resize(n);
  }

  size_t capacity() const { return data_.capacity(); }
  bool empty() const { return data_.empty(); }
  void reserve(size_t n) { data_.reserve(n); }

  reference operator[](size_t n) { return begin()[n]; }
  const_reference operator[](size_t n) const { return begin()[n]; }

  reference at(size_t n) {
    return begin()[&data_.at(n) - &*data_.begin()];
  }
  const_reference at(size_t n) const {
    return begin()[&data_.at(n) - &*data_.begin()];
  }

  reference front() { return *begin(); }
  const_reference front() const { return *begin(); }
  reference back() { return *(end() - 1); }
  const_reference back() const { return *(end() - 1); }

  void pop_back() {
    erase(end() - 1);
  }

  // In C++11, you can use vector<unique_ptr<T>> instead of PointerVector<T>,
  // however if you do so, push_back won't work.  emplace_back does, however.
  // Therefore, for ease of future adaptation, use emplace_back.
  void emplace_back(T *x) {
    data_.push_back(Holder(x));
  }

  iterator emplace(iterator pos, T *x) {
    size_t offset = pos - begin();
    data_.insert(data_.begin() + offset, Holder(x));
    return begin() + offset;
  }

  // Deprecated. Use emplace_back instead.
  void push_back(T *x) { emplace_back(x); }

  // Deprecated. Use emplace instead.
  iterator insert(iterator pos, T *x) { return emplace(pos, x); }

  iterator erase(iterator pos);
  iterator erase(iterator first, iterator last);

  void clear() { erase(begin(), end()); }

  void swap(PointerVector &x) {
    data_.swap(x.data_);
  }

 private:
  BaseVector data_;

  DISALLOW_COPY_AND_ASSIGN(PointerVector);
};

template <typename T>
class PointerVector<T>::Element {
 private:
  template <typename U>
  struct Policy {
    typedef U type;
    static void Destroy(U* p) { delete p; }
  };

  template <typename U>
  struct Policy<U[]> {
    typedef U type;
    static void Destroy(U* p) { delete[] p; }
  };

 public:
  typedef typename Policy<T>::type element_type;

  typedef element_type& reference;
  typedef const element_type& const_reference;
  typedef element_type* pointer;
  typedef const element_type* const_pointer;

  pointer get() const { return p_; }
  pointer operator->() const { return get(); }
  reference operator*() const { return *get(); }

  void reset(pointer p = NULL) { Element(p).swap(*this); }

  pointer release() {
    pointer t = p_;
    p_ = NULL;
    return t;
  }

  void swap(Element& o) {
    using std::swap;
    swap(p_, o.p_);
  }

 private:
  // Only PointerVector::Holder can make Element objects.
  friend class Holder;

  typedef Element Self;

  // Accessible only to PointerVector.
  Element() : p_(NULL) {}
  explicit Element(pointer p) : p_(p) {}
  ~Element() { Policy<T>::Destroy(p_); }

  // copy and assign are private and used only in
  // controlled circumstances.
  Element(const Element& o) : p_(o.p_) { }
  Element& operator=(const Element& o) {
    p_ = o.p_;
    return *this;
  }

  static bool RawLess(const_pointer a, const_pointer b) {
    // std::less, because p < q is undefined unless p and q point
    // to the same array. std::less is specified to overcome this.
    return std::less<const_pointer>()(a, b);
  }

  friend void swap(Self& a, Self& b) { a.swap(b); }

  // Comparisons with self.
  friend bool operator==(const Self& a, const Self& b) {
    return a.get() == b.get();
  }
  friend bool operator!=(const Self& a, const Self& b) { return !(a == b); }
  friend bool operator<(const Self& a, const Self& b) {
    return RawLess(a.get(), b.get());
  }
  friend bool operator>(const Self& a, const Self& b) { return b < a; }
  friend bool operator<=(const Self& a, const Self& b) { return !(b < a); }
  friend bool operator>=(const Self& a, const Self& b) { return !(a < b); }

  // DEPRECATED(billydonahue): Comparisons to raw pointer are going away.
  // Comparisons to nullptr will continue to be supported.
  friend bool operator==(const Self& a, const_pointer b) {
    return a.get() == b;
  }
  friend bool operator==(const_pointer a, const Self& b) { return b == a; }
  friend bool operator!=(const Self& a, const_pointer b) { return !(a == b); }
  friend bool operator!=(const_pointer a, const Self& b) { return b != a; }
  friend bool operator<(const Self& a, const_pointer b) {
    return RawLess(a.get(), b);
  }
  friend bool operator<(const_pointer a, const Self& b) {
    return RawLess(a, b.get());
  }
  friend bool operator>(const Self& a, const_pointer b) { return b < a; }
  friend bool operator>(const_pointer a, const Self& b) { return b < a; }
  friend bool operator<=(const Self& a, const_pointer b) { return !(b < a); }
  friend bool operator<=(const_pointer a, const Self& b) { return !(b < a); }
  friend bool operator>=(const Self& a, const_pointer b) { return !(a < b); }
  friend bool operator>=(const_pointer a, const Self& b) { return !(a < b); }

  pointer p_;
};

template <typename T>
class PointerVector<T>::Holder {
 public:
  Holder() {}
  explicit Holder(T* p) : e_(p) {}
  ~Holder() { e_.release(); }
  Holder(const Holder& o) : e_(o.e_) {}
  Holder& operator=(const Holder& o) {
    e_.release();
    e_ = o.e_;
    return *this;
  }

  Element* get() { return &e_; }
  const Element* get() const { return &e_; }

 private:
  Element e_;
};

// Single iterator template provies both const and nonconst iterators.
// "The Standard Librarian : Defining Iterators and Const Iterators"
//     Matt Austern, DDJ, 2001.
// http://www.drdobbs.com/the-standard-librarian-defining-iterato/184401331
template <typename T>
template <bool IsConst>
class PointerVector<T>::Iterator {
 public:
  typedef std::random_access_iterator_tag iterator_category;
  typedef typename PointerVector::difference_type difference_type;
  typedef typename PointerVector::value_type value_type;

  typedef typename base::if_<IsConst,
                             const value_type&,
                             value_type&>::type reference;
  typedef typename base::if_<IsConst,
                             const value_type*,
                             value_type*>::type pointer;
  typedef typename base::if_<IsConst,
                             typename BaseVector::const_iterator,
                             typename BaseVector::iterator>::type Base;

  Iterator() { }

  // If IsConst, defines an implicit conversion.
  // If !IsConst, defines a copy constructor.
  Iterator(const Iterator<false>& other) : base_(other.base()) { }

  pointer get() const { return base()->get(); }

  reference operator*() const { return *get(); }
  pointer operator->() const { return get(); }
  reference operator[](difference_type n) const { return *(*this + n); }

  Iterator& operator+=(difference_type n) { base_ += n; return *this; }
  Iterator& operator-=(difference_type n) { return *this += -n; }

  Iterator& operator++() { return *this += 1; }
  Iterator& operator--() { return *this -= 1; }
  Iterator operator++(int /* x */) { Iterator t = *this; ++*this; return t; }
  Iterator operator--(int /* x */) { Iterator t = *this; --*this; return t; }

  friend Iterator operator+(Iterator a, difference_type n) { return a += n; }
  friend Iterator operator-(Iterator a, difference_type n) { return a -= n; }

  friend difference_type operator-(Iterator a, Iterator b) {
    return a.base() - b.base();
  }

  friend bool operator==(Iterator a, Iterator b) {
    return a.base() == b.base();
  }
  friend bool operator<(Iterator a, Iterator b) {
    return a.base() < b.base();
  }
  friend bool operator<=(Iterator a, Iterator b) { return !(b < a); }
  friend bool operator>(Iterator a, Iterator b) { return b < a; }
  friend bool operator>=(Iterator a, Iterator b) { return !(a < b); }
  friend bool operator!=(Iterator a, Iterator b) { return !(a == b); }

 private:
  friend class PointerVector;

  Base base() const { return base_; }
  explicit Iterator(Base base) : base_(base) { }

  Base base_;
};

// Definitions of out-of-line member functions.

template <typename T>
PointerVector<T>::~PointerVector() {
  clear();
}

template<typename T>
typename PointerVector<T>::iterator PointerVector<T>::erase(iterator pos) {
  pos->reset();
  return iterator(data_.erase(pos.base()));
}

template<typename T>
typename PointerVector<T>::iterator PointerVector<T>::erase(iterator first,
                                                            iterator last) {
  for (iterator it = first; it != last; ++it) {
    it->reset();
  }
  return iterator(data_.erase(first.base(), last.base()));
}

// The following nonmember functions are part of the public vector interface.

template <typename T>
inline void swap(PointerVector<T> &x, PointerVector<T> &y) {
  x.swap(y);
}

template <typename T>
inline std::ostream& operator<<(std::ostream& out,
                                const PointerVector<T>& seq) {
  util::gtl::LogRangeToStream(out, seq.begin(), seq.end(),
                              util::gtl::LogDefault());
  return out;
}

}  // namespace gtl
}  // namespace util

#endif  // SENSEI_UTIL_GTL_POINTER_VECTOR_H_
