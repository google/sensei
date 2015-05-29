// Copyright 2005 Google Inc. All Rights Reserved.
//
// This file defines some iterator adapters for working on:
// - containers where the value_type is pair<>, such as either
//   hash_map<K,V>, or list<pair<>>.
// - containers where the value_type is a pointer like type.
// - containers that you need to iterate backwards.
//
// Maintainers: This file is part of portable google3 (go/pg3).

#ifndef SENSEI_UTIL_GTL_ITERATOR_ADAPTORS_H_
#define SENSEI_UTIL_GTL_ITERATOR_ADAPTORS_H_

#include <iterator>
#include <memory>

#include "sensei/base/port.h"

#if defined(LANG_CXX11)
#include <type_traits>  // NOLINT
#endif  // defined(LANG_CXX11)

#include "sensei/base/template_util.h"
#include "sensei/base/type_traits.h"

namespace util {
namespace gtl {
namespace internal {

#ifdef LANG_CXX11
template <typename U>
struct PointeeTypeImpl {
  typedef typename std::pointer_traits<U>::element_type type;
};
#else  // !LANG_CXX11 follows
template <typename U>
struct PointeeTypeImpl {
  typedef typename U::element_type type;
};
template <typename U>
struct PointeeTypeImpl<U*> {
  typedef U type;
};
#endif  // !LANG_CXX11

// Extract the pointee type of a (possibly smart) pointer type.
// Top-level cv-qualifications on T are ignored.
// In C++11, this can be done with std::pointer_traits.
// In C++98, T has to be a raw pointer or export an 'element_type'.
template <typename T>
struct PointeeType
    : PointeeTypeImpl<typename base::remove_cv<T>::type> {};

template <typename> struct IsConst : base::false_type {};
template <typename T> struct IsConst<const T> : base::true_type {};

// value == true if Iter prohibits modification of its pointees.
template <typename Iter>
struct IsConstIter
    : IsConst<typename base::remove_reference<
                  typename std::iterator_traits<Iter>::reference>::type> {};

template <bool Cond, typename T>
struct AddConstIf : base::if_<Cond, const T, T> {};

// SynthIterTraits propagates the constness of the 'BaseIter' iterator
// type to its own exported 'pointer' and 'reference' typedefs.
template<typename BaseIter, typename Val>
struct SynthIterTraits : std::iterator_traits<BaseIter> {
 private:
  enum { kIterConst = IsConstIter<BaseIter>::value };
 public:
  typedef typename base::remove_cv<Val>::type value_type;
  typedef typename AddConstIf<kIterConst, Val>::type* pointer;
  typedef typename AddConstIf<kIterConst, Val>::type& reference;
};

// PointeeSynthIterTraits is similar to SynthIterTraits, but the 'Ptr'
// parameter is a pointer-like type, and value_type is the pointee.
template<typename BaseIter, typename Ptr>
struct PointeeSynthIterTraits : std::iterator_traits<BaseIter> {
 private:
  enum { kIterConst = IsConstIter<BaseIter>::value };
 public:
  typedef typename internal::PointeeType<Ptr>::type value_type;
  typedef typename AddConstIf<kIterConst, value_type>::type* pointer;
  typedef typename AddConstIf<kIterConst, value_type>::type& reference;
};

// CRTP base class for generating iterator adaptors.
// 'Sub' is the derived type, and 'Policy' encodes
// all of the behavior for the adaptor.
// Policy requirements:
//   - type 'underlying_iterator': the underlying iterator type.
//   - type 'adapted_traits': the traits of the adaptor.
//   - static 'Extract(underlying_iterator)': convert iterator to reference.
//
template <typename Sub, typename Policy>
class IteratorAdaptorBase {
 private:
  // Everything needed from the Policy type is expressed in this section.
  typedef typename Policy::underlying_iterator It;
  typedef typename Policy::adapted_traits OutTraits;
  static typename OutTraits::reference Extract(const It& it) {
    return Policy::Extract(it);
  }

 public:
  typedef typename OutTraits::iterator_category  iterator_category;
  typedef typename OutTraits::value_type         value_type;
  typedef typename OutTraits::pointer            pointer;
  typedef typename OutTraits::reference          reference;
  typedef typename OutTraits::difference_type    difference_type;

  IteratorAdaptorBase() : it_() {}
  IteratorAdaptorBase(It it) : it_(it) {}  // NOLINT(runtime/explicit)

  Sub& sub() { return static_cast<Sub&>(*this); }
  const Sub& sub() const { return static_cast<const Sub&>(*this); }

  const It& base() const { return it_; }

  reference get() const { return Extract(base()); }
  reference operator*() const { return get(); }
  pointer operator->() const { return &get(); }
  reference operator[](difference_type d) const { return *(sub() + d); }

  Sub& operator++() { ++it_; return sub(); }
  Sub& operator--() { --it_; return sub(); }
  Sub operator++(int /*unused*/) { return it_++; }
  Sub operator--(int /*unused*/) { return it_--; }

  Sub& operator+=(difference_type d) { it_ += d; return sub(); }
  Sub& operator-=(difference_type d) { it_ -= d; return sub(); }

  // TODO(user): These relational operators should be nonmembers
  // like the others, but some callers have become dependent on the left
  // side not being implicitly converted. Fix those callers and make these
  // relationals nonmember functions.
  bool operator==(Sub b) const { return base() == b.base(); }
  bool operator!=(Sub b) const { return base() != b.base(); }
  // These shouldn't be necessary, as implicit conversion from 'It'
  // should be enough to make such comparisons work.
  bool operator==(It b) const { return *this == Sub(b); }
  bool operator!=(It b) const { return *this != Sub(b); }

  friend Sub operator+(Sub it, difference_type d) { return it.base() + d; }
  friend Sub operator+(difference_type d, Sub it) { return it + d; }
  friend Sub operator-(Sub it, difference_type d) { return it.base() - d; }
  friend difference_type operator-(Sub a, Sub b) { return a.base() - b.base(); }

  friend bool operator<(Sub a, Sub b) { return a.base() < b.base(); }
  friend bool operator>(Sub a, Sub b) { return a.base() > b.base(); }
  friend bool operator<=(Sub a, Sub b) { return a.base() <= b.base(); }
  friend bool operator>=(Sub a, Sub b) { return a.base() >= b.base(); }

 private:
  It it_;
};

template <typename It>
struct FirstPolicy {
  typedef It underlying_iterator;
  typedef SynthIterTraits<underlying_iterator,
      typename std::iterator_traits<underlying_iterator>
          ::value_type::first_type> adapted_traits;
  static typename adapted_traits::reference Extract(
      const underlying_iterator& it) {
    return it->first;
  }
};

template <typename It>
struct SecondPolicy {
  typedef It underlying_iterator;
  typedef SynthIterTraits<underlying_iterator,
      typename std::iterator_traits<underlying_iterator>
          ::value_type::second_type> adapted_traits;
  static typename adapted_traits::reference Extract(
      const underlying_iterator& it) {
    return it->second;
  }
};

template <typename It>
struct SecondPtrPolicy {
  typedef It underlying_iterator;
  typedef PointeeSynthIterTraits<underlying_iterator,
      typename std::iterator_traits<underlying_iterator>
          ::value_type::second_type> adapted_traits;
  static typename adapted_traits::reference Extract(
      const underlying_iterator& it) {
    return *it->second;
  }
};

template <typename It>
struct PtrPolicy {
  typedef It underlying_iterator;
  typedef PointeeSynthIterTraits<underlying_iterator,
      typename std::iterator_traits<underlying_iterator>
          ::value_type> adapted_traits;
  static typename adapted_traits::reference Extract(
      const underlying_iterator& it) {
    return **it;
  }
};

}  // namespace internal

// In both iterator adaptors, iterator_first<> and iterator_second<>,
// we build a new iterator based on a parameterized iterator type, "It".
// The value type, "Val" is determined by "It::value_type::first" or
// "It::value_type::second", respectively.

// iterator_first<> adapts an iterator to return the first value of a pair.
// It is equivalent to calling it->first on every value.
// Example:
//
// hash_map<string, int> values;
// values["foo"] = 1;
// values["bar"] = 2;
// for (iterator_first<hash_map<string, int>::iterator> x = values.begin();
//      x != values.end(); ++x) {
//   printf("%s", x->c_str());
// }
template<typename It>
struct iterator_first
    : internal::IteratorAdaptorBase<iterator_first<It>,
                                    internal::FirstPolicy<It> > {
  typedef internal::IteratorAdaptorBase<iterator_first<It>,
                                        internal::FirstPolicy<It> > Base;
  iterator_first() {}
  iterator_first(It it)  // NOLINT(runtime/explicit)
      : Base(it) {}
  template<typename It2>
  iterator_first(iterator_first<It2> o)  // NOLINT(runtime/explicit)
      : Base(o.base()) {}
};

template<typename It>
iterator_first<It> make_iterator_first(It it) {
  return iterator_first<It>(it);
}

// iterator_second<> adapts an iterator to return the second value of a pair.
// It is equivalent to calling it->second on every value.
// Example:
//
// hash_map<string, int> values;
// values["foo"] = 1;
// values["bar"] = 2;
// for (iterator_second<hash_map<string, int>::iterator> x = values.begin();
//      x != values.end(); ++x) {
//   int v = *x;
//   printf("%d", v);
// }
template<typename It>
struct iterator_second
    : internal::IteratorAdaptorBase<iterator_second<It>,
                                    internal::SecondPolicy<It> > {
  typedef internal::IteratorAdaptorBase<iterator_second<It>,
                                        internal::SecondPolicy<It> > Base;
  iterator_second() {}
  iterator_second(It it)  // NOLINT(runtime/explicit)
      : Base(it) {}
  template<typename It2>
  iterator_second(iterator_second<It2> o)  // NOLINT(runtime/explicit)
      : Base(o.base()) {}
};

template<typename It>
iterator_second<It> make_iterator_second(It it) {
  return iterator_second<It>(it);
}

// iterator_second_ptr<> adapts an iterator to return the dereferenced second
// value of a pair.
// It is equivalent to calling *it->second on every value.
// The same result can be achieved by composition
// iterator_ptr<iterator_second<> >
// Can be used with maps where values are regular pointers or pointers wrapped
// into linked_ptr. This iterator adaptor can be used by classes to give their
// clients access to some of their internal data without exposing too much of
// it.
//
// Example:
// class MyClass {
//  public:
//   MyClass(const string& s);
//   string DebugString() const;
// };
// typedef hash_map<string, linked_ptr<MyClass> > MyMap;
// typedef iterator_second_ptr<MyMap::iterator> MyMapValuesIterator;
// MyMap values;
// values["foo"].reset(new MyClass("foo"));
// values["bar"].reset(new MyClass("bar"));
// for (MyMapValuesIterator it = values.begin(); it != values.end(); ++it) {
//   printf("%s", it->DebugString().c_str());
// }
template <typename It>
struct iterator_second_ptr
    : internal::IteratorAdaptorBase<iterator_second_ptr<It>,
                                    internal::SecondPtrPolicy<It> > {
  typedef internal::IteratorAdaptorBase<iterator_second_ptr<It>,
                                        internal::SecondPtrPolicy<It> > Base;
  iterator_second_ptr() {}
  iterator_second_ptr(It it)  // NOLINT(runtime/explicit)
      : Base(it) {}
  template<typename It2>
  iterator_second_ptr(iterator_second_ptr<It2> o)  // NOLINT(runtime/explicit)
      : Base(o.base()) {}
};

template<typename It>
iterator_second_ptr<It> make_iterator_second_ptr(It it) {
  return iterator_second_ptr<It>(it);
}

// iterator_ptr<> adapts an iterator to return the dereferenced value.
// With this adaptor you can write *it instead of **it, or it->something instead
// of (*it)->something.
// Can be used with vectors and lists where values are regular pointers
// or pointers wrapped into linked_ptr. This iterator adaptor can be used by
// classes to give their clients access to some of their internal data without
// exposing too much of it.
//
// Example:
// class MyClass {
//  public:
//   MyClass(const string& s);
//   string DebugString() const;
// };
// typedef vector<linked_ptr<MyClass> > MyVector;
// typedef iterator_ptr<MyVector::iterator> DereferencingIterator;
// MyVector values;
// values.push_back(make_linked_ptr(new MyClass("foo")));
// values.push_back(make_linked_ptr(new MyClass("bar")));
// for (DereferencingIterator it = values.begin(); it != values.end(); ++it) {
//   printf("%s", it->DebugString().c_str());
// }
//
// Without iterator_ptr you would have to do (*it)->DebugString()
template<typename It, typename Ptr /* ignored */ = void>
struct iterator_ptr
    : internal::IteratorAdaptorBase<iterator_ptr<It, Ptr>,
                                    internal::PtrPolicy<It> > {
  typedef internal::IteratorAdaptorBase<iterator_ptr<It, Ptr>,
                                        internal::PtrPolicy<It> > Base;
  iterator_ptr() {}
  iterator_ptr(It it)  // NOLINT(runtime/explicit)
      : Base(it) {}
  template<typename It2>
  iterator_ptr(iterator_ptr<It2> o)  // NOLINT(runtime/explicit)
      : Base(o.base()) {}
};

template<typename It>
iterator_ptr<It> make_iterator_ptr(It it) {
  return iterator_ptr<It>(it);
}

namespace internal {

// Template that uses SFINAE to inspect Container abilities:
// . Set has_size_type true, iff T::size_type is defined
// . Define size_type as T::size_type if defined, or size_t otherwise
template<typename C>
struct container_traits {
 private:
  // Provide Yes and No to make the SFINAE tests clearer.
  typedef base::small_  Yes;
  typedef base::big_    No;

  // Test for availability of C::size_typae.
  template<typename U>
  static Yes test_size_type(typename U::size_type*);
  template<typename>
  static No test_size_type(...);

  // Conditional provisioning of a size_type which defaults to size_t.
  template<bool Cond, typename U = void>
  struct size_type_def {
    typedef typename U::size_type type;
  };
  template<typename U>
  struct size_type_def<false, U> {
    typedef size_t type;
  };

 public:
  // Determine whether C::size_type is available.
  static const bool has_size_type = sizeof(test_size_type<C>(0)) == sizeof(Yes);

  // Provide size_type as either C::size_type if available, or as size_t.
  typedef typename size_type_def<has_size_type, C>::type size_type;
};

template<typename C>
struct IterGenerator {
  typedef C container_type;
  typedef typename C::iterator iterator;
  typedef typename C::const_iterator const_iterator;

  static iterator begin(container_type& c) {  // NOLINT(runtime/references)
    return c.begin();
  }
  static iterator end(container_type& c) {  // NOLINT(runtime/references)
    return c.end();
  }
  static const_iterator begin(const container_type& c) { return c.begin(); }
  static const_iterator end(const container_type& c) { return c.end(); }
};

template<typename SubIterGenerator>
struct ReversingIterGeneratorAdaptor {
  typedef typename SubIterGenerator::container_type container_type;
  typedef std::reverse_iterator<typename SubIterGenerator::iterator> iterator;
  typedef std::reverse_iterator<typename SubIterGenerator::const_iterator>
      const_iterator;

  static iterator begin(container_type& c) {  // NOLINT(runtime/references)
    return iterator(SubIterGenerator::end(c));
  }
  static iterator end(container_type& c) {  // NOLINT(runtime/references)
    return iterator(SubIterGenerator::begin(c));
  }
  static const_iterator begin(const container_type& c) {
    return const_iterator(SubIterGenerator::end(c));
  }
  static const_iterator end(const container_type& c) {
    return const_iterator(SubIterGenerator::begin(c));
  }
};


// C:             the container type
// Iter:          the type of mutable iterator to generate
// ConstIter:     the type of constant iterator to generate
// IterGenerator: a policy type that returns native iterators from a C
template<typename C, typename Iter, typename ConstIter,
         typename IterGenerator = util::gtl::internal::IterGenerator<C> >
class iterator_view_helper {
 public:
  typedef C container_type;
  typedef Iter iterator;
  typedef ConstIter const_iterator;
  typedef typename std::iterator_traits<iterator>::value_type value_type;
  typedef typename util::gtl::internal::container_traits<C>::size_type
      size_type;

  explicit iterator_view_helper(
      container_type& c)  // NOLINT(runtime/references)
      : c_(&c) {
  }

  iterator begin() { return iterator(IterGenerator::begin(container())); }
  iterator end() { return iterator(IterGenerator::end(container())); }
  const_iterator begin() const {
    return const_iterator(IterGenerator::begin(container()));
  }
  const_iterator end() const {
    return const_iterator(IterGenerator::end(container()));
  }
  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }
  const container_type& container() const { return *c_; }
  container_type& container() { return *c_; }

  bool empty() const { return begin() == end(); }
  size_type size() const { return c_->size(); }

 private:
  // TODO(user): Investigate making ownership be via IterGenerator.
  container_type* c_;
};

// TODO(user): Investigate unifying const_iterator_view_helper
// with iterator_view_helper.
template<typename C, typename ConstIter,
         typename IterGenerator = util::gtl::internal::IterGenerator<C> >
class const_iterator_view_helper {
 public:
  typedef C container_type;
  typedef ConstIter const_iterator;
  typedef typename std::iterator_traits<const_iterator>::value_type value_type;
  typedef typename util::gtl::internal::container_traits<C>::size_type
      size_type;

  explicit const_iterator_view_helper(const container_type& c) : c_(&c) { }

  // Allow implicit conversion from the corresponding iterator_view_helper.
  // Erring on the side of constness should be allowed. E.g.:
  //    MyMap m;
  //    key_view_type<MyMap>::type keys = key_view(m);  // ok
  //    key_view_type<const MyMap>::type const_keys = key_view(m);  // ok
  template<typename Iter>
  const_iterator_view_helper(
      const iterator_view_helper<container_type, Iter, const_iterator,
                                 IterGenerator>& v)
      : c_(&v.container()) { }

  const_iterator begin() const {
    return const_iterator(IterGenerator::begin(container()));
  }
  const_iterator end() const {
    return const_iterator(IterGenerator::end(container()));
  }
  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }
  const container_type& container() const { return *c_; }

  bool empty() const { return begin() == end(); }
  size_type size() const { return c_->size(); }

 private:
  const container_type* c_;
};

}  // namespace internal
}  // namespace gtl
}  // namespace util

// Note: The views like value_view, key_view should be in util::gtl namespace.
// Currently there are lot of callers that reference the methods in the global
// namespace.
//
// Traits to provide a typedef abstraction for the return value
// of the key_view() and value_view() functions, such that
// they can be declared as:
//
//    template <typename C>
//    typename key_view_type<C>::type key_view(C& c);
//
//    template <typename C>
//    typename value_view_type<C>::type value_view(C& c);
//
// This abstraction allows callers of these functions to use readable
// type names, and allows the maintainers of iterator_adaptors.h to
// change the return types if needed without updating callers.

template<typename C>
struct key_view_type {
  typedef util::gtl::internal::iterator_view_helper<
      C,
      util::gtl::iterator_first<typename C::iterator>,
      util::gtl::iterator_first<typename C::const_iterator> > type;
};

template<typename C>
struct key_view_type<const C> {
  typedef util::gtl::internal::const_iterator_view_helper<
      C,
      util::gtl::iterator_first<typename C::const_iterator> > type;
};

template<typename C>
struct value_view_type {
  typedef util::gtl::internal::iterator_view_helper<
      C,
      util::gtl::iterator_second<typename C::iterator>,
      util::gtl::iterator_second<typename C::const_iterator> > type;
};

template<typename C>
struct value_view_type<const C> {
  typedef util::gtl::internal::const_iterator_view_helper<
      C,
      util::gtl::iterator_second<typename C::const_iterator> > type;
};

// The key_view and value_view functions provide pretty ways to iterate either
// the keys or the values of a map using range based for loops.
//
// Example:
//    hash_map<int, string> my_map;
//    ...
//    for (string val : value_view(my_map)) {
//      ...
//    }

template<typename C>
typename key_view_type<C>::type
key_view(C& map) {  // NOLINT(runtime/references)
  return typename key_view_type<C>::type(map);
}

template<typename C>
typename key_view_type<const C>::type key_view(const C& map) {
  return typename key_view_type<const C>::type(map);
}

template<typename C>
typename value_view_type<C>::type
value_view(C& map) {  // NOLINT(runtime/references)
  return typename value_view_type<C>::type(map);
}

template<typename C>
typename value_view_type<const C>::type value_view(const C& map) {
  return typename value_view_type<const C>::type(map);
}

namespace util {
namespace gtl {

// Abstract container view that dereferences the pointer-like .second member
// of a container's std::pair elements, such as the elements of std::map<K,V*>
// or of std::vector<std::pair<K,V*>>.
//
// Example:
//   map<int, string*> elements;
//   for (const string& element : deref_second_view(elements)) {
//     ...
//   }
//
// Note: If you pass a temporary container to deref_second_view, be careful that
// the temporary container outlives the deref_second_view to avoid dangling
// references.
// This is fine:  PublishAll(deref_second_view(Make());
// This is not:   for (const auto& v : deref_second_view(Make())) {
//                  Publish(v);
//                }

template<typename C>
struct deref_second_view_type {
  typedef util::gtl::internal::iterator_view_helper<
      C,
      iterator_second_ptr<typename C::iterator>,
      iterator_second_ptr<typename C::const_iterator> > type;
};

template<typename C>
struct deref_second_view_type<const C> {
  typedef util::gtl::internal::const_iterator_view_helper<
      C,
      iterator_second_ptr<typename C::const_iterator> > type;
};

template<typename C>
typename deref_second_view_type<C>::type
deref_second_view(C& map) {  // NOLINT(runtime/references)
  return typename deref_second_view_type<C>::type(map);
}

template<typename C>
typename deref_second_view_type<const C>::type deref_second_view(const C& map) {
  return typename deref_second_view_type<const C>::type(map);
}

// Abstract container view that dereferences pointer elements.
//
// Example:
//   vector<string*> elements;
//   for (const string& element : deref_view(elements)) {
//     ...
//   }
//
// Note: If you pass a temporary container to deref_view, be careful that the
// temporary container outlives the deref_view to avoid dangling references.
// This is fine:  PublishAll(deref_view(Make());
// This is not:   for (const auto& v : deref_view(Make())) { Publish(v); }

template<typename C>
struct deref_view_type {
  typedef internal::iterator_view_helper<
      C,
      util::gtl::iterator_ptr<typename C::iterator>,
      util::gtl::iterator_ptr<typename C::const_iterator> > type;
};

template<typename C>
struct deref_view_type<const C> {
  typedef internal::const_iterator_view_helper<
      C,
      util::gtl::iterator_ptr<typename C::const_iterator> > type;
};

template<typename C>
typename deref_view_type<C>::type
deref_view(C& map) {  // NOLINT(runtime/references)
  return typename deref_view_type<C>::type(map);
}

template<typename C>
typename deref_view_type<const C>::type deref_view(const C& map) {
  return typename deref_view_type<const C>::type(map);
}

// Abstract container view that iterates backwards.
//
// Example:
//   vector<string> elements;
//   for (const string& element : reversed_view(elements)) {
//     ...
//   }
//
// Note: If you pass a temporary container to reversed_view_type, be careful
// that the temporary container outlives the reversed_view to avoid dangling
// references. This is fine:  PublishAll(reversed_view(Make());
// This is not:   for (const auto& v : reversed_view(Make())) { Publish(v); }

template<typename C>
struct reversed_view_type {
 private:
  typedef internal::ReversingIterGeneratorAdaptor<
      internal::IterGenerator<C> > policy;

 public:
  typedef internal::iterator_view_helper<
      C,
      typename policy::iterator,
      typename policy::const_iterator,
      policy> type;
};

template<typename C>
struct reversed_view_type<const C> {
 private:
  typedef internal::ReversingIterGeneratorAdaptor<
      internal::IterGenerator<C> > policy;

 public:
  typedef internal::const_iterator_view_helper<
     C,
     typename policy::const_iterator,
     policy> type;
};

template<typename C>
typename reversed_view_type<C>::type
reversed_view(C& c) {  // NOLINT(runtime/references)
  return typename reversed_view_type<C>::type(c);
}

template<typename C>
typename reversed_view_type<const C>::type reversed_view(const C& c) {
  return typename reversed_view_type<const C>::type(c);
}

}  // namespace gtl
}  // namespace util

// These names are moving from the global namespace to util::gtl. Please use
// the new util::gtl names. Global aliases are provided here for backward
// compatibility.
// TODO(user): upgrade old callers to the util::gtl namespace.
using util::gtl::iterator_first;
using util::gtl::iterator_second;
using util::gtl::iterator_ptr;
using util::gtl::iterator_second_ptr;
using util::gtl::make_iterator_first;
using util::gtl::make_iterator_second;
using util::gtl::make_iterator_ptr;
using util::gtl::make_iterator_second_ptr;

#endif  // SENSEI_UTIL_GTL_ITERATOR_ADAPTORS_H_
