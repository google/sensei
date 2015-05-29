// Copyright 2001 - 2003 Google, Inc.
//
// Google-specific types

#ifndef SENSEI_BASE_BASICTYPES_H_
#define SENSEI_BASE_BASICTYPES_H_

#include "sensei/base/integral_types.h"
#include "sensei/base/macros.h"

// Argument type used in interfaces that can optionally take ownership
// of a passed in argument.  If TAKE_OWNERSHIP is passed, the called
// object takes ownership of the argument.  Otherwise it does not.
enum Ownership {
  DO_NOT_TAKE_OWNERSHIP,
  TAKE_OWNERSHIP
};

#endif  // SENSEI_BASE_BASICTYPES_H_
