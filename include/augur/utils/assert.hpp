#pragma once
// augur/utils/assert.hpp
//
// One assertion macro, used everywhere (models/, filters/, imm/,
// utils/ itself) instead of every file reaching for a different
// combination of assert()/abort()/exceptions. Compiles to nothing in
// release builds (NDEBUG defined) on every platform this targets --
// including Android release builds, where <cassert>'s assert() is
// already a no-op under NDEBUG, so this just makes that behavior
// explicit and gives a consistent call site everywhere in augur.

#include <cassert>

#if defined(NDEBUG)
  #define AUGUR_ASSERT(condition, message) ((void)0)
#else
  #define AUGUR_ASSERT(condition, message) assert((condition) && (message))
#endif
