#pragma once
// augur/reflect/has_reflection.hpp
//
// Thin facade: the actual feature-detection macros live in
// augur/core/config.hpp (the one place in the whole library allowed to
// touch __has_include / __cpp_* directly). This header just re-exposes
// them as constexpr bools under augur::reflect, so code that only cares
// about reflection doesn't need to know config.hpp exists.

#include "augur/core/config.hpp"

namespace augur::reflect {

inline constexpr bool has_std_reflection = augur::core::BuildInfo::has_std_reflection;
inline constexpr bool has_pfr = augur::core::BuildInfo::has_pfr;

// True if ANY backend is available -- i.e. whether reflect::Descriptor
// (descriptor.hpp) can do anything at all for a given type. If this is
// false on your toolchain, add Boost.PFR as a dependency (it's a
// two-line CPM add, see CMakeLists.txt) rather than waiting for
// compiler-native reflection.
inline constexpr bool has_any_reflection_backend = has_std_reflection || has_pfr;

} // namespace augur::reflect
