#pragma once
// augur/utils/type_traits.hpp
//
// Small, domain-agnostic template helpers with no dependency on augur's
// tracking/filtering types. Everything in utils/ follows this rule: if
// it needs to know what a "MotionModel" is, it doesn't belong here --
// it belongs in core/ or the module that owns the concept. This is
// exactly the kind of "used from six different subfolders" code the
// project structure calls out utils/ for, so keep it that way rather
// than letting domain logic creep in here over time.

#include <type_traits>

namespace augur::utils {

// For the "no, really, this branch is unreachable" case in an `if
// constexpr` chain -- `static_assert(always_false<T>, ...)` instead of
// `static_assert(false, ...)`, which some compilers reject even in a
// discarded branch because it doesn't depend on the template parameter.
template <typename...>
inline constexpr bool always_false = false;

// True if T is some instantiation of the class template Template
// (e.g. is_specialization_of<std::vector<int>, std::vector> -> true).
// Handy in the reflect/ and plugin/ modules for detecting "is this a
// Vector<Scalar, N>-shaped thing" without hardcoding augur::math types.
template <typename T, template <typename...> typename Template>
struct is_specialization_of : std::false_type {};

template <template <typename...> typename Template, typename... Args>
struct is_specialization_of<Template<Args...>, Template> : std::true_type {};

template <typename T, template <typename...> typename Template>
inline constexpr bool is_specialization_of_v = is_specialization_of<T, Template>::value;

} // namespace augur::utils
