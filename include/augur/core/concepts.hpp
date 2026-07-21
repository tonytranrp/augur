#pragma once
// augur/core/concepts.hpp
//
// The handful of concepts everything else in augur is built from. These
// are intentionally small and structural (duck-typed), never inheritance
// based -- that's what makes the whole library "open": a user type
// satisfies MotionModel or FilterBackend just by having the right member
// functions, with no base class, no virtual table, no registration call.
// See models/model_concept.hpp and filters/filter_concept.hpp for the
// fuller, domain-specific refinements of these.

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace augur::core {

// A Scalar is whatever floating-point type the caller wants to run the
// math in -- float for a mobile/embedded budget, double for a desktop
// research build. augur never hardcodes one.
template <typename T>
concept Scalar = std::floating_point<T>;

// StateVector / CovarianceMatrix are left abstract on purpose: augur
// doesn't mandate Eigen at the concept level (only augur::math does),
// so a user could in principle swap the math backend without touching
// any concept definition here.
template <typename T>
concept StateVectorLike = requires(const T& v, std::size_t i) {
    { v.size() } -> std::convertible_to<std::size_t>;
    { v[i] } -> std::convertible_to<typename T::Scalar>;
};

// Anything with a static, compile-time-known dimensionality. Motion
// models and filters both key off this so the library can pick
// fixed-size (stack-allocated, no heap traffic in the hot path) Eigen
// types whenever the dimension is known at compile time.
template <typename T>
concept HasStaticDimension = requires {
    { T::dimension } -> std::convertible_to<std::size_t>;
};

} // namespace augur::core
