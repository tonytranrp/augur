#pragma once
// augur/filters/filter_concept.hpp
//
// A Filter wraps a MotionModel plus a recursive state-estimation rule
// (linear KF, EKF, UKF, particle...). Like models::MotionModel, this is
// a structural concept: augur::imm::Estimator only ever talks to Filter,
// never to a concrete class, so a user-supplied filter (say, a
// UKF variant you wrote yourself) drops into an IMM mix exactly like
// the built-in KalmanFilter does.

#include <concepts>
#include "augur/core/concepts.hpp"
#include "augur/math/backend.hpp"

namespace augur::filters {

template <typename T>
concept Filter = requires(T& filter,
                           const augur::math::Vector<typename T::Scalar, T::dimension>& z,
                           typename T::Scalar dt) {
    typename T::Scalar;
    typename T::Model;
    { T::dimension } -> std::convertible_to<std::size_t>;

    { filter.predict(dt) } -> std::same_as<void>;
    { filter.update(z) } -> std::same_as<void>;
    { filter.state() } -> std::same_as<const augur::math::Vector<typename T::Scalar, T::dimension>&>;
    { filter.covariance() } -> std::same_as<const augur::math::Matrix<typename T::Scalar, T::dimension>&>;

    // Innovation likelihood for the most recent update -- this is what
    // imm::Estimator uses to re-weight each model's mode probability.
    { filter.last_likelihood() } -> std::convertible_to<typename T::Scalar>;
};

} // namespace augur::filters
