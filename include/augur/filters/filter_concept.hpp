#pragma once
// augur/filters/filter_concept.hpp
//
// A Filter wraps a MotionModel plus a recursive state-estimation rule
// (linear KF, EKF, UKF, particle...). Like models::MotionModel, this is
// a structural concept: augur::imm::Estimator only ever talks to Filter,
// never to a concrete class, so a user-supplied filter (say, a
// UKF variant you wrote yourself) drops into an IMM mix exactly like
// the built-in KalmanFilter does.
//
// update() is checked against the filter's OWN typename T::Measurement,
// not a vector synthesized from T::dimension (the state dimension).
// Those two are only equal in the degenerate case of observing the whole
// state directly -- every built-in filter's real update() takes
// Vector<Scalar, MeasDim>, an independent template parameter (see e.g.
// filters/kalman.hpp). Synthesizing the check vector from `dimension`
// meant `static_assert(Filter<KalmanFilter<Model, MeasDim>>)` passed
// even when MeasDim != dimension -- a real signature mismatch this
// concept was supposed to catch (docs/IMPROVEMENT_PLAN.md's investigation
// found this the same way from two independent angles: this file's own
// synthesized-vector approach, and plugin/registry.hpp's IFilterBox/
// FilterBox making the identical mistake in its own type signature).
// Requiring `typename T::Measurement` to exist and using it directly
// closes both at once, and just formalizes what track/track_manager.hpp
// and track/out_of_sequence.hpp already, informally, assumed was true of
// every Filter (see their own `using Measurement = typename
// FilterT::Measurement;` lines).
//
// set_state() is required for the same reason: imm::Estimator,
// imm::HeterogeneousEstimator, and track/out_of_sequence.hpp all call it
// unconditionally every cycle, but it wasn't part of this concept --
// meaning a user who wrote a custom Filter, validated it against
// `static_assert(Filter<YourType>)` per docs/PLUGIN_GUIDE.md, and got a
// clean pass could still hit a confusing deep template error the moment
// they plugged into imm::Estimator. All 6 built-in filters already
// implement it, so requiring it here breaks nothing existing.
#include <concepts>
#include "augur/math/backend.hpp"

namespace augur::filters {

template <typename T>
concept Filter = requires(T& filter, const typename T::Measurement& z, typename T::Scalar dt) {
    typename T::Scalar;
    typename T::Model;
    typename T::Measurement;
    { T::dimension } -> std::convertible_to<std::size_t>;

    { filter.predict(dt) } -> std::same_as<void>;
    { filter.update(z) } -> std::same_as<void>;
    { filter.state() } -> std::same_as<const augur::math::Vector<typename T::Scalar, T::dimension>&>;
    { filter.covariance() } -> std::same_as<const augur::math::Matrix<typename T::Scalar, T::dimension>&>;
    { filter.set_state(filter.state(), filter.covariance()) } -> std::same_as<void>;

    // Innovation likelihood for the most recent update -- this is what
    // imm::Estimator uses to re-weight each model's mode probability.
    { filter.last_likelihood() } -> std::convertible_to<typename T::Scalar>;
};

} // namespace augur::filters
