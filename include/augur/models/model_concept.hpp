#pragma once
// augur/models/model_concept.hpp
//
// This concept IS the plugin system for motion models. There is no
// registry to populate and no base class to inherit: any type with
// these members can be dropped into an augur::imm::Estimator<...> or
// used standalone with augur::filters::KalmanFilter<Model>, including
// types augur has never heard of, defined entirely in your own game
// code. That's what "open instead of hardcoded" means concretely here.
//
// See docs/PLUGIN_GUIDE.md for a walkthrough of writing your own model
// from scratch (e.g. a domain-specific "orbits a waypoint" model that
// has no business being in a general-purpose tracking library).

#include <concepts>
#include "augur/core/concepts.hpp"
#include "augur/math/backend.hpp"

namespace augur::models {

template <typename T>
concept MotionModel = requires(const T& model,
                                const augur::math::Vector<typename T::Scalar, T::dimension>& x,
                                typename T::Scalar dt) {
    typename T::Scalar;
    core::Scalar<typename T::Scalar>;
    { T::dimension } -> std::convertible_to<std::size_t>;

    // x_{k+1} = f(x_k, dt) -- the deterministic part of the model.
    { model.transition(x, dt) } -> std::same_as<augur::math::Vector<typename T::Scalar, T::dimension>>;

    // Jacobian of f w.r.t. x, evaluated at x. For a linear model
    // (CV/CA) this is just the constant state-transition matrix and
    // `x` is ignored; for a nonlinear model (coordinated turn) it
    // genuinely depends on x. Filters use this uniformly (KF and EKF
    // alike) so a model never needs to know which filter is driving it.
    { model.jacobian(x, dt) } -> std::same_as<augur::math::Matrix<typename T::Scalar, T::dimension>>;

    // Process noise covariance Q for a step of length dt.
    { model.process_noise(dt) } -> std::same_as<augur::math::Matrix<typename T::Scalar, T::dimension>>;
};

} // namespace augur::models
