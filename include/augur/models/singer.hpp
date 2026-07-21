#pragma once
// augur/models/singer.hpp
//
// ROADMAP MODEL -- transition matrix is the standard closed form below;
// process_noise() is a simplified placeholder, NOT the exact Singer
// closed-form integral. Verify against the source before trusting this
// in anything real. Unlike constant_velocity/constant_acceleration/
// coordinated_turn (which are load-bearing and covered by
// tests/unit/), this file is an honest sketch: it satisfies the
// MotionModel concept and compiles, but the noise model needs a pass
// before it belongs in the "solid" tier.
//
// Reference: R. A. Singer, "Estimating Optimal Tracking Filter
// Performance for Manned Maneuvering Targets," IEEE Transactions on
// Aerospace and Electronic Systems, AES-6(4), 1970, pp. 473-483.
// See also: Blackman & Popoli, "Design and Analysis of Modern Tracking
// Systems" (1999); Li & Jilkov, "Survey of Maneuvering Target Tracking:
// Dynamic Models" (2000) -- both give the full process-noise
// derivation this file simplifies.
//
// Model: per-axis acceleration is a zero-mean, first-order stationary
// Markov (Ornstein-Uhlenbeck) process with correlation time tau =
// 1/alpha, rather than a random walk (as in ConstantAcceleration) or a
// hard zero (as in ConstantVelocity) -- it captures targets whose
// maneuvers decay back toward "no maneuver" over a characteristic
// timescale, which is a better match for most player-controlled
// dodging/strafing than either extreme.

#include <cmath>
#include "augur/math/backend.hpp"

namespace augur::models {

template <typename ScalarT, int SpatialDim>
class Singer {
public:
    using Scalar = ScalarT;
    static constexpr std::size_t dimension = 3 * SpatialDim; // pos, vel, accel per axis

    using State = augur::math::Vector<Scalar, dimension>;
    using Transition = augur::math::Matrix<Scalar, dimension>;

    // maneuver_time_constant: Singer's tau, typically 5-20s for an
    // "aggressive" maneuverer down to ~60s for a lazy one (values from
    // Singer 1970 / Blackman & Popoli, tuned for aircraft -- expect to
    // retune substantially for game-character timescales).
    // maneuver_std_dev: sigma, the RMS acceleration magnitude expected
    // during a maneuver.
    explicit Singer(Scalar maneuver_time_constant = Scalar(2),
                    Scalar maneuver_std_dev = Scalar(1))
        : alpha_(Scalar(1) / maneuver_time_constant), sigma_(maneuver_std_dev) {}

    [[nodiscard]] State transition(const State& x, Scalar dt) const {
        return build_transition(dt) * x;
    }

    [[nodiscard]] Transition jacobian(const State& /*x*/, Scalar dt) const {
        return build_transition(dt); // linear in the state
    }

    // SIMPLIFIED placeholder -- NOT the exact Singer closed-form Q.
    // See Blackman & Popoli / Li & Jilkov for the full derivation
    // (it involves exp(-alpha*dt) cross-terms between all three
    // per-axis states, not just a diagonal block).
    [[nodiscard]] Transition process_noise(Scalar dt) const {
        Transition Q = Transition::Zero();
        const Scalar q = Scalar(2) * alpha_ * sigma_ * sigma_;
        for (int axis = 0; axis < SpatialDim; ++axis) {
            const int a = 2 * SpatialDim + axis;
            Q(a, a) = q * dt; // acceleration-channel noise only, as a starting point
        }
        return Q;
    }

private:
    [[nodiscard]] Transition build_transition(Scalar dt) const {
        Transition F = Transition::Identity();
        const Scalar e = std::exp(-alpha_ * dt);
        const Scalar f_pv = (alpha_ * dt - Scalar(1) + e) / (alpha_ * alpha_);
        const Scalar f_pa = f_pv; // position<-accel term shares the same closed form here
        const Scalar f_va = (Scalar(1) - e) / alpha_;
        for (int axis = 0; axis < SpatialDim; ++axis) {
            const int p = axis, v = SpatialDim + axis, a = 2 * SpatialDim + axis;
            F(p, v) = dt;
            F(p, a) = f_pa;
            F(v, a) = f_va;
            F(a, a) = e;
        }
        return F;
    }

    Scalar alpha_; // 1 / maneuver time constant
    Scalar sigma_; // maneuver std dev
};

} // namespace augur::models
