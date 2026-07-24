#pragma once
// augur/models/constant_velocity.hpp
//
// The workhorse baseline model: state = [position (Dim), velocity (Dim)],
// assumes velocity is constant between steps (acceleration is treated as
// zero-mean process noise). This is almost always one of the models fed
// into an imm::Estimator alongside constant_acceleration / coordinated_turn
// -- see docs/ARCHITECTURE.md for why IMM wants "wrong but cheap" models
// in the mix rather than one "correct" model.

#include "augur/math/backend.hpp"

namespace augur::models {

// SpatialDim = 2 for a top-down/2D game, 3 for full 3D.
template <typename ScalarT, int SpatialDim>
class ConstantVelocity {
public:
    using Scalar = ScalarT;
    static constexpr std::size_t dimension = 2 * SpatialDim;

    using State = augur::math::Vector<Scalar, dimension>;
    using Transition = augur::math::Matrix<Scalar, dimension>;

    // Continuous-white-noise-acceleration spectral density. Larger =
    // the model expects the target to depart from constant velocity
    // more readily (i.e. it trusts the CV assumption less).
    explicit ConstantVelocity(Scalar accel_noise_density = Scalar(1))
        : noise_density_(accel_noise_density) {}

    [[nodiscard]] State transition(const State& x, Scalar dt) const {
        return build_transition(dt) * x;
    }

    [[nodiscard]] Transition jacobian(const State& /*x*/, Scalar dt) const {
        // Linear model: the Jacobian IS the transition matrix, x-independent.
        return build_transition(dt);
    }

    [[nodiscard]] Transition process_noise(Scalar dt) const {
        // Standard discretized continuous-white-noise-acceleration (CWNA)
        // block, applied independently per spatial axis.
        Transition Q = Transition::Zero();
        const Scalar dt2 = dt * dt;
        const Scalar dt3 = dt2 * dt / Scalar(3);
        const Scalar dt2_2 = dt2 / Scalar(2);
        for (int axis = 0; axis < SpatialDim; ++axis) {
            const int p = axis;               // position row/col for this axis
            const int v = SpatialDim + axis;   // velocity row/col for this axis
            Q(p, p) = noise_density_ * dt3;
            Q(p, v) = Q(v, p) = noise_density_ * dt2_2;
            Q(v, v) = noise_density_ * dt;
        }
        return Q;
    }

private:
    [[nodiscard]] Transition build_transition(Scalar dt) const {
        Transition F = Transition::Identity();
        for (int axis = 0; axis < SpatialDim; ++axis) {
            F(axis, SpatialDim + axis) = dt; // position += velocity * dt
        }
        return F;
    }

    Scalar noise_density_;
};

} // namespace augur::models
