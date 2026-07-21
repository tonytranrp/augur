#pragma once
// augur/models/constant_acceleration.hpp
//
// state = [position (Dim), velocity (Dim), acceleration (Dim)]. Tracks
// targets that speed up/slow down smoothly (a diving or accelerating
// target) better than ConstantVelocity, at the cost of one more state
// per axis. Typically the second model in an IMM mix alongside CV.

#include "augur/math/backend.hpp"

namespace augur::models {

template <typename ScalarT, int SpatialDim>
class ConstantAcceleration {
public:
    using Scalar = ScalarT;
    static constexpr std::size_t dimension = 3 * SpatialDim;

    using State = augur::math::Vector<Scalar, dimension>;
    using Transition = augur::math::Matrix<Scalar, dimension>;

    explicit ConstantAcceleration(Scalar jerk_noise_density = Scalar(1))
        : jerk_density_(jerk_noise_density) {}

    [[nodiscard]] State transition(const State& x, Scalar dt) const {
        return build_transition(dt) * x;
    }

    [[nodiscard]] Transition jacobian(const State& /*x*/, Scalar dt) const {
        return build_transition(dt);
    }

    [[nodiscard]] Transition process_noise(Scalar dt) const {
        // Discretized continuous white-noise-jerk model, per axis.
        Transition Q = Transition::Zero();
        const Scalar dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt, dt5 = dt4 * dt;
        for (int axis = 0; axis < SpatialDim; ++axis) {
            const int p = axis;
            const int v = SpatialDim + axis;
            const int a = 2 * SpatialDim + axis;
            Q(p, p) = jerk_density_ * dt5 / Scalar(20);
            Q(p, v) = Q(v, p) = jerk_density_ * dt4 / Scalar(8);
            Q(p, a) = Q(a, p) = jerk_density_ * dt3 / Scalar(6);
            Q(v, v) = jerk_density_ * dt3 / Scalar(3);
            Q(v, a) = Q(a, v) = jerk_density_ * dt2 / Scalar(2);
            Q(a, a) = jerk_density_ * dt;
        }
        return Q;
    }

private:
    [[nodiscard]] Transition build_transition(Scalar dt) const {
        Transition F = Transition::Identity();
        const Scalar half_dt2 = dt * dt / Scalar(2);
        for (int axis = 0; axis < SpatialDim; ++axis) {
            F(axis, SpatialDim + axis) = dt;                  // pos += vel * dt
            F(axis, 2 * SpatialDim + axis) = half_dt2;        // pos += 1/2 * a * dt^2
            F(SpatialDim + axis, 2 * SpatialDim + axis) = dt; // vel += a * dt
        }
        return F;
    }

    Scalar jerk_density_;
};

} // namespace augur::models
