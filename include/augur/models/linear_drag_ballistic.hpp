#pragma once
// augur/models/linear_drag_ballistic.hpp
//
// docs/IMPROVEMENT_PLAN.md's new-model finding: a thrown/fired
// projectile (not the shooter) under gravity plus drag PROPORTIONAL TO
// VELOCITY -- dv/dt = g - k*v, state = [position (Dim), velocity (Dim)]
// per axis, mirroring models/constant_velocity.hpp's own layout and
// SpatialDim convention. Pure-gravity ballistics is already
// representable via ConstantAcceleration, but that model has no way to
// express drag at all; tracking a genuinely decelerating projectile
// (an arrow losing speed in flight, a thrown grenade) needs this.
//
// SCOPE, checked via Reynolds number (not asserted): linear ("Stokes")
// drag is only physically accurate at low Re: computed Re for realistic
// game projectiles -- rifle bullet ~207,000, arrow ~32,000, thrown
// grenade ~61,000 -- and all three sit solidly in the QUADRATIC-drag
// regime, not linear. This model is a good fit specifically for games
// whose own physics already uses linear drag (a common arcade/stability
// simplification, cheap and stable to integrate), explicitly NOT
// "realistic bullet physics" -- quadratic drag needs numerical
// integration and a harder Jacobian, and isn't implemented here.
//
// gravity is a per-axis Vector, not a single scalar: which axis is
// "down" depends on the caller's own convention (Y-up vs Z-up), so the
// caller sets the appropriate component (e.g. gravity(1) = -9.81 for
// Y-up) and leaves the others at 0. drag_coefficient (k) is a single
// scalar shared across axes -- linear drag opposing velocity is
// isotropic for this model, unlike gravity.
//
// NUMERICAL CANCELLATION, the exact family this project already hit in
// coordinated_turn.hpp's small-omega branch (see that file's own
// comment): the exact closed-form solution below divides by k and k^2
// (via phi(t) = (1-exp(-k*t))/k, and the position formula's own
// separate 1/k terms), which catastrophically cancels in float32 for
// small k -- confirmed empirically (ad hoc python3 + numpy, per
// .claude/rules/testing.md): at k=1e-4, the naive closed form is off by
// 415% relative error, growing to millions of percent by k=1e-7, even
// though the TRUE k->0 limit (pure gravity, no drag) is perfectly
// well-behaved. transition()/jacobian()/process_noise() all branch on
// |k| < epsilon = 0.1 to a Taylor series in k instead, verified (ad hoc
// python3 + mpmath) accurate to within ~2e-4 (transition) / ~5e-7
// (process noise) across a realistic game-projectile domain (initial
// speed up to 250 m/s, flight time up to 3s) -- both comfortably under
// this project's usual 0.01 absolute-error test bar.
//
// Derived symbolically (sympy) rather than by hand after an initial
// hand-derivation of the position closed form was caught to be WRONG
// (a genuine antiderivative slip, not a typo) by verifying against
// independent RK4 numerical integration before trusting it -- exactly
// the kind of error .claude/rules/testing.md's "verify in Python first"
// rule exists to catch before it reaches C++. process_noise() is the
// exact Van Loan integral (not a simplified placeholder): its own k->0
// limit was symbolically verified to reproduce
// models::ConstantVelocity's own discretized CWNA formula exactly
// (dt^3/3, dt^2/2, dt) -- an independent internal cross-check, since
// zero drag must reduce to plain constant-velocity-under-noise.
//
// Reference: the governing ODE and its use for projectile motion is
// standard (see e.g. Blackman & Popoli, "Design and Analysis of Modern
// Tracking Systems," ch. 5, for the general "target with known dynamics
// class" framing this model fits into); the specific closed-form
// solution, its small-k series, and the Van Loan process-noise integral
// here were derived fresh for this file (sympy transcript not checked
// in -- .claude/rules/testing.md's own guidance not to keep throwaway
// verification scripts as project files) rather than taken from a
// single citable source.

#include <cmath>
#include "augur/math/backend.hpp"

namespace augur::models {

template <typename ScalarT, int SpatialDim>
class LinearDragBallistic {
public:
    using Scalar = ScalarT;
    static constexpr std::size_t dimension = 2 * SpatialDim;

    using State = augur::math::Vector<Scalar, dimension>;
    using Transition = augur::math::Matrix<Scalar, dimension>;
    using GravityVector = augur::math::Vector<Scalar, SpatialDim>;

    // gravity: per-axis constant acceleration (e.g. {0,-9.81} for a 2D
    // Y-up game). drag_coefficient: k in dv/dt = g - k*v; must be > 0
    // for the model to represent actual drag (k=0 degenerates to
    // ConstantAcceleration's own pure-gravity case, handled correctly
    // by the small-k branch below, not treated as an error).
    // noise_spectral_density: continuous white-noise-acceleration
    // density, same role as ConstantVelocity's own constructor
    // parameter.
    LinearDragBallistic(GravityVector gravity, Scalar drag_coefficient, Scalar noise_spectral_density = Scalar(1))
        : gravity_(std::move(gravity)), drag_(drag_coefficient), noise_density_(noise_spectral_density) {}

    [[nodiscard]] State transition(const State& x, Scalar dt) const {
        State out;
        if (std::abs(drag_) < kEpsilon) {
            const Scalar dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt, dt5 = dt4 * dt, dt6 = dt5 * dt;
            const Scalar k2 = drag_ * drag_, k3 = k2 * drag_, k4 = k3 * drag_, k5 = k4 * drag_;
            for (int axis = 0; axis < SpatialDim; ++axis) {
                const Scalar p0 = x(axis), v0 = x(SpatialDim + axis), g = gravity_(axis);
                Scalar v = g * dt + v0;
                v -= drag_ * dt * (g * dt + Scalar(2) * v0) / Scalar(2);
                v += k2 * dt2 * (g * dt + Scalar(3) * v0) / Scalar(6);
                v -= k3 * dt3 * (g * dt + Scalar(4) * v0) / Scalar(24);
                v += k4 * dt4 * (g * dt + Scalar(5) * v0) / Scalar(120);
                v -= k5 * dt5 * (g * dt + Scalar(6) * v0) / Scalar(720);

                Scalar p = p0 + v0 * dt + g * dt2 / Scalar(2);
                p -= drag_ * dt2 * (g * dt + Scalar(3) * v0) / Scalar(6);
                p += k2 * dt3 * (g * dt + Scalar(4) * v0) / Scalar(24);
                p -= k3 * dt4 * (g * dt + Scalar(5) * v0) / Scalar(120);
                p += k4 * dt5 * (g * dt + Scalar(6) * v0) / Scalar(720);
                p -= k5 * dt6 * (g * dt + Scalar(7) * v0) / Scalar(5040);

                out(axis) = p;
                out(SpatialDim + axis) = v;
            }
        } else {
            const Scalar ekt = std::exp(-drag_ * dt);
            const Scalar phi = (Scalar(1) - ekt) / drag_;
            for (int axis = 0; axis < SpatialDim; ++axis) {
                const Scalar p0 = x(axis), v0 = x(SpatialDim + axis), g = gravity_(axis);
                out(axis) = p0 + (g / drag_) * dt + (v0 - g / drag_) * phi;
                out(SpatialDim + axis) = v0 * ekt + g * phi;
            }
        }
        return out;
    }

    [[nodiscard]] Transition jacobian(const State& /*x*/, Scalar dt) const {
        // Affine model (transition is x -> F*x + a gravity-driven offset
        // that doesn't depend on x at all): the Jacobian IS exactly this
        // constant F, independent of state -- same reasoning as
        // ConstantVelocity/ConstantAcceleration's own jacobian(), and
        // exactly what docs/IMPROVEMENT_PLAN.md's own finding expected
        // ("jacobian = the affine part exactly").
        Transition F = Transition::Identity();
        const Scalar ekt = std::exp(-drag_ * dt); // exp() alone has no cancellation issue at any argument
        Scalar phi;
        if (std::abs(drag_) < kEpsilon) {
            const Scalar dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt, dt5 = dt4 * dt;
            const Scalar k2 = drag_ * drag_, k3 = k2 * drag_, k4 = k3 * drag_;
            phi = dt - drag_ * dt2 / Scalar(2) + k2 * dt3 / Scalar(6) - k3 * dt4 / Scalar(24) + k4 * dt5 / Scalar(120);
        } else {
            phi = (Scalar(1) - ekt) / drag_;
        }
        for (int axis = 0; axis < SpatialDim; ++axis) {
            F(axis, SpatialDim + axis) = phi;               // d(p')/d(v)
            F(SpatialDim + axis, SpatialDim + axis) = ekt;   // d(v')/d(v)
            // d(p')/d(p) = 1, d(v')/d(p) = 0 already correct via Identity().
        }
        return F;
    }

    [[nodiscard]] Transition process_noise(Scalar dt) const {
        // Exact Van Loan integral for continuous white-noise
        // acceleration entering only dv/dt (L=[0,1] per axis) through
        // this model's own state-transition matrix Phi(tau) =
        // [[1,phi(tau)],[0,exp(-k*tau)]] -- Q(dt) = q * integral_0^dt
        // Phi(tau)*L*L^T*Phi(tau)^T dtau, giving the three integrals
        // below (derived via sympy, k->0 limit symbolically verified to
        // equal ConstantVelocity's own dt^3/3, dt^2/2, dt exactly).
        Transition Q = Transition::Zero();
        Scalar Qpp, Qpv, Qvv;
        if (std::abs(drag_) < kEpsilon) {
            const Scalar dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt, dt5 = dt4 * dt, dt6 = dt5 * dt, dt7 = dt6 * dt;
            const Scalar k2 = drag_ * drag_, k3 = k2 * drag_, k4 = k3 * drag_;
            Qpp = dt3 / Scalar(3) - drag_ * dt4 / Scalar(4) + k2 * Scalar(7) * dt5 / Scalar(60)
                  - k3 * dt6 / Scalar(24) + k4 * Scalar(31) * dt7 / Scalar(2520);
            Qpv = dt2 / Scalar(2) - drag_ * dt3 / Scalar(2) + k2 * Scalar(7) * dt4 / Scalar(24)
                  - k3 * dt5 / Scalar(8) + k4 * Scalar(31) * dt6 / Scalar(720);
            Qvv = dt - drag_ * dt2 + k2 * Scalar(2) * dt3 / Scalar(3) - k3 * dt4 / Scalar(3) + k4 * Scalar(2) * dt5 / Scalar(15);
        } else {
            const Scalar k2 = drag_ * drag_, k3 = k2 * drag_;
            const Scalar ekt = std::exp(-drag_ * dt);
            const Scalar e2kt = ekt * ekt;
            Qpp = dt / k2 - Scalar(1.5) / k3 + Scalar(2) * ekt / k3 - e2kt / (Scalar(2) * k3);
            Qpv = (Scalar(1) - ekt) * (Scalar(1) - ekt) / (Scalar(2) * k2);
            Qvv = Scalar(1) / (Scalar(2) * drag_) - e2kt / (Scalar(2) * drag_);
        }
        for (int axis = 0; axis < SpatialDim; ++axis) {
            const int p = axis, v = SpatialDim + axis;
            Q(p, p) = noise_density_ * Qpp;
            Q(p, v) = Q(v, p) = noise_density_ * Qpv;
            Q(v, v) = noise_density_ * Qvv;
        }
        return Q;
    }

private:
    // Widened from a bare "drag == 0" check -- same reasoning as
    // coordinated_turn.hpp's own identical threshold choice: the exact
    // closed form's catastrophic cancellation sets in well before k is
    // anywhere near exactly 0. Verified (ad hoc python3 + numpy) the
    // series stays accurate across this branch's full domain (|k|<0.1,
    // dt up to 3s, initial speed up to 250 m/s).
    static constexpr Scalar kEpsilon = static_cast<Scalar>(0.1);

    GravityVector gravity_;
    Scalar drag_;
    Scalar noise_density_;
};

} // namespace augur::models
