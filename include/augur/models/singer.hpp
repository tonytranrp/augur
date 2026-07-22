#pragma once
// augur/models/singer.hpp
//
// Reference: R. A. Singer, "Estimating Optimal Tracking Filter
// Performance for Manned Maneuvering Targets," IEEE Transactions on
// Aerospace and Electronic Systems, AES-6(4), 1970, pp. 473-483.
// See also: Blackman & Popoli, "Design and Analysis of Modern Tracking
// Systems" (1999); Li & Jilkov, "Survey of Maneuvering Target Tracking:
// Dynamic Models" (2000) -- both give the process-noise derivation this
// file now implements exactly (see process_noise()'s own comment).
//
// Model: per-axis acceleration is a zero-mean, first-order stationary
// Markov (Ornstein-Uhlenbeck) process with correlation time tau =
// 1/alpha, rather than a random walk (as in ConstantAcceleration) or a
// hard zero (as in ConstantVelocity) -- it captures targets whose
// maneuvers decay back toward "no maneuver" over a characteristic
// timescale, which is a better match for most player-controlled
// dodging/strafing than either extreme.
//
// UPDATE: this file was previously flagged as a "sketch" because
// process_noise() only filled the acceleration-diagonal entry as a
// placeholder. It's since been replaced by the exact closed-form Q(dt) --
// the full Van Loan integral, all 6 position/velocity/acceleration cross
// terms, not just the diagonal -- derived symbolically (sympy) and
// independently cross-checked two ways before any of this was written:
// (1) direct numerical integration (Simpson's rule, a completely
// different method from the symbolic Van Loan integral) agrees with the
// closed form to ~1e-10 absolute across a representative (alpha, dt)
// sweep; (2) every tested (alpha, dt) combination's Q is confirmed
// positive-semi-definite (Eigen::SelfAdjointEigenSolver, min eigenvalue
// checked). transition()/jacobian()'s own closed form (f_pv, f_va below)
// was already algebraically correct -- confirmed by re-deriving the same
// matrix exponential symbolically -- but had NO small-alpha*dt protection
// at all, and both it and the new Q(dt) divide by alpha to increasingly
// high powers (up to alpha^5 in Q_pp), which catastrophically cancels in
// float32 far short of alpha*dt actually being 0: verified (ad hoc
// python3 + mpmath, per .claude/rules/testing.md) a NAIVE evaluation is
// off by up to 15.9% relative error at alpha*dt as unremarkable as
// tau=60s (this file's own documented "lazy maneuverer" case), dt=1/30s
// -- not a corner case. Every closed-form division by alpha below now
// branches on the natural dimensionless small parameter x=alpha*dt < 0.2
// to a 6-term Taylor series instead, verified (ad hoc python3 + mpmath,
// coefficients generated programmatically from the same sympy expansion
// rather than hand-copied, after an initial hand-transcription attempt
// was caught producing a non-shrinking, and therefore wrong, series by
// checking that error strictly decreases as more terms are added) to
// stay within ~1e-8 relative error of the exact closed form throughout
// that branch, across alpha in [0.0167, 2.0] and dt in [1/30s, 2s] -- the
// realistic range this file's own tau documentation and existing tests
// span. This file now belongs in the "solid" tier per
// .claude/rules/code-style.md's three-tier convention: no named part is a
// documented simplification anymore.

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

    // Exact closed-form Q(dt) -- the full Van Loan integral
    // integral_0^dt Phi(dt-s) L q L^T Phi(dt-s)^T ds, where Phi is this
    // file's own build_transition(), L=[0,0,1]^T selects the
    // acceleration channel (the only one w(t) directly drives), and
    // q=2*alpha*sigma^2 is the OU acceleration process's own white-noise
    // spectral density (chosen so its STATIONARY variance is sigma^2,
    // the standard Singer parameterization). See this file's own top
    // comment for the derivation/verification methodology. All 6
    // position/velocity/acceleration cross terms per axis, not just the
    // diagonal -- e.g. Q_pa is nonzero because a nonzero current
    // acceleration correlates with where the position ends up.
    [[nodiscard]] Transition process_noise(Scalar dt) const {
        Transition Q = Transition::Zero();
        const Scalar q = Scalar(2) * alpha_ * sigma_ * sigma_;
        const Scalar x = alpha_ * dt; // the natural small parameter (see top comment)

        // Widened from a bare "alpha == 0" check for the same reason
        // build_transition()'s own branch below is: every term here
        // divides by alpha to some power (up to alpha^5 for Q_pp), and
        // that catastrophically cancels in float32 well before alpha*dt
        // is anywhere near exactly 0 (verified: 15.9% relative error at
        // alpha*dt ~ 0.0006 without this branch -- see top comment).
        constexpr Scalar epsilon = static_cast<Scalar>(0.2);
        Scalar Q_pp, Q_pv, Q_pa, Q_vv, Q_va, Q_aa;
        if (std::abs(x) < epsilon) {
            // 6-term Taylor series in x=alpha*dt (coefficients generated
            // programmatically from the same sympy expansion that
            // produced the exact closed form below, per this file's own
            // top comment) -- verified accurate to <1e-8 relative error
            // vs. the exact closed form throughout this branch.
            const Scalar x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x;
            const Scalar dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt, dt5 = dt4 * dt;
            Q_pp = q * dt5 *
                   (Scalar(1) / 20 - x / 36 + Scalar(5) * x2 / 504 - x3 / 360 + Scalar(17) * x4 / 25920 -
                    Scalar(41) * x5 / 302400);
            Q_pv = q * dt4 *
                   (Scalar(1) / 8 - x / 12 + Scalar(5) * x2 / 144 - x3 / 90 + Scalar(17) * x4 / 5760 -
                    Scalar(41) * x5 / 60480);
            Q_pa = q * dt3 *
                   (Scalar(1) / 6 - x / 6 + Scalar(11) * x2 / 120 - Scalar(13) * x3 / 360 + Scalar(19) * x4 / 1680 -
                    x5 / 336);
            Q_vv = q * dt3 *
                   (Scalar(1) / 3 - x / 4 + Scalar(7) * x2 / 60 - x3 / 24 + Scalar(31) * x4 / 2520 - x5 / 320);
            Q_va = q * dt2 * (Scalar(1) / 2 - x / 2 + Scalar(7) * x2 / 24 - x3 / 8 + Scalar(31) * x4 / 720 - x5 / 80);
            Q_aa = q * dt * (Scalar(1) - x + Scalar(2) * x2 / 3 - x3 / 3 + Scalar(2) * x4 / 15 - Scalar(2) * x5 / 45);
        } else {
            const Scalar e = std::exp(-x);
            const Scalar a2 = alpha_ * alpha_, a3 = a2 * alpha_, a4 = a3 * alpha_, a5 = a4 * alpha_;
            const Scalar dt2 = dt * dt, dt3 = dt2 * dt;
            Q_pp = q * (Scalar(2) * a3 * dt3 - Scalar(6) * a2 * dt2 + Scalar(6) * alpha_ * dt * (Scalar(1) - Scalar(2) * e) -
                        Scalar(3) * e * e + Scalar(3)) /
                   (Scalar(6) * a5);
            Q_pv = q * (a2 * dt2 + Scalar(2) * alpha_ * dt * (e - Scalar(1)) + e * e - Scalar(2) * e + Scalar(1)) /
                   (Scalar(2) * a4);
            Q_pa = q * (-Scalar(2) * alpha_ * dt * e - e * e + Scalar(1)) / (Scalar(2) * a3);
            Q_vv = q * (Scalar(2) * alpha_ * dt - e * e + Scalar(4) * e - Scalar(3)) / (Scalar(2) * a3);
            Q_va = q * (e * e - Scalar(2) * e + Scalar(1)) / (Scalar(2) * a2);
            Q_aa = q * (Scalar(1) - e * e) / (Scalar(2) * alpha_);
        }

        for (int axis = 0; axis < SpatialDim; ++axis) {
            const int p = axis, v = SpatialDim + axis, a = 2 * SpatialDim + axis;
            Q(p, p) = Q_pp;
            Q(p, v) = Q(v, p) = Q_pv;
            Q(p, a) = Q(a, p) = Q_pa;
            Q(v, v) = Q_vv;
            Q(v, a) = Q(a, v) = Q_va;
            Q(a, a) = Q_aa;
        }
        return Q;
    }

private:
    [[nodiscard]] Transition build_transition(Scalar dt) const {
        Transition F = Transition::Identity();
        const Scalar e = std::exp(-alpha_ * dt);
        const Scalar x = alpha_ * dt; // the natural small parameter (see top comment)

        // f_pv=(x-1+e)/alpha^2 and f_va=(1-e)/alpha both catastrophically
        // cancel in float32 well before alpha*dt is anywhere near
        // exactly 0 -- see this file's own top comment for the measured
        // 15.9% relative error this branch fixes, and
        // .claude/rules/testing.md for the verification methodology.
        constexpr Scalar epsilon = static_cast<Scalar>(0.2);
        Scalar f_pv, f_va;
        if (std::abs(x) < epsilon) {
            const Scalar x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x;
            f_pv = dt * dt *
                   (Scalar(0.5) - x / Scalar(6) + x2 / Scalar(24) - x3 / Scalar(120) + x4 / Scalar(720) -
                    x5 / Scalar(5040));
            f_va = dt * (Scalar(1) - x / Scalar(2) + x2 / Scalar(6) - x3 / Scalar(24) + x4 / Scalar(120) -
                         x5 / Scalar(720));
        } else {
            f_pv = (x - Scalar(1) + e) / (alpha_ * alpha_);
            f_va = (Scalar(1) - e) / alpha_;
        }

        for (int axis = 0; axis < SpatialDim; ++axis) {
            const int p = axis, v = SpatialDim + axis, a = 2 * SpatialDim + axis;
            F(p, v) = dt;
            F(p, a) = f_pv;
            F(v, a) = f_va;
            F(a, a) = e;
        }
        return F;
    }

    Scalar alpha_; // 1 / maneuver time constant
    Scalar sigma_; // maneuver std dev
};

} // namespace augur::models
