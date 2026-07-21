#pragma once
// augur/models/coordinated_turn.hpp
//
// State = [px, py, vx, vy, omega] (turn rate, rad/s) -- a classic
// nonlinear "nearly-constant-turn" model for a target circling or
// strafing rather than moving in a straight line. This is the model
// that makes an IMM mix noticeably better than a lone CV/CA filter
// whenever a target juke/strafes, which is most of the time in a
// MOBA/shooter context.
//
// This is inherently a planar (2D) model; a 3D extension (e.g. a
// vertical CV channel bolted on, or a full SO(3)-based turn model via
// Sophus) is listed as a roadmap item in docs/ROADMAP.md rather than
// implemented here, to keep this file's math honest and checkable.
//
// NOTE: the transition/jacobian formulas below follow the standard
// "nearly constant turn" derivation used throughout the tracking
// literature (e.g. Blackman & Popoli, "Design and Analysis of Modern
// Tracking Systems", and the Li & Jilkov survey cited in
// docs/ROADMAP.md).
//
// jacobian()'s small-omega branch was checked against an mpmath
// reference (per .claude/rules/testing.md) and a real bug was found and
// fixed there: the original version left F(0,4)/F(1,4)/F(2,4)/F(3,4) at
// their Identity default (0) near omega=0, which meant a target
// starting at (or passing through) zero turn rate could never pick up a
// nonzero omega estimate from position measurements alone -- the
// turn-rate state was silently unobservable. See the comment on
// jacobian() itself for the fix and tests/unit/ for regression coverage.
// Still run any further change here against a reference trajectory
// before trusting it.

#include <cmath>
#include "augur/math/backend.hpp"

namespace augur::models {

template <typename ScalarT>
class CoordinatedTurn {
public:
    using Scalar = ScalarT;
    static constexpr std::size_t dimension = 5; // px, py, vx, vy, omega

    using State = augur::math::Vector<Scalar, dimension>;
    using Transition = augur::math::Matrix<Scalar, dimension>;

    // process_noise_position/turn control how much the model expects
    // straight-line motion and turn rate (respectively) to wander.
    explicit CoordinatedTurn(Scalar process_noise_position = Scalar(1),
                             Scalar process_noise_turn_rate = Scalar(0.1))
        : q_pos_(process_noise_position), q_turn_(process_noise_turn_rate) {}

    [[nodiscard]] State transition(const State& x, Scalar dt) const {
        const Scalar vx = x(2), vy = x(3), omega = x(4);
        State out = x;

        constexpr Scalar epsilon = static_cast<Scalar>(1e-5);
        if (std::abs(omega) < epsilon) {
            // Degenerates cleanly to constant velocity as omega -> 0.
            out(0) = x(0) + vx * dt;
            out(1) = x(1) + vy * dt;
            return out;
        }

        const Scalar s = std::sin(omega * dt);
        const Scalar c = std::cos(omega * dt);
        out(0) = x(0) + (vx * s - vy * (Scalar(1) - c)) / omega;
        out(1) = x(1) + (vy * s + vx * (Scalar(1) - c)) / omega;
        out(2) = vx * c - vy * s;
        out(3) = vx * s + vy * c;
        out(4) = omega; // turn rate persists (it's the mode being estimated)
        return out;
    }

    [[nodiscard]] Transition jacobian(const State& x, Scalar dt) const {
        const Scalar vx = x(2), vy = x(3), omega = x(4);
        Transition F = Transition::Identity();
        const Scalar s = std::sin(omega * dt);
        const Scalar c = std::cos(omega * dt);

        // Widened from a bare "omega == 0" check. F(0,4)/F(1,4) below
        // both divide by omega^2, and evaluating that closed form
        // directly in float32 loses essentially all precision (>100%
        // relative error, confirmed via ad hoc python3 + mpmath per
        // .claude/rules/testing.md) for any |omega| below roughly
        // 0.1-0.2 rad/s, not just at exactly omega=0. The series used
        // below for those two terms stays accurate to <0.01% (vs. the
        // mpmath reference) out to |omega| ~ 10 rad/s, so this
        // threshold has wide margin on both sides.
        constexpr Scalar epsilon = static_cast<Scalar>(0.2);
        if (std::abs(omega) < epsilon) {
            F(0, 2) = dt;
            F(1, 3) = dt;
            // F(0,3)/F(1,2) omitted: their omega->0 limit is 0 and the
            // next term is O(omega * dt^2), negligible next to dt itself.
            // F(2,4)/F(3,4) have no division by omega, so the plain
            // closed form (real s/c, computed above) is already exact --
            // no series needed for these two.
            F(2, 4) = -dt * vx * s - dt * vy * c;
            F(3, 4) = dt * vx * c - dt * vy * s;

            // omega^4 Taylor expansion of F(0,4)/F(1,4) around omega=0
            // (derived symbolically, cross-checked numerically -- see
            // .claude/rules/testing.md). Without this, a target that
            // starts at or passes through omega=0 can never develop a
            // nonzero turn-rate estimate: the Kalman gain's coupling
            // between position innovations and the omega state comes
            // entirely through this term.
            const Scalar dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt, dt5 = dt4 * dt, dt6 = dt5 * dt;
            const Scalar w2 = omega * omega, w3 = w2 * omega, w4 = w3 * omega;
            F(0, 4) = -vy * dt2 / Scalar(2) - vx * dt3 * omega / Scalar(3)
                      + vy * dt4 * w2 / Scalar(8) + vx * dt5 * w3 / Scalar(30)
                      - vy * dt6 * w4 / Scalar(144);
            F(1, 4) = vx * dt2 / Scalar(2) - vy * dt3 * omega / Scalar(3)
                      - vx * dt4 * w2 / Scalar(8) + vy * dt5 * w3 / Scalar(30)
                      + vx * dt6 * w4 / Scalar(144);
            return F;
        }

        const Scalar omega2 = omega * omega;

        F(0, 2) = s / omega;
        F(0, 3) = -(Scalar(1) - c) / omega;
        F(0, 4) = (dt * vx * c - dt * vy * s) / omega - (vx * s - vy * (Scalar(1) - c)) / omega2;

        F(1, 2) = (Scalar(1) - c) / omega;
        F(1, 3) = s / omega;
        F(1, 4) = (dt * vx * s + dt * vy * c) / omega - (vy * s + vx * (Scalar(1) - c)) / omega2;

        F(2, 2) = c;
        F(2, 3) = -s;
        F(2, 4) = -dt * vx * s - dt * vy * c;

        F(3, 2) = s;
        F(3, 3) = c;
        F(3, 4) = dt * vx * c - dt * vy * s;

        // F(4, 4) stays 1: omega is a random walk between updates.
        return F;
    }

    [[nodiscard]] Transition process_noise(Scalar dt) const {
        Transition Q = Transition::Zero();
        const Scalar dt2 = dt * dt, dt3 = dt2 * dt / Scalar(3);
        Q(0, 0) = Q(1, 1) = q_pos_ * dt3;
        Q(2, 2) = Q(3, 3) = q_pos_ * dt;
        Q(4, 4) = q_turn_ * dt;
        return Q;
    }

private:
    Scalar q_pos_;
    Scalar q_turn_;
};

} // namespace augur::models
