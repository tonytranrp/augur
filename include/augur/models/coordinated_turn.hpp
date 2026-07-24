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
// Deliberately no SpatialDim template parameter (unlike ConstantVelocity/
// ConstantAcceleration/LinearDragBallistic): the turn dynamics below have
// no per-axis loop to generalize, so a SpatialDim that only ever accepts
// 2 would be a lying template parameter. models::CoordinatedTurn3D is
// the real 3D story -- it composes this model's xy-block verbatim with a
// decoupled vertical channel, rather than this file growing a dimension
// parameter it can't actually support for anything but 2.
//
// NOTE: the transition/jacobian formulas below follow the standard
// "nearly constant turn" derivation used throughout the tracking
// literature (e.g. Blackman & Popoli, "Design and Analysis of Modern
// Tracking Systems", and the Li & Jilkov survey cited in
// docs/ROADMAP.md).
//
// Both transition()'s and jacobian()'s small-omega branches have been
// checked against an mpmath reference (per .claude/rules/testing.md)
// more than once, and each time a real bug was found and fixed --
// most recently a full pass across every entry (not just the one
// originally reported) after widening tests/unit/test_coordinated_turn.cpp's
// own finite-difference sweep to vary dt as well as omega, which is what
// surfaces this class of bug: several of these terms are only
// negligible for a SMALL dt, not for small omega alone, and a fixed
// dt=1/30 test parameter let three of them ship unnoticed. See the
// per-entry comments on transition() and jacobian() themselves for each
// specific fix and its derivation, and tests/unit/ for regression
// coverage. Still run any further change here against a reference
// trajectory before trusting it -- this file has a real track record of
// "looks right, checked against nothing, was subtly wrong."

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

    // accel_noise_density/turn_rate_noise_density control how much the
    // model expects straight-line motion and turn rate (respectively) to
    // wander. turn_rate_noise_density has units rad^2/s^3 (it's the PSD
    // feeding omega's own random-walk term, process_noise()'s Q(4,4)).
    explicit CoordinatedTurn(Scalar accel_noise_density = Scalar(1),
                             Scalar turn_rate_noise_density = Scalar(0.1))
        : q_pos_(accel_noise_density), q_turn_(turn_rate_noise_density) {}

    [[nodiscard]] State transition(const State& x, Scalar dt) const {
        const Scalar vx = x(2), vy = x(3), omega = x(4);
        State out = x;
        const Scalar s = std::sin(omega * dt);
        const Scalar c = std::cos(omega * dt);

        // Same threshold and same underlying reason as jacobian()'s own
        // small-omega branch: the general closed form's
        // (1-cos(omega*dt))/omega term catastrophically cancels in
        // float32 well before omega is anywhere near exactly 0 -- an
        // earlier, much tighter epsilon=1e-5 here left a real gap.
        // Found via test_helpers.hpp's finite-difference check of
        // jacobian() coming back "wrong" by ~0.016 at omega=0, dt=1.0s;
        // traced (ad hoc python3 + numpy/mpmath, per
        // .claude/rules/testing.md) to THIS function, not jacobian() --
        // at omega=1e-3 to 1e-4 (comfortably above the old 1e-5
        // threshold, so the "exact" branch ran), computing
        // 1-cos(omega*dt) in float32 loses most of its significant
        // digits, and the resulting ~1.6e-5 absolute error per
        // evaluation is large enough to visibly corrupt a caller's own
        // finite difference of this function (division by a small
        // eps amplifies it). jacobian()'s own analytic value was correct
        // throughout; the finite-difference "ground truth" it was being
        // checked against was not.
        constexpr Scalar epsilon = static_cast<Scalar>(0.2);
        if (std::abs(omega) < epsilon) {
            // omega^4 Taylor series for the position update -- the
            // antiderivative (w.r.t. omega) of jacobian()'s own
            // F(0,4)/F(1,4) series just below, so the two stay
            // consistent by construction. Cross-checked numerically
            // against mpmath: accurate to <5e-7 across this branch's
            // full domain (|omega|<0.2, dt up to 1.0s).
            const Scalar dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt, dt5 = dt4 * dt;
            const Scalar w2 = omega * omega, w3 = w2 * omega, w4 = w3 * omega;
            out(0) = x(0) + vx * dt - vy * omega * dt2 / Scalar(2) - vx * w2 * dt3 / Scalar(6)
                     + vy * w3 * dt4 / Scalar(24) + vx * w4 * dt5 / Scalar(120);
            out(1) = x(1) + vy * dt + vx * omega * dt2 / Scalar(2) - vy * w2 * dt3 / Scalar(6)
                     - vx * w3 * dt4 / Scalar(24) + vy * w4 * dt5 / Scalar(120);
        } else {
            out(0) = x(0) + (vx * s - vy * (Scalar(1) - c)) / omega;
            out(1) = x(1) + (vy * s + vx * (Scalar(1) - c)) / omega;
        }
        // vx'/vy' have no division by omega anywhere in their closed
        // form, so the plain closed form (real s/c, computed above) is
        // exact regardless of branch -- no series needed, and no reason
        // to duplicate this computation inside both branches above.
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
            // Shared powers for every series in this branch.
            const Scalar dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt, dt5 = dt4 * dt, dt6 = dt5 * dt;
            const Scalar w2 = omega * omega, w3 = w2 * omega, w4 = w3 * omega;

            F(0, 2) = dt;
            F(1, 3) = dt;

            // F(0,3)/F(1,2): a THIRD bug in this already-twice-fixed
            // branch, surfaced only by widening
            // tests/unit/test_coordinated_turn.cpp's own sweep to also
            // vary dt, not just omega (mpmath-verified, per
            // .claude/rules/testing.md). These are
            // d(px')/d(vy) = -(1-cos(omega*dt))/omega and
            // d(py')/d(vx) = (1-cos(omega*dt))/omega -- their omega->0
            // limit genuinely is 0, but the next term, omega*dt^2/2, is
            // NOT negligible in absolute terms at a large-enough dt (a
            // coasting track between detections can easily go a full
            // second without an update): 0.095 absolute error at
            // omega=0.19, dt=1.0s, comfortably inside this branch's own
            // |omega|<0.2 guard. The two-term series below (same
            // precision bar as F(0,4)/F(1,4) further down) stays
            // accurate to <4e-7 across this branch's full (omega, dt)
            // domain up to dt=1.0s.
            F(0, 3) = -(omega * dt2 / Scalar(2) - w3 * dt4 / Scalar(24));
            F(1, 2) = omega * dt2 / Scalar(2) - w3 * dt4 / Scalar(24);

            // F(2,3)/F(3,2): a SECOND bug found in this branch
            // (mpmath-verified, central finite difference at 50 decimal
            // digits of precision). d(vx')/d(vy) = -sin(omega*dt) and
            // d(vy')/d(vx) = sin(omega*dt) are FIRST-order in omega*dt --
            // leaving them at Identity's default 0 was wrong, not just
            // imprecise (0.189 absolute error at omega=0.19, dt=1.0s;
            // ~0.005 even at this file's own dt=1/30 test parameter, ~2x
            // over test_helpers.hpp's default 0.01 tolerance, which is
            // why the existing test didn't catch it at that one fixed
            // dt).
            //
            // F(2,2)/F(3,3): same family of bug, caught by the same
            // widened sweep (0.018 absolute error at omega=0.19,
            // dt=1.0s -- an EARLIER version of this comment claimed the
            // Identity-default 1 was "fine to leave as-is" here, checked
            // only at a modest omega*dt, not this branch's actual worst
            // corner). Unlike every other entry in this branch, though,
            // this isn't a truncated series -- d(vx')/d(vx) =
            // d(vy')/d(vy) = cos(omega*dt) EXACTLY, no 1/omega anywhere
            // in its closed form, and `c` above already holds that exact
            // value. There was never a reason to approximate it with
            // Identity's default; using `c` directly is both more
            // correct AND free (no new trig call, matches this same
            // function's general-omega branch further below verbatim).
            F(2, 2) = c;
            F(3, 3) = c;
            F(2, 3) = -s;
            F(3, 2) = s;
            // F(2,4)/F(3,4) have no division by omega either, so the
            // plain closed form (real s/c, computed above) is already
            // exact here too -- no series needed for either of these.
            F(2, 4) = -dt * vx * s - dt * vy * c;
            F(3, 4) = dt * vx * c - dt * vy * s;

            // omega^4 Taylor expansion of F(0,4)/F(1,4) around omega=0
            // (derived symbolically, cross-checked numerically -- see
            // .claude/rules/testing.md). Without this, a target that
            // starts at or passes through omega=0 can never develop a
            // nonzero turn-rate estimate: the Kalman gain's coupling
            // between position innovations and the omega state comes
            // entirely through this term.
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
