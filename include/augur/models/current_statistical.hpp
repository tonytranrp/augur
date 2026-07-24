#pragma once
// augur/models/current_statistical.hpp
//
// ROADMAP MODEL -- docs/ROADMAP.md item 1, "Current Statistical (CS)
// model": an adaptive-mean refinement of models/singer.hpp. Singer
// models acceleration as a zero-mean Ornstein-Uhlenbeck process, always
// decaying back toward "no maneuver." CS replaces the fixed zero mean
// with a mean_acceleration() that adapts toward whatever the filter's
// own posterior acceleration estimate has recently been -- letting it
// represent a genuinely SUSTAINED maneuver instead of always dragging
// the estimate back toward straight-line motion. See models/singer.hpp
// for the base derivation and citations this extends.
//
// DESIGN, stated plainly (this took real trial and error -- see
// .claude/rules/testing.md's workflow, followed exactly here):
//
// mean_acceleration() is a MUTABLE MODEL PARAMETER (like Singer's own
// tau/sigma), not a Kalman state component. The first design tried
// here folded it into the state vector instead (state = [p,v,a,a_bar]
// per axis, with a_bar_{k+1} = a_k as an extra equation) -- that's
// REJECTED: ad hoc python3 verification (per .claude/rules/testing.md)
// showed the coupled [a, a_bar] subsystem always has an eigenvalue of
// exactly 1 (a provable consequence of both its rows summing to 1,
// regardless of tuning), which combined with Kalman covariance
// propagation caused the acceleration estimate to grow WITHOUT BOUND
// (observed reaching 10-20x the true value well within tens of steps).
//
// Keeping mean_acceleration() external instead makes jacobian() and
// process_noise() IDENTICAL to Singer's own (a constant offset in
// transition() doesn't change the Jacobian at all), which is clean --
// but the feedback loop of "the filter's own posterior updates the
// mean it's pulled toward" is still, unavoidably, a self-referential
// adaptive estimator, and a second round of ad hoc python3 testing
// found it ALSO drifts without bound if mean_acceleration() is updated
// from the raw (undamped, unclamped) posterior acceleration every
// cycle -- same failure family the vanilla Sage-Husa formulation is
// already known for (see filters/sage_husa.hpp's own citation
// of this). The mitigation used here, verified to actually bound the
// growth (see filters/current_statistical_filter.hpp and
// tests/unit/test_current_statistical.cpp): update
// mean_acceleration() via a slow exponential smoothing of the posterior
// estimate (a small adaptation_rate, not a direct copy) AND clamp its
// magnitude to a caller-supplied bound. This stops runaway divergence,
// but does NOT eliminate the underlying bias -- verified behavior is
// that the mean estimate drifts toward its clamp rather than settling
// tightly on the true sustained acceleration. Treat this model as a
// genuine "flagged sketch," same tier as Singer itself: it satisfies
// MotionModel and its core dynamics are verified correct, but the
// adaptive-mean feedback needs real tuning (or a more rigorous fix --
// e.g. accounting for the mean's own uncertainty when feeding it back,
// as later "Current" Statistical literature discusses) before trusting
// it in anything real.

#include <algorithm>
#include <cmath>
#include "augur/math/backend.hpp"
#include "augur/models/singer.hpp"

namespace augur::models {

template <typename ScalarT, int SpatialDim>
class CurrentStatistical {
public:
    using Scalar = ScalarT;
    static constexpr std::size_t dimension = 3 * SpatialDim; // pos, vel, accel per axis -- same shape as Singer
    static constexpr int spatial_dim = SpatialDim; // for filters/current_statistical_filter.hpp's use

    using State = augur::math::Vector<Scalar, dimension>;
    using Transition = augur::math::Matrix<Scalar, dimension>;
    using AccelVector = augur::math::Vector<Scalar, SpatialDim>;

    // maneuver_time_constant/maneuver_std_dev: same meaning as
    // models::Singer (this reuses Singer's own transition/jacobian/
    // process_noise formulas verbatim -- see the file comment above for
    // why the ONLY difference is the mean acceleration transition()
    // pulls toward).
    explicit CurrentStatistical(Scalar maneuver_time_constant = Scalar(2),
                                Scalar maneuver_std_dev = Scalar(1))
        : singer_(maneuver_time_constant, maneuver_std_dev),
          alpha_(Scalar(1) / maneuver_time_constant),
          mean_acceleration_(AccelVector::Zero()) {}

    [[nodiscard]] const AccelVector& mean_acceleration() const { return mean_acceleration_; }
    void set_mean_acceleration(const AccelVector& mean_accel) { mean_acceleration_ = mean_accel; }

    [[nodiscard]] State transition(const State& x, Scalar dt) const {
        State out = x;
        const Scalar e = std::exp(-alpha_ * dt);
        const Scalar f_pa = (alpha_ * dt - Scalar(1) + e) / (alpha_ * alpha_); // identical to Singer's own
        const Scalar f_va = (Scalar(1) - e) / alpha_;
        const Scalar g_p = dt * dt / Scalar(2) - f_pa; // mean-acceleration correction terms (see file comment)
        const Scalar g_v = dt - f_va;
        for (int axis = 0; axis < SpatialDim; ++axis) {
            const int p = axis, v = SpatialDim + axis, a = 2 * SpatialDim + axis;
            const Scalar mean_a = mean_acceleration_(axis);
            const Scalar deviation = x(a) - mean_a; // "Singer-shaped" zero-mean part
            out(p) = x(p) + x(v) * dt + deviation * f_pa + mean_a * g_p;
            out(v) = x(v) + deviation * f_va + mean_a * g_v;
            out(a) = mean_a + deviation * e;
        }
        return out;
    }

    // Identical to Singer::jacobian(): mean_acceleration() is a
    // constant w.r.t. x (a model parameter, not a state component), so
    // it only shifts transition() by a state-independent additive
    // offset -- verified via finite-difference (ad hoc python3) that
    // this doesn't change the Jacobian relative to Singer at all.
    [[nodiscard]] Transition jacobian(const State& x, Scalar dt) const {
        return singer_.jacobian(x, dt);
    }

    [[nodiscard]] Transition process_noise(Scalar dt) const {
        return singer_.process_noise(dt);
    }

private:
    Singer<Scalar, SpatialDim> singer_; // reused for jacobian()/process_noise(), which are unchanged from Singer
    Scalar alpha_;                      // 1 / maneuver time constant, duplicated from singer_ for transition()'s own use
    AccelVector mean_acceleration_;
};

} // namespace augur::models
