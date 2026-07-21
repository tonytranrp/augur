#pragma once
// augur/filters/current_statistical_filter.hpp
//
// augur::filters::CurrentStatisticalFilter<Inner> -- the adaptive-mean
// half of docs/ROADMAP.md item 1 ("Current Statistical model"). Composes
// with an existing Filter (e.g. KalmanFilter<CurrentStatistical<...>,...>)
// rather than duplicating its predict/update math, the same pattern
// filters/adaptive/sage_husa.hpp's own stub describes: after every
// update(), it reads the inner filter's own posterior acceleration
// estimate and feeds it back into the model's mean_acceleration() for
// the next predict() cycle.
//
// See models/current_statistical.hpp's file comment for why this
// feedback is deliberately DAMPED (a small adaptation_rate, exponential
// smoothing rather than a direct copy) and CLAMPED (to +/- max_mean_accel):
// ad hoc python3 verification (per .claude/rules/testing.md) found that
// feeding the raw, undamped, unclamped posterior estimate back every
// cycle diverges without bound over a sustained maneuver, even though
// position/velocity tracking stays fine throughout -- a latent bug that
// would only surface if anything downstream actually consumed the
// acceleration or mean_acceleration() state. Damping+clamping bounds
// the growth (verified: it settles at the clamp rather than escaping
// it) but does not eliminate the underlying bias toward that bound --
// tune adaptation_rate/max_mean_accel to the actual expected maneuver
// profile, don't treat the defaults as universal.

#include <algorithm>
#include <cmath>
#include "augur/filters/filter_concept.hpp"

namespace augur::filters {

template <filters::Filter Inner>
class CurrentStatisticalFilter {
public:
    using Scalar = typename Inner::Scalar;
    using Model = typename Inner::Model;
    static constexpr std::size_t dimension = Inner::dimension;

    explicit CurrentStatisticalFilter(Inner inner,
                                       Scalar adaptation_rate = Scalar(0.05),
                                       Scalar max_mean_accel = Scalar(12))
        : inner_(std::move(inner)), adaptation_rate_(adaptation_rate), max_mean_accel_(max_mean_accel) {}

    void predict(Scalar dt) { inner_.predict(dt); }

    template <typename Measurement>
    void update(const Measurement& z) {
        inner_.update(z);

        constexpr int spatial_dim = Model::spatial_dim;
        typename Model::AccelVector posterior_accel;
        for (int axis = 0; axis < spatial_dim; ++axis) {
            posterior_accel(axis) = inner_.state()(2 * spatial_dim + axis);
        }

        auto mean_accel = inner_.model().mean_acceleration();
        mean_accel = (Scalar(1) - adaptation_rate_) * mean_accel + adaptation_rate_ * posterior_accel;
        mean_accel = mean_accel.cwiseMax(-max_mean_accel_).cwiseMin(max_mean_accel_);
        inner_.model().set_mean_acceleration(mean_accel);
    }

    [[nodiscard]] const auto& state() const { return inner_.state(); }
    [[nodiscard]] const auto& covariance() const { return inner_.covariance(); }
    [[nodiscard]] Scalar last_likelihood() const { return inner_.last_likelihood(); }
    [[nodiscard]] const Model& model() const { return inner_.model(); }

    void set_state(const typename Inner::StateVector& x, const typename Inner::StateCovariance& P) {
        inner_.set_state(x, P);
    }

private:
    Inner inner_;
    Scalar adaptation_rate_;
    Scalar max_mean_accel_;
};

} // namespace augur::filters
