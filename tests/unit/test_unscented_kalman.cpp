// tests/unit/test_unscented_kalman.cpp
//
// Coverage for docs/ROADMAP.md item 3 ("UKF backend",
// filters/unscented_kalman.hpp). The strongest available correctness
// check for a UKF: for a LINEAR transition and measurement function,
// the unscented transform is exact, so it must reproduce
// filters::KalmanFilter's output -- verified independently (ad hoc
// python3 + numpy, per .claude/rules/testing.md) to agree to ~1e-10
// before this was written in C++.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "augur/augur.hpp"
#include "augur/filters/unscented_kalman.hpp"

using Catch::Matchers::WithinAbs;

static_assert(augur::filters::Filter<
              augur::filters::UnscentedKalmanFilter<augur::models::ConstantVelocity<float, 2>, 2>>);

TEST_CASE("UnscentedKalmanFilter matches KalmanFilter for a linear model and measurement", "[filters][ukf]") {
    using CV = augur::models::ConstantVelocity<float, 2>;
    using KF = augur::filters::KalmanFilter<CV, 2>;
    using UKF = augur::filters::UnscentedKalmanFilter<CV, 2>;

    KF::StateVector x0;
    x0 << 1.0f, 2.0f, 0.5f, -0.3f;
    KF::StateCovariance P0 = KF::StateCovariance::Identity() * 2.0f;
    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    KF::MeasurementCovariance R = KF::MeasurementCovariance::Identity() * 0.1f;

    KF kf{CV{0.01f}, x0, P0, H, R};
    UKF ukf{CV{0.01f}, x0, P0,
            [&](const UKF::StateVector& x) -> UKF::Measurement { return H * x; },
            R};

    const float dt = 1.0f / 30.0f;
    kf.predict(dt);
    ukf.predict(dt);
    REQUIRE_THAT((kf.state() - ukf.state()).norm(), WithinAbs(0.0f, 1e-3f));
    REQUIRE_THAT((kf.covariance() - ukf.covariance()).norm(), WithinAbs(0.0f, 1e-3f));

    const KF::Measurement z{1.05f, 1.98f};
    kf.update(z);
    ukf.update(z);
    REQUIRE_THAT((kf.state() - ukf.state()).norm(), WithinAbs(0.0f, 1e-3f));
    REQUIRE_THAT((kf.covariance() - ukf.covariance()).norm(), WithinAbs(0.0f, 1e-3f));
    REQUIRE_THAT(kf.last_likelihood(), WithinAbs(ukf.last_likelihood(), 1e-4f));
}

TEST_CASE("UnscentedKalmanFilter converges tracking a target through a genuinely nonlinear range+bearing measurement",
          "[filters][ukf][regression]") {
    // Range-only was tried first here and dropped: independently
    // verified (ad hoc python3, per .claude/rules/testing.md) that a
    // single stationary range-only sensor is only weakly observable
    // against a non-radially-moving target (the estimate visibly lags
    // and doesn't converge within any reasonable step count -- not a
    // bug, just a genuinely hard estimation problem), so it made a poor
    // regression test. Range+bearing is still genuinely nonlinear
    // (sqrt(x^2+y^2), atan2(y,x)) but fully determines 2D position from
    // one noise-free reading, and converges quickly -- verified in the
    // same ad hoc check to reach ~5e-4 position error by step 60 before
    // this was ported to C++.
    using CV = augur::models::ConstantVelocity<float, 2>;
    using UKF = augur::filters::UnscentedKalmanFilter<CV, /*MeasDim=*/2>;

    UKF::StateVector x0;
    x0 << 8.0f, 8.0f, 0.0f, 0.0f; // a poor initial guess
    UKF::StateCovariance P0 = UKF::StateCovariance::Identity() * 4.0f;
    UKF::MeasurementCovariance R = (UKF::MeasurementCovariance() << 0.01f, 0.0f, 0.0f, 0.001f).finished();

    auto range_bearing_fn = [](const UKF::StateVector& x) -> UKF::Measurement {
        return UKF::Measurement{std::sqrt(x(0) * x(0) + x(1) * x(1)), std::atan2(x(1), x(0))};
    };
    UKF filter{CV{0.05f}, x0, P0, range_bearing_fn, R};

    const float dt = 1.0f / 30.0f;
    float truth_x = 3.0f, truth_y = 4.0f, vx = 1.0f, vy = 0.5f;
    for (int step = 0; step < 60; ++step) {
        truth_x += vx * dt;
        truth_y += vy * dt;
        filter.predict(dt);
        const float true_range = std::sqrt(truth_x * truth_x + truth_y * truth_y);
        const float true_bearing = std::atan2(truth_y, truth_x);
        filter.update(UKF::Measurement{true_range, true_bearing});
    }

    const float pos_error = std::hypot(filter.state()(0) - truth_x, filter.state()(1) - truth_y);
    REQUIRE(pos_error < 0.01f);
}
