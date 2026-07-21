// tests/unit/test_current_statistical.cpp
//
// Coverage for docs/ROADMAP.md item 1 ("Current Statistical model",
// models/current_statistical.hpp + filters/current_statistical_filter.hpp).
// The last two cases are regression tests for the specific instability
// documented in both those files' comments: an earlier design (mean
// acceleration folded into the Kalman state) had a provable unit-root
// divergence, and even the current design (mean kept external) diverges
// without bound if fed back undamped/unclamped -- both found via ad hoc
// python3 verification per .claude/rules/testing.md before any of this
// was implemented in C++.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "augur/augur.hpp"
#include "augur/filters/current_statistical_filter.hpp"
#include "augur/models/current_statistical.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("CurrentStatistical::jacobian matches finite-difference and equals Singer's own", "[models][cs]") {
    using CS = augur::models::CurrentStatistical<float, 2>;
    using Singer = augur::models::Singer<float, 2>;
    CS model{2.0f, 1.0f};
    Singer singer{2.0f, 1.0f};

    CS::State x0;
    x0 << 0.2f, -0.1f, 1.1f, -0.6f, -0.4f, 0.3f;
    const float dt = 1.0f / 30.0f;

    augur_test::check_jacobian_matches_finite_difference(model, x0, dt);

    const auto cs_jac = model.jacobian(x0, dt);
    const auto singer_jac = singer.jacobian(x0, dt);
    REQUIRE_THAT((cs_jac - singer_jac).norm(), WithinAbs(0.0f, 1e-6));
}

TEST_CASE("CurrentStatistical::transition reduces to Singer's when mean_acceleration is zero", "[models][cs]") {
    using CS = augur::models::CurrentStatistical<float, 1>;
    using Singer = augur::models::Singer<float, 1>;
    CS model{2.0f, 1.0f}; // mean_acceleration() defaults to zero
    Singer singer{2.0f, 1.0f};

    CS::State x0;
    x0 << 0.5f, 1.2f, -0.8f;
    const float dt = 1.0f / 30.0f;

    const auto cs_out = model.transition(x0, dt);
    const auto singer_out = singer.transition(x0, dt);
    REQUIRE_THAT((cs_out - singer_out).norm(), WithinAbs(0.0f, 1e-6));
}

TEST_CASE("CurrentStatistical::process_noise is symmetric positive semi-definite", "[models][cs]") {
    using CS = augur::models::CurrentStatistical<float, 2>;
    CS model{2.0f, 1.0f};
    const auto Q = model.process_noise(0.1f);
    REQUIRE_THAT((Q - Q.transpose()).norm(), WithinAbs(0.0f, 1e-6));
    Eigen::SelfAdjointEigenSolver<CS::Transition> solver(Q);
    REQUIRE(solver.eigenvalues().minCoeff() >= -1e-6f);
}

TEST_CASE("CurrentStatisticalFilter clamps mean_acceleration instead of diverging over a long sustained maneuver",
          "[models][cs][filters][regression]") {
    using CS = augur::models::CurrentStatistical<float, 1>;
    using KF = augur::filters::KalmanFilter<CS, /*MeasDim=*/1>;

    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    KF::MeasurementCovariance R = KF::MeasurementCovariance::Identity() * 0.01f;
    KF inner{CS{2.0f, 1.0f}, KF::StateVector::Zero(), KF::StateCovariance::Identity() * 2.0f, H, R};

    const float max_mean_accel = 12.0f;
    augur::filters::CurrentStatisticalFilter<KF> filter{std::move(inner), 0.05f, max_mean_accel};

    const float dt = 1.0f / 30.0f;
    const float true_accel = 3.0f;
    float truth_p = 0.0f, truth_v = 0.0f;

    for (int step = 0; step < 600; ++step) {
        const float a = (step < 10) ? 0.0f : true_accel;
        truth_v += a * dt;
        truth_p += truth_v * dt;
        filter.predict(dt);
        filter.update(KF::Measurement{truth_p});
    }

    // Regression check: before damping+clamping, this diverged without
    // bound (observed reaching >14x the true acceleration and still
    // climbing linearly after 300 steps). It must stay within the
    // configured clamp now.
    REQUIRE(std::abs(filter.model().mean_acceleration()(0)) <= max_mean_accel + 1e-3f);
    // And position tracking -- which stayed fine even during the
    // divergence found above -- should still be accurate.
    REQUIRE(std::abs(truth_p - filter.state()(0)) < 0.1f);
}

TEST_CASE("CurrentStatisticalFilter behaves like Singer for a brief jitter (not a sustained maneuver)",
          "[models][cs][filters]") {
    using CS = augur::models::CurrentStatistical<float, 1>;
    using KF = augur::filters::KalmanFilter<CS, /*MeasDim=*/1>;

    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    KF::MeasurementCovariance R = KF::MeasurementCovariance::Identity() * 0.01f;
    KF inner{CS{2.0f, 1.0f}, KF::StateVector::Zero(), KF::StateCovariance::Identity() * 2.0f, H, R};
    augur::filters::CurrentStatisticalFilter<KF> filter{std::move(inner), 0.05f, 12.0f};

    const float dt = 1.0f / 30.0f;
    float truth_p = 0.0f, truth_v = 0.0f;
    // 10 calm steps, 3 steps of a brief 3.0 accel jitter, then 40 calm steps.
    for (int step = 0; step < 53; ++step) {
        const float a = (step >= 10 && step < 13) ? 3.0f : 0.0f;
        truth_v += a * dt;
        truth_p += truth_v * dt;
        filter.predict(dt);
        filter.update(KF::Measurement{truth_p});
    }

    // A brief jitter shouldn't leave mean_acceleration() pinned near its
    // clamp -- it should have mostly relaxed back down.
    REQUIRE(std::abs(filter.model().mean_acceleration()(0)) < 1.0f);
}
