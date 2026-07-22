// tests/unit/test_kalman_cv.cpp
//
// Two things worth actually testing before trusting the "solid" tier:
//   1. ConstantVelocity::transition() matches hand-computed arithmetic
//      for a trivial case (catches sign/index errors in the per-axis
//      loop -- easy to get wrong when SpatialDim is templated).
//   2. KalmanFilter converges toward a target moving at genuinely
//      constant velocity when fed noisy but unbiased measurements --
//      catches the "predict/update do the wrong linear algebra
//      entirely" class of bug, which unit-testing individual matrix
//      entries wouldn't necessarily catch.
//
// Deliberately NOT covering CoordinatedTurn/Singer/IMM yet -- those
// need reference trajectories to check against, which is its own
// piece of work (see docs/ROADMAP.md, "Benchmark/validation harness").
// This file is a starting point, not full coverage.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "augur/augur.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("ConstantVelocity::transition matches hand-computed arithmetic", "[models][cv]") {
    using CV = augur::models::ConstantVelocity<float, 2>;
    CV model{/*noise_spectral_density=*/1.0f};

    CV::State x0;
    x0 << 0.0f, 0.0f, 2.0f, -1.0f; // px=0, py=0, vx=2, vy=-1
    const float dt = 0.5f;

    const CV::State x1 = model.transition(x0, dt);

    REQUIRE_THAT(x1(0), WithinAbs(1.0f, 1e-6));  // px += vx * dt = 2 * 0.5
    REQUIRE_THAT(x1(1), WithinAbs(-0.5f, 1e-6)); // py += vy * dt = -1 * 0.5
    REQUIRE_THAT(x1(2), WithinAbs(2.0f, 1e-6));  // velocity unchanged
    REQUIRE_THAT(x1(3), WithinAbs(-1.0f, 1e-6));
}

TEST_CASE("ConstantVelocity::jacobian matches a finite-difference derivative", "[models][cv]") {
    using CV = augur::models::ConstantVelocity<float, 2>;
    CV model{1.0f};
    CV::State x;
    x << 3.0f, -2.0f, 2.0f, -1.0f;
    augur_test::check_jacobian_matches_finite_difference(model, x, 0.5f);
}

TEST_CASE("ConstantVelocity::process_noise is symmetric positive semi-definite", "[models][cv]") {
    using CV = augur::models::ConstantVelocity<float, 2>;
    CV model{1.0f};
    const auto Q = model.process_noise(0.1f);

    REQUIRE_THAT((Q - Q.transpose()).norm(), WithinAbs(0.0f, 1e-6));
    Eigen::SelfAdjointEigenSolver<CV::Transition> solver(Q);
    REQUIRE(solver.eigenvalues().minCoeff() >= -1e-6f);
}

TEST_CASE("KalmanFilter converges toward a constant-velocity target", "[filters][kalman]") {
    using CV = augur::models::ConstantVelocity<float, 2>;
    using KF = augur::filters::KalmanFilter<CV, 2>;

    KF::StateVector x0 = KF::StateVector::Zero();
    KF::StateCovariance P0 = KF::StateCovariance::Identity() * 10.0f;
    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    KF::MeasurementCovariance R = KF::MeasurementCovariance::Identity() * 0.1f;

    KF filter{CV{1.0f}, x0, P0, H, R};

    // Ground truth: starts at origin, moves at (1, 0.5) units/s. Feed
    // noise-free measurements (the simplest possible convergence
    // check) and require the filter to end up close to ground truth.
    const float dt = 1.0f / 60.0f;
    const int steps = 120;
    float true_px = 0.0f, true_py = 0.0f;
    const float true_vx = 1.0f, true_vy = 0.5f;

    for (int i = 0; i < steps; ++i) {
        true_px += true_vx * dt;
        true_py += true_vy * dt;
        filter.predict(dt);
        filter.update(KF::Measurement{true_px, true_py});
    }

    const auto& x = filter.state();
    REQUIRE_THAT(x(0), WithinAbs(true_px, 0.05));
    REQUIRE_THAT(x(1), WithinAbs(true_py, 0.05));
    REQUIRE_THAT(x(2), WithinAbs(true_vx, 0.05));
    REQUIRE_THAT(x(3), WithinAbs(true_vy, 0.05));
}
