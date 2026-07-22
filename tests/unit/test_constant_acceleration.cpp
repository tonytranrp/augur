// tests/unit/test_constant_acceleration.cpp
//
// ConstantAcceleration had zero dedicated test coverage before this file
// (confirmed by grep -- its only prior appearance anywhere under tests/
// was a comment in test_predict_query.cpp naming it as an example state
// shape, never actually constructing or exercising the type). Mirrors
// test_kalman_cv.cpp's coverage shape for the same reasons: transition()
// checked against hand-computed arithmetic, process_noise() checked for
// symmetric-PSD, jacobian() checked against a finite-difference
// derivative (per docs/PRODUCTION_ROADMAP.md Phase 3 -- the same check
// that would have caught coordinated_turn.hpp's small-omega bug, applied
// here even though this model's jacobian is linear/x-independent and so
// isn't expected to ever fail this check -- coverage and regression
// protection matter even when a bug here is unlikely), and a KalmanFilter
// convergence check.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "augur/augur.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("ConstantAcceleration::transition matches hand-computed arithmetic", "[models][ca]") {
    using CA = augur::models::ConstantAcceleration<float, 2>;
    CA model{/*jerk_noise_density=*/1.0f};

    CA::State x0;
    x0 << 0.0f, 0.0f, 2.0f, -1.0f, 0.5f, -0.5f; // px,py,vx,vy,ax,ay
    const float dt = 1.0f;

    const CA::State x1 = model.transition(x0, dt);

    // px += vx*dt + 0.5*ax*dt^2 = 2*1 + 0.5*0.5*1 = 2.25
    REQUIRE_THAT(x1(0), WithinAbs(2.25f, 1e-6));
    // py += vy*dt + 0.5*ay*dt^2 = -1*1 + 0.5*(-0.5)*1 = -1.25
    REQUIRE_THAT(x1(1), WithinAbs(-1.25f, 1e-6));
    // vx += ax*dt = 2 + 0.5*1 = 2.5
    REQUIRE_THAT(x1(2), WithinAbs(2.5f, 1e-6));
    // vy += ay*dt = -1 + (-0.5)*1 = -1.5
    REQUIRE_THAT(x1(3), WithinAbs(-1.5f, 1e-6));
    // acceleration unchanged
    REQUIRE_THAT(x1(4), WithinAbs(0.5f, 1e-6));
    REQUIRE_THAT(x1(5), WithinAbs(-0.5f, 1e-6));
}

TEST_CASE("ConstantAcceleration::jacobian matches a finite-difference derivative", "[models][ca]") {
    using CA = augur::models::ConstantAcceleration<float, 2>;
    CA model{1.0f};
    CA::State x;
    x << 3.0f, -2.0f, 2.0f, -1.0f, 0.5f, -0.5f;
    augur_test::check_jacobian_matches_finite_difference(model, x, 0.5f);
}

TEST_CASE("ConstantAcceleration::process_noise is symmetric positive semi-definite", "[models][ca]") {
    using CA = augur::models::ConstantAcceleration<float, 2>;
    CA model{1.0f};
    const auto Q = model.process_noise(0.1f);

    REQUIRE_THAT((Q - Q.transpose()).norm(), WithinAbs(0.0f, 1e-6));
    Eigen::SelfAdjointEigenSolver<CA::Transition> solver(Q);
    REQUIRE(solver.eigenvalues().minCoeff() >= -1e-6f);
}

TEST_CASE("KalmanFilter converges toward a constant-acceleration target", "[filters][kalman][ca]") {
    using CA = augur::models::ConstantAcceleration<float, 2>;
    using KF = augur::filters::KalmanFilter<CA, 2>;

    KF::StateVector x0 = KF::StateVector::Zero();
    KF::StateCovariance P0 = KF::StateCovariance::Identity() * 10.0f;
    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    KF::MeasurementCovariance R = KF::MeasurementCovariance::Identity() * 0.1f;

    KF filter{CA{1.0f}, x0, P0, H, R};

    // Ground truth: starts at origin, constant acceleration (0.4, -0.2)
    // units/s^2 from an initial velocity of (1, 0.5) units/s. Feed
    // noise-free measurements and require convergence to true position,
    // velocity, AND acceleration -- the harder bar CV's own convergence
    // test doesn't need to clear, since CV has no acceleration state.
    const float dt = 1.0f / 60.0f;
    const int steps = 300;
    float true_px = 0.0f, true_py = 0.0f;
    float true_vx = 1.0f, true_vy = 0.5f;
    const float true_ax = 0.4f, true_ay = -0.2f;

    for (int i = 0; i < steps; ++i) {
        true_px += true_vx * dt + 0.5f * true_ax * dt * dt;
        true_py += true_vy * dt + 0.5f * true_ay * dt * dt;
        true_vx += true_ax * dt;
        true_vy += true_ay * dt;
        filter.predict(dt);
        filter.update(KF::Measurement{true_px, true_py});
    }

    const auto& x = filter.state();
    REQUIRE_THAT(x(0), WithinAbs(true_px, 0.1));
    REQUIRE_THAT(x(1), WithinAbs(true_py, 0.1));
    REQUIRE_THAT(x(2), WithinAbs(true_vx, 0.1));
    REQUIRE_THAT(x(3), WithinAbs(true_vy, 0.1));
    REQUIRE_THAT(x(4), WithinAbs(true_ax, 0.1));
    REQUIRE_THAT(x(5), WithinAbs(true_ay, 0.1));
}
