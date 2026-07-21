// tests/unit/test_coordinated_turn.cpp
//
// CoordinatedTurn coverage beyond what test_kalman_cv.cpp exercises for
// the CV case -- docs/ROADMAP.md's "Benchmark/validation harness" item
// flagged this as the biggest actual risk in the codebase: before this
// file, CoordinatedTurn had zero automated coverage. The last test in
// this file is also the regression test for the small-omega jacobian
// bug documented in docs/ROADMAP.md's "Fixes found along the way"
// section -- before that fix, a target starting at omega=0 could never
// develop a nonzero turn-rate estimate, and that test's final assertion
// would fail.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "augur/augur.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("CoordinatedTurn::transition degenerates to constant velocity at omega=0", "[models][ct]") {
    using CT = augur::models::CoordinatedTurn<float>;
    CT model{1.0f, 0.1f};
    CT::State x0;
    x0 << 0.0f, 0.0f, 2.0f, -1.0f, 0.0f;

    const auto x1 = model.transition(x0, 0.5f);

    REQUIRE_THAT(x1(0), WithinAbs(1.0f, 1e-6));   // px += vx*dt = 2*0.5
    REQUIRE_THAT(x1(1), WithinAbs(-0.5f, 1e-6));  // py += vy*dt = -1*0.5
    REQUIRE_THAT(x1(2), WithinAbs(2.0f, 1e-6));   // velocity unchanged
    REQUIRE_THAT(x1(3), WithinAbs(-1.0f, 1e-6));
    REQUIRE_THAT(x1(4), WithinAbs(0.0f, 1e-6));
}

TEST_CASE("CoordinatedTurn::transition matches closed-form circular motion", "[models][ct]") {
    // Unit speed in +x, omega=pi rad/s: turns exactly half a circle in
    // 1 second, ending up moving in -x, displaced by the turn circle's
    // diameter (2/omega = 2/pi) in +y.
    using CT = augur::models::CoordinatedTurn<float>;
    CT model{1.0f, 0.1f};
    CT::State x0;
    x0 << 0.0f, 0.0f, 1.0f, 0.0f, 3.14159265f;

    const auto x1 = model.transition(x0, 1.0f);

    REQUIRE_THAT(x1(0), WithinAbs(0.0f, 1e-3f));
    REQUIRE_THAT(x1(1), WithinAbs(2.0f / 3.14159265f, 1e-3f));
    REQUIRE_THAT(x1(2), WithinAbs(-1.0f, 1e-3f));
    REQUIRE_THAT(x1(3), WithinAbs(0.0f, 1e-3f));
}

TEST_CASE("CoordinatedTurn::jacobian matches finite-difference at the exact state that exposed the original bug",
          "[models][ct][regression]") {
    // omega=0 with BOTH vx and vy nonzero: the missing terms were
    // -vy*dt^2/2 (row 0) and +vx*dt^2/2 (row 1), each zero unless both
    // velocity components are nonzero -- which is exactly why the
    // original bug didn't show up as a crash or an obviously-wrong
    // number, just a silently-unobservable turn-rate state.
    using CT = augur::models::CoordinatedTurn<float>;
    CT model{1.0f, 0.1f};
    CT::State x0;
    x0 << 0.0f, 0.0f, 1.0f, 0.5f, 0.0f;

    augur_test::check_jacobian_matches_finite_difference(model, x0, 1.0f / 30.0f);
}

TEST_CASE("CoordinatedTurn::jacobian matches finite-difference across a range of turn rates", "[models][ct]") {
    using CT = augur::models::CoordinatedTurn<float>;
    CT model{1.0f, 0.1f};
    const float dt = 1.0f / 30.0f;

    for (float omega : {-3.0f, -0.5f, -0.15f, -0.01f, 0.0f, 0.01f, 0.15f, 0.5f, 3.0f}) {
        CT::State x0;
        x0 << 0.5f, -0.3f, 1.2f, -0.7f, omega;
        augur_test::check_jacobian_matches_finite_difference(model, x0, dt);
    }
}

TEST_CASE("CoordinatedTurn::process_noise is symmetric positive semi-definite", "[models][ct]") {
    using CT = augur::models::CoordinatedTurn<float>;
    CT model{1.0f, 0.1f};
    const auto Q = model.process_noise(0.1f);

    REQUIRE_THAT((Q - Q.transpose()).norm(), WithinAbs(0.0f, 1e-6));
    Eigen::SelfAdjointEigenSolver<CT::Transition> solver(Q);
    REQUIRE(solver.eigenvalues().minCoeff() >= -1e-6f);
}

TEST_CASE("KalmanFilter<CoordinatedTurn> learns a nonzero turn rate from a real sustained turn starting at omega=0",
          "[models][ct][filters][kalman][regression]") {
    // Regression test for the fixed jacobian bug: before the fix, omega
    // could never move away from its initial value of 0, no matter how
    // long or hard the target actually turned. Reference behavior
    // (float64, ad hoc python3 per .claude/rules/testing.md): the
    // omega estimate reaches ~4.26 by the last step against a true
    // 4.0 rad/s turn -- the >2.5 bound below has generous margin for
    // float32 vs. float64 while still failing hard if the bug regresses
    // (estimate would stay exactly 0).
    using CT = augur::models::CoordinatedTurn<float>;
    using KF = augur::filters::KalmanFilter<CT, /*MeasDim=*/2>;

    KF::StateVector x0 = KF::StateVector::Zero();
    x0(2) = 1.5f; // initial vx
    KF::StateCovariance P0 = KF::StateCovariance::Identity() * 2.0f;
    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    KF::MeasurementCovariance R = KF::MeasurementCovariance::Identity() * 0.0004f;

    KF filter{CT{0.05f, 8.0f}, x0, P0, H, R};

    CT truth_model{};
    CT::State truth = x0;
    const float dt = 1.0f / 30.0f;
    const float true_omega = 4.0f;

    for (int step = 0; step < 20; ++step) {
        truth(4) = (step < 4) ? 0.0f : true_omega;
        truth = truth_model.transition(truth, dt);
        filter.predict(dt);
        filter.update(KF::Measurement{truth(0), truth(1)});
    }

    REQUIRE(filter.state()(4) > 2.5f);
}
