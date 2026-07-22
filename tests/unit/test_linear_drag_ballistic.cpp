// tests/unit/test_linear_drag_ballistic.cpp
//
// Coverage for docs/IMPROVEMENT_PLAN.md's new-model finding
// (models/linear_drag_ballistic.hpp). Reference values were computed
// independently (ad hoc python3 + sympy + mpmath, per
// .claude/rules/testing.md) before this was written in C++: sympy
// derived the closed form and caught a real hand-derivation error (a
// genuine antiderivative slip) by disagreeing with independent RK4
// numerical integration; mpmath then verified the small-drag Taylor
// series stays accurate across a realistic game-projectile domain.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "augur/augur.hpp"
#include "augur/models/linear_drag_ballistic.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;
using LDB = augur::models::LinearDragBallistic<float, 2>;

static_assert(augur::models::MotionModel<LDB>);

TEST_CASE("LinearDragBallistic::transition matches an independent reference (exact branch, k=0.5)",
          "[models][ballistic]") {
    LDB::GravityVector g{0.0f, -9.81f};
    LDB model{g, 0.5f, 1.0f};
    LDB::State x0;
    x0 << 0.0f, 0.0f, 0.0f, 10.0f;

    const auto x1 = model.transition(x0, 1.0f);

    // Reference: independently computed (python3+mpmath) closed-form
    // solution of dv/dt=g-k*v, cross-checked against RK4 integration to
    // 1e-15 before being used here.
    REQUIRE_THAT(x1(1), WithinAbs(3.689124f, 1e-3f));
    REQUIRE_THAT(x1(3), WithinAbs(-1.654562f, 1e-3f));
    REQUIRE_THAT(x1(0), WithinAbs(0.0f, 1e-6f)); // no gravity/velocity on x-axis
    REQUIRE_THAT(x1(2), WithinAbs(0.0f, 1e-6f));
}

TEST_CASE("LinearDragBallistic::transition matches an independent reference (small-drag branch, k=0.05)",
          "[models][ballistic]") {
    LDB::GravityVector g{0.0f, -9.81f};
    LDB model{g, 0.05f, 1.0f};
    LDB::State x0;
    x0 << 0.0f, 0.0f, 0.0f, 50.0f;

    const auto x1 = model.transition(x0, 2.0f);

    REQUIRE_THAT(x1(1), WithinAbs(76.180554f, 0.01f));
    REQUIRE_THAT(x1(3), WithinAbs(26.570972f, 0.01f));
}

TEST_CASE("LinearDragBallistic reduces exactly to pure-gravity ballistic motion at drag=0",
          "[models][ballistic]") {
    LDB::GravityVector g{0.0f, -9.81f};
    LDB model{g, 0.0f, 1.0f};
    LDB::State x0;
    x0 << 0.0f, 0.0f, 5.0f, 10.0f;

    const auto x1 = model.transition(x0, 1.0f);

    const float expected_py = 10.0f * 1.0f + (-9.81f) * 1.0f * 1.0f / 2.0f;
    const float expected_vy = 10.0f + (-9.81f) * 1.0f;
    REQUIRE_THAT(x1(1), WithinAbs(expected_py, 1e-4f));
    REQUIRE_THAT(x1(3), WithinAbs(expected_vy, 1e-4f));
    REQUIRE_THAT(x1(0), WithinAbs(5.0f, 1e-6f)); // x-axis: no gravity, no drag -> pure constant velocity
}

TEST_CASE("LinearDragBallistic::jacobian matches finite-difference across a wide drag/dt sweep",
          "[models][ballistic]") {
    // Sweeps across the kEpsilon=0.1 branch boundary specifically (both
    // sides) and drag=0 exactly, plus dt up to 2s -- the exact domain
    // this file's own comment claims the small-drag series is verified
    // over.
    for (float k : {0.0f, 0.01f, 0.05f, 0.09f, 0.15f, 0.5f, 2.0f, 5.0f}) {
        for (float dt : {1.0f / 30.0f, 0.1f, 1.0f, 2.0f}) {
            LDB::GravityVector g{0.0f, -9.81f};
            LDB model{g, k, 1.0f};
            LDB::State x0;
            x0 << 1.0f, 2.0f, 3.0f, -4.0f;
            augur_test::check_jacobian_matches_finite_difference(model, x0, dt, 0.01f);
        }
    }
}

TEST_CASE("LinearDragBallistic::process_noise is symmetric positive semi-definite across a drag sweep",
          "[models][ballistic]") {
    for (float k : {0.0f, 0.01f, 0.09f, 0.5f, 3.0f}) {
        LDB::GravityVector g{0.0f, -9.81f};
        LDB model{g, k, 2.0f};
        const auto Q = model.process_noise(0.1f);

        REQUIRE_THAT((Q - Q.transpose()).norm(), WithinAbs(0.0f, 1e-5f));
        Eigen::SelfAdjointEigenSolver<LDB::Transition> solver(Q);
        REQUIRE(solver.eigenvalues().minCoeff() >= -1e-4f);
    }
}

TEST_CASE("LinearDragBallistic::process_noise's drag->0 limit exactly reproduces ConstantVelocity's own CWNA",
          "[models][ballistic]") {
    // Independent internal cross-check (symbolically verified via sympy
    // before this was written): zero drag must reduce to plain
    // constant-velocity-under-noise, since that's exactly what the
    // governing ODE degenerates to.
    using CV = augur::models::ConstantVelocity<float, 2>;
    LDB::GravityVector g{0.0f, 0.0f};
    LDB ldb{g, 0.0f, 3.0f};
    CV cv{3.0f};

    const auto Q_ldb = ldb.process_noise(0.5f);
    const auto Q_cv = cv.process_noise(0.5f);

    REQUIRE_THAT(Q_ldb(0, 0), WithinAbs(Q_cv(0, 0), 1e-4f));
    REQUIRE_THAT(Q_ldb(0, 2), WithinAbs(Q_cv(0, 2), 1e-4f));
    REQUIRE_THAT(Q_ldb(2, 2), WithinAbs(Q_cv(2, 2), 1e-4f));
}

TEST_CASE("LinearDragBallistic runs through KalmanFilter and mixes into imm::Estimator", "[models][ballistic][filters][imm]") {
    using KF = augur::filters::KalmanFilter<LDB, /*MeasDim=*/2>;
    LDB::GravityVector g{0.0f, -9.81f};
    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    KF::MeasurementCovariance R = KF::MeasurementCovariance::Identity() * 0.1f;

    KF filter{LDB{g, 0.3f, 1.0f}, KF::StateVector::Zero(), KF::StateCovariance::Identity(), H, R};
    filter.predict(0.1f);
    filter.update(KF::Measurement{1.0f, 0.5f});
    REQUIRE(filter.state().allFinite());

    // Two differently-tuned instances mixed via imm::Estimator -- the
    // same "several instances of one model" pattern imm::Estimator's
    // own file comment describes for same-order mixing.
    using Tracker = augur::imm::Estimator<KF, KF>;
    Tracker tracker{
        KF{LDB{g, 0.1f, 1.0f}, KF::StateVector::Zero(), KF::StateCovariance::Identity(), H, R},
        KF{LDB{g, 1.0f, 1.0f}, KF::StateVector::Zero(), KF::StateCovariance::Identity(), H, R},
        augur::imm::ModeMatrix<2, float>::uniform(0.9f),
    };
    tracker.predict(0.1f);
    tracker.update(KF::Measurement{0.5f, 0.3f});

    const auto [x, P] = tracker.combined_state();
    REQUIRE(x.allFinite());
    REQUIRE(P.allFinite());
}
