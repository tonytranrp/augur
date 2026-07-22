// tests/unit/test_coordinated_turn_3d.cpp
//
// Coverage for docs/IMPROVEMENT_PLAN.md's "quasi-3D coordinated turn"
// finding (models/coordinated_turn_3d.hpp). The model is a composition of
// two already-independently-tested models (CoordinatedTurn's xy-block,
// ConstantVelocity<.,1>'s vertical channel), so the highest-value checks
// here are less "is the underlying math right" (already covered by
// test_coordinated_turn.cpp / test_kalman_cv.cpp) and more "is the WIRING
// between the two blocks right": exact block-for-block equality against
// standalone instances of the two sub-models, plus an explicit check that
// the jacobian's off-block-diagonal entries are genuinely zero, not just
// small. Reference values for the block-equality checks are the sub-models'
// own already-verified outputs, not independently recomputed.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "augur/augur.hpp"
#include "augur/models/coordinated_turn_3d.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;
using CT = augur::models::CoordinatedTurn<float>;
using CT3D = augur::models::CoordinatedTurn3D<float>;
using CV1 = augur::models::ConstantVelocity<float, 1>;

static_assert(augur::models::MotionModel<CT3D>);
static_assert(CT3D::dimension == 7);

namespace {
CT3D::State make_state(float px, float py, float vx, float vy, float omega, float pz, float vz) {
    CT3D::State x;
    x << px, py, vx, vy, omega, pz, vz;
    return x;
}
} // namespace

TEST_CASE("CoordinatedTurn3D::transition's xy-block exactly matches a standalone CoordinatedTurn", "[models][ct3d]") {
    CT3D model{1.0f, 0.1f, 0.5f};
    CT xy_only{1.0f, 0.1f};
    const auto x0 = make_state(1.0f, -2.0f, 3.0f, 0.5f, 0.8f, 100.0f, -5.0f);

    const auto x1 = model.transition(x0, 0.2f);
    const auto xy1 = xy_only.transition(x0.head<5>(), 0.2f);

    for (int i = 0; i < 5; ++i) {
        REQUIRE_THAT(x1(i), WithinAbs(xy1(i), 1e-6f));
    }
}

TEST_CASE("CoordinatedTurn3D::transition's vertical channel exactly matches a standalone ConstantVelocity<.,1>",
          "[models][ct3d]") {
    CT3D model{1.0f, 0.1f, 0.5f};
    CV1 z_only{0.5f};
    const auto x0 = make_state(1.0f, -2.0f, 3.0f, 0.5f, 0.8f, 100.0f, -5.0f);

    const auto x1 = model.transition(x0, 0.2f);
    const CV1::State z1 = z_only.transition(x0.tail<2>(), 0.2f);

    REQUIRE_THAT(x1(5), WithinAbs(z1(0), 1e-6f));
    REQUIRE_THAT(x1(6), WithinAbs(z1(1), 1e-6f));
}

TEST_CASE("CoordinatedTurn3D::transition vertical channel is plain constant velocity (no gravity/drag/coupling)",
          "[models][ct3d]") {
    CT3D model{1.0f, 0.1f, 0.5f};
    const auto x0 = make_state(0.0f, 0.0f, 1.0f, 0.0f, 2.0f, 50.0f, -3.0f); // large omega, to stress-test decoupling

    const auto x1 = model.transition(x0, 0.5f);

    REQUIRE_THAT(x1(5), WithinAbs(50.0f + (-3.0f) * 0.5f, 1e-4f)); // pz += vz*dt exactly
    REQUIRE_THAT(x1(6), WithinAbs(-3.0f, 1e-6f));                  // vz unchanged
}

TEST_CASE("CoordinatedTurn3D::jacobian is exactly block-diagonal (off-block entries are zero, not just small)",
          "[models][ct3d][regression]") {
    CT3D model{1.0f, 0.1f, 0.5f};
    for (float omega : {-3.0f, -0.5f, -0.19f, 0.0f, 0.19f, 0.5f, 3.0f}) {
        const auto x0 = make_state(0.5f, -0.3f, 1.2f, -0.7f, omega, 10.0f, -1.0f);
        const auto F = model.jacobian(x0, 0.1f);

        for (int i = 0; i < 7; ++i) {
            for (int j = 0; j < 7; ++j) {
                const bool in_xy_block = i < 5 && j < 5;
                const bool in_z_block = i >= 5 && j >= 5;
                if (!in_xy_block && !in_z_block) {
                    REQUIRE(F(i, j) == 0.0f);
                }
            }
        }
    }
}

TEST_CASE("CoordinatedTurn3D::jacobian's blocks exactly match the standalone sub-models", "[models][ct3d]") {
    CT3D model{1.0f, 0.1f, 0.5f};
    CT xy_only{1.0f, 0.1f};
    CV1 z_only{0.5f};
    const auto x0 = make_state(0.5f, -0.3f, 1.2f, -0.7f, 0.6f, 10.0f, -1.0f);

    const auto F = model.jacobian(x0, 0.3f);
    const auto F_xy = xy_only.jacobian(x0.head<5>(), 0.3f);
    const auto F_z = z_only.jacobian(x0.tail<2>(), 0.3f);

    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            REQUIRE_THAT(F(i, j), WithinAbs(F_xy(i, j), 1e-6f));
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            REQUIRE_THAT(F(5 + i, 5 + j), WithinAbs(F_z(i, j), 1e-6f));
}

TEST_CASE("CoordinatedTurn3D::jacobian matches finite-difference across a range of turn rates and step sizes",
          "[models][ct3d][regression]") {
    // Mirrors test_coordinated_turn.cpp's own sweep (varying both omega and
    // dt, per that file's own comment on why a single fixed dt let bugs
    // through previously) -- run here too since CT3D's jacobian is new code
    // (the block assembly), even though the sub-model math itself is not.
    CT3D model{1.0f, 0.1f, 0.5f};
    for (float dt : {1.0f / 30.0f, 0.1f, 1.0f}) {
        for (float omega : {-3.0f, -0.5f, -0.19f, -0.01f, 0.0f, 0.01f, 0.19f, 0.5f, 3.0f}) {
            const auto x0 = make_state(0.5f, -0.3f, 1.2f, -0.7f, omega, 10.0f, -1.0f);
            augur_test::check_jacobian_matches_finite_difference(model, x0, dt);
        }
    }
}

TEST_CASE("CoordinatedTurn3D::process_noise is symmetric positive semi-definite and exactly block-diagonal",
          "[models][ct3d]") {
    CT3D model{1.0f, 0.1f, 0.5f};
    const auto Q = model.process_noise(0.2f);

    REQUIRE_THAT((Q - Q.transpose()).norm(), WithinAbs(0.0f, 1e-6f));
    Eigen::SelfAdjointEigenSolver<CT3D::Transition> solver(Q);
    REQUIRE(solver.eigenvalues().minCoeff() >= -1e-6f);

    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            const bool in_xy_block = i < 5 && j < 5;
            const bool in_z_block = i >= 5 && j >= 5;
            if (!in_xy_block && !in_z_block) {
                REQUIRE(Q(i, j) == 0.0f);
            }
        }
    }
}

TEST_CASE("CoordinatedTurn3D::process_noise's blocks exactly match the standalone sub-models", "[models][ct3d]") {
    CT3D model{1.0f, 0.1f, 0.5f};
    CT xy_only{1.0f, 0.1f};
    CV1 z_only{0.5f};

    const auto Q = model.process_noise(0.2f);
    const auto Q_xy = xy_only.process_noise(0.2f);
    const auto Q_z = z_only.process_noise(0.2f);

    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            REQUIRE_THAT(Q(i, j), WithinAbs(Q_xy(i, j), 1e-6f));
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            REQUIRE_THAT(Q(5 + i, 5 + j), WithinAbs(Q_z(i, j), 1e-6f));
}

TEST_CASE("CoordinatedTurn3D runs through KalmanFilter and mixes into imm::Estimator", "[models][ct3d][filters][imm]") {
    using KF = augur::filters::KalmanFilter<CT3D, /*MeasDim=*/3>;
    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1; // px
    H(1, 1) = 1; // py
    H(2, 5) = 1; // pz
    KF::MeasurementCovariance R = KF::MeasurementCovariance::Identity() * 0.1f;

    KF filter{CT3D{1.0f, 0.1f, 0.5f}, KF::StateVector::Zero(), KF::StateCovariance::Identity(), H, R};
    filter.predict(0.1f);
    filter.update(KF::Measurement{1.0f, 0.5f, -3.0f});
    REQUIRE(filter.state().allFinite());

    // Two differently-tuned instances mixed via imm::Estimator -- the same
    // "several instances of one model" same-dimension-mixing pattern
    // imm::Estimator's own file comment describes.
    using Tracker = augur::imm::Estimator<KF, KF>;
    Tracker tracker{
        KF{CT3D{1.0f, 0.02f, 0.5f}, KF::StateVector::Zero(), KF::StateCovariance::Identity(), H, R},
        KF{CT3D{1.0f, 3.0f, 0.5f}, KF::StateVector::Zero(), KF::StateCovariance::Identity(), H, R},
        augur::imm::ModeMatrix<2, float>::uniform(0.95f),
    };
    tracker.predict(0.1f);
    tracker.update(KF::Measurement{0.5f, 0.3f, -1.0f});

    const auto [x, P] = tracker.combined_state();
    REQUIRE(x.allFinite());
    REQUIRE(P.allFinite());
}
