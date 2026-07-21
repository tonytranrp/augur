// tests/unit/test_singer.cpp
//
// models/singer.hpp is explicitly a "flagged sketch" (see
// .claude/rules/code-style.md's three-tier convention): its
// process_noise() is a documented simplified placeholder, not the exact
// Singer closed-form integral, so this file doesn't try to validate
// that term against the "real" derivation -- only that it stays a
// mathematically sane covariance. transition()/jacobian() ARE meant to
// be exact closed forms, so those get the same finite-difference
// treatment as CoordinatedTurn.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "augur/augur.hpp"
#include "augur/models/singer.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("Singer::jacobian matches finite-difference across spatial dimensions and time constants", "[models][singer]") {
    const float dt = 1.0f / 30.0f;
    for (float tau : {0.5f, 2.0f, 10.0f}) {
        {
            using S1 = augur::models::Singer<float, 1>;
            S1 model{tau, 1.0f};
            S1::State x0;
            x0 << 0.2f, 1.1f, -0.4f;
            augur_test::check_jacobian_matches_finite_difference(model, x0, dt);
        }
        {
            using S2 = augur::models::Singer<float, 2>;
            S2 model{tau, 1.0f};
            S2::State x0;
            x0 << 0.2f, -0.1f, 1.1f, -0.6f, -0.4f, 0.3f;
            augur_test::check_jacobian_matches_finite_difference(model, x0, dt);
        }
    }
}

TEST_CASE("Singer::transition matches hand-computed arithmetic for a trivial case", "[models][singer]") {
    // With acceleration=0 at the start, position/velocity should evolve
    // exactly like constant velocity regardless of tau -- the
    // Ornstein-Uhlenbeck acceleration term only kicks in once
    // acceleration itself is nonzero.
    using S = augur::models::Singer<float, 1>;
    S model{2.0f, 1.0f};
    S::State x0;
    x0 << 0.0f, 2.0f, 0.0f; // pos=0, vel=2, accel=0

    const auto x1 = model.transition(x0, 0.5f);

    REQUIRE_THAT(x1(0), WithinAbs(1.0f, 1e-6));  // 0 + 2*0.5
    REQUIRE_THAT(x1(1), WithinAbs(2.0f, 1e-6));  // velocity unchanged
    REQUIRE_THAT(x1(2), WithinAbs(0.0f, 1e-6));  // 0 * exp(-alpha*dt) = 0
}

TEST_CASE("Singer::process_noise is symmetric positive semi-definite", "[models][singer]") {
    using S = augur::models::Singer<float, 2>;
    S model{2.0f, 1.0f};
    const auto Q = model.process_noise(0.1f);

    REQUIRE_THAT((Q - Q.transpose()).norm(), WithinAbs(0.0f, 1e-6));
    Eigen::SelfAdjointEigenSolver<S::Transition> solver(Q);
    REQUIRE(solver.eigenvalues().minCoeff() >= -1e-6f);
}
