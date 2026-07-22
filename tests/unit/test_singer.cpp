// tests/unit/test_singer.cpp
//
// models/singer.hpp is now solid tier (see .claude/rules/code-style.md's
// three-tier convention, and that file's own top comment): process_noise()
// is the exact Van Loan closed form, not a placeholder, so this file
// validates it against an independent reference -- values below were
// computed ad hoc (python3 + sympy for the closed form, mpmath for
// precision, per .claude/rules/testing.md) before any of the C++ was
// written, and independently cross-checked against direct Simpson's-rule
// numerical integration of the defining integral (a completely different
// method from the symbolic derivation) to ~1e-10 absolute agreement.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "augur/augur.hpp"
#include "augur/models/current_statistical.hpp"
#include "augur/models/singer.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("Singer::jacobian matches finite-difference across spatial dimensions, time constants, and step sizes",
          "[models][singer]") {
    // tau=60 (alpha=1/60) is included specifically: it's this file's own
    // documented "lazy maneuverer" case, and alpha*dt=(1/60)*(1/30) is
    // exactly the parameter combination that exposed a real 15.9%
    // relative error in the old, unbranched closed form (see
    // singer.hpp's own top comment) -- widening this sweep from the
    // original {0.5, 2.0, 10.0} is what would have caught it.
    const float dt_default = 1.0f / 30.0f;
    for (float tau : {0.5f, 2.0f, 10.0f, 60.0f}) {
        {
            using S1 = augur::models::Singer<float, 1>;
            S1 model{tau, 1.0f};
            S1::State x0;
            x0 << 0.2f, 1.1f, -0.4f;
            augur_test::check_jacobian_matches_finite_difference(model, x0, dt_default);
        }
        {
            using S2 = augur::models::Singer<float, 2>;
            S2 model{tau, 1.0f};
            S2::State x0;
            x0 << 0.2f, -0.1f, 1.1f, -0.6f, -0.4f, 0.3f;
            augur_test::check_jacobian_matches_finite_difference(model, x0, dt_default);
        }
    }
    // Also sweep dt at a fixed tau, matching the reasoning
    // test_coordinated_turn.cpp's own sweep documents: a small-parameter
    // branch needs checking against BOTH its own small parameter (here,
    // alpha*dt) AND a range of dt, since the same alpha can put alpha*dt
    // on either side of the branch threshold depending on dt alone.
    using S1 = augur::models::Singer<float, 1>;
    for (float dt : {1.0f / 30.0f, 0.1f, 0.5f, 1.0f, 2.0f}) {
        for (float tau : {0.5f, 10.0f, 60.0f}) {
            S1 model{tau, 1.0f};
            S1::State x0;
            x0 << 0.5f, -0.3f, 0.8f;
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

TEST_CASE("Singer::transition's small-alpha*dt branch matches an independent reference (regression)",
          "[models][singer][regression]") {
    // tau=60s, dt=1/30s: alpha*dt ~ 5.6e-4, comfortably inside the
    // small-parameter branch. Reference (ad hoc python3+mpmath): the OLD
    // unbranched closed form was off by 15.9% relative error here --
    // this exact scenario is what caught it.
    using S = augur::models::Singer<float, 1>;
    S model{60.0f, 1.0f};
    S::State x0;
    x0 << 0.0f, 0.0f, 1.0f; // isolate f_pv/f_va: accel=1, everything else 0

    const auto x1 = model.transition(x0, 1.0f / 30.0f);

    REQUIRE_THAT(x1(0), WithinAbs(0.0005554527f, 1e-7f));  // f_pv
    REQUIRE_THAT(x1(1), WithinAbs(0.0333240758f, 1e-6f));  // f_va
    REQUIRE_THAT(x1(2), WithinAbs(0.9994445987f, 1e-6f));  // e = exp(-alpha*dt)
}

TEST_CASE("Singer::process_noise matches an independent reference (exact branch)", "[models][singer][regression]") {
    // tau=2s (alpha=0.5), dt=1/30s, sigma=1: alpha*dt~0.0167, exact branch.
    // Reference computed at 50 decimal digits of precision (mpmath, per
    // .claude/rules/testing.md) -- NOT plain float64, which turned out to
    // still be corrupted by the same catastrophic cancellation this fix
    // addresses for the smallest entries at small alpha*dt (see the
    // small-alpha*dt branch test below for exactly how badly: an earlier
    // version of this test hardcoded a float64-computed "reference" for
    // Q_pp there that was wrong by 28x, caught only because the real
    // build disagreed with it and the discrepancy was run back down to
    // its root cause rather than the tolerance being loosened).
    using S = augur::models::Singer<float, 1>;
    S model{2.0f, 1.0f};
    const auto Q = model.process_noise(1.0f / 30.0f);

    REQUIRE_THAT(Q(0, 0), WithinRel(2.0386740723995931e-09f, 1e-4f));
    REQUIRE_THAT(Q(0, 1), WithinRel(1.5261815427082971e-07f, 1e-4f));
    REQUIRE_THAT(Q(0, 2), WithinRel(6.0708957607265726e-06f, 1e-4f));
    REQUIRE_THAT(Q(1, 1), WithinRel(1.2192551189556749e-05f, 1e-4f));
    REQUIRE_THAT(Q(1, 2), WithinRel(5.4638567754184612e-04f, 1e-4f));
    REQUIRE_THAT(Q(2, 2), WithinRel(3.2783899517994097e-02f, 1e-4f));
}

TEST_CASE("Singer::process_noise matches an independent reference (small-alpha*dt branch, regression)",
          "[models][singer][regression]") {
    // tau=60s (alpha=1/60), dt=1/30s, sigma=1: the same parameter
    // combination that exposed the 15.9% transition() error above.
    // Reference at 50 decimal digits (mpmath) -- see this test case's
    // sibling above for why that precision floor matters here
    // specifically: Q_pp divides by alpha^5, and even float64 (~15-16
    // significant digits) isn't always enough headroom to survive that
    // much cancellation at this alpha*dt.
    using S = augur::models::Singer<float, 1>;
    S model{60.0f, 1.0f};
    const auto Q = model.process_noise(1.0f / 30.0f);

    REQUIRE_THAT(Q(0, 0), WithinRel(6.8565940963899426e-11f, 1e-3f));
    REQUIRE_THAT(Q(0, 1), WithinRel(5.1421281653717713e-09f, 1e-3f));
    REQUIRE_THAT(Q(0, 2), WithinRel(2.0564703995069670e-07f, 1e-3f));
    REQUIRE_THAT(Q(1, 1), WithinRel(4.1135121042658219e-07f, 1e-3f));
    REQUIRE_THAT(Q(1, 2), WithinRel(1.8508233785976627e-05f, 1e-3f));
    REQUIRE_THAT(Q(2, 2), WithinRel(1.1104940557206868e-03f, 1e-3f));
}

TEST_CASE("Singer::process_noise matches an independent reference (small-alpha*dt branch, larger dt)",
          "[models][singer][regression]") {
    // tau=10s (alpha=0.1), dt=1.0s, sigma=2: alpha*dt=0.1, still inside
    // the small-parameter branch but with a much larger dt than the other
    // two reference cases -- exercises the dt^5/dt^4/... prefactors at a
    // scale where they're not tiny. Reference at 50 decimal digits (mpmath).
    using S = augur::models::Singer<float, 1>;
    S model{10.0f, 2.0f};
    const auto Q = model.process_noise(1.0f);

    REQUIRE_THAT(Q(0, 0), WithinRel(3.7854972039149240e-02f, 1e-4f));
    REQUIRE_THAT(Q(0, 1), WithinRel(9.3602453018507906e-02f, 1e-4f));
    REQUIRE_THAT(Q(0, 2), WithinRel(1.2070532593049069e-01f, 1e-4f));
    REQUIRE_THAT(Q(1, 1), WithinRel(2.4756762634257362e-01f, 1e-4f));
    REQUIRE_THAT(Q(1, 2), WithinRel(3.6223668024250849e-01f, 1e-4f));
    REQUIRE_THAT(Q(2, 2), WithinRel(7.2507698768807261e-01f, 1e-4f));
}

TEST_CASE("Singer::process_noise is symmetric positive semi-definite across a wide (tau, dt) sweep",
          "[models][singer]") {
    using S = augur::models::Singer<float, 2>;
    for (float tau : {0.5f, 2.0f, 10.0f, 60.0f}) {
        for (float dt : {1.0f / 30.0f, 0.1f, 1.0f, 2.0f}) {
            S model{tau, 1.0f};
            const auto Q = model.process_noise(dt);

            REQUIRE_THAT((Q - Q.transpose()).norm(), WithinAbs(0.0f, 1e-6));
            Eigen::SelfAdjointEigenSolver<S::Transition> solver(Q);
            REQUIRE(solver.eigenvalues().minCoeff() >= -1e-6f);
        }
    }
}

TEST_CASE("Singer::process_noise's two branches agree closely at the branch boundary (continuity check)",
          "[models][singer][regression]") {
    // x=alpha*dt just below and just above the branch threshold (0.2)
    // should produce close values -- a coarse but effective check that
    // the two independently-derived branches (series vs. exact closed
    // form) weren't accidentally given inconsistent units/coefficients.
    using S = augur::models::Singer<float, 1>;
    S below{1.0f / 0.199f, 1.0f}; // dt=1 => alpha*dt = 0.199
    S above{1.0f / 0.201f, 1.0f}; // dt=1 => alpha*dt = 0.201
    const auto Qb = below.process_noise(1.0f);
    const auto Qa = above.process_noise(1.0f);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            INFO("entry (" << i << "," << j << "): below=" << Qb(i, j) << " above=" << Qa(i, j));
            REQUIRE_THAT(Qb(i, j), WithinAbs(Qa(i, j), 0.01f));
        }
    }
}

TEST_CASE("CurrentStatistical (which reuses Singer internally) still runs correctly through a real KalmanFilter",
          "[models][singer][current_statistical][regression]") {
    // Guards against a regression in the shared code CurrentStatistical
    // reuses verbatim (see models/current_statistical.hpp's own file
    // comment): its jacobian()/process_noise() delegate directly to a
    // composed Singer instance.
    using CS = augur::models::CurrentStatistical<float, 2>;
    using KF = augur::filters::KalmanFilter<CS, 2>;
    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    KF filter{CS{2.0f, 1.0f}, KF::StateVector::Zero(), KF::StateCovariance::Identity(), H,
              KF::MeasurementCovariance::Identity() * 0.1f};

    filter.predict(1.0f / 30.0f);
    filter.update(KF::Measurement{1.0f, 0.5f});

    REQUIRE(filter.state().allFinite());
    REQUIRE(filter.covariance().allFinite());
}
