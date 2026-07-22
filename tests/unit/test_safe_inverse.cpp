// tests/unit/test_safe_inverse.cpp
//
// docs/IMPROVEMENT_PLAN.md's top finding: math/backend.hpp::safe_inverse()
// had two distinct bugs in degenerate-matrix handling, and had no
// dedicated test despite being the single most safety-critical primitive
// in the codebase (every Kalman-style filter's update() routes its
// innovation-covariance inverse through it).
//
//   Bug A: the fast-path guard was `ldlt.isPositive()` alone, which means
//   positive-SEMI-definite (Eigen's own documented meaning), not strictly
//   positive-definite -- an exactly-singular-but-PSD input wrongly took
//   the fast path, and ldlt.solve(Identity()) silently returned an
//   all-zero "inverse" instead of erroring or falling back.
//   Bug B: even the fallback branch's fixed epsilon=1e-6 nudge + raw
//   .inverse() wasn't robust -- a realistically-scaled indefinite input
//   stayed indefinite after the nudge, and an adversarial input produced
//   literal NaN.
//
// Fixed by requiring every LDLT pivot to clear a small positive floor on
// the fast path, and replacing the fallback with project_to_psd(m).inverse()
// (this file's own more robust primitive). Verified ad hoc via python3 +
// numpy first (per .claude/rules/testing.md) that project_to_psd-then-
// invert stays finite for all three pathological cases below; this file
// is the permanent C++ regression coverage for that.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "augur/augur.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("safe_inverse matches plain .inverse() on well-conditioned input", "[math][safe_inverse]") {
    using Mat = augur::math::Matrix<float, 2>;
    Mat m;
    m << 2.0f, 0.0f,
         0.0f, 3.0f;

    const Mat inv = augur::math::safe_inverse<float, 2>(m);
    const Mat expected = m.inverse();

    REQUIRE_THAT((inv - expected).norm(), WithinAbs(0.0f, 1e-5));
}

TEST_CASE("safe_inverse falls back cleanly for an exactly-zero (singular PSD) matrix", "[math][safe_inverse]") {
    // isPositive() alone (Bug A) treats Zero() as a valid fast-path input
    // since all its eigenvalues are >= 0 -- exactly the "R=Zero(), no
    // sensor noise configured" case docs/IMPROVEMENT_PLAN.md traced into
    // a silent no-op Kalman gain. The fixed pivot-floor guard must reject
    // this and fall back instead of returning a zero "inverse."
    using Mat = augur::math::Matrix<float, 2>;
    const Mat m = Mat::Zero();

    const Mat inv = augur::math::safe_inverse<float, 2>(m);

    REQUIRE(inv.allFinite());
    REQUIRE(inv.norm() > 1.0f); // not the old silent-zero bug
}

TEST_CASE("safe_inverse stays finite for a realistically-scaled indefinite matrix", "[math][safe_inverse]") {
    // Bug B: a fixed epsilon=1e-6 nudge doesn't reliably fix an
    // indefinite matrix at this scale -- diag(-0.5, 100) + 1e-6*I is
    // still indefinite (verified ad hoc via python3/numpy).
    using Mat = augur::math::Matrix<float, 2>;
    Mat m;
    m << -0.5f, 0.0f,
          0.0f, 100.0f;

    const Mat inv = augur::math::safe_inverse<float, 2>(m);

    REQUIRE(inv.allFinite());
}

TEST_CASE("safe_inverse stays finite for an adversarial near-zero-negative-definite matrix", "[math][safe_inverse]") {
    // Bug B's adversarial case: m = -1e-6*I produced literal NaN from
    // the old fallback's raw .inverse() call (verified ad hoc via
    // python3/numpy: (-1e-6*I + 1e-6*I).inverse() divides by zero).
    using Mat = augur::math::Matrix<float, 2>;
    const Mat m = Mat::Identity() * -1e-6f;

    const Mat inv = augur::math::safe_inverse<float, 2>(m);

    REQUIRE(inv.allFinite());
}

TEST_CASE("KalmanFilter::update() no longer silently no-ops when the innovation covariance is singular", "[math][safe_inverse][filters][kalman]") {
    // Integration-level reproduction of docs/IMPROVEMENT_PLAN.md's traced
    // downstream effect. Two measurement rows that both read px (a
    // realistic duplicate/redundant-sensor misconfiguration) make
    // S=H*P*H^T singular regardless of P being well-conditioned; R=Zero()
    // ("no sensor noise configured") keeps it exactly singular rather
    // than just ill-conditioned. Verified ad hoc via python3/numpy first:
    // under the OLD bug, S_inv=0 makes the Kalman gain K=0, so a
    // measurement disagreeing sharply with the prediction is completely
    // ignored (px stays at 0 instead of moving toward the measured 50).
    using CV = augur::models::ConstantVelocity<float, 2>;
    using KF = augur::filters::KalmanFilter<CV, /*MeasDim=*/2>;

    KF::StateVector x0 = KF::StateVector::Zero();
    KF::StateCovariance P0 = KF::StateCovariance::Identity() * 10.0f;
    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1; // row 0 reads px
    H(1, 0) = 1; // row 1 ALSO reads px -- redundant, makes H*P*H^T singular
    KF::MeasurementCovariance R = KF::MeasurementCovariance::Zero();

    KF filter{CV{1.0f}, x0, P0, H, R};
    filter.update(KF::Measurement{50.0f, 50.0f});

    const auto& x = filter.state();
    REQUIRE(x.allFinite());
    // px moved SUBSTANTIALLY toward the measurement -- not the old bug's
    // exact no-op (state staying at 0). The relative eigenvalue-floor
    // regularization needed for float32 safety (see safe_inverse's own
    // comment) means this won't land on exactly 50, so the tolerance
    // here is loose by this test's usual standard -- but 0.5 is still a
    // 100x-tighter bound than "unchanged from 0," which is what the old
    // bug actually produced (verified ad hoc via python3/numpy).
    REQUIRE_THAT(x(0), WithinAbs(50.0f, 0.5f));
    REQUIRE_THAT(x(1), WithinAbs(0.0f, 1e-4f)); // py untouched -- H's rows don't read it
}
