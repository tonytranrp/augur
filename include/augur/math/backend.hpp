#pragma once
// augur/math/backend.hpp
//
// Every other header talks to `augur::math::Vector<N,S>` / `Matrix<N,M,S>`,
// never to Eigen directly. That's a one-file choke point: swapping the
// linear-algebra backend later (or adding a second one for a platform
// where Eigen doesn't fit) means editing this file only.
//
// Why Eigen specifically (see docs/ARCHITECTURE.md "External dependencies"
// for the full comparison against Blaze/Armadillo/xtensor):
//   - header-only -> nothing to link, trivial to vendor via CPM
//   - ARM NEON is used automatically on 64-bit ARM with no extra flags,
//     and Android's NDK enables NEON by default for all API levels, so
//     this "just works" cross-compiled for Android
//   - single-threaded unless YOU explicitly turn on OpenMP -- which augur
//     never does (see core/config.hpp); this is Eigen's documented
//     default behavior, and OpenMP isn't even present in the default
//     Android/iOS/macOS toolchains, so "universal + no forced threads"
//     falls out naturally instead of fighting the library
//   - fixed-size Eigen::Matrix<S,N,1> for compile-time-known dimensions
//     allocates on the stack -- no heap traffic in the predict/update
//     hot path

#include <Eigen/Core>
#include <Eigen/Dense>

namespace augur::math {

// Fixed dimension (the common case: a 6-state CV/CA model, a 2D or 3D
// position+velocity block, etc). Purely stack-allocated.
template <typename Scalar, int N>
using Vector = Eigen::Matrix<Scalar, N, 1>;

template <typename Scalar, int Rows, int Cols = Rows>
using Matrix = Eigen::Matrix<Scalar, Rows, Cols>;

// Dynamic dimension, for the (rarer, opt-in) case of a filter whose
// state size is only known at runtime -- e.g. a GM-PHD component list
// (see track/gm_phd.hpp) whose length changes as targets are born/die.
template <typename Scalar>
using DynVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <typename Scalar>
using DynMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

// Projects a symmetric matrix onto the positive-semi-definite cone by
// eigenvalue-flooring: decompose, clamp any eigenvalue below
// min_eigenvalue up to it, reconstruct. Used by
// filters/adaptive/sage_husa.hpp to repair an adaptively-estimated Q/R
// that's drifted non-PSD -- the "known divergence/non-positive-
// definiteness failure mode" that formulation's own file comment cites
// as needing a fix rather than the vanilla 1969 update applied as-is.
// Verified numerically (ad hoc python3, per .claude/rules/testing.md)
// that this keeps a Sage-Husa-style adaptive update stable through a
// real change in the true noise level, where the unprojected update
// would otherwise be free to go non-PSD. Also used by safe_inverse()
// below (declared after this function specifically so it can call it).
template <typename Scalar, int N>
[[nodiscard]] inline Matrix<Scalar, N> project_to_psd(const Matrix<Scalar, N>& m,
                                                        Scalar min_eigenvalue = static_cast<Scalar>(1e-9)) {
    const Matrix<Scalar, N> symmetric = (m + m.transpose()) * Scalar(0.5);
    Eigen::SelfAdjointEigenSolver<Matrix<Scalar, N>> solver(symmetric);
    const auto clamped = solver.eigenvalues().cwiseMax(min_eigenvalue);
    return solver.eigenvectors() * clamped.asDiagonal() * solver.eigenvectors().transpose();
}

// A numerically-safe symmetric inverse, used by every filter's update
// step (innovation covariance inverse). Falls back to a pseudo-inverse
// if the matrix is (near-)singular rather than propagating NaNs into
// the state -- a sensor dropout or a degenerate covariance shouldn't be
// able to poison an entire track.
//
// The admission floor is scale-RELATIVE (a fraction of the matrix's own
// diagonal magnitude), not a fixed absolute constant. This mattered in
// practice, not just in theory: an earlier version of this fix used an
// absolute 1e-9 floor (matching project_to_psd's own default
// min_eigenvalue) and passed every check at float64 -- but verified (ad
// hoc python3 + numpy, per .claude/rules/testing.md) at REAL float32
// precision, the *same* scenario came back non-finite. Adding 1e-9 to an
// eigenvalue of order 10 (a perfectly ordinary covariance magnitude)
// rounds away to nothing in float32's ~7 decimal digits, so the
// regularization silently did nothing and `.inverse()` on the
// still-exactly-singular result produced Inf/NaN. A relative floor scales
// with whatever magnitude the caller's covariance actually uses.
template <typename Scalar, int N>
[[nodiscard]] inline Matrix<Scalar, N> safe_inverse(const Matrix<Scalar, N>& m) {
    // scale: a cheap, standard proxy for "how big are this covariance's
    // own numbers" -- diagonal entries of a PSD matrix are variances,
    // always its largest-magnitude entries, so no separate norm/SVD call
    // is needed. absolute_floor only matters when scale itself is ~0
    // (e.g. m is the exact zero matrix): without it, relative_eps * 0
    // would floor to 0 too, right back to the original bug.
    constexpr Scalar relative_eps = static_cast<Scalar>(1e-4);
    constexpr Scalar absolute_floor = static_cast<Scalar>(1e-6);
    const Scalar scale = m.diagonal().cwiseAbs().maxCoeff();
    const Scalar floor = std::max(relative_eps * scale, absolute_floor);

    // isPositive() alone means positive-SEMI-definite (Eigen's own
    // documented meaning), not strictly positive-definite -- an
    // exactly-singular-but-PSD input (e.g. a zero measurement-noise
    // covariance, a completely plausible "no sensor noise configured"
    // case) would otherwise wrongly take this fast path, and
    // `ldlt.solve(Identity())` silently returns an all-zero "inverse"
    // instead of erroring or falling back (traced into
    // filters/kalman.hpp::update(): a zero S_inv makes the Kalman gain
    // zero too, silently turning update() into a no-op -- the exact
    // opposite of the correct R->0 behavior of fully trusting the new
    // measurement). Requiring every pivot to clear the floor above is
    // what actually encodes "safe to invert directly."
    Eigen::LDLT<Matrix<Scalar, N>> ldlt(m);
    if (ldlt.info() == Eigen::Success && ldlt.isPositive() && ldlt.vectorD().minCoeff() > floor) {
        return ldlt.solve(Matrix<Scalar, N>::Identity());
    }
    // Degenerate covariance: project onto the PSD cone (eigenvalue
    // floor) and invert that, rather than a blind epsilon*I additive
    // nudge -- verified (ad hoc python3 + numpy) that a fixed nudge
    // doesn't reliably fix an indefinite matrix (e.g. diag(-0.5, 100)
    // stays indefinite after +1e-6*I) and that an adversarial input like
    // -1e-6*I produces literal NaN from the raw .inverse() that used to
    // follow it, while project_to_psd(m, floor).inverse() stays finite
    // for both plus the R=Zero() case above -- reuses this file's own
    // more robust primitive rather than a second regularization scheme.
    return project_to_psd<Scalar, N>(m, floor).inverse();
}

} // namespace augur::math
