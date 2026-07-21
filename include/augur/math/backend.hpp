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

// A numerically-safe symmetric inverse, used by every filter's update
// step (innovation covariance inverse). Falls back to a pseudo-inverse
// if the matrix is (near-)singular rather than propagating NaNs into
// the state -- a sensor dropout or a degenerate covariance shouldn't be
// able to poison an entire track.
template <typename Scalar, int N>
[[nodiscard]] inline Matrix<Scalar, N> safe_inverse(const Matrix<Scalar, N>& m) {
    Eigen::LDLT<Matrix<Scalar, N>> ldlt(m);
    if (ldlt.info() == Eigen::Success && ldlt.isPositive()) {
        return ldlt.solve(Matrix<Scalar, N>::Identity());
    }
    // Degenerate covariance: regularize slightly and retry rather than
    // returning garbage. This is intentionally conservative -- callers
    // that hit this path in practice should treat it as a signal their
    // process/measurement noise needs revisiting.
    constexpr Scalar epsilon = static_cast<Scalar>(1e-6);
    return (m + epsilon * Matrix<Scalar, N>::Identity()).inverse();
}

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
// would otherwise be free to go non-PSD.
template <typename Scalar, int N>
[[nodiscard]] inline Matrix<Scalar, N> project_to_psd(const Matrix<Scalar, N>& m,
                                                        Scalar min_eigenvalue = static_cast<Scalar>(1e-9)) {
    const Matrix<Scalar, N> symmetric = (m + m.transpose()) * Scalar(0.5);
    Eigen::SelfAdjointEigenSolver<Matrix<Scalar, N>> solver(symmetric);
    const auto clamped = solver.eigenvalues().cwiseMax(min_eigenvalue);
    return solver.eigenvectors() * clamped.asDiagonal() * solver.eigenvectors().transpose();
}

} // namespace augur::math
