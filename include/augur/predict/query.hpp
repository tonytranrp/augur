#pragma once
// augur/predict/query.hpp
//
// Small, genuinely-implemented helpers for the two things almost every
// consumer wants out of a tracked/predicted state: "give me just the
// position/velocity part" (every model in models/ agrees on the
// [position..., velocity..., (acceleration...)] layout convention, so
// this is safe to do generically), and "draw the uncertainty so a
// designer can SEE why the aim-assist is behaving the way it is" (see
// docs/ROADMAP.md, "Debug/visualization export" -- this file is that
// idea's first real implementation, not just a stub).

#include <cmath>
#include <numbers>
#include "augur/math/backend.hpp"

namespace augur::predict {

// Assumes the [position(SpatialDim), velocity(SpatialDim), ...] layout
// every model in augur::models follows.
template <typename Scalar, int SpatialDim, int StateDim>
[[nodiscard]] inline augur::math::Vector<Scalar, SpatialDim> position(
    const augur::math::Vector<Scalar, StateDim>& state) {
    return state.template head<SpatialDim>();
}

template <typename Scalar, int SpatialDim, int StateDim>
[[nodiscard]] inline augur::math::Vector<Scalar, SpatialDim> velocity(
    const augur::math::Vector<Scalar, StateDim>& state) {
    return state.template segment<SpatialDim>(SpatialDim);
}

template <typename Scalar>
struct ErrorEllipse2D {
    Scalar semi_major;      // 1-sigma half-length along the major axis
    Scalar semi_minor;      // 1-sigma half-length along the minor axis
    Scalar rotation_radians; // angle of the major axis from +X
};

// Extracts the 2x2 position sub-block of a covariance matrix (assumed
// to start at row/col 0, per the layout convention above) and returns
// the 1-sigma error ellipse -- exactly what a debug-draw call needs to
// show a player-visible "here's how sure the tracker is" gizmo, or to
// feed a confidence-aware firing decision (see docs/ROADMAP.md,
// "Firing-decision / optimal-stopping utilities").
template <typename Scalar, int StateDim>
[[nodiscard]] inline ErrorEllipse2D<Scalar> error_ellipse_2d(
    const augur::math::Matrix<Scalar, StateDim>& covariance) {
    const augur::math::Matrix<Scalar, 2> block = covariance.template topLeftCorner<2, 2>();
    Eigen::SelfAdjointEigenSolver<augur::math::Matrix<Scalar, 2>> solver(block);

    const auto& eigenvalues = solver.eigenvalues();   // ascending order
    const auto& eigenvectors = solver.eigenvectors();

    const Scalar minor_var = eigenvalues(0) > Scalar(0) ? eigenvalues(0) : Scalar(0);
    const Scalar major_var = eigenvalues(1) > Scalar(0) ? eigenvalues(1) : Scalar(0);
    const auto major_axis = eigenvectors.col(1);

    return ErrorEllipse2D<Scalar>{
        std::sqrt(major_var),
        std::sqrt(minor_var),
        std::atan2(major_axis.y(), major_axis.x())
    };
}

// docs/ROADMAP.md item 11's 3D extension of ErrorEllipse2D/error_ellipse_2d
// above, "via the same eigen-decomposition approach, one dimension up."
// Axes are ordered largest-to-smallest, matching the 2D case's
// semi_major-before-semi_minor convention. `orientation`'s columns are
// those axes' unit directions in the same order, forced to a proper
// (determinant +1) rotation -- Eigen::SelfAdjointEigenSolver only
// guarantees an orthonormal eigenvector basis, and either sign of a
// unit eigenvector is equally valid, so the raw result can come out as
// a reflection (det -1) about as often as a rotation; flipping the sign
// of one axis fixes this without changing the ellipsoid it represents
// (v*v^T is unchanged by v -> -v), and a proper rotation matrix is what
// most debug-draw APIs and quaternion conversions expect. Verified
// numerically (ad hoc python3 + numpy, per .claude/rules/testing.md)
// before writing this.
template <typename Scalar>
struct ErrorEllipsoid3D {
    Scalar semi_axis_major;  // 1-sigma half-length, largest axis
    Scalar semi_axis_mid;
    Scalar semi_axis_minor;  // 1-sigma half-length, smallest axis
    augur::math::Matrix<Scalar, 3> orientation; // columns: unit axis directions, major/mid/minor order; proper rotation (det +1)
};

template <typename Scalar, int StateDim>
[[nodiscard]] inline ErrorEllipsoid3D<Scalar> error_ellipsoid_3d(
    const augur::math::Matrix<Scalar, StateDim>& covariance) {
    const augur::math::Matrix<Scalar, 3> block = covariance.template topLeftCorner<3, 3>();
    Eigen::SelfAdjointEigenSolver<augur::math::Matrix<Scalar, 3>> solver(block);

    const auto& eigenvalues = solver.eigenvalues();   // ascending order
    const auto& eigenvectors = solver.eigenvectors();

    const auto clamp = [](Scalar v) { return v > Scalar(0) ? v : Scalar(0); };
    const Scalar major_var = clamp(eigenvalues(2));
    const Scalar mid_var = clamp(eigenvalues(1));
    const Scalar minor_var = clamp(eigenvalues(0));

    augur::math::Matrix<Scalar, 3> orientation;
    orientation.col(0) = eigenvectors.col(2);
    orientation.col(1) = eigenvectors.col(1);
    orientation.col(2) = eigenvectors.col(0);
    if (orientation.determinant() < Scalar(0)) {
        orientation.col(2) = -orientation.col(2);
    }

    return ErrorEllipsoid3D<Scalar>{
        std::sqrt(major_var),
        std::sqrt(mid_var),
        std::sqrt(minor_var),
        orientation,
    };
}

} // namespace augur::predict
