#pragma once
// augur/track/fusion.hpp
//
// docs/ROADMAP.md item 10, "Sensor / detection fusion": combine several
// noisy per-frame detections of the same target (e.g. multiple
// raycast/vision-cone samples) into one fused measurement before it
// ever reaches a filter's update() -- a Filter's update() assumes one
// measurement per step, but real detection pipelines often produce
// several candidate observations of the same thing that should be
// reconciled first.
//
// Inverse-covariance-weighted (information-form) combination: precision
// matrices (covariance inverses) simply add across independent
// measurements of the same quantity, and the fused mean is the
// precision-weighted average. Verified (ad hoc python3 + numpy, per
// .claude/rules/testing.md) to match applying the same detections as
// sequential Kalman updates against a near-uninformative prior -- as it
// must, since this is the same underlying math, just in closed form
// rather than iterated one measurement at a time.

#include <cstddef>
#include "augur/math/backend.hpp"
#include "augur/utils/assert.hpp"
#include "augur/utils/fixed_vector.hpp"

namespace augur::track {

template <typename Scalar, int MeasDim>
struct FusedMeasurement {
    augur::math::Vector<Scalar, MeasDim> mean;
    augur::math::Matrix<Scalar, MeasDim> covariance;
};

// Assumes the detections are conditionally independent given the true
// target state (no shared bias/correlated noise across sensors) --
// the standard assumption this closed form relies on.
template <typename Scalar, int MeasDim, std::size_t MaxDetections>
[[nodiscard]] inline FusedMeasurement<Scalar, MeasDim> fuse_measurements(
    const augur::utils::FixedVector<augur::math::Vector<Scalar, MeasDim>, MaxDetections>& detections,
    const augur::utils::FixedVector<augur::math::Matrix<Scalar, MeasDim>, MaxDetections>& covariances) {
    AUGUR_ASSERT(detections.size() == covariances.size(), "fuse_measurements: detections/covariances size mismatch");
    AUGUR_ASSERT(!detections.empty(), "fuse_measurements: need at least one detection to fuse");

    augur::math::Matrix<Scalar, MeasDim> precision = augur::math::Matrix<Scalar, MeasDim>::Zero();
    augur::math::Vector<Scalar, MeasDim> weighted_sum = augur::math::Vector<Scalar, MeasDim>::Zero();
    for (std::size_t i = 0; i < detections.size(); ++i) {
        const auto covariance_inv = augur::math::safe_inverse<Scalar, MeasDim>(covariances[i]);
        precision += covariance_inv;
        weighted_sum += covariance_inv * detections[i];
    }

    const auto fused_covariance = augur::math::safe_inverse<Scalar, MeasDim>(precision);
    return FusedMeasurement<Scalar, MeasDim>{fused_covariance * weighted_sum, fused_covariance};
}

} // namespace augur::track
