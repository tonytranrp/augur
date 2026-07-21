// examples/14_sensor_fusion/main.cpp
//
// docs/ROADMAP.md item 10: three simulated vision-cone samples of the
// same target in one frame, fused via track/fusion.hpp into a single
// measurement before it reaches the filter's update() -- rather than
// picking one arbitrarily or calling update() three times (which would
// double-count the shared prior and understate the result's
// confidence).

#include <cstdio>
#include "augur/augur.hpp"
#include "augur/track/fusion.hpp"

int main() {
    using Scalar = float;
    using CV = augur::models::ConstantVelocity<Scalar, 2>;
    using KF = augur::filters::KalmanFilter<CV, /*MeasDim=*/2>;

    augur::utils::FixedVector<augur::math::Vector<Scalar, 2>, 4> detections;
    detections.push_back({1.02f, 2.05f});
    detections.push_back({0.95f, 1.90f});
    detections.push_back({1.10f, 1.98f});
    augur::utils::FixedVector<augur::math::Matrix<Scalar, 2>, 4> covariances;
    covariances.push_back(augur::math::Matrix<Scalar, 2>::Identity() * 0.30f); // noisy raycast
    covariances.push_back(augur::math::Matrix<Scalar, 2>::Identity() * 0.10f); // more precise vision-cone sample
    covariances.push_back(augur::math::Matrix<Scalar, 2>::Identity() * 0.20f);

    const auto fused = augur::track::fuse_measurements<Scalar, 2, 4>(detections, covariances);
    std::printf("fused measurement: pos=(%.4f, %.4f)  covariance diag=(%.4f, %.4f)\n",
                fused.mean(0), fused.mean(1), fused.covariance(0, 0), fused.covariance(1, 1));

    KF::StateVector x0 = KF::StateVector::Zero();
    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    KF filter{CV{1.0f}, x0, KF::StateCovariance::Identity() * 5.0f, H, fused.covariance};

    filter.predict(1.0f / 30.0f);
    filter.update(fused.mean);
    std::printf("filter state after update: pos=(%.4f, %.4f)\n", filter.state()(0), filter.state()(1));
    return 0;
}
