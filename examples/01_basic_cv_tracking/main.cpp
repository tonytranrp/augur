// examples/01_basic_cv_tracking/main.cpp
//
// The simplest possible use of augur: one model, one filter, no IMM.
// Feeds a few noisy 2D position "detections" through a constant-
// velocity Kalman filter and prints the smoothed state each step,
// along with the 1-sigma error ellipse (predict/query.hpp) a debug-draw
// gizmo would use to show a player-visible "here's how sure the tracker
// is" indicator. Start here before looking at example 2 (IMM).

#include <cstdio>
#include "augur/augur.hpp"
#include "augur/predict/query.hpp"

int main() {
    using Scalar = float;
    constexpr int SpatialDim = 2;

    using CV = augur::models::ConstantVelocity<Scalar, SpatialDim>;
    using KF = augur::filters::KalmanFilter<CV, /*MeasDim=*/SpatialDim>;

    KF::StateVector x0 = KF::StateVector::Zero();       // [px, py, vx, vy] = 0
    KF::StateCovariance P0 = KF::StateCovariance::Identity() * Scalar(10);

    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1; // measure px directly
    H(1, 1) = 1; // measure py directly

    KF::MeasurementCovariance R = KF::MeasurementCovariance::Identity() * Scalar(0.5);

    KF filter{CV{/*accel_noise_density=*/Scalar(1)}, x0, P0, H, R};

    // A target drifting roughly along +x with a little measurement noise.
    const Scalar dt = Scalar(1.0 / 60.0);
    const KF::Measurement detections[] = {
        {0.10f, 0.02f}, {0.18f, -0.05f}, {0.31f, 0.04f}, {0.39f, 0.01f}, {0.52f, -0.03f},
    };

    for (const auto& z : detections) {
        filter.predict(dt);
        filter.update(z);
        const auto& x = filter.state();
        const auto ellipse = augur::predict::error_ellipse_2d<Scalar, KF::dimension>(filter.covariance());
        std::printf("pos=(%.3f, %.3f)  vel=(%.3f, %.3f)  likelihood=%.4f  "
                    "1-sigma ellipse: major=%.3f minor=%.3f angle=%.1f deg\n",
                    x(0), x(1), x(2), x(3), filter.last_likelihood(),
                    ellipse.semi_major, ellipse.semi_minor, ellipse.rotation_radians * Scalar(180.0 / 3.14159265));
    }
    return 0;
}
