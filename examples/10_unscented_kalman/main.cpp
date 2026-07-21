// examples/10_unscented_kalman/main.cpp
//
// docs/ROADMAP.md item 3: UnscentedKalmanFilter tracking a target from
// range+bearing measurements -- genuinely nonlinear (sqrt(x^2+y^2),
// atan2(y,x)), the kind of sensor model extended_kalman.hpp's own file
// comment names as a UKF use case. Starts from a deliberately poor
// initial guess to show the filter actually converging.
//
// Range-only was tried first while building this and dropped (see
// filters/unscented_kalman.hpp and tests/unit/test_unscented_kalman.cpp):
// independently verified to be only weakly observable from a single
// stationary sensor against a non-radially-moving target, which made it
// a poor demonstration, not because anything was broken.

#include <cstdio>
#include "augur/augur.hpp"
#include "augur/filters/unscented_kalman.hpp"

int main() {
    using Scalar = float;
    using CV = augur::models::ConstantVelocity<Scalar, 2>;
    using UKF = augur::filters::UnscentedKalmanFilter<CV, /*MeasDim=*/2>;

    UKF::StateVector x0;
    x0 << 8.0f, 8.0f, 0.0f, 0.0f; // poor initial guess
    UKF::StateCovariance P0 = UKF::StateCovariance::Identity() * 4.0f;
    UKF::MeasurementCovariance R = (UKF::MeasurementCovariance() << 0.01f, 0.0f, 0.0f, 0.001f).finished();

    auto range_bearing = [](const UKF::StateVector& x) -> UKF::Measurement {
        return UKF::Measurement{std::sqrt(x(0) * x(0) + x(1) * x(1)), std::atan2(x(1), x(0))};
    };
    UKF filter{CV{0.05f}, x0, P0, range_bearing, R};

    const Scalar dt = Scalar(1.0 / 30.0);
    Scalar truth_x = 3.0f, truth_y = 4.0f, vx = 1.0f, vy = 0.5f;

    for (int step = 0; step < 60; ++step) {
        truth_x += vx * dt;
        truth_y += vy * dt;
        filter.predict(dt);
        filter.update(UKF::Measurement{std::sqrt(truth_x * truth_x + truth_y * truth_y), std::atan2(truth_y, truth_x)});

        if (step % 10 == 9 || step == 59) {
            const auto& x = filter.state();
            std::printf("step %2d: truth=(%.3f,%.3f) est=(%.3f,%.3f) pos_err=%.4f\n",
                        step, truth_x, truth_y, x(0), x(1), std::hypot(x(0) - truth_x, x(1) - truth_y));
        }
    }
    return 0;
}
