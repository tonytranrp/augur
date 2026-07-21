// examples/11_particle_filter/main.cpp
//
// docs/ROADMAP.md item 4: ParticleFilter tracking the same kind of
// target basic_cv_tracking (example 01) does, so the two are directly
// comparable -- a particle filter is overkill for a plain linear-
// Gaussian problem like this one (KalmanFilter solves it exactly and
// far more cheaply), but it's the easiest way to confirm the particle
// filter is actually working: its weighted-mean estimate should track
// the same detections a Kalman filter would, closely.

#include <cstdio>
#include "augur/augur.hpp"
#include "augur/filters/particle_filter.hpp"

int main() {
    using Scalar = float;
    using CV = augur::models::ConstantVelocity<Scalar, 2>;
    using PF = augur::filters::ParticleFilter<CV, /*MeasDim=*/2, /*NumParticles=*/1000>;

    PF::StateVector x0 = PF::StateVector::Zero();
    PF::StateCovariance P0 = PF::StateCovariance::Identity() * Scalar(10);
    augur::math::Matrix<Scalar, 2, PF::StateDim> H = augur::math::Matrix<Scalar, 2, PF::StateDim>::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    PF::MeasurementCovariance R = PF::MeasurementCovariance::Identity() * Scalar(0.5);

    PF filter{CV{Scalar(1)}, x0, P0,
              [&](const PF::StateVector& x) -> PF::Measurement { return H * x; },
              R, /*seed=*/42};

    const Scalar dt = Scalar(1.0 / 60.0);
    const PF::Measurement detections[] = {
        {0.10f, 0.02f}, {0.18f, -0.05f}, {0.31f, 0.04f}, {0.39f, 0.01f}, {0.52f, -0.03f},
    };

    for (const auto& z : detections) {
        filter.predict(dt);
        filter.update(z);
        const auto& x = filter.state();
        std::printf("pos=(%.3f, %.3f)  vel=(%.3f, %.3f)  likelihood=%.4f\n",
                    x(0), x(1), x(2), x(3), filter.last_likelihood());
    }
    return 0;
}
