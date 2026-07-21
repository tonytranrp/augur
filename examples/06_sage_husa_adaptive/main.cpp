// examples/06_sage_husa_adaptive/main.cpp
//
// docs/ROADMAP.md item 2: SageHusaAdaptive wraps a KalmanFilter and
// estimates measurement noise (R) online from the innovation sequence,
// instead of requiring a hand-tuned constant. The sensor here starts
// precise and abruptly gets much noisier partway through -- a
// hand-tuned filter would keep trusting its original (now wrong) R;
// this one adapts.
//
// See filters/adaptive/sage_husa.hpp's file comment for the scope this
// is built for (H observes position directly, the convention every
// example in this library already follows) and the honest limitation
// found while testing it: without the eigenvalue-flooring fix
// (math::project_to_psd()) the adaptive estimate can lose positive-
// semi-definiteness and diverge, which is exactly the well-documented
// failure mode of the vanilla 1969 Sage-Husa formulation.

#include <cstdio>
#include <random>
#include "augur/augur.hpp"
#include "augur/filters/adaptive/sage_husa.hpp"

int main() {
    using Scalar = float;
    using CV = augur::models::ConstantVelocity<Scalar, 1>;
    using KF = augur::filters::KalmanFilter<CV, /*MeasDim=*/1>;

    KF::MeasurementMatrix H = KF::MeasurementMatrix::Identity();
    KF::MeasurementCovariance R0 = KF::MeasurementCovariance::Identity() * Scalar(0.05);
    KF inner{CV{Scalar(0.01)}, KF::StateVector::Zero(), KF::StateCovariance::Identity(), H, R0};

    augur::filters::SageHusaAdaptive<KF, 1> filter{
        std::move(inner), R0, KF::StateCovariance::Identity() * Scalar(0.01), Scalar(0.95)};

    const Scalar dt = Scalar(1.0 / 30.0);
    Scalar truth_p = 0, truth_v = Scalar(0.5);
    std::mt19937 rng{42};

    for (int step = 0; step < 200; ++step) {
        const Scalar true_r = (step < 100) ? Scalar(0.05) : Scalar(2.0); // sensor gets much noisier at step 100
        std::normal_distribution<Scalar> noise(Scalar(0), std::sqrt(true_r));
        truth_p += truth_v * dt;
        const Scalar z = truth_p + noise(rng);

        filter.predict(dt);
        filter.update(KF::Measurement{z});

        if (step % 20 == 19 || step == 99 || step == 100) {
            std::printf("step %3d (true_r=%.2f): pos_est=%.3f  R_hat=%.4f\n",
                        step, true_r, filter.state()(0), filter.measurement_noise_estimate()(0, 0));
        }
    }
    return 0;
}
