// examples/12_out_of_sequence/main.cpp
//
// docs/ROADMAP.md item 8: a detection that crosses an unreliable
// network link and arrives late -- z2's true timestamp is between z1's
// and z3's, but it doesn't show up until AFTER z3 has already been
// processed. track/out_of_sequence.hpp's retrodiction rolls the filter
// back and reprocesses everything since in the correct order, instead
// of either discarding z2 or misapplying it as if it were a fresh
// reading at the current time.
//
// Honest note (see tests/unit/test_out_of_sequence.cpp for the numbers):
// for this gentle, low-process-noise trajectory, discarding z2 turns
// out to be fairly benign on its own -- the model's own prediction
// already interpolates the gap well. Misapplying it (predict-by-zero
// then update, treating stale data as fresh) is the one that visibly
// corrupts the estimate. Retrodiction gets the right answer either way,
// without having to reason about which failure mode you'd hit.

#include <cstdio>
#include "augur/augur.hpp"
#include "augur/track/out_of_sequence.hpp"

int main() {
    using Scalar = float;
    using CV = augur::models::ConstantVelocity<Scalar, 2>;
    using KF = augur::filters::KalmanFilter<CV, /*MeasDim=*/2>;
    using OOSM = augur::track::OutOfSequenceEstimator<KF, /*MaxHistory=*/8>;

    KF::StateVector x0 = KF::StateVector::Zero();
    x0(2) = 1.0f;
    x0(3) = 0.5f;
    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    KF kf{CV{0.01f}, x0, KF::StateCovariance::Identity(), H, KF::MeasurementCovariance::Identity() * 0.05f};

    const Scalar dt = Scalar(1.0 / 30.0);
    const KF::Measurement z1{0.033f, 0.017f};
    const KF::Measurement z2{0.067f, 0.033f}; // true timestamp: 2*dt -- arrives late
    const KF::Measurement z3{0.100f, 0.050f}; // true timestamp: 3*dt -- arrives on time, before z2

    OOSM est{std::move(kf)};
    est.step(dt, z1);
    std::printf("after z1 (t=%.3f):        pos=(%.4f,%.4f)\n", dt, est.state()(0), est.state()(1));

    est.step(2 * dt, z3); // z2 hasn't arrived yet
    std::printf("after z3, z2 still lost:  pos=(%.4f,%.4f)\n", est.state()(0), est.state()(1));

    const bool accepted = est.insert_out_of_sequence(2 * dt, z2);
    std::printf("z2 finally arrives (accepted=%s): pos=(%.4f,%.4f)\n",
                accepted ? "true" : "false", est.state()(0), est.state()(1));
    return 0;
}
