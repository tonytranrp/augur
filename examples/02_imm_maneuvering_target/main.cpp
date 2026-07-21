// examples/02_imm_maneuvering_target/main.cpp
//
// Mixes three same-dimension CoordinatedTurn instances -- "calm",
// "juking", and "sharp turn" -- tuned only by how much turn-rate
// process noise each expects. This sidesteps the different-order
// mixing limitation documented in imm/estimator.hpp (a true CV+CA+CT
// mix needs the common-space mapping machinery from Bar-Shalom ch.
// 11.6, which is a documented near-term roadmap item, not implemented
// in this v0.1) while still demonstrating the actual point of IMM:
// watch mode_probability() shift toward "sharp turn" once the
// synthetic target commits to a sustained hard turn, and notice the
// combined estimate track the curve far better than any single regime
// would.
//
// The detections below trace a real coordinated-turn trajectory (omega
// jumps from 0 to 4 rad/s for 10 steps, then back to 0) with small,
// fixed measurement noise added -- generated once via an ad hoc python3
// command per .claude/rules/testing.md, not saved as a project file, so
// this example stays dependency-free and fully deterministic. Earlier
// parameters here (looser noise, a much milder synthetic turn) produced
// a mode-probability shift too small to see at 2 decimal places even
// after models/coordinated_turn.hpp's jacobian() was fixed to actually
// let the turn-rate state become observable -- these parameters were
// retuned so the effect the comment above describes is actually visible.

#include <cstdio>
#include "augur/augur.hpp"

int main() {
    using Scalar = float;
    using CT = augur::models::CoordinatedTurn<Scalar>;
    using KF = augur::filters::KalmanFilter<CT, /*MeasDim=*/2>;

    KF::StateVector x0 = KF::StateVector::Zero();
    x0(2) = Scalar(1.5); // initial vx
    KF::StateCovariance P0 = KF::StateCovariance::Identity() * Scalar(2);

    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    KF::MeasurementCovariance R = KF::MeasurementCovariance::Identity() * Scalar(0.0004);

    augur::imm::Estimator<KF, KF, KF> tracker{
        KF{CT{/*q_pos=*/Scalar(0.05), /*q_turn=*/Scalar(0.001)}, x0, P0, H, R}, // calm
        KF{CT{/*q_pos=*/Scalar(0.05), /*q_turn=*/Scalar(0.2)},  x0, P0, H, R},  // juking
        KF{CT{/*q_pos=*/Scalar(0.05), /*q_turn=*/Scalar(8.0)},  x0, P0, H, R},  // sharp turn
        augur::imm::ModeMatrix<3, Scalar>::uniform(Scalar(0.90))
    };

    const Scalar dt = Scalar(1.0 / 30.0);
    // Straight for 4 steps, a real sustained 4 rad/s turn for 10 steps,
    // then straight again for 6.
    const KF::Measurement detections[] = {
        {0.0500f, 0.0060f},   // straight
        {0.0945f, -0.0178f},
        {0.1409f, -0.0198f},
        {0.2012f, 0.0268f},
        {0.2400f, -0.0091f},  // turn begins
        {0.3086f, 0.0204f},
        {0.3481f, 0.0110f},
        {0.3901f, 0.0660f},
        {0.4050f, 0.0711f},
        {0.4310f, 0.0879f},
        {0.4645f, 0.1471f},
        {0.5030f, 0.1993f},
        {0.5526f, 0.2354f},
        {0.5141f, 0.2760f},
        {0.5753f, 0.3376f},   // straight again
        {0.5574f, 0.3744f},
        {0.5802f, 0.4164f},
        {0.6327f, 0.4650f},
        {0.6226f, 0.5475f},
        {0.6234f, 0.5761f},
    };

    for (const auto& z : detections) {
        tracker.predict(dt);
        tracker.update(z);
        const auto [x, P] = tracker.combined_state();
        (void)P;
        const auto& mode = tracker.mode_probability();
        std::printf("pos=(%.3f,%.3f) modes[calm=%.2f juke=%.2f sharp=%.2f]\n",
                    x(0), x(1), mode[0], mode[1], mode[2]);
    }
    return 0;
}
