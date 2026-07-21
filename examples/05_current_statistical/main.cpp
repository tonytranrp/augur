// examples/05_current_statistical/main.cpp
//
// docs/ROADMAP.md item 1: the Current Statistical (CS) model side by
// side with plain Singer, tracking a target that commits to a real,
// SUSTAINED acceleration (not a brief jitter) -- the scenario CS is
// supposed to handle better, because Singer always decays its
// acceleration estimate back toward zero regardless of how long the
// maneuver actually continues.
//
// Honest note (see models/current_statistical.hpp and
// filters/current_statistical_filter.hpp for the full story, found via
// ad hoc python3 verification per .claude/rules/testing.md before any
// of this was implemented): CS's adaptive mean is damped and clamped
// because an undamped/unclamped version provably diverges without
// bound. With that guard in place, CS's acceleration estimate still
// tends to drift toward its configured clamp rather than settling
// tightly on the true value -- so what this example actually shows is
// "CS's position tracking keeps pace with a sustained maneuver at
// least as well as Singer's, without Singer's built-in bias back
// toward zero," not "CS estimates acceleration perfectly." Tune
// adaptation_rate/max_mean_accel to your own application before
// trusting this beyond that.

#include <cstdio>
#include "augur/augur.hpp"
#include "augur/filters/current_statistical_filter.hpp"
#include "augur/models/current_statistical.hpp"

int main() {
    using Scalar = float;
    using Singer = augur::models::Singer<Scalar, 1>;
    using CS = augur::models::CurrentStatistical<Scalar, 1>;
    using KFSinger = augur::filters::KalmanFilter<Singer, /*MeasDim=*/1>;
    using KFCS = augur::filters::KalmanFilter<CS, /*MeasDim=*/1>;

    KFSinger::MeasurementMatrix H = KFSinger::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    KFSinger::MeasurementCovariance R = KFSinger::MeasurementCovariance::Identity() * Scalar(0.01);

    KFSinger singer_filter{Singer{Scalar(2.0), Scalar(1.0)}, KFSinger::StateVector::Zero(),
                            KFSinger::StateCovariance::Identity() * Scalar(2), H, R};

    KFCS cs_inner{CS{Scalar(2.0), Scalar(1.0)}, KFCS::StateVector::Zero(),
                  KFCS::StateCovariance::Identity() * Scalar(2), H, R};
    augur::filters::CurrentStatisticalFilter<KFCS> cs_filter{std::move(cs_inner), Scalar(0.05), Scalar(12)};

    const Scalar dt = Scalar(1.0 / 30.0);
    const Scalar true_accel = Scalar(3.0);
    Scalar truth_p = 0, truth_v = 0;

    for (int step = 0; step < 60; ++step) {
        const Scalar a = (step < 10) ? Scalar(0) : true_accel; // calm, then a real sustained maneuver
        truth_v += a * dt;
        truth_p += truth_v * dt;

        singer_filter.predict(dt);
        singer_filter.update(KFSinger::Measurement{truth_p});
        cs_filter.predict(dt);
        cs_filter.update(KFCS::Measurement{truth_p});

        if (step % 10 == 9 || step == 59) {
            std::printf("step %2d: truth_p=%.3f | Singer p=%.3f a_est=%.3f | CS p=%.3f a_est=%.3f mean_a=%.3f\n",
                        step, truth_p,
                        singer_filter.state()(0), singer_filter.state()(2),
                        cs_filter.state()(0), cs_filter.state()(2), cs_filter.model().mean_acceleration()(0));
        }
    }
    return 0;
}
