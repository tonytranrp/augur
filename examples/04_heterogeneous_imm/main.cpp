// examples/04_heterogeneous_imm/main.cpp
//
// The "textbook full IMM example" docs/ROADMAP.md names as the goal for
// item 0 ("different-order IMM mixing"): mixing ConstantVelocity (4-dim),
// ConstantAcceleration (6-dim), and CoordinatedTurn (5-dim) together in
// one IMM, despite their three different state dimensions -- something
// augur::imm::Estimator can't do (it requires every mixed filter to
// share a dimension; see docs/ARCHITECTURE.md §5). This uses
// augur::imm::HeterogeneousEstimator instead, which mixes by expanding
// every filter's native state into a shared 7-component augmented space
// (core/state_component.hpp) and restricting back down before each
// filter's own predict().
//
// Watch the "modes[...]" column as this runs: uniform for the first few
// straight-line steps, then CT's probability climbs steadily (crossing
// 0.5 partway through) as the sustained turn's evidence accumulates,
// while CV's declines correspondingly -- IMM correctly recognizing the
// true motion mode as it becomes evident, not switching abruptly. The
// combined position estimate tracks the true trajectory tightly
// throughout regardless of which model currently dominates.
//
// Tuning history, stated honestly rather than left implicit: an earlier
// version of imm/heterogeneous_mixing.hpp's big_unknown_variance() padding
// constant (1e4, since fixed to 100 -- see that file's comment,
// docs/PRODUCTION_ROADMAP.md P0 item 5) was large enough that mixing a
// dominant model's padded "no information" belief into a non-dominant
// model could nearly erase that model's own prior in a single cycle --
// this example's own mode probabilities used to snap almost to 1.0/0.0
// mid-turn and the combined estimate briefly went wrong-signed right
// around the crossover. The current constant keeps that from happening
// at the cost of a real, smaller, still-present characteristic: a
// non-dominant model's own sub-estimate for a state the others don't
// share (e.g. CoordinatedTurn's turn rate while CV still dominates) gets
// partly reset toward the dominant model's belief every mixing cycle, so
// it can be noisier than a standalone filter's would be. See
// tests/unit/test_heterogeneous_imm.cpp for both findings turned into
// permanent regression checks.

#include <cstdio>
#include "augur/augur.hpp"
#include "augur/imm/heterogeneous_estimator.hpp"

int main() {
    using Scalar = float;
    using CV = augur::models::ConstantVelocity<Scalar, 2>;
    using CA = augur::models::ConstantAcceleration<Scalar, 2>;
    using CT = augur::models::CoordinatedTurn<Scalar>;
    using KFCV = augur::filters::KalmanFilter<CV, /*MeasDim=*/2>;
    using KFCA = augur::filters::KalmanFilter<CA, /*MeasDim=*/2>;
    using KFCT = augur::filters::KalmanFilter<CT, /*MeasDim=*/2>;

    KFCV::MeasurementMatrix H_cv = KFCV::MeasurementMatrix::Zero();
    H_cv(0, 0) = 1;
    H_cv(1, 1) = 1;
    KFCV::MeasurementCovariance R = KFCV::MeasurementCovariance::Identity() * Scalar(0.0004);

    KFCV::StateVector x0_cv = KFCV::StateVector::Zero();
    x0_cv(2) = Scalar(1.5);
    KFCV kf_cv{CV{Scalar(0.05)}, x0_cv, KFCV::StateCovariance::Identity() * Scalar(2), H_cv, R};

    KFCA::MeasurementMatrix H_ca = KFCA::MeasurementMatrix::Zero();
    H_ca(0, 0) = 1;
    H_ca(1, 1) = 1;
    KFCA::StateVector x0_ca = KFCA::StateVector::Zero();
    x0_ca(2) = Scalar(1.5);
    KFCA kf_ca{CA{Scalar(0.05)}, x0_ca, KFCA::StateCovariance::Identity() * Scalar(2), H_ca, R};

    KFCT::MeasurementMatrix H_ct = KFCT::MeasurementMatrix::Zero();
    H_ct(0, 0) = 1;
    H_ct(1, 1) = 1;
    KFCT::StateVector x0_ct = KFCT::StateVector::Zero();
    x0_ct(2) = Scalar(1.5);
    KFCT kf_ct{CT{Scalar(0.05), Scalar(2.0)}, x0_ct, KFCT::StateCovariance::Identity() * Scalar(2), H_ct, R};

    augur::imm::HeterogeneousEstimator<KFCV, KFCA, KFCT> tracker{
        std::move(kf_cv), std::move(kf_ca), std::move(kf_ct),
        augur::imm::ModeMatrix<3, Scalar>::uniform(Scalar(0.9)),
    };

    // Ground truth: straight, then a real sustained 4 rad/s turn --
    // generated with CoordinatedTurn's own transition() so the "truth"
    // and the tracker's own CT model agree on the dynamics.
    CT truth_model{};
    CT::State truth = CT::State::Zero();
    truth(2) = Scalar(1.5);
    const Scalar dt = Scalar(1.0 / 30.0);

    const auto px_idx = static_cast<int>(augur::core::augmented_index_of(augur::core::StateComponent::PositionX));
    const auto py_idx = static_cast<int>(augur::core::augmented_index_of(augur::core::StateComponent::PositionY));

    for (int step = 0; step < 20; ++step) {
        truth(4) = (step < 4) ? Scalar(0) : Scalar(4.0);
        truth = truth_model.transition(truth, dt);

        tracker.predict(dt);
        tracker.update(KFCT::Measurement{truth(0), truth(1)});

        const auto [x_aug, P] = tracker.combined_state();
        (void)P;
        const auto& mode = tracker.mode_probability();
        std::printf("truth=(%.3f,%.3f) combined=(%.3f,%.3f) modes[cv=%.2f ca=%.2f ct=%.2f]\n",
                    truth(0), truth(1), x_aug(px_idx), x_aug(py_idx), mode[0], mode[1], mode[2]);
    }
    return 0;
}
