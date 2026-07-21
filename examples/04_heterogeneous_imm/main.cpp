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
// Honest note on what this demo shows and doesn't: an ad hoc python3
// exploration (per .claude/rules/testing.md) of this same scenario found
// that once one model's mode probability comes to dominate, the
// "large-but-finite unknown variance" padding used for components other
// models don't track (imm/heterogeneous_mixing.hpp) effectively resets
// the OTHER models' confidence about their own unique states (e.g.
// CoordinatedTurn's turn rate) every mixing cycle -- so per-model
// sub-estimates for those unique states can be noisy rather than
// smoothly convergent. What stays reliable, and what this example
// actually demonstrates, is the COMBINED position/velocity estimate:
// it tracks the true trajectory well throughout, and mode probabilities
// remain valid, finite, and sum to 1 regardless of which model
// currently dominates. See tests/unit/test_heterogeneous_imm.cpp for
// the same finding turned into a permanent regression check.

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
