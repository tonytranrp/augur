// tests/unit/test_heterogeneous_imm.cpp
//
// Coverage for docs/ROADMAP.md item 0, "different-order IMM mixing"
// (imm/heterogeneous_estimator.hpp, imm/heterogeneous_mixing.hpp,
// imm/augmented_layout.hpp, core/state_component.hpp).
//
// The first three cases hand-verify expand_state/expand_covariance/
// restrict_state/restrict_covariance plus the reused imm::mix() against
// an independently-computed 2-model reference (ad hoc python3 + numpy,
// per .claude/rules/testing.md) -- exactly the check that rule asks for
// before trusting this kind of change. The last case is an
// integration-level robustness check, not a "clean mode-probability
// swing" demo: an ad hoc python3/C++ exploration of a CV+CoordinatedTurn
// mix during a real sustained turn (docs/PRODUCTION_ROADMAP.md P0 item 5)
// found that once CV's mode probability dominates, the big-unknown-
// variance padding (imm/heterogeneous_mixing.hpp) effectively resets
// CoordinatedTurn's turn-rate confidence every mixing cycle, producing a
// noisy (not smoothly-converging) omega estimate. That exploration
// initially also found the COMBINED position estimate going badly wrong
// (even wrong-signed) for several steps right around a mode-probability
// crossover -- traced to the padding constant itself (1e4) being large
// enough to nearly erase a recovering model's own prior in one mixing
// cycle; see imm/heterogeneous_mixing.hpp's file comment for the swept
// fix (100). With that fix, the 20-step window this test actually
// exercises never reaches a crossover, so what it checks -- valid
// probabilities and bounded position error -- holds throughout; a longer
// run that does cross over is exactly what caught the original issue,
// which is why this comment states the finding rather than only the
// clean result.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "augur/augur.hpp"
#include "augur/imm/heterogeneous_estimator.hpp"

using Catch::Matchers::WithinAbs;

namespace {
constexpr float kTol = 1e-3f;
}

TEST_CASE("expand_state/expand_covariance/restrict round-trip through imm::mix matches a hand-computed reference",
          "[imm][heterogeneous]") {
    using CV = augur::models::ConstantVelocity<float, 2>;
    using CT = augur::models::CoordinatedTurn<float>;

    CV::State x_cv;
    x_cv << 0.0f, 0.0f, 1.0f, 0.5f;
    CV::Transition P_cv = CV::Transition::Identity() * 2.0f;

    CT::State x_ct;
    x_ct << 0.1f, -0.1f, 1.1f, 0.4f, 0.05f;
    CT::Transition P_ct = CT::Transition::Identity() * 1.5f;

    augur::imm::ModeMatrix<2, float> Pi{
        (augur::imm::ModeMatrix<2, float>::Matrix() << 0.9f, 0.1f, 0.15f, 0.85f).finished()};
    const std::array<float, 2> mode_p{0.5f, 0.5f};

    const std::array<augur::imm::AugmentedVector<float>, 2> states_aug{
        augur::imm::expand_state<CV>(x_cv),
        augur::imm::expand_state<CT>(x_ct),
    };
    const std::array<augur::imm::AugmentedMatrix<float>, 2> covs_aug{
        augur::imm::expand_covariance<CV>(P_cv),
        augur::imm::expand_covariance<CT>(P_ct),
    };

    const auto mixed = augur::imm::mix<2, augur::imm::kAugmentedDim, float>(states_aug, covs_aug, mode_p, Pi);

    const auto cv_state = augur::imm::restrict_state<CV>(mixed.state[0]);
    const auto ct_state = augur::imm::restrict_state<CT>(mixed.state[1]);
    const auto ct_cov = augur::imm::restrict_covariance<CT>(mixed.covariance[1]);

    // Reference values from an independent numpy computation.
    REQUIRE_THAT(cv_state(0), WithinAbs(0.01428571f, kTol));
    REQUIRE_THAT(cv_state(1), WithinAbs(-0.01428571f, kTol));
    REQUIRE_THAT(cv_state(2), WithinAbs(1.01428571f, kTol));
    REQUIRE_THAT(cv_state(3), WithinAbs(0.48571429f, kTol));

    REQUIRE_THAT(ct_state(0), WithinAbs(0.08947368f, kTol));
    REQUIRE_THAT(ct_state(4), WithinAbs(0.04473684f, kTol)); // mixed omega: pulled toward 0 by CV's "no opinion" padding
    REQUIRE_THAT(ct_cov(4, 4), WithinAbs(11.8686565f, 0.01f)); // stays large (vs CT's own 1.5 prior): CT shouldn't be overconfident about omega here
}

TEST_CASE("HeterogeneousEstimator<ConstantVelocity, CoordinatedTurn> stays numerically sane over a real turn",
          "[imm][heterogeneous][filters][regression]") {
    using CV = augur::models::ConstantVelocity<float, 2>;
    using CT = augur::models::CoordinatedTurn<float>;
    using KFCV = augur::filters::KalmanFilter<CV, /*MeasDim=*/2>;
    using KFCT = augur::filters::KalmanFilter<CT, /*MeasDim=*/2>;

    KFCV::StateVector x0_cv = KFCV::StateVector::Zero();
    x0_cv(2) = 1.5f;
    KFCV::MeasurementMatrix H_cv = KFCV::MeasurementMatrix::Zero();
    H_cv(0, 0) = 1;
    H_cv(1, 1) = 1;
    KFCV::MeasurementCovariance R = KFCV::MeasurementCovariance::Identity() * 0.0004f;
    KFCV kf_cv{CV{0.05f}, x0_cv, KFCV::StateCovariance::Identity() * 2.0f, H_cv, R};

    KFCT::StateVector x0_ct = KFCT::StateVector::Zero();
    x0_ct(2) = 1.5f;
    KFCT::MeasurementMatrix H_ct = KFCT::MeasurementMatrix::Zero();
    H_ct(0, 0) = 1;
    H_ct(1, 1) = 1;
    KFCT kf_ct{CT{0.05f, 2.0f}, x0_ct, KFCT::StateCovariance::Identity() * 2.0f, H_ct, R};

    augur::imm::HeterogeneousEstimator<KFCV, KFCT> tracker{
        std::move(kf_cv), std::move(kf_ct),
        augur::imm::ModeMatrix<2, float>::uniform(0.9f),
    };

    CT truth_model{};
    CT::State truth = KFCT::StateVector::Zero();
    truth(2) = 1.5f;
    const float dt = 1.0f / 30.0f;
    float max_pos_error = 0.0f;

    for (int step = 0; step < 20; ++step) {
        truth(4) = (step < 4) ? 0.0f : 4.0f;
        truth = truth_model.transition(truth, dt);
        tracker.predict(dt);
        tracker.update(KFCT::Measurement{truth(0), truth(1)});

        const auto [x_aug, P_aug] = tracker.combined_state();
        const auto px_idx = static_cast<int>(augur::core::augmented_index_of(augur::core::StateComponent::PositionX));
        const auto py_idx = static_cast<int>(augur::core::augmented_index_of(augur::core::StateComponent::PositionY));

        REQUIRE(std::isfinite(x_aug(px_idx)));
        REQUIRE(std::isfinite(x_aug(py_idx)));
        REQUIRE(P_aug(px_idx, px_idx) > 0.0f);

        const float dx = truth(0) - x_aug(px_idx);
        const float dy = truth(1) - x_aug(py_idx);
        max_pos_error = std::max(max_pos_error, std::sqrt(dx * dx + dy * dy));

        const auto& mode_p = tracker.mode_probability();
        REQUIRE_THAT(mode_p[0] + mode_p[1], WithinAbs(1.0f, 1e-4f));
        REQUIRE(mode_p[0] >= 0.0f);
        REQUIRE(mode_p[1] >= 0.0f);
    }

    // Verified max position error over this exact scenario is ~0.0096
    // (down from ~0.014 before the big_unknown_variance fix above); this
    // bound has generous margin while still catching a real divergence
    // (e.g. a sign error in expand/restrict would blow this up far past 0.1).
    REQUIRE(max_pos_error < 0.1f);
}
