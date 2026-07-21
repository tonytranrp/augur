// tests/unit/test_imm_mixing.cpp
//
// imm/mixing.hpp had zero automated coverage before this file (see
// docs/ROADMAP.md, "Benchmark/validation harness"). The first three
// test cases hand-verify mix()/update_mode_probability()/combine()
// against a tiny 2-model, 2-state numeric example computed
// independently (ad hoc python3 + numpy, per .claude/rules/testing.md)
// -- exactly the check that rule asks for before trusting a change to
// this file. The last test is an integration-level regression check
// that a real IMM mix actually re-weights toward the better-fitting
// model during a sustained maneuver, which only became possible once
// the CoordinatedTurn jacobian bug (docs/ROADMAP.md, "Fixes found along
// the way") was fixed.

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "augur/augur.hpp"

using Catch::Matchers::WithinAbs;

namespace {
constexpr float kTol = 1e-4f;
}

TEST_CASE("imm::mix matches a hand-computed 2-model reference", "[imm][mixing]") {
    using augur::imm::ModeMatrix;
    using augur::imm::mix;

    ModeMatrix<2, float> Pi{(ModeMatrix<2, float>::Matrix() << 0.9f, 0.1f, 0.2f, 0.8f).finished()};
    const std::array<float, 2> mode_p{0.6f, 0.4f};
    const std::array<augur::math::Vector<float, 2>, 2> state{
        augur::math::Vector<float, 2>{1.0f, 2.0f},
        augur::math::Vector<float, 2>{3.0f, 4.0f},
    };
    const std::array<augur::math::Matrix<float, 2>, 2> cov{
        augur::math::Matrix<float, 2>::Identity(),
        augur::math::Matrix<float, 2>::Identity() * 2.0f,
    };

    const auto mixed = mix<2, 2, float>(state, cov, mode_p, Pi);

    // Reference values from an independent numpy computation.
    REQUIRE_THAT(mixed.state[0](0), WithinAbs(1.25806452f, kTol));
    REQUIRE_THAT(mixed.state[0](1), WithinAbs(2.25806452f, kTol));
    REQUIRE_THAT(mixed.state[1](0), WithinAbs(2.68421053f, kTol));
    REQUIRE_THAT(mixed.state[1](1), WithinAbs(3.68421053f, kTol));

    REQUIRE_THAT(mixed.covariance[0](0, 0), WithinAbs(1.578564f, 1e-3f));
    REQUIRE_THAT(mixed.covariance[0](0, 1), WithinAbs(0.44953174f, 1e-3f));
    REQUIRE_THAT(mixed.covariance[1](0, 0), WithinAbs(2.37396122f, 1e-3f));
    REQUIRE_THAT(mixed.covariance[1](0, 1), WithinAbs(0.53185596f, 1e-3f));
}

TEST_CASE("imm::update_mode_probability matches a hand-computed 2-model reference", "[imm][mixing]") {
    using augur::imm::ModeMatrix;
    using augur::imm::update_mode_probability;

    ModeMatrix<2, float> Pi{(ModeMatrix<2, float>::Matrix() << 0.9f, 0.1f, 0.2f, 0.8f).finished()};
    const std::array<float, 2> mode_p{0.6f, 0.4f};
    const std::array<float, 2> likelihood{0.02f, 0.08f}; // model 1 fit far better this step

    const auto updated = update_mode_probability<2, float>(mode_p, Pi, likelihood);

    REQUIRE_THAT(updated[0], WithinAbs(0.28971963f, kTol));
    REQUIRE_THAT(updated[1], WithinAbs(0.71028037f, kTol));
}

TEST_CASE("imm::combine matches a hand-computed 2-model reference", "[imm][mixing]") {
    using augur::imm::combine;

    const std::array<float, 2> mode_p{0.28971963f, 0.71028037f};
    const std::array<augur::math::Vector<float, 2>, 2> state{
        augur::math::Vector<float, 2>{1.1f, 2.2f},
        augur::math::Vector<float, 2>{2.9f, 3.7f},
    };
    const std::array<augur::math::Matrix<float, 2>, 2> cov{
        augur::math::Matrix<float, 2>::Identity() * 0.5f,
        augur::math::Matrix<float, 2>::Identity() * 0.7f,
    };

    const auto [x, P] = combine<2, 2, float>(state, cov, mode_p);

    REQUIRE_THAT(x(0), WithinAbs(2.37850467f, kTol));
    REQUIRE_THAT(x(1), WithinAbs(3.26542056f, kTol));
    REQUIRE_THAT(P(0, 0), WithinAbs(1.30879029f, 1e-3f));
    REQUIRE_THAT(P(0, 1), WithinAbs(0.55561184f, 1e-3f));
    REQUIRE_THAT(P(1, 1), WithinAbs(1.10506594f, 1e-3f));
}

TEST_CASE("imm::Estimator<CoordinatedTurn x3> shifts mode probability toward the sharp-turn model during a real turn",
          "[imm][filters][ct][regression]") {
    // This only works at all because of the CoordinatedTurn jacobian
    // fix (docs/ROADMAP.md, "Fixes found along the way") -- before that
    // fix, all three filters would produce identical likelihoods
    // forever and mode probability could never move off its uniform
    // 1/3 prior. Same configuration as examples/02_imm_maneuvering_target
    // (verified there against an independent Python/numpy reference:
    // "sharp" peaks around 0.42 mid-turn and settles back to ~0.38).
    using CT = augur::models::CoordinatedTurn<float>;
    using KF = augur::filters::KalmanFilter<CT, /*MeasDim=*/2>;

    KF::StateVector x0 = KF::StateVector::Zero();
    x0(2) = 1.5f;
    KF::StateCovariance P0 = KF::StateCovariance::Identity() * 2.0f;
    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    KF::MeasurementCovariance R = KF::MeasurementCovariance::Identity() * 0.0004f;

    augur::imm::Estimator<KF, KF, KF> tracker{
        KF{CT{0.05f, 0.001f}, x0, P0, H, R}, // calm
        KF{CT{0.05f, 0.2f}, x0, P0, H, R},   // juking
        KF{CT{0.05f, 8.0f}, x0, P0, H, R},   // sharp turn
        augur::imm::ModeMatrix<3, float>::uniform(0.90f),
    };

    CT truth_model{};
    CT::State truth = x0;
    const float dt = 1.0f / 30.0f;
    float peak_sharp_probability = 0.0f;

    for (int step = 0; step < 20; ++step) {
        truth(4) = (step < 4) ? 0.0f : 4.0f; // straight, then a real sustained 4 rad/s turn
        truth = truth_model.transition(truth, dt);
        tracker.predict(dt);
        tracker.update(KF::Measurement{truth(0), truth(1)});
        peak_sharp_probability = std::max(peak_sharp_probability, tracker.mode_probability()[2]);
    }

    // Verified peak (Python/numpy reference) is ~0.42; this leaves
    // generous margin while still failing hard if the bug regresses
    // (peak would stay exactly at the 1/3 uniform prior).
    REQUIRE(peak_sharp_probability > 0.38f);
}
