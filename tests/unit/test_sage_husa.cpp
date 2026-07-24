// tests/unit/test_sage_husa.cpp
//
// Coverage for docs/ROADMAP.md item 2 ("Adaptive / self-tuning process
// noise", filters/sage_husa.hpp). The scenario in the first two
// cases mirrors what was verified in Python before any of this was
// written in C++ (per .claude/rules/testing.md): a sensor/process that
// changes its true noise level partway through a run, checking that the
// adaptive estimate tracks the change and -- the actual point, given the
// vanilla 1969 formulation's well-documented divergence failure mode --
// that the estimate stays positive-semi-definite and the filter never
// diverges throughout.

#include <random>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "augur/augur.hpp"
#include "augur/filters/sage_husa.hpp"

using Catch::Matchers::WithinAbs;

namespace {

template <typename Matrix>
bool is_psd_n(const Matrix& m, float tol = -1e-6f) {
    Eigen::SelfAdjointEigenSolver<Matrix> solver(m);
    return solver.eigenvalues().minCoeff() >= tol;
}

} // namespace

TEST_CASE("SageHusaAdaptive::measurement_noise_estimate tracks a real change in sensor noise without diverging",
          "[filters][adaptive][sage_husa][regression]") {
    using CV = augur::models::ConstantVelocity<float, 1>;
    using KF = augur::filters::KalmanFilter<CV, /*MeasDim=*/1>;

    KF::MeasurementMatrix H = KF::MeasurementMatrix::Identity();
    KF::MeasurementCovariance R0 = KF::MeasurementCovariance::Identity() * 0.05f;
    KF inner{CV{0.01f}, KF::StateVector::Zero(), KF::StateCovariance::Identity(), H, R0};

    augur::filters::SageHusaAdaptive<KF, 1> filter{
        std::move(inner), R0, KF::StateCovariance::Identity() * 0.01f, 0.95f};

    const float dt = 1.0f / 30.0f;
    float truth_p = 0.0f, truth_v = 0.5f;
    std::mt19937 rng{42};

    for (int step = 0; step < 200; ++step) {
        const float true_r = (step < 100) ? 0.05f : 2.0f; // sensor gets much noisier halfway through
        std::normal_distribution<float> noise(0.0f, std::sqrt(true_r));
        truth_p += truth_v * dt;
        const float z = truth_p + noise(rng);

        filter.predict(dt);
        filter.update(KF::Measurement{z});

        REQUIRE(is_psd_n(filter.measurement_noise_estimate()));
        REQUIRE(is_psd_n(filter.process_noise_estimate()));
        REQUIRE(std::isfinite(filter.state()(0)));
        REQUIRE(std::isfinite(filter.covariance()(0, 0)));
    }

    // The estimate should have moved meaningfully toward the new,
    // higher true noise level -- not stayed pinned near the old one.
    REQUIRE(filter.measurement_noise_estimate()(0, 0) > 0.3f);
}

TEST_CASE("SageHusaAdaptive with forgetting_factor=1 never updates Q/R (sanity check on the update formula)",
          "[filters][adaptive][sage_husa]") {
    using CV = augur::models::ConstantVelocity<float, 1>;
    using KF = augur::filters::KalmanFilter<CV, /*MeasDim=*/1>;

    KF::MeasurementMatrix H = KF::MeasurementMatrix::Identity();
    KF::MeasurementCovariance R0 = KF::MeasurementCovariance::Identity() * 0.2f;
    KF::StateCovariance Q0 = KF::StateCovariance::Identity() * 0.05f;
    KF inner{CV{0.01f}, KF::StateVector::Zero(), KF::StateCovariance::Identity(), H, R0};

    augur::filters::SageHusaAdaptive<KF, 1> filter{std::move(inner), R0, Q0, /*forgetting_factor=*/1.0f};

    std::mt19937 rng{7};
    std::normal_distribution<float> noise(0.0f, 0.5f);
    for (int step = 0; step < 30; ++step) {
        filter.predict(1.0f / 30.0f);
        filter.update(KF::Measurement{noise(rng)});
    }

    REQUIRE_THAT(filter.measurement_noise_estimate()(0, 0), WithinAbs(0.2f, 1e-4f));
    REQUIRE_THAT(filter.process_noise_estimate()(0, 0), WithinAbs(0.05f, 1e-4f));
}

TEST_CASE("SageHusaAdaptive stays stable and PSD through an abrupt process-noise change", "[filters][adaptive][sage_husa]") {
    using CV = augur::models::ConstantVelocity<float, 1>;
    using KF = augur::filters::KalmanFilter<CV, /*MeasDim=*/1>;

    KF::MeasurementMatrix H = KF::MeasurementMatrix::Identity();
    KF::MeasurementCovariance R0 = KF::MeasurementCovariance::Identity() * 0.05f;
    KF inner{CV{0.01f}, KF::StateVector::Zero(), KF::StateCovariance::Identity(), H, R0};
    augur::filters::SageHusaAdaptive<KF, 1> filter{
        std::move(inner), R0, KF::StateCovariance::Identity() * 0.01f, 0.9f};

    const float dt = 1.0f / 30.0f;
    float truth_p = 0.0f, truth_v = 0.5f;
    std::mt19937 rng{99};
    std::normal_distribution<float> meas_noise(0.0f, std::sqrt(0.05f));

    for (int step = 0; step < 150; ++step) {
        // an unmodeled maneuver kicks in partway through: real velocity
        // starts jittering hard, well beyond what Q0 assumed.
        if (step >= 75) {
            std::normal_distribution<float> jitter(0.0f, 2.0f);
            truth_v += jitter(rng) * dt;
        }
        truth_p += truth_v * dt;
        filter.predict(dt);
        filter.update(KF::Measurement{truth_p + meas_noise(rng)});

        REQUIRE(is_psd_n(filter.measurement_noise_estimate()));
        REQUIRE(is_psd_n(filter.process_noise_estimate()));
        REQUIRE(std::isfinite(filter.state()(0)));
    }
}
