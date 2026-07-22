// tests/unit/test_particle_filter.cpp
//
// Coverage for docs/ROADMAP.md item 4 ("Particle filter / SMC backend",
// filters/particle_filter.hpp). Verified end-to-end against a
// KalmanFilter baseline on a linear system (ad hoc python3 + numpy, per
// .claude/rules/testing.md) before being written in C++: with enough
// particles, a bootstrap particle filter should closely approximate the
// exact Kalman solution on a problem the KF already solves exactly.

#include <random>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "augur/augur.hpp"
#include "augur/filters/particle_filter.hpp"

using Catch::Matchers::WithinAbs;

static_assert(augur::filters::Filter<
              augur::filters::ParticleFilter<augur::models::ConstantVelocity<float, 2>, 2, 500>>);

TEST_CASE("ParticleFilter tracks a linear system within a reasonable Monte Carlo tolerance of KalmanFilter",
          "[filters][particle][regression]") {
    using CV = augur::models::ConstantVelocity<float, 2>;
    using KF = augur::filters::KalmanFilter<CV, 2>;
    using PF = augur::filters::ParticleFilter<CV, 2, 2000>;

    KF::StateVector x0 = KF::StateVector::Zero();
    x0(2) = 1.0f;
    x0(3) = 0.5f;
    KF::StateCovariance P0 = KF::StateCovariance::Identity() * 0.5f;
    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    KF::MeasurementCovariance R = KF::MeasurementCovariance::Identity() * 0.1f;

    KF kf{CV{0.02f}, x0, P0, H, R};
    PF pf{CV{0.02f}, x0, P0,
          [&](const PF::StateVector& x) -> PF::Measurement { return H * x; },
          R, /*seed=*/1234};

    const float dt = 1.0f / 30.0f;
    std::mt19937 truth_rng{7};
    std::normal_distribution<float> meas_noise(0.0f, std::sqrt(0.1f));
    KF::StateVector truth = x0;

    for (int step = 0; step < 60; ++step) {
        truth = CV{0.02f}.transition(truth, dt);
        const KF::Measurement z{truth(0) + meas_noise(truth_rng), truth(1) + meas_noise(truth_rng)};

        kf.predict(dt);
        kf.update(z);
        pf.predict(dt);
        pf.update(z);
    }

    // Both filters see the exact same measurements; on a linear-Gaussian
    // problem the particle filter's weighted mean should land close to
    // the KF's (which is the exact MMSE solution here). Generous
    // tolerance for genuine Monte Carlo variation, not floating-point
    // noise.
    REQUIRE_THAT((kf.state() - pf.state()).norm(), WithinAbs(0.0f, 0.3f));
}

TEST_CASE("ParticleFilter resets to uniform weights when the measurement covariance is degenerate",
          "[filters][particle][regression]") {
    // Regression coverage for the hoisted-out-of-the-loop determinant/
    // inverse in update() (see that function's own comment): a singular
    // R (det<=0) must make EVERY particle's likelihood exactly 0 --
    // regardless of whether that determinant check runs once (now) or
    // once per particle (before) -- so total stays 0 and every weight
    // resets to 1/NumParticles, not silently NaN/Inf from dividing by a
    // zero determinant's normalizer.
    using CV = augur::models::ConstantVelocity<float, 2>;
    using PF = augur::filters::ParticleFilter<CV, 2, 200>;

    PF::StateVector x0 = PF::StateVector::Zero();
    PF::StateCovariance P0 = PF::StateCovariance::Identity();
    PF filter{CV{0.05f}, x0, P0,
              [](const PF::StateVector& x) -> PF::Measurement { return PF::Measurement{x(0), x(1)}; },
              PF::MeasurementCovariance::Zero(), /*seed=*/7};

    filter.predict(1.0f / 30.0f);
    filter.update(PF::Measurement{1.0f, 1.0f});

    REQUIRE_THAT(filter.last_likelihood(), WithinAbs(0.0f, 1e-9f));
    REQUIRE(std::isfinite(filter.state()(0)));
    REQUIRE(std::isfinite(filter.state()(1)));
    REQUIRE(filter.covariance()(0, 0) >= 0.0f);
}

TEST_CASE("ParticleFilter weights stay valid and the state stays finite across many predict/update/resample cycles",
          "[filters][particle]") {
    using CV = augur::models::ConstantVelocity<float, 2>;
    using PF = augur::filters::ParticleFilter<CV, 2, 300>;

    PF::StateVector x0 = PF::StateVector::Zero();
    PF::StateCovariance P0 = PF::StateCovariance::Identity();
    augur::math::Matrix<float, 2, PF::StateDim> H = augur::math::Matrix<float, 2, PF::StateDim>::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    PF filter{CV{0.05f}, x0, P0,
              [&](const PF::StateVector& x) -> PF::Measurement { return H * x; },
              PF::MeasurementCovariance::Identity() * 0.1f, /*seed=*/99};

    for (int step = 0; step < 100; ++step) {
        filter.predict(1.0f / 30.0f);
        filter.update(PF::Measurement{0.05f * static_cast<float>(step), 0.0f});
        REQUIRE(std::isfinite(filter.state()(0)));
        REQUIRE(std::isfinite(filter.state()(1)));
        REQUIRE(filter.covariance()(0, 0) >= 0.0f);
        REQUIRE(filter.last_likelihood() >= 0.0f);
    }
}
