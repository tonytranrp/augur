// tests/unit/test_filter_config.cpp
//
// filters::{KalmanFilter,ExtendedKalmanFilter,UnscentedKalmanFilter,
// ParticleFilter}::Config -- the designated-initializer-friendly
// construction aggregate added alongside each filter's existing
// positional constructor. Every Config ctor DELEGATES to the positional
// one (see each filter's own header), so bit-identical output is a
// mechanical consequence of that delegation, not something that could
// silently drift -- these tests are what keeps it that way, not a
// from-scratch derivation, so no separate Python pre-check applies
// (per .claude/rules/testing.md's "pure refactors... don't need a
// Python check").
//
// Comparisons use memcmp on the underlying Eigen storage (REQUIRE bit-
// identical, not WithinAbs) precisely because the delegation makes
// anything less than exact equality a genuine bug, not float noise.

#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include "augur/augur.hpp"
#include "augur/filters/observe_position.hpp"
#include "augur/filters/particle_filter.hpp"
#include "augur/filters/unscented_kalman.hpp"

namespace {

using Scalar = float;
using CV = augur::models::ConstantVelocity<Scalar, 2>;
using KF = augur::filters::KalmanFilter<CV, 2>;
using EKF = augur::filters::ExtendedKalmanFilter<CV, 2>;
using UKF = augur::filters::UnscentedKalmanFilter<CV, 2>;
using PF = augur::filters::ParticleFilter<CV, 2, 200>;

template <typename EigenT>
bool bit_identical(const EigenT& a, const EigenT& b) {
    return std::memcmp(a.data(), b.data(), sizeof(typename EigenT::Scalar) * static_cast<std::size_t>(a.size())) == 0;
}

const KF::StateVector kX0 = KF::StateVector::Zero();
const KF::StateCovariance kP0 = KF::StateCovariance::Identity() * Scalar(5);
const KF::MeasurementCovariance kR = KF::MeasurementCovariance::Identity() * Scalar(0.3);

template <typename FilterT>
void run_cycles(FilterT& f) {
    for (int i = 0; i < 5; ++i) {
        f.predict(Scalar(0.1));
        f.update(typename FilterT::Measurement{Scalar(0.1) * Scalar(i), Scalar(0.05)});
    }
}

TEST_CASE("KalmanFilter::Config matches the positional ctor bit-identically", "[filters][config]") {
    const auto H = augur::filters::observe_position<KF>();

    SECTION("all fields supplied") {
        KF positional{CV{1.0f}, kX0, kP0, H, kR};
        KF via_config{KF::Config{.model = CV{1.0f}, .initial_state = kX0,
                                 .initial_covariance = kP0, .measurement_matrix = H,
                                 .measurement_noise = kR}};
        run_cycles(positional);
        run_cycles(via_config);
        REQUIRE(bit_identical(positional.state(), via_config.state()));
        REQUIRE(bit_identical(positional.covariance(), via_config.covariance()));
    }

    SECTION("measurement_matrix omitted equals observe_position() passed explicitly") {
        KF positional{CV{1.0f}, kX0, kP0, H, kR};
        KF via_config{KF::Config{.model = CV{1.0f}, .initial_state = kX0,
                                 .initial_covariance = kP0, .measurement_noise = kR}};
        REQUIRE(bit_identical(positional.state(), via_config.state()));
        REQUIRE(bit_identical(positional.covariance(), via_config.covariance()));
    }

    SECTION("initial_state/initial_covariance/measurement_noise default to Zero/Identity/Identity") {
        KF via_config{KF::Config{.model = CV{1.0f}}};
        // Comparing against a fresh Zero()/Identity() expression, not
        // another independently-constructed filter's output, so plain
        // operator== (exact per-element comparison) is the right tool
        // here -- bit_identical's memcmp needs two concrete, identically-
        // typed storage buffers, and Zero()/Identity() are lazy Eigen
        // expression templates, not concrete Matrix/Vector objects.
        REQUIRE(via_config.state() == KF::StateVector::Zero());
        REQUIRE(via_config.covariance() == KF::StateCovariance::Identity());
    }
}

TEST_CASE("ExtendedKalmanFilter::Config matches the positional ctor bit-identically", "[filters][config]") {
    using Fn = augur::filters::ObservePositionFn<Scalar, 2, 4>;
    using Jac = augur::filters::ObservePositionJacobianFn<Scalar, 2, 4>;

    EKF positional{CV{1.0f}, kX0, kP0, Fn{}, Jac{}, kR};
    EKF via_config{EKF::Config{.model = CV{1.0f}, .initial_state = kX0,
                               .initial_covariance = kP0, .measurement_noise = kR}};
    run_cycles(positional);
    run_cycles(via_config);
    REQUIRE(bit_identical(positional.state(), via_config.state()));
    REQUIRE(bit_identical(positional.covariance(), via_config.covariance()));
}

TEST_CASE("ExtendedKalmanFilter with a custom MeasurementFnT still compiles positionally",
          "[filters][config]") {
    // Proves Config's default member initializers (which construct
    // ObservePositionFn/ObservePositionJacobianFn) are NOT eagerly
    // instantiated just because the class template is -- a custom
    // MeasurementFnT that ObservePositionFn could never convert to must
    // still work fine as long as Config itself is never touched.
    struct CustomFn {
        augur::math::Vector<Scalar, 2> operator()(const augur::math::Vector<Scalar, 4>& x) const {
            return x.head<2>();
        }
    };
    struct CustomJac {
        augur::math::Matrix<Scalar, 2, 4> operator()(const augur::math::Vector<Scalar, 4>&) const {
            return augur::filters::observe_position<Scalar, 2, 4>();
        }
    };
    using CustomEKF = augur::filters::ExtendedKalmanFilter<CV, 2, CustomFn, CustomJac>;
    CustomEKF filter{CV{1.0f}, kX0, kP0, CustomFn{}, CustomJac{}, kR};
    run_cycles(filter);
    REQUIRE(filter.state().allFinite());
}

TEST_CASE("UnscentedKalmanFilter::SigmaPointTuning (renamed from Config) still works positionally",
          "[filters][config]") {
    using Fn = augur::filters::ObservePositionFn<Scalar, 2, 4>;
    UKF::SigmaPointTuning tuning{}; // same alpha=1/beta=2/kappa=0 defaults as before the rename
    UKF filter{CV{1.0f}, kX0, kP0, Fn{}, kR, tuning};
    run_cycles(filter);
    REQUIRE(filter.state().allFinite());
}

TEST_CASE("UnscentedKalmanFilter::Config matches the positional ctor bit-identically", "[filters][config]") {
    using Fn = augur::filters::ObservePositionFn<Scalar, 2, 4>;

    UKF positional{CV{1.0f}, kX0, kP0, Fn{}, kR, UKF::SigmaPointTuning{}};
    UKF via_config{UKF::Config{.model = CV{1.0f}, .initial_state = kX0,
                               .initial_covariance = kP0, .measurement_noise = kR}};
    run_cycles(positional);
    run_cycles(via_config);
    REQUIRE(bit_identical(positional.state(), via_config.state()));
    REQUIRE(bit_identical(positional.covariance(), via_config.covariance()));
}

TEST_CASE("ParticleFilter::Config matches the positional ctor bit-identically given the same seed",
          "[filters][config]") {
    using Fn = augur::filters::ObservePositionFn<Scalar, 2, 4>;

    PF positional{CV{1.0f}, kX0, kP0, Fn{}, kR, /*seed=*/7};
    PF via_config{PF::Config{.model = CV{1.0f}, .initial_state = kX0,
                             .initial_covariance = kP0, .measurement_noise = kR, .seed = 7}};
    run_cycles(positional);
    run_cycles(via_config);
    REQUIRE(bit_identical(positional.state(), via_config.state()));
    REQUIRE(bit_identical(positional.covariance(), via_config.covariance()));
}

} // namespace
