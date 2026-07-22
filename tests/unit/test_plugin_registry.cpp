// tests/unit/test_plugin_registry.cpp
//
// docs/IMPROVEMENT_PLAN.md: this file didn't exist before, and its
// absence is exactly how plugin/registry.hpp shipped broken -- grep
// confirmed zero existing callers of FilterRegistry/FilterBox anywhere
// in examples/ or tests/, despite docs/ARCHITECTURE.md listing this file
// as "Solid ... Covered by tests/unit/". The template had likely never
// actually been instantiated by a compiler before that investigation.
//
// The specific bug: IFilterBox<Scalar,Dim>/FilterBox<F> typed update()'s
// measurement argument as Vector<Scalar,Dim> where Dim is the filter's
// STATE dimension, not MeasDim -- an independent template parameter,
// only equal to state dimension in the degenerate case of observing the
// whole state directly. Registering a real
// KalmanFilter<ConstantVelocity<float,2>, MeasDim=2> (dimension=4,
// exactly the README's own teaser shape) failed to compile at all.
// Fixed by deriving measurement_dimension from F::Measurement's own
// compile-time row count (see filter_concept.hpp's parallel fix, which
// makes F::Measurement a required part of filters::Filter itself).

#include <algorithm>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "augur/augur.hpp"
#include "augur/plugin/registry.hpp"

using Catch::Matchers::WithinAbs;

static_assert(augur::filters::Filter<
              augur::filters::KalmanFilter<augur::models::ConstantVelocity<float, 2>, /*MeasDim=*/2>>);

TEST_CASE("FilterRegistry registers and runs a real KalmanFilter with dimension != MeasDim", "[plugin][registry]") {
    using CV = augur::models::ConstantVelocity<float, 2>; // dimension = 4 (px,py,vx,vy)
    using KF = augur::filters::KalmanFilter<CV, /*MeasDim=*/2>; // MeasDim = 2 -- deliberately != dimension
    using Box = augur::plugin::FilterBox<KF>;

    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    KF::MeasurementCovariance R = KF::MeasurementCovariance::Identity() * 0.1f;

    augur::plugin::FilterRegistry<float, KF::dimension, 2> registry;
    registry.register_factory("constant_velocity", [=]() -> std::unique_ptr<Box> {
        return std::make_unique<Box>(KF{CV{0.1f}, KF::StateVector::Zero(), KF::StateCovariance::Identity(), H, R});
    });

    REQUIRE(registry.contains("constant_velocity"));
    REQUIRE_FALSE(registry.contains("nonexistent"));

    auto box = registry.create("constant_velocity");
    REQUIRE(box != nullptr);
    REQUIRE(registry.create("nonexistent") == nullptr);

    // The type-erasure adapter must not change the math at all -- run
    // the same inputs through a directly-used KalmanFilter and require
    // bit-for-bit (well, float-precision-for-float-precision) agreement.
    KF direct{CV{0.1f}, KF::StateVector::Zero(), KF::StateCovariance::Identity(), H, R};

    const float dt = 1.0f / 60.0f;
    box->predict(dt);
    direct.predict(dt);
    REQUIRE_THAT((box->state() - direct.state()).norm(), WithinAbs(0.0f, 1e-6f));

    box->update(augur::math::Vector<float, 2>{1.0f, 0.5f});
    direct.update(KF::Measurement{1.0f, 0.5f});
    REQUIRE_THAT((box->state() - direct.state()).norm(), WithinAbs(0.0f, 1e-6f));
    REQUIRE_THAT((box->covariance() - direct.covariance()).norm(), WithinAbs(0.0f, 1e-6f));
}

TEST_CASE("FilterRegistry::names()/unregister() enumerate and retract registered factories", "[plugin][registry]") {
    using CV = augur::models::ConstantVelocity<float, 2>;
    using KF = augur::filters::KalmanFilter<CV, 2>;
    using Box = augur::plugin::FilterBox<KF>;

    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    KF::MeasurementCovariance R = KF::MeasurementCovariance::Identity();

    auto factory = [=]() -> std::unique_ptr<Box> {
        return std::make_unique<Box>(KF{CV{0.1f}, KF::StateVector::Zero(), KF::StateCovariance::Identity(), H, R});
    };

    augur::plugin::FilterRegistry<float, KF::dimension, 2> registry;
    REQUIRE(registry.names().empty());

    registry.register_factory("a", factory);
    registry.register_factory("b", factory);

    auto names = registry.names();
    std::sort(names.begin(), names.end());
    REQUIRE(names == std::vector<std::string>{"a", "b"});

    REQUIRE(registry.unregister("a"));
    REQUIRE_FALSE(registry.unregister("a")); // already gone -- no-op, not an error
    REQUIRE_FALSE(registry.contains("a"));
    REQUIRE(registry.contains("b"));
}
