// tests/unit/test_observe_position.cpp
//
// filters/observe_position.hpp -- the packaged "observe position
// directly" measurement model. Pure selector-matrix construction, no
// equations, so no separate Python pre-check was needed (per
// .claude/rules/testing.md's own exemption); every case below compares
// against the exact hand-built matrix the examples previously wrote
// inline.
//
// Deliberately NOT covered: CoordinatedTurn3D, whose pz lives at state
// index 5 rather than 2 -- observe_position() selecting the leading
// components is documented (in the header's own comment) as wrong for
// observing 3D position from that model. examples/17 keeps the
// hand-built H(2,5)=1 form as the counter-example.

#include <catch2/catch_test_macros.hpp>

#include "augur/filters/kalman.hpp"
#include "augur/filters/observe_position.hpp"
#include "augur/models/constant_velocity.hpp"

namespace {

using augur::filters::observe_position;

TEST_CASE("observe_position matches the hand-built selector matrix",
          "[observe_position]") {
    SECTION("2 measured of 4 states (2D constant velocity)") {
        const auto H = observe_position<float, 2, 4>();
        augur::math::Matrix<float, 2, 4> expected =
            augur::math::Matrix<float, 2, 4>::Zero();
        expected(0, 0) = 1.0f;
        expected(1, 1) = 1.0f;
        REQUIRE(H == expected);
    }

    SECTION("2 measured of 5 states (coordinated turn)") {
        const auto H = observe_position<float, 2, 5>();
        augur::math::Matrix<float, 2, 5> expected =
            augur::math::Matrix<float, 2, 5>::Zero();
        expected(0, 0) = 1.0f;
        expected(1, 1) = 1.0f;
        REQUIRE(H == expected);
    }

    SECTION("3 measured of 9 states, double precision") {
        const auto H = observe_position<double, 3, 9>();
        augur::math::Matrix<double, 3, 9> expected =
            augur::math::Matrix<double, 3, 9>::Zero();
        expected(0, 0) = 1.0;
        expected(1, 1) = 1.0;
        expected(2, 2) = 1.0;
        REQUIRE(H == expected);
    }
}

TEST_CASE("observe_position<FilterT> deduces dimensions from the filter alias",
          "[observe_position]") {
    using CV = augur::models::ConstantVelocity<float, 2>;
    using KF = augur::filters::KalmanFilter<CV, 2>;

    const KF::MeasurementMatrix from_alias = observe_position<KF>();
    const auto from_dims = observe_position<float, 2, 4>();

    static_assert(std::is_same_v<decltype(observe_position<KF>()),
                                 KF::MeasurementMatrix>);
    REQUIRE(from_alias == from_dims);
}

TEST_CASE("ObservePositionFn agrees with the matrix form exactly",
          "[observe_position]") {
    const augur::filters::ObservePositionFn<float, 2, 5> h;
    augur::math::Vector<float, 5> x;
    x << 3.25f, -1.5f, 0.75f, 2.0f, 0.125f; // exact in float on purpose

    const augur::math::Vector<float, 2> via_fn = h(x);
    const augur::math::Vector<float, 2> via_matrix =
        observe_position<float, 2, 5>() * x;

    REQUIRE(via_fn == via_matrix);
    REQUIRE(via_fn(0) == 3.25f);
    REQUIRE(via_fn(1) == -1.5f);
}

TEST_CASE("ObservePositionJacobianFn returns the observe_position matrix",
          "[observe_position]") {
    const augur::filters::ObservePositionJacobianFn<double, 3, 6> jac;
    const augur::math::Vector<double, 6> anywhere =
        augur::math::Vector<double, 6>::Constant(42.0); // h is linear: x must not matter

    REQUIRE(jac(anywhere) == (observe_position<double, 3, 6>()));
}

TEST_CASE("observe_position feeds a KalmanFilter identically to the manual H",
          "[observe_position]") {
    using CV = augur::models::ConstantVelocity<float, 2>;
    using KF = augur::filters::KalmanFilter<CV, 2>;

    KF::MeasurementMatrix manual_H = KF::MeasurementMatrix::Zero();
    manual_H(0, 0) = 1.0f;
    manual_H(1, 1) = 1.0f;

    const KF::StateVector x0 = KF::StateVector::Zero();
    const KF::StateCovariance P0 = KF::StateCovariance::Identity() * 10.0f;
    const KF::MeasurementCovariance R = KF::MeasurementCovariance::Identity() * 0.5f;

    KF manual{CV{1.0f}, x0, P0, manual_H, R};
    KF helper{CV{1.0f}, x0, P0, observe_position<KF>(), R};

    // Identical inputs through identical code paths -- outputs must be
    // bit-identical, not merely close.
    for (int step = 0; step < 5; ++step) {
        manual.predict(1.0f / 30.0f);
        helper.predict(1.0f / 30.0f);
        const KF::Measurement z{0.1f * static_cast<float>(step), 0.05f};
        manual.update(z);
        helper.update(z);
    }
    REQUIRE(manual.state() == helper.state());
    REQUIRE(manual.covariance() == helper.covariance());
}

} // namespace
