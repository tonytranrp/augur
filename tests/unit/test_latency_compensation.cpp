// tests/unit/test_latency_compensation.cpp
//
// Coverage for docs/ROADMAP.md item 9 ("Latency compensation",
// predict/latency_compensation.hpp). Mostly bookkeeping (interpolation,
// horizon arithmetic) rather than novel estimation math, so this is
// verified directly in C++ rather than via a separate ad hoc python3
// pass -- consistent with .claude/rules/testing.md's own guidance that
// pure bookkeeping doesn't need that step.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "augur/augur.hpp"
#include "augur/predict/latency_compensation.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("SnapshotBuffer::rewind_to returns exact snapshots and interpolates between them", "[predict][latency]") {
    using State = augur::math::Vector<float, 2>;
    augur::predict::SnapshotBuffer<float, State, 8> buffer;

    buffer.record(0.0f, State{0.0f, 0.0f});
    buffer.record(1.0f, State{10.0f, 0.0f});
    buffer.record(2.0f, State{20.0f, 10.0f});

    REQUIRE_THAT((buffer.rewind_to(0.0f) - State{0.0f, 0.0f}).norm(), WithinAbs(0.0f, 1e-5f));
    REQUIRE_THAT((buffer.rewind_to(1.0f) - State{10.0f, 0.0f}).norm(), WithinAbs(0.0f, 1e-5f));
    REQUIRE_THAT((buffer.rewind_to(2.0f) - State{20.0f, 10.0f}).norm(), WithinAbs(0.0f, 1e-5f));

    // Interpolated: halfway between t=0 and t=1.
    REQUIRE_THAT((buffer.rewind_to(0.5f) - State{5.0f, 0.0f}).norm(), WithinAbs(0.0f, 1e-5f));
    // Interpolated: a quarter of the way between t=1 and t=2.
    REQUIRE_THAT((buffer.rewind_to(1.25f) - State{12.5f, 2.5f}).norm(), WithinAbs(0.0f, 1e-4f));
}

TEST_CASE("SnapshotBuffer::rewind_to clamps to the oldest/newest retained snapshot outside the window", "[predict][latency]") {
    using State = augur::math::Vector<float, 1>;
    augur::predict::SnapshotBuffer<float, State, 4> buffer;
    buffer.record(1.0f, State{5.0f});
    buffer.record(2.0f, State{15.0f});

    REQUIRE_THAT(buffer.rewind_to(-10.0f)(0), WithinAbs(5.0f, 1e-5f));  // before the oldest -> clamp
    REQUIRE_THAT(buffer.rewind_to(100.0f)(0), WithinAbs(15.0f, 1e-5f)); // after the newest -> clamp
}

TEST_CASE("SnapshotBuffer evicts the oldest snapshot once full (sliding window)", "[predict][latency]") {
    using State = augur::math::Vector<float, 1>;
    augur::predict::SnapshotBuffer<float, State, 2> buffer;
    buffer.record(0.0f, State{0.0f});
    buffer.record(1.0f, State{10.0f});
    buffer.record(2.0f, State{20.0f}); // evicts t=0

    REQUIRE(buffer.size() == 2);
    REQUIRE_THAT(buffer.rewind_to(0.0f)(0), WithinAbs(10.0f, 1e-5f)); // oldest retained is now t=1
}

TEST_CASE("predict_to_render_time matches a direct model.transition() call at the combined horizon", "[predict][latency]") {
    using CV = augur::models::ConstantVelocity<float, 2>;
    using KF = augur::filters::KalmanFilter<CV, /*MeasDim=*/2>;

    KF::StateVector x0;
    x0 << 1.0f, 2.0f, 0.5f, -0.3f;
    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    KF filter{CV{1.0f}, x0, KF::StateCovariance::Identity(), H, KF::MeasurementCovariance::Identity()};

    const float last_update_time = 10.0f;
    const float render_time = 10.2f;
    const float one_way_latency = 0.05f;

    const auto predicted = augur::predict::predict_to_render_time(filter, last_update_time, render_time, one_way_latency);
    const auto expected = CV{1.0f}.transition(x0, (render_time - last_update_time) + one_way_latency);

    REQUIRE_THAT((predicted - expected).norm(), WithinAbs(0.0f, 1e-5f));
}
