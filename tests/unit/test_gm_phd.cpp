// tests/unit/test_gm_phd.cpp
//
// Coverage for docs/ROADMAP.md item 7 ("GM-PHD filter", track/gm_phd.hpp)
// -- the highest-effort roadmap item. The first case's reference values
// come from an independent numpy computation (ad hoc python3, per
// .claude/rules/testing.md) of the exact same 2-component, 2-detection
// scenario, done before any of this was written in C++.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "augur/augur.hpp"
#include "augur/track/gm_phd.hpp"

using Catch::Matchers::WithinAbs;

namespace {
using CV = augur::models::ConstantVelocity<float, 2>;
using PHD = augur::track::GmPhdFilter<CV, /*MeasDim=*/2, /*MaxComponents=*/64>;

PHD::MeasurementMatrix make_H() {
    PHD::MeasurementMatrix H = PHD::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    return H;
}
} // namespace

TEST_CASE("GmPhdFilter::update matches a hand-computed reference for two components and two detections",
          "[track][gm_phd]") {
    PHD::Config config;
    config.detection_probability = 0.9f;
    config.clutter_intensity = 0.01f;
    PHD filter{CV{1.0f}, make_H(), PHD::MeasurementCovariance::Identity() * 0.1f, config};

    augur::utils::FixedVector<PHD::GaussianComponent, 8> birth;
    PHD::GaussianComponent c0, c1;
    c0.weight = 0.9f;
    c0.mean = PHD::State::Zero();
    c0.covariance = PHD::StateCovariance::Identity();
    c1.weight = 0.8f;
    c1.mean = PHD::State::Zero();
    c1.mean(0) = 5.0f;
    c1.covariance = PHD::StateCovariance::Identity();
    birth.push_back(c0);
    birth.push_back(c1);

    // Directly seed components_ via predict() with zero survivors and
    // these as "births", then update -- this reproduces the exact
    // reference scenario (no prior propagation to muddy the numbers).
    filter.predict(0.0f, birth);

    augur::utils::FixedVector<PHD::Measurement, 8> detections;
    detections.push_back({0.1f, 0.05f});
    detections.push_back({5.2f, -0.1f});
    filter.update(detections);

    REQUIRE(filter.components().size() == 6); // 2 predicted * (1 missed + 2 detections)

    float total_weight = 0.0f;
    for (const auto& c : filter.components()) total_weight += c.weight;
    // Reference from an independent numpy computation matching this
    // exact scenario (4D CV state, not a simplified 2D stand-in -- an
    // earlier, less careful check using a 2D toy state omitted the
    // detection_probability factor from the per-detection weight
    // formula and got 2.0171, which this test originally (wrongly)
    // asserted; re-deriving carefully against the real 4D case matches
    // this implementation, confirming the C++ was right and that first
    // check was the bug).
    REQUIRE_THAT(total_weight, WithinAbs(2.00155132f, 1e-3f));

    // The two dominant (correctly-matched) components should each be
    // near their own detection with weight close to 1.
    int high_weight_count = 0;
    for (const auto& c : filter.components()) {
        if (c.weight > 0.8f) {
            ++high_weight_count;
            const bool near_first = std::abs(c.mean(0) - 0.0909f) < 0.01f;
            const bool near_second = std::abs(c.mean(0) - 5.1818f) < 0.01f;
            REQUIRE((near_first || near_second));
        }
    }
    REQUIRE(high_weight_count == 2);
}

TEST_CASE("GmPhdFilter::prune_and_merge preserves total weight when merging nearby components", "[track][gm_phd]") {
    PHD::Config config;
    config.merge_mahalanobis_threshold = 4.0f;
    config.prune_weight_threshold = 1e-6f;
    config.max_components_after_prune = 32;
    PHD filter{CV{1.0f}, make_H(), PHD::MeasurementCovariance::Identity() * 0.1f, config};

    augur::utils::FixedVector<PHD::GaussianComponent, 8> birth;
    for (int i = 0; i < 3; ++i) {
        PHD::GaussianComponent c;
        c.weight = 0.3f;
        c.mean = PHD::State::Zero();
        c.mean(0) = 0.05f * static_cast<float>(i); // three components very close together
        c.covariance = PHD::StateCovariance::Identity();
        birth.push_back(c);
    }
    filter.predict(0.0f, birth);

    float weight_before = 0.0f;
    for (const auto& c : filter.components()) weight_before += c.weight;

    filter.prune_and_merge();

    REQUIRE(filter.components().size() == 1); // all three should merge into one
    float weight_after = 0.0f;
    for (const auto& c : filter.components()) weight_after += c.weight;
    REQUIRE_THAT(weight_after, WithinAbs(weight_before, 1e-4f));
}

TEST_CASE("GmPhdFilter::prune_and_merge discards components below the weight threshold", "[track][gm_phd]") {
    PHD::Config config;
    config.prune_weight_threshold = 0.1f;
    PHD filter{CV{1.0f}, make_H(), PHD::MeasurementCovariance::Identity() * 0.1f, config};

    augur::utils::FixedVector<PHD::GaussianComponent, 8> birth;
    PHD::GaussianComponent strong, weak;
    strong.weight = 0.9f;
    strong.mean = PHD::State::Zero();
    strong.covariance = PHD::StateCovariance::Identity();
    weak.weight = 0.001f;
    weak.mean = PHD::State::Zero();
    weak.mean(0) = 50.0f; // far from strong, so it won't be merged into it
    weak.covariance = PHD::StateCovariance::Identity();
    birth.push_back(strong);
    birth.push_back(weak);
    filter.predict(0.0f, birth);

    filter.prune_and_merge();

    REQUIRE(filter.components().size() == 1);
    REQUIRE_THAT(filter.components()[0].weight, WithinAbs(0.9f, 1e-4f));
}

TEST_CASE("GmPhdFilter extract_targets and a full predict/update/prune cycle track two separated targets",
          "[track][gm_phd][regression]") {
    PHD::Config config;
    config.survival_probability = 0.99f;
    config.detection_probability = 0.9f;
    config.clutter_intensity = 0.001f;
    config.prune_weight_threshold = 1e-3f;
    config.merge_mahalanobis_threshold = 2.0f;
    config.max_components_after_prune = 16;
    PHD filter{CV{0.1f}, make_H(), PHD::MeasurementCovariance::Identity() * 0.05f, config};

    augur::utils::FixedVector<PHD::GaussianComponent, 8> birth;
    PHD::GaussianComponent b0, b1;
    b0.weight = 0.5f;
    b0.mean = PHD::State::Zero();
    b0.covariance = PHD::StateCovariance::Identity() * 2.0f;
    b1.weight = 0.5f;
    b1.mean = PHD::State::Zero();
    b1.mean(0) = 10.0f;
    b1.covariance = PHD::StateCovariance::Identity() * 2.0f;
    birth.push_back(b0);
    birth.push_back(b1);

    const float dt = 1.0f / 30.0f;
    for (int step = 0; step < 15; ++step) {
        filter.predict(dt, birth);
        augur::utils::FixedVector<PHD::Measurement, 8> detections;
        detections.push_back({0.0f, 0.0f});
        detections.push_back({10.0f, 0.0f});
        filter.update(detections);
        filter.prune_and_merge();
    }

    const auto targets = filter.extract_targets<8>(0.5f);
    REQUIRE(targets.size() == 2);
    bool found_near_0 = false, found_near_10 = false;
    for (const auto& t : targets) {
        if (std::abs(t.mean(0) - 0.0f) < 1.0f) found_near_0 = true;
        if (std::abs(t.mean(0) - 10.0f) < 1.0f) found_near_10 = true;
    }
    REQUIRE(found_near_0);
    REQUIRE(found_near_10);
}
