// tests/unit/test_fusion.cpp
//
// Coverage for docs/ROADMAP.md item 10 ("Sensor / detection fusion",
// track/fusion.hpp). The two-detection reference values match an
// independent numpy computation (ad hoc python3, per
// .claude/rules/testing.md) that also cross-checked the closed form
// against sequential Kalman updates before this was written in C++.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "augur/augur.hpp"
#include "augur/track/fusion.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("fuse_measurements matches a hand-computed reference for two independent detections", "[track][fusion]") {
    augur::utils::FixedVector<augur::math::Vector<float, 2>, 4> detections;
    detections.push_back({1.0f, 2.0f});
    detections.push_back({1.2f, 1.9f});
    augur::utils::FixedVector<augur::math::Matrix<float, 2>, 4> covariances;
    covariances.push_back((augur::math::Matrix<float, 2>() << 0.5f, 0.0f, 0.0f, 0.3f).finished());
    covariances.push_back((augur::math::Matrix<float, 2>() << 0.2f, 0.0f, 0.0f, 0.6f).finished());

    const auto fused = augur::track::fuse_measurements<float, 2, 4>(detections, covariances);

    REQUIRE_THAT(fused.mean(0), WithinAbs(1.14285714f, 1e-4f));
    REQUIRE_THAT(fused.mean(1), WithinAbs(1.96666667f, 1e-4f));
    REQUIRE_THAT(fused.covariance(0, 0), WithinAbs(0.14285714f, 1e-4f));
    REQUIRE_THAT(fused.covariance(1, 1), WithinAbs(0.2f, 1e-4f));
}

TEST_CASE("fuse_measurements of a single detection returns it unchanged", "[track][fusion]") {
    augur::utils::FixedVector<augur::math::Vector<float, 2>, 4> detections;
    detections.push_back({3.0f, -1.0f});
    augur::utils::FixedVector<augur::math::Matrix<float, 2>, 4> covariances;
    covariances.push_back(augur::math::Matrix<float, 2>::Identity() * 0.4f);

    const auto fused = augur::track::fuse_measurements<float, 2, 4>(detections, covariances);

    REQUIRE_THAT(fused.mean(0), WithinAbs(3.0f, 1e-5f));
    REQUIRE_THAT(fused.mean(1), WithinAbs(-1.0f, 1e-5f));
    REQUIRE_THAT(fused.covariance(0, 0), WithinAbs(0.4f, 1e-4f));
}

TEST_CASE("fuse_measurements of N identical detections divides covariance by N and keeps the same mean",
          "[track][fusion]") {
    augur::utils::FixedVector<augur::math::Vector<float, 1>, 8> detections;
    augur::utils::FixedVector<augur::math::Matrix<float, 1>, 8> covariances;
    for (int i = 0; i < 4; ++i) {
        detections.push_back(augur::math::Vector<float, 1>(5.0f));
        covariances.push_back(augur::math::Matrix<float, 1>::Identity() * 0.8f);
    }

    const auto fused = augur::track::fuse_measurements<float, 1, 8>(detections, covariances);

    REQUIRE_THAT(fused.mean(0), WithinAbs(5.0f, 1e-4f));
    REQUIRE_THAT(fused.covariance(0, 0), WithinAbs(0.2f, 1e-4f)); // 0.8 / 4
}
