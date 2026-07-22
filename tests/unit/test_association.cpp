// tests/unit/test_association.cpp
//
// Coverage for docs/ROADMAP.md item 5 ("Multi-target data association",
// track/association.hpp). The JPDA reference values were computed
// independently (ad hoc python3, per .claude/rules/testing.md) via
// direct joint-event enumeration before any of this was written in C++.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "augur/track/association.hpp"

using Catch::Matchers::WithinAbs;
using augur::math::Matrix;
using augur::math::Vector;
using augur::utils::FixedVector;

namespace {
constexpr std::size_t kMaxTracks = 4;
constexpr std::size_t kMaxDetections = 4;
} // namespace

TEST_CASE("nearest_neighbor assigns well-separated tracks to their obvious detections", "[track][association]") {
    FixedVector<Vector<float, 2>, kMaxTracks> predictions;
    predictions.push_back({0.0f, 0.0f});
    predictions.push_back({10.0f, 0.0f});
    FixedVector<Matrix<float, 2>, kMaxTracks> covariances;
    covariances.push_back(Matrix<float, 2>::Identity());
    covariances.push_back(Matrix<float, 2>::Identity());
    FixedVector<Vector<float, 2>, kMaxDetections> detections;
    detections.push_back({0.2f, 0.1f});
    detections.push_back({10.1f, -0.2f});

    const auto assignments = augur::track::nearest_neighbor<float, 2, kMaxTracks, kMaxDetections>(
        predictions, covariances, detections, 3.0f);

    REQUIRE(assignments.size() == 2);
    for (const auto& a : assignments) {
        REQUIRE(a.track_index == a.detection_index); // track i's obvious match is detection i here
    }
}

TEST_CASE("nearest_neighbor's global greedy resolves a conflict a per-track-independent choice would miss",
          "[track][association]") {
    // Both tracks' single nearest detection is the SAME one (0,0); the
    // globally-best pair must win it, leaving the other track its
    // second-best (still gated) option -- not simply skip it.
    FixedVector<Vector<float, 2>, kMaxTracks> predictions;
    predictions.push_back({0.0f, 0.0f});  // track 0: detection 0 is its exact match
    predictions.push_back({0.5f, 0.0f});  // track 1: detection 0 is closer than detection 1, but detection 0 will be taken
    FixedVector<Matrix<float, 2>, kMaxTracks> covariances;
    covariances.push_back(Matrix<float, 2>::Identity());
    covariances.push_back(Matrix<float, 2>::Identity());
    FixedVector<Vector<float, 2>, kMaxDetections> detections;
    detections.push_back({0.0f, 0.0f});
    detections.push_back({2.0f, 0.0f});

    const auto assignments = augur::track::nearest_neighbor<float, 2, kMaxTracks, kMaxDetections>(
        predictions, covariances, detections, 5.0f);

    REQUIRE(assignments.size() == 2);
    bool track0_got_det0 = false, track1_got_det1 = false;
    for (const auto& a : assignments) {
        if (a.track_index == 0) track0_got_det0 = (a.detection_index == 0);
        if (a.track_index == 1) track1_got_det1 = (a.detection_index == 1);
    }
    REQUIRE(track0_got_det0); // track 0 wins the globally-best (exact) match
    REQUIRE(track1_got_det1); // track 1 gets its remaining, still-gated option
}

TEST_CASE("nearest_neighbor respects the gate threshold", "[track][association]") {
    FixedVector<Vector<float, 2>, kMaxTracks> predictions;
    predictions.push_back({0.0f, 0.0f});
    FixedVector<Matrix<float, 2>, kMaxTracks> covariances;
    covariances.push_back(Matrix<float, 2>::Identity());
    FixedVector<Vector<float, 2>, kMaxDetections> detections;
    detections.push_back({100.0f, 100.0f}); // far outside any reasonable gate

    const auto assignments = augur::track::nearest_neighbor<float, 2, kMaxTracks, kMaxDetections>(
        predictions, covariances, detections, 3.0f);

    REQUIRE(assignments.size() == 0);
}

TEST_CASE("joint_probabilistic_data_association matches a hand-computed reference for well-separated tracks",
          "[track][association][jpda]") {
    FixedVector<Vector<float, 2>, kMaxTracks> predictions;
    predictions.push_back({0.0f, 0.0f});
    predictions.push_back({5.0f, 0.0f});
    FixedVector<Matrix<float, 2>, kMaxTracks> covariances;
    covariances.push_back(Matrix<float, 2>::Identity());
    covariances.push_back(Matrix<float, 2>::Identity());
    FixedVector<Vector<float, 2>, kMaxDetections> detections;
    detections.push_back({0.2f, 0.1f});
    detections.push_back({5.1f, -0.2f});

    const auto result = augur::track::joint_probabilistic_data_association<float, 2, kMaxTracks, kMaxDetections>(
        predictions, covariances, detections, 3.0f, 0.01f);

    REQUIRE_THAT(result.beta(0, 0), WithinAbs(0.93947662f, 1e-4f));
    REQUIRE_THAT(result.beta(0, 1), WithinAbs(0.0f, 1e-6f));
    REQUIRE_THAT(result.beta_missed(0), WithinAbs(0.06052338f, 1e-4f));
    REQUIRE_THAT(result.beta(1, 1), WithinAbs(0.93947662f, 1e-4f));
}

TEST_CASE("joint_probabilistic_data_association matches a hand-computed reference for ambiguous, close tracks",
          "[track][association][jpda]") {
    FixedVector<Vector<float, 2>, kMaxTracks> predictions;
    predictions.push_back({0.0f, 0.0f});
    predictions.push_back({1.0f, 0.0f});
    FixedVector<Matrix<float, 2>, kMaxTracks> covariances;
    covariances.push_back(Matrix<float, 2>::Identity());
    covariances.push_back(Matrix<float, 2>::Identity());
    FixedVector<Vector<float, 2>, kMaxDetections> detections;
    detections.push_back({0.3f, 0.0f});
    detections.push_back({0.7f, 0.0f});

    const auto result = augur::track::joint_probabilistic_data_association<float, 2, kMaxTracks, kMaxDetections>(
        predictions, covariances, detections, 3.0f, 0.01f);

    // Reference values from independent joint-event enumeration in numpy.
    REQUIRE_THAT(result.beta(0, 0), WithinAbs(0.55688957f, 1e-4f));
    REQUIRE_THAT(result.beta(0, 1), WithinAbs(0.37839122f, 1e-4f));
    REQUIRE_THAT(result.beta_missed(0), WithinAbs(0.06471921f, 1e-4f));
    REQUIRE_THAT(result.beta(1, 0), WithinAbs(0.37839122f, 1e-4f));
    REQUIRE_THAT(result.beta(1, 1), WithinAbs(0.55688957f, 1e-4f));

    for (std::size_t i = 0; i < 2; ++i) {
        float row_sum = result.beta_missed(static_cast<int>(i));
        for (std::size_t j = 0; j < 2; ++j) row_sum += result.beta(static_cast<int>(i), static_cast<int>(j));
        REQUIRE_THAT(row_sum, WithinAbs(1.0f, 1e-4f));
    }
}

TEST_CASE("joint_probabilistic_data_association's clustering is lossless across two disjoint clusters",
          "[track][association][jpda]") {
    // docs/IMPROVEMENT_PLAN.md's finding: solving 2 disjoint clusters
    // jointly (one recursion spanning both) costs the PRODUCT of their
    // individual tree sizes, when the correct cost is their SUM.
    // Verified provably lossless ad hoc (python3, per
    // .claude/rules/testing.md) before touching the implementation --
    // this test checks the C++ side. Cluster A is exactly the previous
    // test's "ambiguous, close tracks" scenario (reusing its
    // independently-verified reference values); cluster B is the same
    // scenario translated by +1000 in x, far enough that no cross-
    // cluster pair can possibly gate. Mahalanobis distance depends only
    // on (detection - prediction), which a uniform translation of both
    // leaves unchanged, so cluster B's beta values must be BIT-IDENTICAL
    // to cluster A's, not just "also correct" -- a strong, cheap way to
    // check clustering didn't perturb anything.
    constexpr std::size_t kBigTracks = 4;
    constexpr std::size_t kBigDetections = 4;
    FixedVector<Vector<float, 2>, kBigTracks> predictions;
    predictions.push_back({0.0f, 0.0f});        // cluster A, track 0
    predictions.push_back({1.0f, 0.0f});        // cluster A, track 1
    predictions.push_back({1000.0f, 0.0f});     // cluster B, track 2 (=track 0 + 1000)
    predictions.push_back({1001.0f, 0.0f});     // cluster B, track 3 (=track 1 + 1000)
    FixedVector<Matrix<float, 2>, kBigTracks> covariances;
    for (int i = 0; i < 4; ++i) covariances.push_back(Matrix<float, 2>::Identity());
    FixedVector<Vector<float, 2>, kBigDetections> detections;
    detections.push_back({0.3f, 0.0f});         // cluster A, det 0
    detections.push_back({0.7f, 0.0f});         // cluster A, det 1
    detections.push_back({1000.3f, 0.0f});      // cluster B, det 2
    detections.push_back({1000.7f, 0.0f});      // cluster B, det 3

    const auto result = augur::track::joint_probabilistic_data_association<float, 2, kBigTracks, kBigDetections>(
        predictions, covariances, detections, 3.0f, 0.01f);

    // Cluster A: identical reference values to the previous test.
    REQUIRE_THAT(result.beta(0, 0), WithinAbs(0.55688957f, 1e-4f));
    REQUIRE_THAT(result.beta(0, 1), WithinAbs(0.37839122f, 1e-4f));
    REQUIRE_THAT(result.beta_missed(0), WithinAbs(0.06471921f, 1e-4f));
    REQUIRE_THAT(result.beta(1, 0), WithinAbs(0.37839122f, 1e-4f));
    REQUIRE_THAT(result.beta(1, 1), WithinAbs(0.55688957f, 1e-4f));
    // Zero cross-cluster leakage.
    REQUIRE_THAT(result.beta(0, 2), WithinAbs(0.0f, 1e-6f));
    REQUIRE_THAT(result.beta(0, 3), WithinAbs(0.0f, 1e-6f));

    // Cluster B: matches cluster A to float32 precision (translation
    // invariance) -- WithinAbs rather than exact ==, since
    // 1000.3f-1000.0f isn't guaranteed bit-identical to 0.3f-0.0f (float32
    // has coarser absolute precision at the larger magnitude).
    REQUIRE_THAT(result.beta(2, 2), WithinAbs(result.beta(0, 0), 1e-4f));
    REQUIRE_THAT(result.beta(2, 3), WithinAbs(result.beta(0, 1), 1e-4f));
    REQUIRE_THAT(result.beta_missed(2), WithinAbs(result.beta_missed(0), 1e-4f));
    REQUIRE_THAT(result.beta(3, 2), WithinAbs(result.beta(1, 0), 1e-4f));
    REQUIRE_THAT(result.beta(3, 3), WithinAbs(result.beta(1, 1), 1e-4f));
    REQUIRE_THAT(result.beta(2, 0), WithinAbs(0.0f, 1e-6f));
    REQUIRE_THAT(result.beta(2, 1), WithinAbs(0.0f, 1e-6f));
}

TEST_CASE("joint_probabilistic_data_association gives an isolated track (no gated detections) beta_missed=1",
          "[track][association][jpda]") {
    // A track with zero gated detections is the degenerate case of a
    // 1-track, 0-detection cluster -- verified ad hoc (python3) this
    // falls out of the clustering fix with no special-casing needed.
    FixedVector<Vector<float, 2>, kMaxTracks> predictions;
    predictions.push_back({0.0f, 0.0f});
    predictions.push_back({500.0f, 500.0f}); // isolated: nowhere near any detection
    FixedVector<Matrix<float, 2>, kMaxTracks> covariances;
    covariances.push_back(Matrix<float, 2>::Identity());
    covariances.push_back(Matrix<float, 2>::Identity());
    FixedVector<Vector<float, 2>, kMaxDetections> detections;
    detections.push_back({0.2f, 0.1f});

    const auto result = augur::track::joint_probabilistic_data_association<float, 2, kMaxTracks, kMaxDetections>(
        predictions, covariances, detections, 3.0f, 0.01f);

    REQUIRE_THAT(result.beta_missed(1), WithinAbs(1.0f, 1e-6f));
    REQUIRE_THAT(result.beta(1, 0), WithinAbs(0.0f, 1e-6f));
}
