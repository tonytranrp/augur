// tests/unit/test_track_manager.cpp
//
// Coverage for docs/ROADMAP.md item 6 ("Track lifecycle management",
// track/track_manager.hpp). Mostly state-machine bookkeeping (per that
// item's own effort estimate), so these are lifecycle-transition tests
// rather than math-verification ones.

#include <catch2/catch_test_macros.hpp>
#include "augur/augur.hpp"
#include "augur/track/track_manager.hpp"

using CV = augur::models::ConstantVelocity<float, 2>;
using KF = augur::filters::KalmanFilter<CV, /*MeasDim=*/2>;
using TM = augur::track::TrackManager<KF, /*MaxTracks=*/4>;

namespace {

KF make_filter_at(const KF::Measurement& z) {
    KF::StateVector x0 = KF::StateVector::Zero();
    x0(0) = z(0);
    x0(1) = z(1);
    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    return KF{CV{1.0f}, x0, KF::StateCovariance::Identity(), H, KF::MeasurementCovariance::Identity() * 0.1f};
}

augur::utils::FixedVector<KF::Measurement, 8> dets(std::initializer_list<KF::Measurement> zs) {
    augur::utils::FixedVector<KF::Measurement, 8> out;
    for (const auto& z : zs) out.push_back(z);
    return out;
}

} // namespace

TEST_CASE("TrackManager spawns a tentative track for an unmatched detection", "[track][track_manager]") {
    TM manager{TM::Config{}};
    manager.step(dets({{1.0f, 2.0f}}), 1.0f / 30.0f, make_filter_at);

    REQUIRE(manager.tracks().size() == 1);
    REQUIRE(manager.tracks()[0].status == augur::track::TrackStatus::Tentative);
    REQUIRE(manager.tracks()[0].hit_streak == 1);
}

TEST_CASE("TrackManager confirms a track after enough consecutive hits", "[track][track_manager]") {
    TM::Config config;
    config.confirmation_hits = 3;
    TM manager{config};

    KF::Measurement z{0.0f, 0.0f};
    for (int i = 0; i < 3; ++i) {
        manager.step(dets({z}), 1.0f / 30.0f, make_filter_at);
        REQUIRE(manager.tracks().size() == 1);
    }
    REQUIRE(manager.tracks()[0].status == augur::track::TrackStatus::Confirmed);
}

TEST_CASE("TrackManager coasts a confirmed track through a single miss instead of dropping it", "[track][track_manager]") {
    TM::Config config;
    config.confirmation_hits = 2;
    config.coast_limit = 3;
    TM manager{config};

    KF::Measurement z{0.0f, 0.0f};
    manager.step(dets({z}), 1.0f / 30.0f, make_filter_at); // spawn (tentative, hit_streak=1)
    manager.step(dets({z}), 1.0f / 30.0f, make_filter_at); // confirm (hit_streak=2 >= confirmation_hits)
    REQUIRE(manager.tracks()[0].status == augur::track::TrackStatus::Confirmed);

    manager.step(dets({}), 1.0f / 30.0f, make_filter_at); // miss -- should coast, not drop
    REQUIRE(manager.tracks().size() == 1);
    REQUIRE(manager.tracks()[0].status == augur::track::TrackStatus::Coasting);
    const auto track_id = manager.tracks()[0].id;

    // A single post-coast match is NOT enough to reconfirm (see
    // docs/IMPROVEMENT_PLAN.md's reacquisition finding, and the next
    // test below for the full regression coverage) -- reacquisition
    // needs the same fresh confirmation_hits a brand-new track does.
    manager.step(dets({z}), 1.0f / 30.0f, make_filter_at); // first post-coast hit
    REQUIRE(manager.tracks().size() == 1);
    REQUIRE(manager.tracks()[0].id == track_id); // same track, id unchanged
    REQUIRE(manager.tracks()[0].status == augur::track::TrackStatus::Tentative);

    manager.step(dets({z}), 1.0f / 30.0f, make_filter_at); // second post-coast hit (== confirmation_hits)
    REQUIRE(manager.tracks().size() == 1);
    REQUIRE(manager.tracks()[0].id == track_id);
    REQUIRE(manager.tracks()[0].status == augur::track::TrackStatus::Confirmed);
}

TEST_CASE("TrackManager refuses silent reacquisition: a coasting track needs fresh confirmation, not one match",
          "[track][track_manager][regression]") {
    // docs/IMPROVEMENT_PLAN.md's exact finding, reproduced as a unit
    // test: real TrackManager (GNN), a track coasts one frame, then a
    // DIFFERENT nearby object -- not the one that originally spawned
    // this track -- passes the gate and matches it. Before this fix,
    // that single match silently reconfirmed the track (status jumping
    // straight back to Confirmed), transferring the track's id to the
    // wrong real-world object with no visible signal. Now it must sit
    // at Tentative, needing confirmation_hits consecutive matches --
    // and, being Tentative, it's dropped outright on its very next miss
    // rather than lingering as a falsely-Confirmed track.
    TM::Config config;
    config.confirmation_hits = 3;
    config.coast_limit = 3;
    TM manager{config};

    const KF::Measurement original{0.0f, 0.0f};
    for (int i = 0; i < 3; ++i) manager.step(dets({original}), 1.0f / 30.0f, make_filter_at);
    REQUIRE(manager.tracks()[0].status == augur::track::TrackStatus::Confirmed);
    const auto original_id = manager.tracks()[0].id;

    manager.step(dets({}), 1.0f / 30.0f, make_filter_at); // original object leaves frame -- coast
    REQUIRE(manager.tracks()[0].status == augur::track::TrackStatus::Coasting);

    // A DIFFERENT object, close enough to gate, enters and matches.
    const KF::Measurement different_object{0.3f, 0.2f};
    manager.step(dets({different_object}), 1.0f / 30.0f, make_filter_at);
    REQUIRE(manager.tracks().size() == 1);
    REQUIRE(manager.tracks()[0].id == original_id); // still the same tracked id (minimal fix, not a re-ID)
    REQUIRE(manager.tracks()[0].status == augur::track::TrackStatus::Tentative); // NOT silently Confirmed
    REQUIRE(manager.tracks()[0].hit_streak == 1);

    // Being Tentative, it now gets no coast grace -- if the "reacquisition"
    // was in fact wrong and the object doesn't show up again, the track
    // is dropped immediately rather than persisting under a stolen id.
    manager.step(dets({}), 1.0f / 30.0f, make_filter_at);
    REQUIRE(manager.tracks().size() == 0);
}

TEST_CASE("TrackManager drops a track once it exceeds the coast limit", "[track][track_manager]") {
    TM::Config config;
    config.confirmation_hits = 1;
    config.coast_limit = 2;
    TM manager{config};

    KF::Measurement z{0.0f, 0.0f};
    manager.step(dets({z}), 1.0f / 30.0f, make_filter_at); // spawn + immediately confirmed (confirmation_hits=1)
    REQUIRE(manager.tracks().size() == 1);

    manager.step(dets({}), 1.0f / 30.0f, make_filter_at); // miss 1/2
    REQUIRE(manager.tracks().size() == 1);
    manager.step(dets({}), 1.0f / 30.0f, make_filter_at); // miss 2/2 (== coast_limit, still tolerated)
    REQUIRE(manager.tracks().size() == 1);
    manager.step(dets({}), 1.0f / 30.0f, make_filter_at); // miss 3 (> coast_limit) -- dropped
    REQUIRE(manager.tracks().size() == 0);
}

TEST_CASE("TrackManager drops a tentative track on its first miss", "[track][track_manager]") {
    TM::Config config;
    config.confirmation_hits = 5; // won't be reached
    TM manager{config};

    manager.step(dets({{0.0f, 0.0f}}), 1.0f / 30.0f, make_filter_at); // spawn (tentative)
    REQUIRE(manager.tracks().size() == 1);
    manager.step(dets({}), 1.0f / 30.0f, make_filter_at); // miss -- tentative tracks get no coast grace
    REQUIRE(manager.tracks().size() == 0);
}

TEST_CASE("TrackManager tracks two well-separated targets independently", "[track][track_manager]") {
    TM::Config config;
    config.confirmation_hits = 1;
    TM manager{config};

    for (int i = 0; i < 5; ++i) {
        const float x = 1.0f + 0.1f * static_cast<float>(i);
        const float y = 10.0f + 0.1f * static_cast<float>(i);
        manager.step(dets({{x, 0.0f}, {y, 5.0f}}), 1.0f / 30.0f, make_filter_at);
    }

    REQUIRE(manager.tracks().size() == 2);
    for (const auto& track : manager.tracks()) {
        REQUIRE(track.status == augur::track::TrackStatus::Confirmed);
    }
}
