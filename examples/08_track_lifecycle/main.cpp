// examples/08_track_lifecycle/main.cpp
//
// docs/ROADMAP.md item 6: track/track_manager.hpp driving a full
// multi-target scenario -- one steady target present the whole time, a
// second target that appears partway through (spawns as Tentative,
// gets Confirmed after enough hits), and a run of missed detections
// for the first target (Coasting, then recovers) -- all through one
// TrackManager, with no per-target bookkeeping written by hand.

#include <cstdio>
#include "augur/augur.hpp"
#include "augur/track/track_manager.hpp"

namespace {
using Scalar = float;
using CV = augur::models::ConstantVelocity<Scalar, 2>;
using KF = augur::filters::KalmanFilter<CV, /*MeasDim=*/2>;
using TM = augur::track::TrackManager<KF, /*MaxTracks=*/8>;

KF make_filter_at(const KF::Measurement& z) {
    KF::StateVector x0 = KF::StateVector::Zero();
    x0(0) = z(0);
    x0(1) = z(1);
    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    return KF{CV{1.0f}, x0, KF::StateCovariance::Identity(), H, KF::MeasurementCovariance::Identity() * 0.1f};
}

const char* status_name(augur::track::TrackStatus s) {
    switch (s) {
        case augur::track::TrackStatus::Tentative: return "Tentative";
        case augur::track::TrackStatus::Confirmed: return "Confirmed";
        case augur::track::TrackStatus::Coasting: return "Coasting";
    }
    return "?";
}
} // namespace

int main() {
    TM::Config config;
    config.confirmation_hits = 3;
    config.coast_limit = 3;
    config.gate_threshold = Scalar(5.0);
    TM manager{config};

    const Scalar dt = Scalar(1.0 / 30.0);
    for (int frame = 0; frame < 15; ++frame) {
        augur::utils::FixedVector<KF::Measurement, 8> detections;

        // Target A: present every frame except a run of misses in the middle.
        const bool a_missed = (frame >= 6 && frame < 9);
        if (!a_missed) detections.push_back({0.1f * static_cast<float>(frame), 0.0f});

        // Target B: appears starting frame 5.
        if (frame >= 5) detections.push_back({10.0f + 0.1f * static_cast<float>(frame - 5), 5.0f});

        manager.step(detections, dt, make_filter_at);

        std::printf("frame %2d:", frame);
        for (const auto& track : manager.tracks()) {
            std::printf("  [id=%zu %s pos=(%.2f,%.2f)]", track.id, status_name(track.status),
                        track.filter->state()(0), track.filter->state()(1));
        }
        std::printf("\n");
    }
    return 0;
}
