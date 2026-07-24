#pragma once
// augur/track/track_manager.hpp
//
// docs/ROADMAP.md, "Track lifecycle management": the practical glue
// every real multi-target user of this library eventually needs. Given
// a stream of (possibly missing, possibly spurious) per-frame
// detections, decides when to spawn a new track, when an existing
// track should "coast" on prediction alone through a few missed
// detections rather than being killed immediately, and when to finally
// drop it. Sits directly on top of track/association.hpp (this frame's
// assignments, via nearest_neighbor() -- the cheap default; wire in
// joint_probabilistic_data_association() yourself around this class if
// close-track ambiguity needs the fuller treatment) and
// imm::Estimator/filters::Filter (each individual track's actual state
// estimate).
//
// Mostly state-machine bookkeeping, not novel math (matches
// docs/ROADMAP.md's own effort estimate for this item) -- built on
// utils::FixedVector so a game with a known max entity count never has
// this module touching the heap per frame.
//
// SCOPE, stated plainly, matching filters/sage_husa.hpp's own
// stated assumption: association gating assumes the measurement
// observes the state's first MeasDim components directly (position, no
// scaling/rotation) -- Filter's interface doesn't expose a general H to
// build a fully generic version around, and this is the convention
// every example in this library already follows.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include "augur/filters/filter_concept.hpp"
#include "augur/track/association.hpp"
#include "augur/utils/fixed_vector.hpp"

namespace augur::track {

enum class TrackStatus { Tentative, Confirmed, Coasting };

template <filters::Filter FilterT, std::size_t MaxTracks>
class TrackManager {
public:
    using Scalar = typename FilterT::Scalar;
    using Measurement = typename FilterT::Measurement; // every Filter shipped in this library defines this, even though it's not part of the base filters::Filter concept itself
    static constexpr int MeasDim = static_cast<int>(Measurement::RowsAtCompileTime);
    static constexpr int StateDim = static_cast<int>(FilterT::dimension);

    struct Track {
        std::size_t id = 0;
        // std::optional, not a plain FilterT: FixedVector's fixed-size
        // std::array storage default-constructs every slot up front
        // (see utils/fixed_vector.hpp), but FilterT (e.g. KalmanFilter)
        // has no default constructor by design -- a filter without a
        // model/state/H/R doesn't mean anything. Always holds a value
        // once a Track exists; nullopt only ever appears transiently
        // inside FixedVector's own unused capacity.
        std::optional<FilterT> filter;
        std::uint32_t hit_streak = 0;
        std::uint32_t missed_frames = 0;
        TrackStatus status = TrackStatus::Tentative;
    };

    struct Config {
        std::uint32_t confirmation_hits = 3;  // consecutive hits (including the spawning one) before Tentative -> Confirmed
        std::uint32_t coast_limit = 5;        // consecutive misses a Confirmed/Coasting track tolerates before being dropped
        Scalar gate_threshold = Scalar(9);     // Mahalanobis distance gate passed to nearest_neighbor()
        augur::math::Matrix<Scalar, MeasDim> measurement_noise =
            augur::math::Matrix<Scalar, MeasDim>::Identity(); // R, for gating only (see file comment: H = direct position read)
    };

    explicit TrackManager(Config config = Config{}) : config_(config) {}

    // make_filter(detection) constructs a fresh FilterT to seed a new
    // tentative track for an unmatched detection -- the caller decides
    // initial state/covariance/model tuning, this class only decides
    // *when* to spawn one.
    template <typename FilterFactory, std::size_t MaxDetections>
    void step(const augur::utils::FixedVector<Measurement, MaxDetections>& detections, Scalar dt, FilterFactory&& make_filter) {
        for (auto& track : tracks_) track.filter->predict(dt);

        augur::utils::FixedVector<augur::math::Vector<Scalar, MeasDim>, MaxTracks> predictions;
        augur::utils::FixedVector<augur::math::Matrix<Scalar, MeasDim>, MaxTracks> innovation_covariances;
        for (auto& track : tracks_) {
            predictions.push_back(track.filter->state().template head<MeasDim>());
            innovation_covariances.push_back(
                track.filter->covariance().template topLeftCorner<MeasDim, MeasDim>() + config_.measurement_noise);
        }

        const auto assignments = nearest_neighbor<Scalar, MeasDim, MaxTracks, MaxDetections>(
            predictions, innovation_covariances, detections, config_.gate_threshold);

        augur::utils::FixedVector<bool, MaxTracks> track_matched;
        for (std::size_t i = 0; i < tracks_.size(); ++i) track_matched.push_back(false);
        augur::utils::FixedVector<bool, MaxDetections> detection_matched;
        for (std::size_t j = 0; j < detections.size(); ++j) detection_matched.push_back(false);

        for (const auto& a : assignments) {
            auto& track = tracks_[a.track_index];
            track.filter->update(detections[a.detection_index]);
            track.missed_frames = 0;
            // A match right after a coast is not proof of continuity --
            // it only means SOMETHING passed the gate, which could be a
            // different real-world object that happened to be nearby
            // (docs/IMPROVEMENT_PLAN.md demonstrated exactly this on
            // real pedestrian trajectory data: a track's id silently
            // transferring to a different person after one coasted
            // frame). Refuse instant reconfirmation on a single
            // post-coast hit -- drop back to Tentative and require the
            // SAME fresh confirmation_hits a brand-new track needs,
            // exposing the uncertainty via `status` instead of hiding
            // it. This narrows the identity-swap window from "one
            // plausible match" to "confirmation_hits consecutive
            // plausible matches, none of which the track's own gate
            // rejects" -- a real, measurable improvement, not a complete
            // fix (this is still gate-based, not appearance-based;
            // Filter/measurement types carry no feature hook to do
            // better today -- see docs/IMPROVEMENT_PLAN.md's own "fuller
            // fix" note for what that would take).
            if (track.status == TrackStatus::Coasting) {
                track.status = TrackStatus::Tentative;
                track.hit_streak = 0;
            }
            ++track.hit_streak;
            if (track.status == TrackStatus::Tentative && track.hit_streak >= config_.confirmation_hits) {
                track.status = TrackStatus::Confirmed;
            }
            track_matched[a.track_index] = true;
            detection_matched[a.detection_index] = true;
        }

        // Rebuild tracks_ keeping only tracks that survive this frame,
        // rather than swap_remove-while-iterating (which would desync
        // track_matched's indices from tracks_'s as elements move).
        augur::utils::FixedVector<Track, MaxTracks> survivors;
        for (std::size_t i = 0; i < tracks_.size(); ++i) {
            if (track_matched[i]) {
                survivors.push_back(std::move(tracks_[i]));
                continue;
            }
            auto& track = tracks_[i];
            ++track.missed_frames;
            track.hit_streak = 0;
            const bool drop = (track.status == TrackStatus::Tentative) // a tentative track gets one shot: any miss drops it
                              || (track.missed_frames > config_.coast_limit);
            if (drop) continue;
            track.status = TrackStatus::Coasting;
            survivors.push_back(std::move(track));
        }
        tracks_ = std::move(survivors);

        for (std::size_t j = 0; j < detections.size(); ++j) {
            if (detection_matched[j] || tracks_.full()) continue;
            Track track;
            track.id = next_id_++;
            track.filter = make_filter(detections[j]);
            track.hit_streak = 1;
            track.status = (config_.confirmation_hits <= 1) ? TrackStatus::Confirmed : TrackStatus::Tentative;
            tracks_.push_back(std::move(track));
        }
    }

    [[nodiscard]] const augur::utils::FixedVector<Track, MaxTracks>& tracks() const { return tracks_; }

private:
    augur::utils::FixedVector<Track, MaxTracks> tracks_;
    std::size_t next_id_ = 0;
    Config config_;
};

} // namespace augur::track
