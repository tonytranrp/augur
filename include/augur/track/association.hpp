#pragma once
// augur/track/association.hpp
//
// docs/ROADMAP.md, "Multi-target data association". Everything under
// models/, filters/, imm/ tracks ONE target. The moment there are
// several tracks and several detections per frame, something has to
// answer "which detection belongs to which track" before any
// Estimator::update() gets called -- that's what this file is for.
//
// Two techniques, in increasing order of cost/quality:
//
//   nearest_neighbor() (GNN): greedily assigns the globally best
//   (smallest gated Mahalanobis distance) unclaimed track-detection
//   pair, repeatedly, until no valid pairs remain. Cheap, fine when
//   tracks are well-separated relative to sensor noise.
//
//   joint_probabilistic_data_association() (JPDA): weights every
//   track-detection pair by its TRUE marginal association probability,
//   computed by enumerating every feasible joint assignment (each track
//   gets at most one detection, each detection used by at most one
//   track) and summing each pair's weight across all events that
//   include it. This is what actually holds up when tracks are close
//   together and a naive per-track-independent approximation would
//   double-count a detection across multiple tracks -- verified
//   numerically (ad hoc python3, per .claude/rules/testing.md) that
//   this implementation's marginals differ meaningfully from that
//   naive approximation on exactly the ambiguous case JPDA exists for.
//   Reference: Fortmann, T., Bar-Shalom, Y., Scheffe, M., "Sonar
//   Tracking of Multiple Targets Using Joint Probabilistic Data
//   Association," IEEE J. Oceanic Engineering, 8(3), 1983, pp. 173-184.
//
// Bounded, no-heap FixedVector-based API throughout (see
// utils/fixed_vector.hpp), matching track/track_manager.hpp's own
// stated convention for this module, rather than the std::vector shape
// this file originally sketched -- a bounded max track/detection count
// is exactly the assumption the rest of track/ is built around.
// joint_probabilistic_data_association()'s event enumeration uses a
// bitmask over detections, hence the MaxDetections <= 32 limit below.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <numbers>
#include "augur/math/backend.hpp"
#include "augur/utils/fixed_vector.hpp"

namespace augur::track {

template <typename Scalar, int MeasDim>
struct Assignment {
    std::size_t track_index;
    std::size_t detection_index;
    Scalar gate_distance; // Mahalanobis distance (not squared)
};

namespace detail {

// Both nearest_neighbor() and joint_probabilistic_data_association() loop
// tracks-outer, detections-inner, but a track's innovation covariance S_i
// depends only on the track's own predicted covariance (via H/R) -- not
// on which detection it's being compared against. Precomputing S_i's
// inverse and determinant once per track (this struct) and reusing it
// across every detection in the inner loop turns what was previously
// O(tracks * detections) safe_inverse/determinant calls (mahalanobis_
// distance and gaussian_likelihood each recomputed both from scratch, per
// pair) into O(tracks). A pure hoist: same formulas, same results, so the
// existing hand-computed JPDA reference values in
// tests/unit/test_association.cpp are unchanged by this.
template <typename Scalar, int MeasDim>
struct InnovationGate {
    augur::math::Matrix<Scalar, MeasDim> S_inv;
    Scalar det_s;
};

template <typename Scalar, int MeasDim>
[[nodiscard]] inline InnovationGate<Scalar, MeasDim> make_innovation_gate(
    const augur::math::Matrix<Scalar, MeasDim>& innovation_covariance) {
    return {augur::math::safe_inverse<Scalar, MeasDim>(innovation_covariance), innovation_covariance.determinant()};
}

template <typename Scalar, int MeasDim>
[[nodiscard]] inline Scalar mahalanobis_distance(const augur::math::Vector<Scalar, MeasDim>& innovation,
                                                  const InnovationGate<Scalar, MeasDim>& gate) {
    const Scalar squared = (innovation.transpose() * gate.S_inv * innovation)(0, 0);
    return std::sqrt(squared > Scalar(0) ? squared : Scalar(0));
}

template <typename Scalar, int MeasDim>
[[nodiscard]] inline Scalar gaussian_likelihood(const augur::math::Vector<Scalar, MeasDim>& innovation,
                                                 const InnovationGate<Scalar, MeasDim>& gate) {
    const Scalar exponent = Scalar(-0.5) * (innovation.transpose() * gate.S_inv * innovation)(0, 0);
    if (!(gate.det_s > Scalar(0))) return Scalar(0);
    const Scalar normalizer = std::sqrt(std::pow(Scalar(2) * std::numbers::pi_v<Scalar>, Scalar(MeasDim)) * gate.det_s);
    return std::exp(exponent) / normalizer;
}

} // namespace detail

// Global Nearest Neighbor: repeatedly assigns the smallest remaining
// gated Mahalanobis distance among all (track, unclaimed detection)
// pairs, until none remain within gate_threshold. This is a GLOBAL
// greedy match (not "each track independently picks its own nearest"),
// so two tracks can never both claim the same detection.
template <typename Scalar, int MeasDim, std::size_t MaxTracks, std::size_t MaxDetections>
[[nodiscard]] inline augur::utils::FixedVector<Assignment<Scalar, MeasDim>, MaxTracks> nearest_neighbor(
    const augur::utils::FixedVector<augur::math::Vector<Scalar, MeasDim>, MaxTracks>& track_predictions,
    const augur::utils::FixedVector<augur::math::Matrix<Scalar, MeasDim>, MaxTracks>& track_innovation_covariances,
    const augur::utils::FixedVector<augur::math::Vector<Scalar, MeasDim>, MaxDetections>& detections,
    Scalar gate_threshold) {
    const std::size_t num_tracks = track_predictions.size();
    const std::size_t num_detections = detections.size();

    augur::math::Matrix<Scalar, static_cast<int>(MaxTracks), static_cast<int>(MaxDetections)> distance;
    for (std::size_t i = 0; i < num_tracks; ++i) {
        const auto gate = detail::make_innovation_gate<Scalar, MeasDim>(track_innovation_covariances[i]);
        for (std::size_t j = 0; j < num_detections; ++j) {
            const auto innovation = detections[j] - track_predictions[i];
            distance(static_cast<int>(i), static_cast<int>(j)) =
                detail::mahalanobis_distance<Scalar, MeasDim>(innovation, gate);
        }
    }

    augur::utils::FixedVector<Assignment<Scalar, MeasDim>, MaxTracks> assignments;
    augur::utils::FixedVector<bool, MaxTracks> track_claimed;
    augur::utils::FixedVector<bool, MaxDetections> detection_claimed;
    for (std::size_t i = 0; i < num_tracks; ++i) track_claimed.push_back(false);
    for (std::size_t j = 0; j < num_detections; ++j) detection_claimed.push_back(false);

    for (std::size_t iteration = 0; iteration < num_tracks; ++iteration) {
        Scalar best = gate_threshold;
        std::size_t best_i = num_tracks, best_j = num_detections;
        for (std::size_t i = 0; i < num_tracks; ++i) {
            if (track_claimed[i]) continue;
            for (std::size_t j = 0; j < num_detections; ++j) {
                if (detection_claimed[j]) continue;
                const Scalar d = distance(static_cast<int>(i), static_cast<int>(j));
                if (d <= best) {
                    best = d;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        if (best_i == num_tracks) break; // nothing left within gate
        track_claimed[best_i] = true;
        detection_claimed[best_j] = true;
        assignments.push_back(Assignment<Scalar, MeasDim>{best_i, best_j, best});
    }
    return assignments;
}

template <typename Scalar, std::size_t MaxTracks, std::size_t MaxDetections>
struct JpdaResult {
    // beta(i,j): probability detection j (of num_detections actually
    // present this frame) belongs to track i. beta_missed(i):
    // probability track i has no valid detection this frame (gated out
    // entirely, or lost to another track under exclusivity, or
    // genuinely clutter-only). Only the first num_tracks/num_detections
    // rows/cols are meaningful.
    augur::math::Matrix<Scalar, static_cast<int>(MaxTracks), static_cast<int>(MaxDetections)> beta;
    augur::math::Vector<Scalar, static_cast<int>(MaxTracks)> beta_missed;
};

// Joint Probabilistic Data Association. clutter_density is the assumed
// spatial density of false-alarm detections (same units as the
// Gaussian likelihood, i.e. probability per unit measurement volume) --
// higher values make the model more tolerant of an unexplained
// detection being clutter rather than forcing it onto a track.
template <typename Scalar, int MeasDim, std::size_t MaxTracks, std::size_t MaxDetections>
[[nodiscard]] inline JpdaResult<Scalar, MaxTracks, MaxDetections> joint_probabilistic_data_association(
    const augur::utils::FixedVector<augur::math::Vector<Scalar, MeasDim>, MaxTracks>& track_predictions,
    const augur::utils::FixedVector<augur::math::Matrix<Scalar, MeasDim>, MaxTracks>& track_innovation_covariances,
    const augur::utils::FixedVector<augur::math::Vector<Scalar, MeasDim>, MaxDetections>& detections,
    Scalar gate_threshold,
    Scalar clutter_density) {
    static_assert(MaxDetections <= 32, "joint_probabilistic_data_association enumerates events via a 32-bit detection bitmask");

    const std::size_t num_tracks = track_predictions.size();
    const std::size_t num_detections = detections.size();

    augur::math::Matrix<Scalar, static_cast<int>(MaxTracks), static_cast<int>(MaxDetections)> likelihood =
        augur::math::Matrix<Scalar, static_cast<int>(MaxTracks), static_cast<int>(MaxDetections)>::Zero();
    for (std::size_t i = 0; i < num_tracks; ++i) {
        const auto gate = detail::make_innovation_gate<Scalar, MeasDim>(track_innovation_covariances[i]);
        for (std::size_t j = 0; j < num_detections; ++j) {
            const auto innovation = detections[j] - track_predictions[i];
            const Scalar maha = detail::mahalanobis_distance<Scalar, MeasDim>(innovation, gate);
            if (maha <= gate_threshold) {
                likelihood(static_cast<int>(i), static_cast<int>(j)) =
                    detail::gaussian_likelihood<Scalar, MeasDim>(innovation, gate);
            }
        }
    }

    JpdaResult<Scalar, MaxTracks, MaxDetections> result;
    result.beta.setZero();
    result.beta_missed.setZero();
    Scalar total_weight = Scalar(0);
    std::array<int, MaxTracks> current_assignment; // -1 = no detection this event
    current_assignment.fill(-1);

    // Recursive enumeration of every feasible joint event (each track
    // gets at most one detection, each detection used by at most one
    // track, only gated pairs allowed), accumulating each event's
    // weight into beta/total_weight as it's completed rather than
    // materializing the (potentially large) event list.
    //
    // GCC's -Warray-bounds= flags the current_assignment[track_idx]/
    // current_assignment[i] accesses below (docs/PRODUCTION_ROADMAP.md
    // P0 item 4), reportedly with out-of-range subscripts against
    // current_assignment's MaxTracks-element std::array. That's a false
    // positive: `enumerate` only ever indexes with `i < num_tracks` (the
    // loop bound in the base case) or `track_idx` while the recursion
    // guard `track_idx == num_tracks` above has NOT yet triggered --
    // i.e. `track_idx < num_tracks` -- and num_tracks is `.size()` of a
    // FixedVector<..., MaxTracks>, which that container's own push_back
    // assert guarantees is <= MaxTracks. GCC's value-range analysis
    // appears to lose that bound through several levels of inlined
    // recursive generic-lambda self-capture (`self(self, ...)`), a
    // known pattern for this class of false positive. Verified instead
    // with what this environment can actually check (no GCC available
    // here -- confirmed its "gcc"/"g++" are clang aliases): a standalone
    // repro instantiating this function at full MaxTracks=4/
    // MaxDetections=4 capacity, fully-connected gating (every track
    // gates every detection, the worst case for this recursion's branch
    // factor), under clang -fsanitize=address,undefined -- clean exit,
    // no report from either sanitizer. Suppressed narrowly (this block
    // only, GCC only) rather than restructured, since restructuring
    // provably-correct code on the strength of a warning from a compiler
    // not available to verify against isn't a fix, just an unverified
    // guess -- re-evaluate on a real GCC toolchain if this ever needs
    // revisiting (e.g. Phase 2's CI matrix).
#if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Warray-bounds="
#endif
    const auto enumerate = [&](auto&& self, std::size_t track_idx, std::uint32_t used_mask,
                                Scalar weight_so_far, std::size_t num_assigned) -> void {
        if (track_idx == num_tracks) {
            Scalar weight = weight_so_far * std::pow(clutter_density, Scalar(num_detections - num_assigned));
            total_weight += weight;
            for (std::size_t i = 0; i < num_tracks; ++i) {
                if (current_assignment[i] >= 0) {
                    result.beta(static_cast<int>(i), current_assignment[i]) += weight;
                }
            }
            return;
        }
        self(self, track_idx + 1, used_mask, weight_so_far, num_assigned); // track_idx gets no detection
        for (std::size_t j = 0; j < num_detections; ++j) {
            const std::uint32_t bit = std::uint32_t(1) << j;
            if ((used_mask & bit) != 0) continue;
            const Scalar L = likelihood(static_cast<int>(track_idx), static_cast<int>(j));
            if (!(L > Scalar(0))) continue;
            current_assignment[track_idx] = static_cast<int>(j);
            self(self, track_idx + 1, used_mask | bit, weight_so_far * L, num_assigned + 1);
            current_assignment[track_idx] = -1;
        }
    };
#if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC diagnostic pop
#endif
    enumerate(enumerate, 0, 0u, Scalar(1), 0);

    if (total_weight > Scalar(0)) {
        for (std::size_t i = 0; i < num_tracks; ++i) {
            Scalar assigned_prob = Scalar(0);
            for (std::size_t j = 0; j < num_detections; ++j) {
                result.beta(static_cast<int>(i), static_cast<int>(j)) /= total_weight;
                assigned_prob += result.beta(static_cast<int>(i), static_cast<int>(j));
            }
            result.beta_missed(static_cast<int>(i)) = Scalar(1) - assigned_prob;
        }
    }
    return result;
}

} // namespace augur::track
