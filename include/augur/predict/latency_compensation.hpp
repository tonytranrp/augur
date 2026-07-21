#pragma once
// augur/predict/latency_compensation.hpp
//
// docs/ROADMAP.md, "Latency compensation / predict-to-render-time".
// imm::Estimator::predict_ahead() and standalone Filter::predict()
// already give the raw building block (propagate state forward by an
// arbitrary horizon); this file is the bookkeeping layer real netcode
// needs on top of that:
//
//   - SnapshotBuffer<Scalar, State, MaxHistory>: a bounded ring buffer
//     of (timestamp, state) snapshots, so the server can rewind a
//     target to where a shooting client saw it (classic lag
//     compensation), or so a client can interpolate another player's
//     track between two known-good snapshots.
//   - predict_to_render_time(): takes a filter's own state + its last
//     update timestamp + the local render clock + a one-way latency
//     estimate, and returns the state extrapolated to the right horizon
//     via the filter's own model -- for a single Filter, not an IMM
//     mix (imm::Estimator already has predict_ahead() for that case;
//     use it directly rather than this helper when tracking via IMM).
//
// DEVIATION from this file's own original stub sketch, stated plainly:
// timestamps here are plain Scalar (seconds, caller-defined epoch), not
// std::chrono::steady_clock::time_point. Every other timing quantity in
// this library (predict(dt), track/out_of_sequence.hpp's history) is
// already a plain Scalar -- introducing a std::chrono type here would
// be the only place in augur that forces a specific clock type on the
// caller, working against the "universal, no forced anything" posture
// docs/ARCHITECTURE.md describes. Convert from whatever clock you use
// to seconds at the call site.
//
// References (found during the original research pass):
//   - Gambetta, G., "Fast-Paced Multiplayer" series -- Client-Side
//     Prediction and Server Reconciliation; Entity Interpolation; Lag
//     Compensation. gabrielgambetta.com.
//   - Bernier, Y., "Latency Compensating Methods in Client/Server
//     In-game Protocol Design and Optimization," GDC 2001 (the
//     original Valve/Source engine lag-compensation write-up).

#include <cstddef>
#include <cstdint>
#include <span>
#include "augur/reflect/serialize.hpp"
#include "augur/utils/assert.hpp"
#include "augur/utils/fixed_vector.hpp"

namespace augur::predict {

// State must support `a + b` and `scalar * a` (any augur::math::Vector
// does, being an Eigen type) -- rewind_to()'s interpolation is a plain
// linear blend between the two bracketing snapshots.
template <typename Scalar, typename State, std::size_t MaxHistory>
class SnapshotBuffer {
public:
    struct Snapshot {
        Scalar timestamp = Scalar(0);
        State state{};
    };

    // Upper bound (not exact -- Eigen's alignment padding on State can
    // make sizeof(Snapshot) a little larger than the packed byte count
    // reflect::serialize() actually writes) on the buffer serialize()
    // needs. Safe to size a destination buffer with, since ByteWriter/
    // ByteReader track real bytes written/read independently of this.
    static constexpr std::size_t kMaxSerializedBytes =
        sizeof(std::uint32_t) + MaxHistory * sizeof(Snapshot);

    // Timestamps must be non-decreasing (this is a time-ordered log,
    // not a general associative container) -- asserts rather than
    // silently accepting an out-of-order snapshot, since rewind_to()'s
    // bracketing search assumes sorted order.
    void record(Scalar timestamp, const State& state) {
        AUGUR_ASSERT(history_.empty() || timestamp >= history_[history_.size() - 1].timestamp,
                     "SnapshotBuffer::record: timestamps must be non-decreasing");
        if (history_.full()) {
            for (std::size_t i = 1; i < history_.size(); ++i) history_[i - 1] = history_[i];
            history_[history_.size() - 1] = Snapshot{timestamp, state};
        } else {
            history_.push_back(Snapshot{timestamp, state});
        }
    }

    // Server-side rewind ("what did the shooting client actually see
    // at time t") or client-side interpolation between two known-good
    // snapshots. Clamps to the oldest/newest retained snapshot if t
    // falls outside the retained window, rather than extrapolating --
    // MaxHistory must be sized for the actual latency/interpolation
    // window this needs to cover.
    [[nodiscard]] State rewind_to(Scalar t) const {
        AUGUR_ASSERT(!history_.empty(), "SnapshotBuffer::rewind_to: no snapshots recorded");
        if (t <= history_[0].timestamp) return history_[0].state;
        const std::size_t last = history_.size() - 1;
        if (t >= history_[last].timestamp) return history_[last].state;

        for (std::size_t i = 0; i < last; ++i) {
            if (history_[i].timestamp <= t && t <= history_[i + 1].timestamp) {
                const Scalar span = history_[i + 1].timestamp - history_[i].timestamp;
                const Scalar alpha = (span > Scalar(0)) ? (t - history_[i].timestamp) / span : Scalar(0);
                return State(history_[i].state + alpha * (history_[i + 1].state - history_[i].state));
            }
        }
        return history_[last].state; // unreachable given the clamps above; kept for a defined return
    }

    [[nodiscard]] std::size_t size() const { return history_.size(); }
    [[nodiscard]] bool empty() const { return history_.empty(); }

    // Whole-buffer save-state / network-snapshot support (docs/ROADMAP.md,
    // "Reflection-driven (de)serialization"): count-prefixed, then each
    // retained Snapshot in oldest-to-newest order via reflect::serialize,
    // which walks {timestamp, state} through reflect::Descriptor -- state
    // recurses into VectorBackend (Eigen has no PFR path of its own),
    // timestamp bottoms out directly since Scalar is arithmetic. Returns
    // the number of bytes actually written (<= kMaxSerializedBytes).
    [[nodiscard]] std::size_t serialize(std::span<std::byte> destination) const {
        reflect::ByteWriter writer(destination);
        writer.write(static_cast<std::uint32_t>(history_.size()));
        for (std::size_t i = 0; i < history_.size(); ++i) {
            reflect::serialize(history_[i], writer);
        }
        return writer.bytes_written();
    }

    // Replaces the current contents with what serialize() produced.
    // Requires the same Scalar/State/MaxHistory as the writer -- there's
    // no type tag or schema on the wire to check that for you (see
    // reflect/serialize.hpp's file comment on why not).
    void deserialize(std::span<const std::byte> source) {
        reflect::ByteReader reader(source);
        const auto count = reader.template read<std::uint32_t>();
        AUGUR_ASSERT(count <= MaxHistory, "SnapshotBuffer::deserialize: snapshot count exceeds MaxHistory");
        history_.clear();
        for (std::uint32_t i = 0; i < count; ++i) {
            Snapshot snapshot{};
            reflect::deserialize(snapshot, reader);
            history_.push_back(snapshot);
        }
    }

private:
    augur::utils::FixedVector<Snapshot, MaxHistory> history_;
};

// last_update_time: when the filter's state was last corrected by a
// real measurement. render_time: the local clock's current render
// timestamp, same units/epoch as last_update_time. one_way_latency:
// estimated network delay from wherever the measurement originated to
// here. All three are Scalar seconds (see file comment).
template <typename FilterT>
[[nodiscard]] typename FilterT::StateVector predict_to_render_time(
    const FilterT& filter,
    typename FilterT::Scalar last_update_time,
    typename FilterT::Scalar render_time,
    typename FilterT::Scalar one_way_latency) {
    const auto horizon = (render_time - last_update_time) + one_way_latency;
    return filter.model().transition(filter.state(), horizon);
}

} // namespace augur::predict
