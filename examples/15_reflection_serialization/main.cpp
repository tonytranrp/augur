// examples/15_reflection_serialization/main.cpp
//
// docs/ROADMAP.md item 12: reflect::Descriptor<T>-driven binary
// (de)serialization. Two backends feed the same Descriptor<T>/serialize()/
// deserialize() API:
//   - a plain user-defined aggregate (TrackSummary below) via Boost.PFR,
//     which also gives real field *names* (px, py, ...) for a debug dump;
//   - augur::math::Vector<Scalar,N> (an Eigen type) via a hand-written
//     VectorBackend, since Eigen's matrix type isn't an aggregate and PFR
//     categorically cannot reflect it (see
//     reflect/backends/vector_backend.hpp's file comment for how that was
//     confirmed) -- this is what lets SnapshotBuffer's {timestamp, state}
//     Snapshot serialize as a whole, State included.
//
// The scenario: a server records a target's track history into a
// SnapshotBuffer every tick (exactly like examples/13_latency_compensation),
// then serializes the whole buffer to a byte array -- standing in for a
// save-file write or a network snapshot send. A fresh process (simulated
// here by a second, freshly-constructed buffer) deserializes those bytes
// and reproduces identical rewind_to() results.

#include <array>
#include <cstdio>
#include <span>
#include "augur/augur.hpp"
#include "augur/predict/latency_compensation.hpp"
#include "augur/reflect/descriptor.hpp"
#include "augur/reflect/serialize.hpp"

// Deliberately NOT in an anonymous namespace: boost::pfr::get_name (used
// by Descriptor<T>::field_name() below) requires T to have external
// linkage, which an anonymous-namespace type does not have.
struct TrackSummary {
    float px = 0.0f;
    float py = 0.0f;
    std::uint32_t track_id = 0;
};

int main() {
    using Scalar = float;
    using CV = augur::models::ConstantVelocity<Scalar, 2>;
    using KF = augur::filters::KalmanFilter<CV, /*MeasDim=*/2>;
    using Buffer = augur::predict::SnapshotBuffer<Scalar, KF::StateVector, 32>;

    // --- Part 1: a plain aggregate through the PFR backend, with names ---
    TrackSummary summary{12.5f, -3.0f, 7u};
    std::printf("TrackSummary has %zu fields:\n", augur::reflect::Descriptor<TrackSummary>::field_count());
    augur::reflect::Descriptor<TrackSummary>::for_each_field(summary, [i = 0](const auto& field) mutable {
        std::printf("  field %d = %g\n", i++, static_cast<double>(field));
    });
    for (std::size_t i = 0; i < augur::reflect::Descriptor<TrackSummary>::field_count(); ++i) {
        std::printf("  name(%zu) = %s\n", i, std::string(augur::reflect::Descriptor<TrackSummary>::field_name(i)).c_str());
    }

    std::array<std::byte, 64> summary_bytes{};
    augur::reflect::ByteWriter summary_writer(summary_bytes);
    augur::reflect::serialize(summary, summary_writer);
    std::printf("TrackSummary serialized to %zu bytes\n\n", summary_writer.bytes_written());

    // --- Part 2: a whole SnapshotBuffer (Vector-valued State) round trip ---
    KF::StateVector x0 = KF::StateVector::Zero();
    x0(2) = 2.0f; // vx
    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    KF filter{CV{0.1f}, x0, KF::StateCovariance::Identity(), H, KF::MeasurementCovariance::Identity() * 0.1f};

    Buffer server_history;
    const Scalar dt = Scalar(1.0 / 30.0);
    Scalar t = 0;
    for (int tick = 0; tick < 10; ++tick) {
        filter.predict(dt);
        filter.update(KF::Measurement{2.0f * t, 0.0f});
        t += dt;
        server_history.record(t, filter.state());
    }

    std::array<std::byte, Buffer::kMaxSerializedBytes> wire{};
    const std::size_t written = server_history.serialize(wire);
    std::printf("SnapshotBuffer: %zu snapshots serialized to %zu bytes (capacity %zu)\n",
                server_history.size(), written, Buffer::kMaxSerializedBytes);

    // Simulates a fresh process/machine that only has the byte array --
    // a default-constructed buffer, never touched by record(). deserialize()
    // returns false on a truncated/malformed source (checked unconditionally,
    // even in this Release build -- see reflect/serialize.hpp's file
    // comment) rather than silently misparsing; a real save/network
    // loader should check this the same way.
    Buffer restored_history;
    if (!restored_history.deserialize(std::span<const std::byte>(wire.data(), written))) {
        std::fprintf(stderr, "SnapshotBuffer::deserialize failed: source was truncated or malformed\n");
        return 1;
    }

    const Scalar query_time = t - Scalar(0.1);
    const auto original_state = server_history.rewind_to(query_time);
    const auto restored_state = restored_history.rewind_to(query_time);
    std::printf("rewind_to(%.4f): original=(%.4f, %.4f)  restored=(%.4f, %.4f)  diff=%.2e\n",
                query_time, original_state(0), original_state(1), restored_state(0), restored_state(1),
                static_cast<double>((original_state - restored_state).norm()));
    return 0;
}
