// tests/unit/test_reflection_serialization.cpp
//
// Coverage for docs/ROADMAP.md item 12 ("Reflection-driven (de)serialization").
// Structural template-dispatch code (which backend Descriptor<T> picks,
// whether byte round-trips are exact), not a numerical estimation
// algorithm -- there's no meaningful ad hoc python3 equivalent for "does
// this C++ template resolve to the right backend," so per
// .claude/rules/testing.md this is verified directly in C++. The one
// genuinely non-obvious fact behind this design -- that Boost.PFR cannot
// reflect augur::math::Vector<Scalar,N> at all, since Eigen's matrix type
// isn't an aggregate -- was verified ad hoc via a throwaway
// boost::pfr::tuple_size_v<Eigen::Matrix<float,4,1>> compile check before
// any of this was written; see reflect/backends/vector_backend.hpp's file
// comment.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <array>
#include <cstddef>
#include "augur/augur.hpp"
#include "augur/predict/latency_compensation.hpp"
#include "augur/reflect/descriptor.hpp"
#include "augur/reflect/serialize.hpp"

using Catch::Matchers::WithinAbs;

// Deliberately NOT in an anonymous namespace: boost::pfr::get_name (used
// by Descriptor<T>::field_name()) forms an `extern const T` reference
// internally and requires T to have external linkage -- an anonymous-
// namespace type fails to compile with "cannot be defined in any other
// translation unit because its type does not have linkage". field_count()/
// for_each_field() don't need this (confirmed: only the field_name() test
// below exercises get_name), but any type meant to use field_name() needs
// ordinary (non-anonymous-namespace) scope.

// A plain user-defined aggregate -- the PfrBackend path. No private
// members, no user-declared constructors, so it's aggregate-initializable
// and Boost.PFR can destructure it.
struct TrackSnapshot {
    float px = 0.0f;
    float py = 0.0f;
    std::uint32_t track_id = 0;
};

// An aggregate that nests a Vector field -- exercises the recursive case
// serialize()/deserialize() actually needs for SnapshotBuffer::Snapshot:
// the PfrBackend walks {timestamp, state} as 2 direct members, then
// `state`'s own type dispatches to VectorBackend when serialize()
// recurses into it.
struct NestedSnapshot {
    float timestamp = 0.0f;
    augur::math::Vector<float, 3> state{};
};

TEST_CASE("Descriptor<Vector<N>> reflects N components with no PFR involved", "[reflect]") {
    using Vec4 = augur::math::Vector<float, 4>;
    STATIC_REQUIRE(augur::reflect::Descriptor<Vec4>::field_count() == 4);

    Vec4 v;
    v << 1.0f, 2.0f, 3.0f, 4.0f;

    std::array<float, 4> seen{};
    std::size_t i = 0;
    augur::reflect::Descriptor<Vec4>::for_each_field(v, [&](const auto& field) { seen[i++] = field; });

    REQUIRE(i == 4);
    REQUIRE_THAT(seen[0], WithinAbs(1.0f, 1e-6f));
    REQUIRE_THAT(seen[1], WithinAbs(2.0f, 1e-6f));
    REQUIRE_THAT(seen[2], WithinAbs(3.0f, 1e-6f));
    REQUIRE_THAT(seen[3], WithinAbs(4.0f, 1e-6f));

    // Vector components have no real names -- synthesized as "0".."3".
    REQUIRE(augur::reflect::Descriptor<Vec4>::field_name(0) == "0");
    REQUIRE(augur::reflect::Descriptor<Vec4>::field_name(3) == "3");
}

TEST_CASE("Descriptor<T> reflects a plain aggregate's real field names via PFR", "[reflect]") {
    STATIC_REQUIRE(augur::reflect::Descriptor<TrackSnapshot>::field_count() == 3);
    REQUIRE(augur::reflect::Descriptor<TrackSnapshot>::field_name(0) == "px");
    REQUIRE(augur::reflect::Descriptor<TrackSnapshot>::field_name(1) == "py");
    REQUIRE(augur::reflect::Descriptor<TrackSnapshot>::field_name(2) == "track_id");
}

TEST_CASE("reflect::serialize/deserialize round-trips a plain aggregate", "[reflect]") {
    TrackSnapshot original{3.5f, -2.25f, 42u};

    std::array<std::byte, 64> bytes{};
    augur::reflect::ByteWriter writer(bytes);
    augur::reflect::serialize(original, writer);
    REQUIRE(writer.bytes_written() == sizeof(float) * 2 + sizeof(std::uint32_t));

    TrackSnapshot restored{};
    augur::reflect::ByteReader reader(std::span<const std::byte>(bytes.data(), writer.bytes_written()));
    augur::reflect::deserialize(restored, reader);

    REQUIRE_THAT(restored.px, WithinAbs(original.px, 1e-6f));
    REQUIRE_THAT(restored.py, WithinAbs(original.py, 1e-6f));
    REQUIRE(restored.track_id == original.track_id);
}

TEST_CASE("reflect::serialize/deserialize round-trips an aggregate nesting a Vector field", "[reflect]") {
    NestedSnapshot original{1.5f, augur::math::Vector<float, 3>{7.0f, 8.0f, 9.0f}};

    std::array<std::byte, 64> bytes{};
    augur::reflect::ByteWriter writer(bytes);
    augur::reflect::serialize(original, writer);
    REQUIRE(writer.bytes_written() == sizeof(float) * 4);

    NestedSnapshot restored{};
    augur::reflect::ByteReader reader(std::span<const std::byte>(bytes.data(), writer.bytes_written()));
    augur::reflect::deserialize(restored, reader);

    REQUIRE_THAT(restored.timestamp, WithinAbs(original.timestamp, 1e-6f));
    REQUIRE_THAT((restored.state - original.state).norm(), WithinAbs(0.0f, 1e-6f));
}

TEST_CASE("SnapshotBuffer::serialize/deserialize round-trips the whole history", "[reflect][predict]") {
    using State = augur::math::Vector<float, 2>;
    using Buffer = augur::predict::SnapshotBuffer<float, State, 8>;

    Buffer original;
    original.record(0.0f, State{0.0f, 0.0f});
    original.record(1.0f, State{10.0f, 5.0f});
    original.record(2.0f, State{20.0f, 10.0f});

    std::array<std::byte, Buffer::kMaxSerializedBytes> bytes{};
    const std::size_t written = original.serialize(bytes);
    REQUIRE(written <= Buffer::kMaxSerializedBytes);

    Buffer restored;
    restored.deserialize(std::span<const std::byte>(bytes.data(), written));

    REQUIRE(restored.size() == original.size());
    REQUIRE_THAT(restored.rewind_to(0.0f)(0), WithinAbs(0.0f, 1e-5f));
    REQUIRE_THAT(restored.rewind_to(1.0f)(0), WithinAbs(10.0f, 1e-5f));
    REQUIRE_THAT(restored.rewind_to(2.0f)(1), WithinAbs(10.0f, 1e-5f));
    // Interpolation still works post-round-trip -- not just exact hits.
    REQUIRE_THAT(restored.rewind_to(0.5f)(0), WithinAbs(5.0f, 1e-5f));
}
