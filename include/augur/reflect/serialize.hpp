#pragma once
// augur/reflect/serialize.hpp
//
// Generic binary (de)serialization for any type reflect::Descriptor<T> can
// walk -- a plain user-defined aggregate (via PfrBackend) or a fixed-size
// augur::math::Vector<Scalar,N> (via VectorBackend), including aggregates
// that nest a Vector as one of their fields (e.g. predict/
// latency_compensation.hpp's SnapshotBuffer<Scalar,State,MaxHistory>::
// Snapshot, which is {Scalar timestamp; State state;}). This is the
// concrete wire format docs/ROADMAP.md's "Reflection-driven
// (de)serialization" item asked for: a fixed-width positional binary
// layout (each arithmetic leaf written as its raw bytes, in declaration/
// component order), chosen deliberately over anything self-describing
// (length-prefixed field names, type tags, a schema version) -- this is
// an internal engine wire format for a single build (save-states, replay,
// same-version network snapshots), not a cross-version interchange
// format, so nothing here needs to tolerate a field being added, removed,
// or reordered between the writer and the reader.
//
// Writes into a caller-provided std::span<std::byte> rather than owning a
// growable buffer -- every type this library ships (a State vector, an
// IMM mode matrix, a SnapshotBuffer of bounded MaxHistory) has a size
// that's either compile-time-fixed or bounded by an existing template
// parameter, so the caller sizes the destination once (see
// SnapshotBuffer::kMaxSerializedBytes for how that bound is computed)
// rather than this header owning a heap-backed byte buffer.

#include <cstddef>
#include <cstring>
#include <span>
#include <type_traits>

#include "augur/reflect/descriptor.hpp"
#include "augur/utils/assert.hpp"

namespace augur::reflect {

class ByteWriter {
public:
    explicit ByteWriter(std::span<std::byte> destination) : buffer_(destination) {}

    template <typename T>
    void write(const T& value) {
        static_assert(std::is_arithmetic_v<T>, "ByteWriter::write: T must be an arithmetic leaf type");
        AUGUR_ASSERT(offset_ + sizeof(T) <= buffer_.size(), "ByteWriter::write: destination buffer too small");
        std::memcpy(buffer_.data() + offset_, &value, sizeof(T));
        offset_ += sizeof(T);
    }

    [[nodiscard]] std::size_t bytes_written() const { return offset_; }

private:
    std::span<std::byte> buffer_;
    std::size_t offset_ = 0;
};

class ByteReader {
public:
    explicit ByteReader(std::span<const std::byte> source) : buffer_(source) {}

    template <typename T>
    [[nodiscard]] T read() {
        static_assert(std::is_arithmetic_v<T>, "ByteReader::read: T must be an arithmetic leaf type");
        AUGUR_ASSERT(offset_ + sizeof(T) <= buffer_.size(), "ByteReader::read: source buffer exhausted");
        T value{};
        std::memcpy(&value, buffer_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    [[nodiscard]] std::size_t bytes_read() const { return offset_; }

private:
    std::span<const std::byte> buffer_;
    std::size_t offset_ = 0;
};

// Recurses field-by-field via Descriptor<T> until it bottoms out at
// arithmetic leaves (a Scalar, or a Vector's individual components).
template <typename T>
void serialize(const T& value, ByteWriter& writer) {
    if constexpr (std::is_arithmetic_v<T>) {
        writer.write(value);
    } else {
        Descriptor<T>::for_each_field(value, [&writer](const auto& field) { serialize(field, writer); });
    }
}

// Mirrors serialize()'s field order exactly -- deserialize() does not
// re-derive layout from the bytes themselves (there are no type tags or
// field names on the wire), so the caller must deserialize<T> with the
// same T that produced the bytes.
template <typename T>
void deserialize(T& value, ByteReader& reader) {
    if constexpr (std::is_arithmetic_v<T>) {
        value = reader.template read<T>();
    } else {
        Descriptor<T>::for_each_field(value, [&reader](auto& field) { deserialize(field, reader); });
    }
}

} // namespace augur::reflect
