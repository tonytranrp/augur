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
//
// ByteReader's bounds check is UNCONDITIONAL -- checked even in
// Release/NDEBUG builds, unlike every AUGUR_ASSERT elsewhere in this
// codebase (which compiles to nothing under NDEBUG, the build type
// CLAUDE.md/README.md document as the first example command). That's a
// deliberate exception, not an inconsistency: every other assert in
// augur documents an internal invariant the caller never violates by
// construction; this is the one path that parses externally-sourced
// bytes (a save file, a network packet) -- a materially different trust
// boundary. docs/IMPROVEMENT_PLAN.md reproduced a real, ASan-confirmed
// out-of-bounds read here: a game changing tracked state shape between
// patches, or a save truncated by a full disk, silently spliced bytes
// from beyond the intended field into the result under the documented
// default Release build -- not an error, not garbage, just quietly wrong
// data. ByteWriter is unaffected (still AUGUR_ASSERT-gated): its
// destination buffer is caller-owned and caller-sized, an ordinary
// internal invariant like everywhere else in augur, not untrusted input.
//
// ByteReader signals failure via a sticky flag (mirroring
// std::istream's failbit) rather than an exception or an
// std::optional-returning read(): once tripped, every subsequent read()
// short-circuits to a safe default without touching the buffer again,
// so the recursive deserialize<T>() below needs no signature change at
// any recursion depth to propagate failure -- the shared ByteReader
// already carries it, and the one place that matters (the outermost
// deserialize<T> call) checks it once at the end.

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

    // Returns a default-constructed T (not whatever bytes happen to be
    // at the current offset) and sets failed() once the source is
    // exhausted -- see this file's own top comment for why this check is
    // unconditional rather than AUGUR_ASSERT-gated like the rest of the
    // codebase. Safe to keep calling after failure: every subsequent
    // read() short-circuits before touching buffer_ at all.
    template <typename T>
    [[nodiscard]] T read() {
        static_assert(std::is_arithmetic_v<T>, "ByteReader::read: T must be an arithmetic leaf type");
        if (failed_ || offset_ + sizeof(T) > buffer_.size()) {
            failed_ = true;
            return T{};
        }
        T value{};
        std::memcpy(&value, buffer_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    [[nodiscard]] std::size_t bytes_read() const { return offset_; }
    [[nodiscard]] bool failed() const { return failed_; }

private:
    std::span<const std::byte> buffer_;
    std::size_t offset_ = 0;
    bool failed_ = false;
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
//
// Returns false if the source was exhausted anywhere during the walk
// (reader.failed()) -- `value` may have been partially overwritten in
// that case (whatever fields were read before truncation hit, plus
// default values for the rest), so a caller that needs all-or-nothing
// semantics on failure should stage into a temporary and only commit it
// after checking the return value (see SnapshotBuffer::deserialize() for
// exactly that pattern).
template <typename T>
[[nodiscard]] bool deserialize(T& value, ByteReader& reader) {
    if constexpr (std::is_arithmetic_v<T>) {
        value = reader.template read<T>();
    } else {
        // Inner recursive calls' own bool is deliberately unchecked here
        // -- correctness comes from checking the shared reader's
        // failed() once at the end of this function, not from
        // threading a return value through for_each_field's callback
        // (which doesn't collect one). (void) makes the discard
        // explicit rather than tripping this same [[nodiscard]].
        Descriptor<T>::for_each_field(value, [&reader](auto& field) { (void)deserialize(field, reader); });
    }
    return !reader.failed();
}

} // namespace augur::reflect
