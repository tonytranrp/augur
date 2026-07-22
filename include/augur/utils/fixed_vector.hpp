#pragma once
// augur/utils/fixed_vector.hpp
//
// A std::vector-like container backed by inline storage with a
// compile-time capacity -- no heap allocation, ever. Used wherever the
// library needs a bounded collection in a hot path: track/track_manager.hpp
// (a game realistically has at most N active tracks), predict/ batching,
// and anywhere else "small N, known upper bound, called every frame"
// applies. This is the single implementation every one of those modules
// reuses rather than each hand-rolling its own inline array + count.
//
// Deliberately minimal (push_back/size/operator[]/iteration only) --
// this is not trying to be a general-purpose container library, just
// enough for augur's own internal use and for consumers who want the
// same "no heap in the hot path" property in their own code.
//
// STORAGE, stated plainly (docs/IMPROVEMENT_PLAN.md, found independently
// by two separate investigations): an earlier version of this file used
// `std::array<T, Capacity> storage_{}`, which unconditionally
// default-constructs all Capacity slots up front regardless of logical
// size -- both a real cost (measured 10.5x a placement-new equivalent
// for constructing+filling 16-of-256 slots; 154x for
// TrackManager::step()'s real per-frame pattern of rebuilding a
// generously-capacity'd FixedVector every frame) and a real usability
// gap (requires T to be default-constructible at all, which is exactly
// why track/track_manager.hpp's own Track struct has to wrap its filter
// field in std::optional<FilterT> as a workaround).
//
// A first attempt at fixing this used `std::array<std::optional<T>,
// Capacity>` instead -- simpler and much lower-risk than hand-rolled
// placement-new, since every special member function could stay
// compiler-generated, correct by construction via std::optional's own
// already-correct lifetime management. It fixed the default-
// constructible requirement and the double-construct-then-assign cost
// in push_back/emplace_back, but NOT the actual O(Capacity) headline
// cost the plan measured -- confirmed by direct A/B timing before
// committing to it (ad hoc clang++, per .claude/rules/testing.md): even
// bare `std::array<std::optional<Payload>,1024>{}` alone was SLOWER
// than `std::array<Payload,1024>{}` (1.43us vs 0.95us on this
// toolchain), because each optional's one-byte "engaged" discriminant
// sits at a non-contiguous stride (sizeof(optional<T>) > sizeof(T)),
// which defeats the compiler's ability to vectorize "clear 1024 flags"
// into one contiguous memset the way it can for "value-initialize 1024
// plain structs." Reported honestly rather than shipped anyway --
// exactly the kind of measured, not assumed, correction this project's
// own culture asks for (docs/ROADMAP.md's "Fixes found along the way").
//
// This version instead uses raw, uninitialized storage (an array of
// Capacity byte-sized "slots," each individually aligned for T) plus
// explicit placement-new/manual destructor calls for exactly the
// `size_` live elements -- mirroring the soon-to-be-standardized
// std::inplace_vector/P0843's own actual internal representation, and
// the docs/IMPROVEMENT_PLAN.md recommendation. Declaring the storage
// array WITHOUT `{}` (default-, not value-initialization) means
// constructing an empty-or-mostly-empty FixedVector, at ANY Capacity,
// touches zero bytes of that storage -- genuinely O(size_), not
// O(Capacity), confirmed by the SAME timing comparison that caught the
// std::optional attempt's regression (see the commit message for this
// file's history for the actual before/after numbers). std::launder is
// used at every point a T is actually accessed through a pointer formed
// before that T's own placement-new (required by [basic.life] -- the
// same object-lifetime rule std::vector's own implementations have to
// satisfy for their own raw-storage-backed internals), not merely for
// pointer arithmetic/formation, which doesn't require it.
//
// push_back/emplace_back past capacity, and operator[]/swap_remove past
// size, remain AUGUR_ASSERT-gated (a no-op under NDEBUG/Release) --
// these document an internal invariant the caller controls (how many
// elements they push relative to Capacity), the same category as most
// asserts elsewhere in this codebase, not the externally-sourced-bytes
// trust boundary reflect/serialize.hpp's ByteReader has to treat
// differently.

#include <array>
#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

#include "augur/utils/assert.hpp"

namespace augur::utils {

template <typename T, std::size_t Capacity>
class FixedVector {
public:
    using value_type = T;
    using iterator = T*;
    using const_iterator = const T*;

    FixedVector() = default;

    FixedVector(const FixedVector& other) { assign_copy(other); }
    FixedVector(FixedVector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) { assign_move(other); }

    FixedVector& operator=(const FixedVector& other) {
        if (this == &other) return *this;
        clear();
        assign_copy(other);
        return *this;
    }
    FixedVector& operator=(FixedVector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
        if (this == &other) return *this;
        clear();
        assign_move(other);
        return *this;
    }

    ~FixedVector() { clear(); }

    void push_back(const T& value) { emplace_back(value); }
    void push_back(T&& value) { emplace_back(std::move(value)); }

    template <typename... Args>
    T& emplace_back(Args&&... args) {
        AUGUR_ASSERT(size_ < Capacity, "FixedVector: emplace_back would exceed compile-time capacity");
        T* p = ::new (static_cast<void*>(&storage_[size_])) T(std::forward<Args>(args)...);
        ++size_;
        return *p;
    }

    // Swap-and-pop removal -- O(1), does not preserve order. Fine for
    // track lists where order doesn't carry meaning; if you need
    // stable order, don't reach for this.
    void swap_remove(std::size_t index) {
        AUGUR_ASSERT(index < size_, "FixedVector: swap_remove index out of range");
        (*this)[index] = std::move((*this)[size_ - 1]); // both slots already live: ordinary move-assignment
        (*this)[size_ - 1].~T(); // destroys the now-duplicate tail element
        --size_;
    }

    void clear() {
        for (std::size_t i = 0; i < size_; ++i) (*this)[i].~T();
        size_ = 0;
    }

    [[nodiscard]] std::size_t size() const { return size_; }
    [[nodiscard]] static constexpr std::size_t capacity() { return Capacity; }
    [[nodiscard]] bool empty() const { return size_ == 0; }
    [[nodiscard]] bool full() const { return size_ == Capacity; }

    T& operator[](std::size_t i) { return *std::launder(reinterpret_cast<T*>(&storage_[i])); }
    const T& operator[](std::size_t i) const { return *std::launder(reinterpret_cast<const T*>(&storage_[i])); }

    // Pointer arithmetic/formation (unlike dereferencing) doesn't
    // require std::launder -- only operator[]/operator* above, which
    // actually access a T through the pointer, do.
    iterator begin() { return reinterpret_cast<T*>(&storage_[0]); }
    iterator end() { return reinterpret_cast<T*>(&storage_[0]) + size_; }
    const_iterator begin() const { return reinterpret_cast<const T*>(&storage_[0]); }
    const_iterator end() const { return reinterpret_cast<const T*>(&storage_[0]) + size_; }

private:
    // Alignas(T) on the slot type, not the array, is what guarantees
    // every element -- not just the first -- is correctly aligned for
    // T: sizeof(T) is always a multiple of alignof(T) (a standard
    // guarantee, so arrays of T work at all), so a slot exactly
    // sizeof(T) bytes wide keeps every subsequent slot's offset a
    // multiple of alignof(T) too. Deliberately NOT std::aligned_storage
    // (deprecated in C++23; this project targets C++20 as its floor but
    // avoids reaching for facilities already on their way out).
    struct alignas(T) Slot {
        std::byte bytes[sizeof(T)];
    };

    void assign_copy(const FixedVector& other) {
        for (std::size_t i = 0; i < other.size_; ++i) emplace_back(other[i]);
    }
    void assign_move(FixedVector& other) {
        for (std::size_t i = 0; i < other.size_; ++i) emplace_back(std::move(other[i]));
        other.clear();
    }

    std::array<Slot, Capacity> storage_; // deliberately NOT {}-initialized: see this file's own top comment
    std::size_t size_ = 0;
};

} // namespace augur::utils
