#pragma once
// augur/utils/ring_buffer.hpp
//
// docs/IMPROVEMENT_PLAN.md: track/out_of_sequence.hpp and
// predict/latency_compensation.hpp each hand-rolled an identical
// shift-on-full history buffer (once a FixedVector reaches its
// MaxHistory capacity, appending shifts every existing element down by
// one -- O(MaxHistory) per push, and the same ~10 lines duplicated
// twice). A head-index ring buffer makes the full case O(1) instead --
// overwrite the oldest slot and advance where "oldest" points to,
// rather than moving every other element -- and gives both call sites
// one shared implementation instead of two hand-copied ones.
//
// Deliberately NOT the same placement-new/raw-storage design as
// utils/fixed_vector.hpp: that complexity earns its keep there because
// FixedVector is used at both trivially-small AND deliberately-
// oversized capacities (docs/IMPROVEMENT_PLAN.md's own finding was
// specifically about a large Capacity with few live elements), and
// needs to support types with no default constructor. This class's two
// actual call sites use small, fixed-size history buffers (Entry/
// Snapshot structs, both already default-constructible with sensible
// member defaults) where a plain std::array<T,Capacity> storage_{}'s
// upfront construction cost is exactly what
// docs/IMPROVEMENT_PLAN.md's own measurement called negligible. Keeping
// this simple matches the plan's own effort estimate ("~15-20 lines")
// and this project's "don't add complexity beyond what the task
// requires" convention -- reach for FixedVector's heavier design only
// if a future consumer of THIS class also needs non-default-
// constructible elements or a much larger capacity.

#include <array>
#include <cstddef>
#include <utility>
#include "augur/utils/assert.hpp"

namespace augur::utils {

template <typename T, std::size_t Capacity>
class RingBuffer {
public:
    void push_back(const T& value) { emplace_slot() = value; }
    void push_back(T&& value) { emplace_slot() = std::move(value); }

    // Removes the MOST RECENTLY pushed element -- O(1), no data
    // movement (unlike FixedVector::swap_remove(), this never needs to
    // relocate a different element into the removed slot's place,
    // since "most recent" is always physically at
    // (head_+size_-1)%Capacity, so shrinking size_ by one is already
    // correct on its own). This is the operation
    // track/out_of_sequence.hpp's insert_out_of_sequence() actually
    // needs (repeatedly discarding its own most-recent entries to make
    // room for a replay) -- a general swap_remove(index) isn't, since a
    // ring buffer would need to shift or rotate to keep the remaining
    // elements' relative time-order contiguous, which defeats the
    // point of this class.
    void pop_back() {
        AUGUR_ASSERT(size_ > 0, "RingBuffer: pop_back on an empty buffer");
        --size_;
    }

    void clear() { head_ = 0; size_ = 0; }

    [[nodiscard]] std::size_t size() const { return size_; }
    [[nodiscard]] static constexpr std::size_t capacity() { return Capacity; }
    [[nodiscard]] bool empty() const { return size_ == 0; }
    [[nodiscard]] bool full() const { return size_ == Capacity; }

    // Index 0 is always the OLDEST retained element, index size()-1 the
    // newest -- callers see the same chronologically-ordered view a
    // shift-based buffer already gave them, the wraparound is entirely
    // internal.
    T& operator[](std::size_t i) { return storage_[physical_index(i)]; }
    const T& operator[](std::size_t i) const { return storage_[physical_index(i)]; }

private:
    [[nodiscard]] std::size_t physical_index(std::size_t logical_index) const {
        AUGUR_ASSERT(logical_index < size_, "RingBuffer: index out of range");
        return (head_ + logical_index) % Capacity;
    }

    T& emplace_slot() {
        if (size_ < Capacity) {
            return storage_[(head_ + size_++) % Capacity];
        }
        T& slot = storage_[head_];
        head_ = (head_ + 1) % Capacity;
        return slot;
    }

    std::array<T, Capacity> storage_{};
    std::size_t head_ = 0;
    std::size_t size_ = 0;
};

} // namespace augur::utils
