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

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "augur/utils/assert.hpp"

namespace augur::utils {

template <typename T, std::size_t Capacity>
class FixedVector {
public:
    using value_type = T;
    using iterator = typename std::array<T, Capacity>::iterator;
    using const_iterator = typename std::array<T, Capacity>::const_iterator;

    FixedVector() = default;

    void push_back(const T& value) {
        AUGUR_ASSERT(size_ < Capacity, "FixedVector: push_back would exceed compile-time capacity");
        storage_[size_++] = value;
    }

    void push_back(T&& value) {
        AUGUR_ASSERT(size_ < Capacity, "FixedVector: push_back would exceed compile-time capacity");
        storage_[size_++] = std::move(value);
    }

    template <typename... Args>
    T& emplace_back(Args&&... args) {
        AUGUR_ASSERT(size_ < Capacity, "FixedVector: emplace_back would exceed compile-time capacity");
        storage_[size_] = T(std::forward<Args>(args)...);
        return storage_[size_++];
    }

    // Swap-and-pop removal -- O(1), does not preserve order. Fine for
    // track lists where order doesn't carry meaning; if you need
    // stable order, don't reach for this.
    void swap_remove(std::size_t index) {
        AUGUR_ASSERT(index < size_, "FixedVector: swap_remove index out of range");
        storage_[index] = std::move(storage_[size_ - 1]);
        --size_;
    }

    void clear() { size_ = 0; }

    [[nodiscard]] std::size_t size() const { return size_; }
    [[nodiscard]] static constexpr std::size_t capacity() { return Capacity; }
    [[nodiscard]] bool empty() const { return size_ == 0; }
    [[nodiscard]] bool full() const { return size_ == Capacity; }

    T& operator[](std::size_t i) { return storage_[i]; }
    const T& operator[](std::size_t i) const { return storage_[i]; }

    iterator begin() { return storage_.begin(); }
    iterator end() { return storage_.begin() + static_cast<std::ptrdiff_t>(size_); }
    const_iterator begin() const { return storage_.begin(); }
    const_iterator end() const { return storage_.begin() + static_cast<std::ptrdiff_t>(size_); }

private:
    std::array<T, Capacity> storage_{};
    std::size_t size_ = 0;
};

} // namespace augur::utils
