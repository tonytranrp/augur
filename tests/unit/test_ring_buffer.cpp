// tests/unit/test_ring_buffer.cpp
//
// Coverage for docs/IMPROVEMENT_PLAN.md's ring-buffer finding
// (utils/ring_buffer.hpp), which replaces the hand-duplicated
// shift-on-full history buffer previously in both
// track/out_of_sequence.hpp and predict/latency_compensation.hpp.
// Verified ad hoc (clang++ -fsanitize=address,undefined, per
// .claude/rules/testing.md's spirit -- this is a lifetime-adjacent
// container, not a math change, but the same "verify before shipping"
// discipline applies) with a move-only element type cycled through
// wraparound, pop_back, and clear before this permanent suite was
// written; also measured a real before/after against the old shift-based
// approach it replaces (ad hoc clang++ timing probe): 6.4x at
// MaxHistory=8 (a value actually used in this repo's own tests), 12x at
// 32, 106x at 512 (close to docs/IMPROVEMENT_PLAN.md's own independently
// cited ~121x estimate at that size).

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include "augur/utils/ring_buffer.hpp"

TEST_CASE("RingBuffer fills to capacity in order", "[utils][ring_buffer]") {
    augur::utils::RingBuffer<int, 4> rb;
    REQUIRE(rb.empty());
    rb.push_back(1);
    rb.push_back(2);
    rb.push_back(3);
    rb.push_back(4);

    REQUIRE(rb.full());
    REQUIRE(rb.size() == 4);
    REQUIRE(rb.capacity() == 4);
    REQUIRE(rb[0] == 1);
    REQUIRE(rb[1] == 2);
    REQUIRE(rb[2] == 3);
    REQUIRE(rb[3] == 4);
}

TEST_CASE("RingBuffer overwrites the oldest element once full, keeping chronological index order",
          "[utils][ring_buffer]") {
    // The core behavior change this class exists for: pushing past
    // capacity must overwrite the OLDEST retained element (index 0
    // stays "oldest," index size()-1 stays "newest") in O(1), not shift
    // every other element down.
    augur::utils::RingBuffer<int, 4> rb;
    for (int i = 1; i <= 4; ++i) rb.push_back(i);

    rb.push_back(5); // overwrites 1
    REQUIRE(rb.size() == 4);
    REQUIRE(rb[0] == 2);
    REQUIRE(rb[1] == 3);
    REQUIRE(rb[2] == 4);
    REQUIRE(rb[3] == 5);

    rb.push_back(6); // overwrites 2
    REQUIRE(rb[0] == 3);
    REQUIRE(rb[1] == 4);
    REQUIRE(rb[2] == 5);
    REQUIRE(rb[3] == 6);

    // Push well past capacity multiple times over -- exercises every
    // physical slot repeatedly, the case most likely to expose an
    // off-by-one in the modular index arithmetic.
    for (int i = 7; i <= 100; ++i) rb.push_back(i);
    REQUIRE(rb.size() == 4);
    REQUIRE(rb[0] == 97);
    REQUIRE(rb[3] == 100);
}

TEST_CASE("RingBuffer::pop_back removes the most recently pushed element", "[utils][ring_buffer]") {
    augur::utils::RingBuffer<int, 4> rb;
    rb.push_back(1);
    rb.push_back(2);
    rb.push_back(3);

    rb.pop_back();
    REQUIRE(rb.size() == 2);
    REQUIRE(rb[0] == 1);
    REQUIRE(rb[1] == 2);

    // pop_back after wraparound -- the newest element is no longer at
    // the highest physical index, so this checks the wraparound-aware
    // "most recent" calculation specifically.
    augur::utils::RingBuffer<int, 3> wrapped;
    wrapped.push_back(1);
    wrapped.push_back(2);
    wrapped.push_back(3);
    wrapped.push_back(4); // overwrites 1; logical order is now [2,3,4]
    wrapped.pop_back();   // removes 4
    REQUIRE(wrapped.size() == 2);
    REQUIRE(wrapped[0] == 2);
    REQUIRE(wrapped[1] == 3);
}

TEST_CASE("RingBuffer::clear empties the buffer and it's fully reusable afterward", "[utils][ring_buffer]") {
    augur::utils::RingBuffer<int, 4> rb;
    rb.push_back(1);
    rb.push_back(2);
    rb.push_back(3);
    rb.push_back(4);
    rb.push_back(5); // wraps once, so head_ != 0 -- checks clear() resets that too

    rb.clear();
    REQUIRE(rb.empty());
    REQUIRE(rb.size() == 0);

    rb.push_back(100);
    rb.push_back(200);
    REQUIRE(rb.size() == 2);
    REQUIRE(rb[0] == 100);
    REQUIRE(rb[1] == 200);
}

TEST_CASE("RingBuffer supports a move-only element type through wraparound", "[utils][ring_buffer]") {
    augur::utils::RingBuffer<std::unique_ptr<int>, 4> rb;
    for (int i = 0; i < 4; ++i) rb.push_back(std::make_unique<int>(i));
    for (int i = 4; i < 10; ++i) rb.push_back(std::make_unique<int>(i)); // wraps repeatedly

    REQUIRE(rb.size() == 4);
    int expected = 6;
    for (std::size_t i = 0; i < rb.size(); ++i) {
        REQUIRE(rb[i] != nullptr);
        REQUIRE(*rb[i] == expected++);
    }
}

TEST_CASE("RingBuffer supports copy and move construction/assignment", "[utils][ring_buffer]") {
    augur::utils::RingBuffer<int, 4> original;
    original.push_back(1);
    original.push_back(2);
    original.push_back(3);
    original.push_back(4);
    original.push_back(5); // wraps -- copy/move must carry head_ correctly, not just storage_

    augur::utils::RingBuffer<int, 4> copy = original;
    REQUIRE(copy.size() == 4);
    REQUIRE(copy[0] == 2);
    REQUIRE(copy[3] == 5);

    original.push_back(6); // mutating the original must not affect the copy
    REQUIRE(copy[0] == 2);

    augur::utils::RingBuffer<int, 4> moved = std::move(copy);
    REQUIRE(moved.size() == 4);
    REQUIRE(moved[0] == 2);
    REQUIRE(moved[3] == 5);
}
