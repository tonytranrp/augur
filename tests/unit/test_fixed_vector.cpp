// tests/unit/test_fixed_vector.cpp
//
// docs/IMPROVEMENT_PLAN.md: FixedVector's std::array<T,Capacity>-backed
// storage unconditionally default-constructed all Capacity slots
// regardless of logical size, requiring T to be default-constructible at
// all -- both fixed by switching to std::array<std::optional<T>,Capacity>
// internally (see fixed_vector.hpp's own file comment for the full
// reasoning). This file specifically exercises the properties that
// change directly: a non-default-constructible element type actually
// compiles and works now, and swap_remove()/clear() genuinely destroy
// elements rather than just decrementing size_ (a real risk any manual-
// or optional-based lifetime scheme could get wrong).

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <memory>
#include "augur/utils/fixed_vector.hpp"

namespace {

// No default constructor at all -- the exact type the OLD
// std::array<T,Capacity>-backed FixedVector could not have held.
struct NotDefaultConstructible {
    explicit NotDefaultConstructible(int v) : value(v) {}
    int value;
};

// Tracks live construction/destruction counts so swap_remove()/clear()
// can be checked for ACTUALLY destroying elements, not just forgetting
// about them.
struct DtorCounter {
    static inline int live = 0;
    int value = 0;

    DtorCounter() { ++live; }
    explicit DtorCounter(int v) : value(v) { ++live; }
    DtorCounter(const DtorCounter& other) : value(other.value) { ++live; }
    DtorCounter(DtorCounter&& other) noexcept : value(other.value) { ++live; }
    DtorCounter& operator=(const DtorCounter&) = default;
    DtorCounter& operator=(DtorCounter&&) = default;
    ~DtorCounter() { --live; }
};

} // namespace

TEST_CASE("FixedVector supports a non-default-constructible element type", "[utils][fixed_vector]") {
    augur::utils::FixedVector<NotDefaultConstructible, 4> v;
    v.push_back(NotDefaultConstructible{1});
    v.emplace_back(2);

    REQUIRE(v.size() == 2);
    REQUIRE(v[0].value == 1);
    REQUIRE(v[1].value == 2);
}

TEST_CASE("FixedVector supports a move-only element type", "[utils][fixed_vector]") {
    augur::utils::FixedVector<std::unique_ptr<int>, 4> v;
    v.push_back(std::make_unique<int>(42));
    v.emplace_back(std::make_unique<int>(7));

    REQUIRE(v.size() == 2);
    REQUIRE(*v[0] == 42);
    REQUIRE(*v[1] == 7);
}

TEST_CASE("FixedVector::push_back/emplace_back construct in place, matching std::vector semantics", "[utils][fixed_vector]") {
    augur::utils::FixedVector<int, 4> v;
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);

    REQUIRE(v.size() == 3);
    REQUIRE(v[0] == 1);
    REQUIRE(v[1] == 2);
    REQUIRE(v[2] == 3);
    REQUIRE_FALSE(v.empty());
    REQUIRE_FALSE(v.full());

    v.push_back(4);
    REQUIRE(v.full());
    REQUIRE(v.capacity() == 4);
}

TEST_CASE("FixedVector::swap_remove genuinely destroys the removed element", "[utils][fixed_vector]") {
    DtorCounter::live = 0;
    {
        augur::utils::FixedVector<DtorCounter, 4> v;
        v.emplace_back(1);
        v.emplace_back(2);
        v.emplace_back(3);
        REQUIRE(DtorCounter::live == 3);

        v.swap_remove(0); // moves index 2's value into slot 0, destroys the (now-duplicate) tail
        REQUIRE(v.size() == 2);
        REQUIRE(DtorCounter::live == 2); // not 3 -- the old tail slot must be destroyed, not just orphaned
        REQUIRE(v[0].value == 3);
        REQUIRE(v[1].value == 2);
    }
    REQUIRE(DtorCounter::live == 0); // container destructor cleaned up the rest
}

TEST_CASE("FixedVector::clear genuinely destroys every element", "[utils][fixed_vector]") {
    DtorCounter::live = 0;
    {
        augur::utils::FixedVector<DtorCounter, 8> v;
        for (int i = 0; i < 5; ++i) v.emplace_back(i);
        REQUIRE(DtorCounter::live == 5);

        v.clear();
        REQUIRE(v.size() == 0);
        REQUIRE(v.empty());
        REQUIRE(DtorCounter::live == 0); // not 5 -- clear() must destroy, not just reset size_

        v.emplace_back(99); // reusable after clear()
        REQUIRE(v.size() == 1);
        REQUIRE(DtorCounter::live == 1);
    }
    REQUIRE(DtorCounter::live == 0);
}

TEST_CASE("FixedVector's destructor destroys exactly the live elements, none of the unused capacity", "[utils][fixed_vector]") {
    DtorCounter::live = 0;
    {
        augur::utils::FixedVector<DtorCounter, 256> v; // large capacity, few live elements
        v.emplace_back(1);
        v.emplace_back(2);
        REQUIRE(DtorCounter::live == 2); // NOT 256 -- unused capacity never constructs a T at all
    }
    REQUIRE(DtorCounter::live == 0);
}

TEST_CASE("FixedVector's iterators satisfy std::sort's requirements", "[utils][fixed_vector]") {
    // Matches track/gm_phd.hpp's own real usage pattern
    // (std::sort(order.begin(), order.end(), ...)) -- std::sort needs a
    // genuine LegacyRandomAccessIterator, not just forward iteration.
    augur::utils::FixedVector<int, 8> v;
    for (int x : {5, 3, 8, 1, 9, 2}) v.push_back(x);

    std::sort(v.begin(), v.end());
    REQUIRE(std::is_sorted(v.begin(), v.end()));
    REQUIRE(v[0] == 1);
    REQUIRE(v[5] == 9);

    std::sort(v.begin(), v.end(), std::greater<int>{});
    REQUIRE(v[0] == 9);
    REQUIRE(v[5] == 1);
}

TEST_CASE("FixedVector supports range-based for and const iteration", "[utils][fixed_vector]") {
    augur::utils::FixedVector<int, 4> v;
    v.push_back(10);
    v.push_back(20);
    v.push_back(30);

    int sum = 0;
    for (int x : v) sum += x;
    REQUIRE(sum == 60);

    const auto& cv = v;
    int const_sum = 0;
    for (const int& x : cv) const_sum += x;
    REQUIRE(const_sum == 60);

    REQUIRE(std::distance(v.begin(), v.end()) == 3);
}

TEST_CASE("FixedVector copy and move construction/assignment work correctly", "[utils][fixed_vector]") {
    augur::utils::FixedVector<int, 4> original;
    original.push_back(1);
    original.push_back(2);

    augur::utils::FixedVector<int, 4> copy = original;
    REQUIRE(copy.size() == 2);
    REQUIRE(copy[0] == 1);
    REQUIRE(copy[1] == 2);
    original.push_back(3); // mutating the original must not affect the copy
    REQUIRE(copy.size() == 2);

    augur::utils::FixedVector<int, 4> moved = std::move(original);
    REQUIRE(moved.size() == 3);
    REQUIRE(moved[2] == 3);

    augur::utils::FixedVector<int, 4> assigned;
    assigned.push_back(99);
    assigned = copy;
    REQUIRE(assigned.size() == 2);
    REQUIRE(assigned[0] == 1);
}
