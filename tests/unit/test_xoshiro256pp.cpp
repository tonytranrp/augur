// tests/unit/test_xoshiro256pp.cpp
//
// docs/IMPROVEMENT_PLAN.md: filters/particle_filter.hpp now uses this
// hand-vendored engine instead of std::mt19937_64 + std::normal_distribution
// (see xoshiro256pp.hpp's own file comment for the full reasoning:
// measured 1.4-1.8x faster per sample, ~200x faster to construct, and
// std::normal_distribution's actual output was confirmed to genuinely
// diverge between libc++ and MSVC STL even given an identical engine
// and seed). Reference values below were computed independently (ad hoc
// python3, per .claude/rules/testing.md) before this file was written,
// including bit-exact raw generator output for a fixed seed -- the
// strongest possible check that this port matches the reference
// algorithm (Blackman & Vigna 2021) exactly, not just "looks
// statistically fine."

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <random>
#include "augur/utils/xoshiro256pp.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("Xoshiro256PlusPlus matches an independently-computed reference sequence exactly", "[utils][rng]") {
    // seed=42, first 5 raw draws -- computed independently in python3
    // (SplitMix64 seeding + the reference xoshiro256++ algorithm) before
    // this file was written. Bit-exact agreement here means this port
    // has no transcription error anywhere in the algorithm.
    augur::utils::Xoshiro256PlusPlus rng(42);
    constexpr std::uint64_t expected[5] = {
        15021278609987233951ULL, 5881210131331364753ULL, 18149643915985481100ULL,
        12933668939759105464ULL, 14637574242682825331ULL,
    };
    for (std::uint64_t e : expected) {
        REQUIRE(rng() == e);
    }
}

TEST_CASE("Xoshiro256PlusPlus is reproducible for the same seed and distinct for different seeds", "[utils][rng]") {
    augur::utils::Xoshiro256PlusPlus a(100);
    augur::utils::Xoshiro256PlusPlus b(100);
    augur::utils::Xoshiro256PlusPlus c(101);

    bool any_differs_from_c = false;
    for (int i = 0; i < 10; ++i) {
        const auto va = a();
        const auto vb = b();
        const auto vc = c();
        REQUIRE(va == vb); // same seed -> same sequence
        if (va != vc) any_differs_from_c = true;
    }
    REQUIRE(any_differs_from_c); // different seed -> (overwhelmingly likely) different sequence
}

TEST_CASE("Xoshiro256PlusPlus satisfies UniformRandomBitGenerator", "[utils][rng]") {
    // Required for use with std::*_distribution (e.g.
    // particle_filter.hpp's own std::uniform_real_distribution in
    // systematic_resample()) -- a compile-time check via std::uniform_random_bit_generator.
    STATIC_REQUIRE(std::uniform_random_bit_generator<augur::utils::Xoshiro256PlusPlus>);
}

TEST_CASE("Xoshiro256PlusPlus's raw output is statistically uniform", "[utils][rng]") {
    augur::utils::Xoshiro256PlusPlus rng(123);
    constexpr int N = 200000;
    double sum = 0, sumsq = 0;
    for (int i = 0; i < N; ++i) {
        const double u = static_cast<double>(rng()) / 18446744073709551616.0; // / 2^64
        sum += u;
        sumsq += u * u;
    }
    const double mean = sum / N;
    const double variance = sumsq / N - mean * mean;
    REQUIRE_THAT(mean, WithinAbs(0.5, 0.01));
    REQUIRE_THAT(std::sqrt(variance), WithinAbs(1.0 / std::sqrt(12.0), 0.01)); // theoretical uniform(0,1) std
}

TEST_CASE("StandardNormalSampler produces correctly-distributed standard normal samples", "[utils][rng]") {
    augur::utils::Xoshiro256PlusPlus rng(7);
    augur::utils::StandardNormalSampler<double> sampler;
    constexpr int N = 500000;
    double sum = 0, sumsq = 0;
    for (int i = 0; i < N; ++i) {
        const double v = sampler(rng);
        sum += v;
        sumsq += v * v;
    }
    const double mean = sum / N;
    const double variance = sumsq / N - mean * mean;
    REQUIRE_THAT(mean, WithinAbs(0.0, 0.01));
    REQUIRE_THAT(std::sqrt(variance), WithinAbs(1.0, 0.01));
}

TEST_CASE("StandardNormalSampler works with Scalar=float, matching this library's default", "[utils][rng]") {
    augur::utils::Xoshiro256PlusPlus rng(7);
    augur::utils::StandardNormalSampler<float> sampler;
    constexpr int N = 200000;
    float sum = 0, sumsq = 0;
    for (int i = 0; i < N; ++i) {
        const float v = sampler(rng);
        sum += v;
        sumsq += v * v;
    }
    const float mean = sum / static_cast<float>(N);
    const float variance = sumsq / static_cast<float>(N) - mean * mean;
    REQUIRE_THAT(mean, WithinAbs(0.0f, 0.02f));
    REQUIRE_THAT(std::sqrt(variance), WithinAbs(1.0f, 0.02f));
}
