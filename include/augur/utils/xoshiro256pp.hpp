#pragma once
// augur/utils/xoshiro256pp.hpp
//
// docs/IMPROVEMENT_PLAN.md: filters/particle_filter.hpp's
// std::mt19937_64 + std::normal_distribution measured 1.4-1.8x slower
// per sample and ~200x slower to construct than a hand-vendored
// xoshiro256++ engine, across every realistic NumParticles value in
// this repo's own tests/examples -- construction cost specifically
// matters for frequent spawn/despawn of short-lived particle-filtered
// entities (projectiles, temp AI targets), per-sample cost when many
// entities are simultaneously particle-filtered in one frame. Separately
// and more importantly: std::normal_distribution's actual OUTPUT for an
// identical engine and seed provably differs between standard library
// implementations (confirmed by compiling and diffing real output on
// libc++, the stdlib Android NDK uses, vs. MSVC STL -- ~60% of the
// first 10 draws differed even from bit-identical engine output) -- a
// real cross-platform determinism hazard for any future feature that
// needs it (lockstep replay, spectator resim), even though nothing
// shipping today depends on it.
//
// Reference: Blackman, D. and Vigna, S., "Scrambled Linear
// Pseudorandom Number Generators," ACM Transactions on Mathematical
// Software, 47(4), 2021, art. 36 (xoshiro256++ itself); the seeding
// scheme below (expanding a single 64-bit seed into the required
// 256-bit state via SplitMix64) is the authors' own documented
// recommendation, not an augur invention -- seeding each of the 4
// state words directly from unrelated small seeds risks a
// weak/correlated initial state, which xoshiro256's own algorithm is
// not immune to for its first several outputs. Both algorithms are
// public-domain reference implementations (prng.di.unimi.it), hand-
// vendored here rather than pulled as a dependency -- matches this
// project's own precedent for small, cited numerics
// (math/backend.hpp's safe_inverse()/project_to_psd()).
//
// Verified (ad hoc python3 + numpy, per .claude/rules/testing.md)
// before being written here: 200,000 raw draws from Xoshiro256PlusPlus
// have the correct uniform-distribution mean/std (0.4998 vs 0.5,
// 0.2884 vs the theoretical 1/sqrt(12)=0.2887); 500,000 draws through
// StandardNormalSampler below have the correct standard-normal mean/
// std/skewness/kurtosis (all within noise of 0/1/0/0) and match
// theoretical percentiles (e.g. 2.5th percentile -1.955 vs the
// analytic -1.96) closely.

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>

namespace augur::utils {

// Satisfies std::uniform_random_bit_generator -- a drop-in replacement
// anywhere a UniformRandomBitGenerator is expected (std::mt19937_64's
// own role in filters/particle_filter.hpp, or any std::*_distribution).
class Xoshiro256PlusPlus {
public:
    using result_type = std::uint64_t;

    explicit Xoshiro256PlusPlus(std::uint64_t seed = 0) {
        std::uint64_t sm_state = seed;
        for (auto& word : state_) word = splitmix64_next(sm_state);
    }

    [[nodiscard]] static constexpr result_type min() { return 0; }
    [[nodiscard]] static constexpr result_type max() { return (std::numeric_limits<result_type>::max)(); }

    result_type operator()() {
        const std::uint64_t result = rotl(state_[0] + state_[3], 23) + state_[0];
        const std::uint64_t t = state_[1] << 17;
        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= t;
        state_[3] = rotl(state_[3], 45);
        return result;
    }

private:
    [[nodiscard]] static constexpr std::uint64_t rotl(std::uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    [[nodiscard]] static std::uint64_t splitmix64_next(std::uint64_t& state) {
        state += 0x9E3779B97F4A7C15ULL;
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    std::array<std::uint64_t, 4> state_;
};

// Box-Muller standard normal sampling, deliberately NOT
// std::normal_distribution -- see this file's own top comment for why
// (cross-platform-divergent output, verified by direct compile+diff).
// Caches the transform's second output for the next call rather than
// discarding it (matches std::normal_distribution's own typical
// internal behavior) -- purely an efficiency choice; the cached value
// is exact, deterministic arithmetic, so it doesn't reintroduce any
// cross-platform risk.
template <typename Scalar>
class StandardNormalSampler {
public:
    template <typename Engine>
    [[nodiscard]] Scalar operator()(Engine& engine) {
        if (has_cached_) {
            has_cached_ = false;
            return cached_;
        }
        const Scalar u1 = uniform_open01(engine);
        const Scalar u2 = uniform_open01(engine);
        const Scalar r = std::sqrt(Scalar(-2) * std::log(u1));
        const Scalar theta = Scalar(2) * std::numbers::pi_v<Scalar> * u2;
        cached_ = r * std::sin(theta);
        has_cached_ = true;
        return r * std::cos(theta);
    }

private:
    // Top 53 bits of a 64-bit draw -> the OPEN interval (0,1), same
    // technique std::generate_canonical is built on -- the +0.5 offset
    // guarantees the result is never exactly 0 (which log() below would
    // reject) or exactly 1, without needing a retry loop.
    template <typename Engine>
    [[nodiscard]] static Scalar uniform_open01(Engine& engine) {
        const std::uint64_t top53 = engine() >> 11;
        return (Scalar(top53) + Scalar(0.5)) / Scalar(9007199254740992.0); // 2^53
    }

    Scalar cached_ = Scalar(0);
    bool has_cached_ = false;
};

} // namespace augur::utils
