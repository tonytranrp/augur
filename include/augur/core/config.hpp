#pragma once
// augur/core/config.hpp
//
// Central feature-detection point. Nothing in this file does any work --
// it only decides, at compile time, which backend the rest of the library
// should use for reflection, SIMD, etc. Every other header asks *this*
// file "is X available", never the raw __has_include/__cpp_* macros
// directly, so the detection logic lives in exactly one place.
//
// Why this matters for a "universal" library: the same augur.hpp is
// included on a desktop build with a bleeding-edge Clang trunk AND on an
// Android NDK build with a toolchain that's a year or two behind. Every
// "modern" feature has to be opt-in-if-available, never assumed.

// ---------------------------------------------------------------------
// C++26 static reflection (P2996) -- voted into the C++26 working draft
// at the June 2025 WG21 Sofia meeting, but as of this writing it ships
// only in experimental branches (Bloomberg's clang-p2996 fork, GCC
// trunk); MSVC has no public support/ETA, and Android NDK's Clang is
// several releases behind mainline, so it does NOT have it either.
// This macro will simply flip to 1 on its own once a compiler defines
// the real feature-test macro -- no other file needs to change.
// ---------------------------------------------------------------------
#if defined(__cpp_impl_reflection) || defined(__cpp_lib_reflection)
  #define AUGUR_HAS_STD_REFLECTION 1
#else
  #define AUGUR_HAS_STD_REFLECTION 0
#endif

// ---------------------------------------------------------------------
// Boost.PFR -- pure-library, C++14/17 "poor man's reflection" via
// aggregate structured-binding tricks. No compiler support required
// beyond a conforming C++17 front end, so it works TODAY on every
// platform this library targets, Android NDK included. This is the
// default reflection backend until AUGUR_HAS_STD_REFLECTION is 1.
// ---------------------------------------------------------------------
#if __has_include(<boost/pfr.hpp>)
  #define AUGUR_HAS_PFR 1
#else
  #define AUGUR_HAS_PFR 0
#endif

// ---------------------------------------------------------------------
// glm interop is opt-in via CMake (AUGUR_WITH_GLM), which defines
// AUGUR_WITH_GLM=1 on the `augur` target when enabled.
// ---------------------------------------------------------------------
#if !defined(AUGUR_WITH_GLM)
  #define AUGUR_WITH_GLM 0
#endif

// ---------------------------------------------------------------------
// Threading policy: deliberate non-feature.
//
// augur never spawns a thread, never requires a thread pool, and never
// links a threading runtime. Every algorithm here is a plain function
// of its inputs, safe to call from whatever thread *you* happen to be
// on (main thread, a job-system worker, an Android UI thread, doesn't
// matter) as long as you don't share one Estimator/Filter instance
// across threads without your own synchronization -- same rule as
// std::vector. This is a deliberate design constraint, not a TODO:
// a single-threaded, reentrant core is what makes the library equally
// at home on a phone SoC and a 32-core desktop. If you want to batch
// predictions across many tracks in parallel, do that OUTSIDE augur
// (e.g. a parallel `for_each` over a std::vector<Estimator<...>> in
// your own job system) -- augur will never fight your threading model.
// ---------------------------------------------------------------------
#define AUGUR_SINGLE_THREADED 1

// Eigen must not silently pull in OpenMP: on Android/iOS/macOS default
// toolchains it isn't linked anyway, but we make the intent explicit
// and platform-independent rather than relying on that default.
#ifndef EIGEN_DONT_PARALLELIZE
  #define EIGEN_DONT_PARALLELIZE
#endif

namespace augur::core {

// Compile-time facts callers can query without preprocessor `#if` soup
// in their own code.
struct BuildInfo {
    static constexpr bool has_std_reflection = static_cast<bool>(AUGUR_HAS_STD_REFLECTION);
    static constexpr bool has_pfr            = static_cast<bool>(AUGUR_HAS_PFR);
    static constexpr bool has_glm_interop    = static_cast<bool>(AUGUR_WITH_GLM);
    static constexpr bool single_threaded    = true;
};

} // namespace augur::core
