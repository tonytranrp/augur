# augur

C++20 motion-prediction / target-tracking library for game and mod developers,
built around an Interacting Multiple Model (IMM) estimator: mix several motion
models (constant velocity, constant acceleration, coordinated turn, ...) so the
tracker itself notices when a target stops moving in a straight line, instead
of switching models by hand. `augur` is a placeholder name — rename freely.

## Design philosophy

Header-only, C++20-concepts-constrained (`models::MotionModel`,
`filters::Filter`) instead of inheritance — any type with the right member
functions works everywhere a built-in model does, no base class, no
registration call, no vtable. No forced multithreading, ever: every algorithm
is a plain reentrant function of its inputs, so the same code runs
identically on Android and desktop without imposing a threading model on
either. Reflection uses Boost.PFR today (works on Android NDK right now)
behind a feature-detected seam for C++26 static reflection once compilers
actually ship it. Full reasoning, dependency rationale, and the "known
limitation" (same-order IMM mixing only) are in `docs/ARCHITECTURE.md` — read
it before touching `imm/`.

## Folder map

- `include/augur/core/` — concepts + feature-detection config. Foundation
  everything else depends on.
- `include/augur/utils/` — domain-agnostic helpers (`FixedVector`,
  `Stopwatch`, ...). Reused everywhere; must never know what a `MotionModel`
  is.
- `include/augur/math/` — the one choke point for linear algebra (Eigen
  aliases) and optional glm interop.
- `include/augur/models/` — motion models (constant velocity/acceleration,
  coordinated turn; Singer is a flagged sketch).
- `include/augur/filters/` — linear/extended Kalman filters;
  `filters/adaptive/` is a roadmap stub.
- `include/augur/imm/` — the star of the library: mode matrix, mixing math,
  `Estimator<Filters...>` (same-dimension only), plus the opt-in
  `HeterogeneousEstimator<Filters...>` for different-order mixing.
- `include/augur/predict/` — turns filter state into position/velocity/
  error-ellipse queries; `latency_compensation.hpp` is a roadmap stub.
- `include/augur/track/` — multi-target association/lifecycle/GM-PHD — all
  roadmap stubs.
- `include/augur/reflect/`, `include/augur/plugin/` — the mechanisms that
  make this "open" rather than hardcoded: a reflection facade over
  Boost.PFR, and the concepts-based (plus opt-in runtime registry) plugin
  system.
- `examples/`, `tests/unit/`, `docs/` — see `docs/ARCHITECTURE.md` §4 for the
  full annotated tree.

## The one standing rule

**Before implementing or modifying anything, read the actual existing source
you're touching or extending first.** This file and `.claude/rules/` are a
quick-orientation summary, not a replacement for the real code — they will
drift out of date and the code won't. When in doubt, open the file.

Detailed, topic-specific rules live in `.claude/rules/` (code style, testing,
subagent orchestration) and load automatically — no need to `@`-import them,
just know they're there.

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/examples/basic_cv_tracking
```

Tests:

```sh
cmake -B build -DAUGUR_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

Android: build with the NDK's own CMake toolchain file
(`-DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24`) — see
`docs/GETTING_STARTED.md` for the full invocation, the CMake options table
(`AUGUR_WITH_GLM`, `AUGUR_BUILD_EXAMPLES`, `AUGUR_BUILD_TESTS`), and how to
consume `augur` from your own CMake project via CPM.
