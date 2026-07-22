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

Every item in `docs/ROADMAP.md` (14 extension ideas plus the different-order
IMM mixing structural improvement) is implemented and tested — nothing
below is an unimplemented stub anymore. "Flagged sketch" means it works and
is tested, but a named part is a documented simplification; see the file's
own top comment and `docs/ROADMAP.md` for specifics. Full tier breakdown:
`docs/ARCHITECTURE.md` §6.

- `include/augur/core/` — concepts + feature-detection config +
  `state_component.hpp` (augmented-layout enum for different-order IMM
  mixing). Foundation everything else depends on.
- `include/augur/utils/` — domain-agnostic helpers (`FixedVector`,
  `Stopwatch`, ...). Reused everywhere; must never know what a `MotionModel`
  is.
- `include/augur/math/` — the one choke point for linear algebra (Eigen
  aliases, `safe_inverse`/`project_to_psd`) and optional glm interop.
- `include/augur/models/` — constant velocity/acceleration and coordinated
  turn are solid; Singer and Current Statistical are flagged sketches
  (Singer's process-noise term is a documented simplification, not its
  exact closed-form integral; Current Statistical's adaptive-mean feedback
  needed damping/clamping to avoid unbounded divergence, and still biases
  toward its clamp under a very long sustained maneuver).
- `include/augur/filters/` — linear, extended, unscented Kalman, and
  particle filters are solid; `filters/adaptive/sage_husa.hpp` is a flagged
  sketch (assumes a direct-position measurement model, stated in its own
  file comment).
- `include/augur/imm/` — the star of the library: mode matrix, mixing math,
  `Estimator<Filters...>` (same-dimension only), plus the opt-in
  `HeterogeneousEstimator<Filters...>` for different-order mixing — a
  flagged sketch (works and is tested, with a documented characteristic
  around how confident a non-dominant model can be about state the others
  don't share; see `docs/ROADMAP.md` item 0).
- `include/augur/predict/` — `query.hpp` (2D/3D error-ellipse queries) and
  `latency_compensation.hpp` (`SnapshotBuffer` + `predict_to_render_time`)
  are both solid.
- `include/augur/track/` — data association (GNN + JPDA), track lifecycle
  management, out-of-sequence measurement handling, and sensor fusion are
  solid; `gm_phd.hpp` is a flagged sketch (`extract_targets()` doesn't yet
  split an above-weight-1 component into multiple targets).
- `include/augur/reflect/`, `include/augur/plugin/` — the mechanisms that
  make this "open" rather than hardcoded: a reflection facade
  (`descriptor.hpp`) dispatching between a Boost.PFR backend (plain
  aggregates) and a hand-written Eigen-vector backend (PFR can't reflect
  Eigen's non-aggregate matrix type), driving `serialize.hpp`'s binary
  (de)serialization; and the concepts-based (plus opt-in runtime registry)
  plugin system. Both solid.
- `examples/`, `tests/unit/`, `docs/` — one example and one test file per
  roadmap item (15 examples, 17 test files); see `docs/ARCHITECTURE.md` §4
  for the full annotated tree.

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
