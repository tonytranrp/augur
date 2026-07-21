# augur — Architecture & Design Document

`augur` is a C++20 motion-prediction / target-tracking library built around an
Interacting Multiple Model (IMM) estimator, aimed at game and mod developers who
need to answer "where is this thing going to be" — for aim assist, AI target
tracking, or netcode extrapolation — without hand-deriving Kalman filter math
every time. This document is the precise plan: what it covers, why each design
decision was made, what's actually solid today versus what's a documented
roadmap stub, and how to build it. `docs/ROADMAP.md`, `docs/GETTING_STARTED.md`,
and `docs/PLUGIN_GUIDE.md` go deeper on their respective slices; this file is
the map.

`augur` is a suggested/placeholder name — rename freely, it doesn't appear
anywhere load-bearing outside the namespace.

## 1. What this library covers

- **Motion models** (`models/`): the deterministic + stochastic dynamics of a
  moving target — constant velocity, constant acceleration, coordinated turn,
  and (as a flagged sketch) the Singer maneuvering-target model.
- **Filters** (`filters/`): the recursive estimation rule applied to a model —
  linear Kalman filter and Extended Kalman filter today; UKF, particle filter,
  and adaptive noise estimation are roadmap items.
- **IMM** (`imm/`): the actual point of the library — mixing several
  models/filters together so the tracker itself notices when a target stops
  moving in a straight line, instead of you manually switching models.
- **Prediction queries** (`predict/`): turning a filter/estimator's internal
  state into what a caller actually wants — a position, a velocity, an
  uncertainty ellipse for debug draw, a state extrapolated to a render or
  network-compensation horizon.
- **Multi-target scaffolding** (`track/`): data association, track lifecycle,
  and (as an advanced stretch item) GM-PHD for unknown target counts —
  documented interface-level stubs, not implemented yet.
- **Reflection** (`reflect/`) and **plugin** (`plugin/`) layers: the mechanisms
  that make the rest of this "open" rather than hardcoded (see §3 and §4).

What it deliberately does **not** cover: rendering, input handling, or the
firing decision itself (that's the separate `aim`-style library from the
original survey — `predict/` exposes what such a library would consume).

## 2. Design philosophy

### 2.1 Template + namespace hybrid API

Every type in `augur` lives in a nested namespace (`augur::models`,
`augur::filters`, `augur::imm`, ...) for logical grouping, **and** the
interesting configuration happens through template parameters, not through
runtime flags or inheritance:

```cpp
using CV = augur::models::ConstantVelocity<float, 3>;      // 3D, float precision
using KF = augur::filters::KalmanFilter<CV, /*MeasDim=*/3>;
augur::imm::Estimator<KF, KF, KF> tracker{ /* ... */ };
```

The `<>` isn't decoration — `ConstantVelocity<float, 3>` and
`ConstantVelocity<double, 2>` are different, unrelated types the compiler
generates independently, with zero-overhead abstraction and no vtable
anywhere in the hot path. The `::` isn't decoration either — it's what lets
`augur::filters::Filter` and `augur::models::MotionModel` exist as distinctly
named concepts you constrain against, rather than one flat pile of names.

### 2.2 Open architecture: concepts, not inheritance

The plugin mechanism is `models::MotionModel` and `filters::Filter` — two
C++20 concepts (`core/concepts.hpp`, refined in `models/model_concept.hpp` and
`filters/filter_concept.hpp`). Any type with the right member functions
satisfies them, full stop:

```cpp
struct MyCustomModel {
    using Scalar = float;
    static constexpr std::size_t dimension = 4;
    /* transition(), jacobian(), process_noise() */
};
static_assert(augur::models::MotionModel<MyCustomModel>); // just works
```

No base class, no factory registration, no virtual dispatch — `imm::Estimator`
and `filters::KalmanFilter` accept `MyCustomModel` exactly like a built-in
model, defined entirely outside `augur`'s own source tree (see
`examples/03_custom_plugin_model/`). This is the primary sense in which the
library is "open instead of hardcoded": the contract is structural, so nothing
about the library needs to change, be recompiled, or even know your type
exists.

A **second**, heavier, opt-in plugin mechanism exists for the genuinely
different problem of choosing a model at **runtime** (a config file names
which model to use, or a separately-compiled mod DLL wants to hand a host
application a new model without recompiling the host) — see
`plugin/registry.hpp`'s type-erased `FilterRegistry`. This costs a vtable
indirection and is never on the default path; it exists because a modding
context specifically (unlike a single compiled game binary) sometimes
genuinely needs runtime extensibility, not just compile-time genericity.

### 2.3 No forced multithreading

Every algorithm in `augur` is a plain, reentrant function of its inputs — no
thread is ever spawned, no thread pool is required, no threading runtime is
linked. This is a deliberate constraint (`core/config.hpp` documents it,
`AUGUR_SINGLE_THREADED` names it), not a missing feature, for one concrete
reason: **"universal" and "has an opinion about your threading model" are in
tension.** A library that assumes `std::thread`, or OpenMP, or a specific job
system behaves differently on Android (where OpenMP isn't in the default
toolchain at all) than on a 32-core desktop. Keeping the core single-threaded
and reentrant means you can safely call it from whatever thread you're
already on — main thread, a job-system worker, doesn't matter — and if you
want to batch predictions across many tracks in parallel, that's a
`parallel_for` in **your** job system over a `std::vector<Estimator<...>>`,
entirely outside this library. `augur` will never fight your threading model
because it never has one of its own.

### 2.4 Reflection: today's portable path, not tomorrow's promise

C++26 static reflection (P2996) was voted into the C++26 working draft at the
June 2025 WG21 Sofia meeting — but as of this writing it's experimental-only
(Bloomberg's `clang-p2996` fork, GCC trunk), MSVC has no public support or ETA,
and Android NDK's Clang lags mainline releases by enough that it isn't there
either. Building a "universal, works on Android and desktop today" library
around a feature that doesn't exist on one of its two named target platforms
would be a mistake, so:

- `core/config.hpp` defines `AUGUR_HAS_STD_REFLECTION`, which is `0` on every
  toolchain in practice today and flips to `1` on its own, with no other file
  changing, once a compiler defines the real feature-test macro.
- `reflect/backends/pfr_backend.hpp` implements the same operations against
  **Boost.PFR** — pure C++14/17 aggregate-destructuring, no compiler
  extension required, works identically on desktop and Android NDK right now.
  PFR only reflects actual aggregates, though, and `augur::math::Vector<
  Scalar,N>` (an Eigen type with private storage) categorically isn't one —
  confirmed by compiling a throwaway `boost::pfr::tuple_size_v<Eigen::
  Matrix<float,4,1>>` check, which fails with "Type must be aggregate
  initializable." `reflect/backends/vector_backend.hpp` covers that case
  instead, reflecting a fixed-size Eigen column vector structurally (N
  contiguous Scalars) without going through PFR at all.
- `reflect/descriptor.hpp` is the one API the rest of the codebase (and your
  code) calls; it picks a backend internally (`VectorBackend` for a fixed-
  size `Vector<Scalar,N>`, `PfrBackend` for a plain aggregate) and never
  leaks which one.
- `reflect/serialize.hpp` builds the actual (de)serialization docs/
  ROADMAP.md's "Reflection-driven (de)serialization" item asks for on top
  of `Descriptor<T>`: a fixed-width positional binary layout, recursing
  through nested fields (a struct that has a `Vector` member serializes
  correctly, composing both backends) down to arithmetic leaves.
  `predict/latency_compensation.hpp`'s `SnapshotBuffer` is the real
  consumer — `serialize()`/`deserialize()` round-trip its whole snapshot
  history for save-states or network transport.

This mirrors what the Glaze JSON library ships today: P2996 support is
opt-in over an existing portable path, not a replacement for it, with the
stated expectation that P2996 becomes preferred "as C++26 compilers mature and
[it] becomes widely available" — not before.

## 3. External dependencies

| Library | Role | Why |
|---|---|---|
| **Eigen** (header-only) | Core linear algebra (`math/backend.hpp`) | Auto-vectorizes with ARM NEON on 64-bit ARM with no extra flags, and Android enables NEON by default for all API levels. Single-threaded unless **you** explicitly enable OpenMP — which isn't even present in the default Android/iOS/macOS toolchains, so "universal + no forced threads" falls out of Eigen's own default behavior instead of augur fighting it. Used in production tracking code elsewhere (e.g. the `mherb/kalman` C++ library). Header-only means CPM only needs `DOWNLOAD_ONLY`, never a real build. |
| **Boost.PFR** (header-only, no Boost dependency despite the name) | Portable reflection backend (`reflect/backends/pfr_backend.hpp`) | Works on any C++14/17-conforming front end — no compiler extension, no code-gen step, so it's available on Android NDK today, unlike P2996. |
| **glm** (header-only, optional via `AUGUR_WITH_GLM`) | Ergonomic interop (`math/interop_glm.hpp`) | Game code already has positions as `glm::vec3` — this avoids forcing manual Eigen conversions at every call site. Entirely absent from the build if you don't opt in. |
| **Catch2** (test-only, via `AUGUR_BUILD_TESTS`) | Unit tests | Never fetched unless you're building tests — consumers of the library never pay for it. |

Libraries considered and **not** chosen, for the record: **Blaze** and
**Armadillo** typically want a BLAS/LAPACK backend to hit their best
performance, which is a real portability tax on Android; **xtensor** is a
fine numpy-style array library but doesn't buy anything over Eigen for
fixed-size Kalman-filter-shaped linear algebra specifically. **Sophus**
(Lie groups, SO(3)/SE(3)) is worth revisiting if/when a 3D coordinated-turn
model is built (see `docs/ROADMAP.md`) but isn't a dependency of anything
that exists today.

## 4. Folder structure

```
augur/
├── CMakeLists.txt              root build, options, cross-platform notes
├── cmake/
│   └── get_cpm.cmake           CPM.cmake bootstrap (not FetchContent)
├── include/augur/
│   ├── augur.hpp                umbrella header (solid tier only, see §6)
│   ├── core/                    concepts.hpp, config.hpp, state_component.hpp (augmented-layout enum) — foundation everything else builds on
│   ├── math/                    backend.hpp (Eigen aliases + safe_inverse/project_to_psd), interop_glm.hpp
│   ├── models/                  model_concept.hpp + constant_velocity/constant_acceleration/coordinated_turn/singer/current_statistical
│   ├── filters/                 filter_concept.hpp + kalman/extended_kalman/unscented_kalman/particle_filter/current_statistical_filter, filters/adaptive/sage_husa.hpp
│   ├── imm/                     mode_matrix.hpp, mixing.hpp, estimator.hpp (same-dimension IMM) + augmented_layout.hpp/heterogeneous_mixing.hpp/heterogeneous_estimator.hpp (opt-in different-order mixing, §5)
│   ├── predict/                 query.hpp (2D/3D error ellipse) + latency_compensation.hpp (SnapshotBuffer, predict_to_render_time)
│   ├── track/                   association.hpp (NN + JPDA), track_manager.hpp, gm_phd.hpp, out_of_sequence.hpp, fusion.hpp
│   ├── reflect/                 descriptor.hpp (dispatches VectorBackend/PfrBackend), has_reflection.hpp, serialize.hpp (binary wire format), backends/{pfr_backend,vector_backend}.hpp
│   ├── plugin/                  concepts.hpp (the real plugin mechanism), registry.hpp (runtime variant)
│   └── utils/                   type_traits/fixed_vector/assert/timing — reused from every module above
├── examples/                    01_basic_cv_tracking .. 15_reflection_serialization (one per roadmap item, see docs/ROADMAP.md)
├── tests/unit/                  one test file per module — see tests/CMakeLists.txt for the current list; test_helpers.hpp holds the shared finite-difference jacobian checker
└── docs/                        this file, ROADMAP.md, GETTING_STARTED.md, PLUGIN_GUIDE.md
```

Every module that isn't `core/` or `utils/` reaches back into both of them —
that's the intended reuse direction. Nothing in `utils/` is allowed to know
what a `MotionModel` is; the day it does, that code has drifted out of
`utils/` and belongs in `core/` or the specific module that owns the concept.

## 5. Known limitation: same-order IMM mixing only (in `imm::Estimator` specifically)

`imm::Estimator<Filters...>` requires every mixed filter to share a state
**dimension** (enforced with a `static_assert`, not a silent bug), and stays
that way deliberately — it's the zero-overhead default path, and relaxing it
would cost every user of the common case something they don't need.

Bar-Shalom's general IMM formulation (Bar-Shalom, Li & Kirubarajan,
*Estimation with Applications to Tracking and Navigation*, Wiley, 2001, ch.
11.6) allows mixing models of genuinely different order — a 5-state
coordinated-turn model against a 6-state constant-velocity model — via linear
mapping matrices onto a common augmented state space. That's now implemented
as a **separate, opt-in** mechanism instead of a change to `imm::Estimator`
itself: `imm::HeterogeneousEstimator<Filters...>` (`imm/heterogeneous_estimator.hpp`),
built on `core/state_component.hpp` and `imm/augmented_layout.hpp`. It reuses
`imm::mix()`/`imm::combine()` from `imm/mixing.hpp` directly rather than
duplicating the mixing math. See `docs/ROADMAP.md` item 0 for the honest
characteristics found while building and testing it (the "components some
models don't track" padding heuristic, and what it does to per-model
confidence when one model's mode probability dominates) and
`examples/04_heterogeneous_imm` for the CV+CA+CT demonstration this section
used to say wasn't possible.

The original three-`CoordinatedTurn` examples (calm / juking / sharp-turn,
same dimension) remain the simplest, cheapest way to get IMM's core benefit
when heterogeneous model types aren't actually needed — reach for
`HeterogeneousEstimator` only when the mixed models genuinely differ in
state dimension.

## 6. What's solid vs. what's a roadmap stub

Being upfront about this distinction matters more for a numerically tricky
domain like tracking than for most libraries — silently shipping
unvalidated Kalman math is worse than not shipping it.

**Solid — implemented and reasoned through carefully:**
`core/` (including `state_component.hpp`), `math/` (both files),
`models/constant_velocity.hpp`, `models/constant_acceleration.hpp`,
`models/coordinated_turn.hpp`, `filters/kalman.hpp`,
`filters/extended_kalman.hpp`, `filters/unscented_kalman.hpp`,
`filters/particle_filter.hpp`, `imm/mode_matrix.hpp`, `imm/mixing.hpp`,
`imm/estimator.hpp`, `predict/query.hpp`, `predict/latency_compensation.hpp`,
`track/association.hpp`, `track/track_manager.hpp`,
`track/out_of_sequence.hpp`, `track/fusion.hpp`, `reflect/` (all five files:
`descriptor.hpp`, `has_reflection.hpp`, `serialize.hpp`,
`backends/pfr_backend.hpp`, `backends/vector_backend.hpp`),
`plugin/concepts.hpp`, `plugin/registry.hpp`, all of `utils/`. Covered by
`tests/unit/` (see `docs/ROADMAP.md` item 13 for what that now includes).

**Flagged sketch — compiles and satisfies its concept, needs verification
before trusting it:** `models/singer.hpp` (transition matrix is the standard
closed form; the process-noise term is a simplified placeholder, not Singer's
exact closed-form integral — the file says so directly); `models/current_statistical.hpp`
+ `filters/current_statistical_filter.hpp` (docs/ROADMAP.md item 1 — the core
dynamics are verified, but the adaptive-mean feedback needed damping and
clamping to avoid an unbounded divergence found during testing, and still
biases toward its clamp under a very long sustained maneuver); `imm/augmented_layout.hpp`,
`imm/heterogeneous_mixing.hpp`, `imm/heterogeneous_estimator.hpp` (the
different-order IMM mixing from `docs/ROADMAP.md` item 0 — the expand/mix/
restrict math itself is verified, but the "unknown variance padding" heuristic
for components a lower-order model doesn't track has a documented, tested
characteristic: a non-dominant model's confidence about its own unique state
can be noisy under repeated mixing — see that item's write-up for specifics);
`filters/adaptive/sage_husa.hpp` (docs/ROADMAP.md item 2 — the eigenvalue-
flooring fix for the known divergence failure mode is verified, but the
implementation assumes a direct-position measurement model, stated in its
own file comment); `track/gm_phd.hpp` (docs/ROADMAP.md item 7 — the update/
merge math is verified, but `extract_targets()` doesn't yet split a
much-higher-than-1-weight component into multiple targets, a stated,
not-yet-implemented refinement).

All roadmap items with a dedicated stub file now have a status recorded
directly in `docs/ROADMAP.md` — see that file rather than a stale list here.

`augur.hpp` (the umbrella header) deliberately only includes the solid tier —
pulling in a roadmap stub requires an explicit `#include`, so an accidental
"I just included augur.hpp" can't silently hand you an unimplemented type.

## 7. Build system

See `docs/GETTING_STARTED.md` for copy-pasteable build commands. In short:
CMake + **CPM.cmake** (not `FetchContent` — CPM gives source caching across
builds via `CPM_SOURCE_CACHE`, works uniformly whether a dependency comes from
GitHub, GitLab, or a plain URL, and doesn't wait until build time the way
`ExternalProject` does, which matters once you're cross-compiling). The root
`CMakeLists.txt` fetches Eigen (`DOWNLOAD_ONLY`, header-only, no need for
Eigen's own heavier CMake machinery), optionally glm and Boost.PFR, and
(only when `AUGUR_BUILD_TESTS=ON`) Catch2.

Cross-platform posture: the library targets desktop (Linux/macOS/Windows) and
Android (via the NDK's own CMake toolchain file) as co-equal first-class
targets, not "desktop plus an afterthought Android port." Concretely:
nothing in the solid tier requires anything past a conforming C++20 front
end (concepts), which recent NDKs provide; NEON is on by default for
`arm64-v8a`, so Eigen vectorizes without extra flags; and the library never
enables OpenMP itself, so there's no threading-related toolchain surprise to
debug on either platform.
