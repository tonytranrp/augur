---
paths:
  - "include/**/*.hpp"
  - "tests/**/*.cpp"
  - "examples/**/*.cpp"
---

# Code style

Precise and example-backed; see `docs/ARCHITECTURE.md` for the *why* behind
any of this. This file is the *how to write code that fits in*.

## File structure

Every header: `#pragma once` (never include guards), then a
`// augur/path/to/file.hpp` comment on line 2 giving the file's real include
path, then a prose block explaining what the file is for and why, with
cross-references to related files/docs — not a line-by-line description of
what the code does. E.g. `core/config.hpp`:

```cpp
#pragma once
// augur/core/config.hpp
//
// Central feature-detection point. Nothing in this file does any work --
// it only decides, at compile time, which backend the rest of the library
// should use for reflection, SIMD, etc. ...
```

Inside `.hpp` comments use `--` (ASCII double-hyphen) for a parenthetical
dash, not an em dash (real em dashes are for prose docs under `docs/*.md`
only). Explain *why* right where a reader would ask ("position += velocity *
dt", "Linear model: the Jacobian IS the transition matrix, x-independent").

## Naming

- Types/concepts: `PascalCase` — `ConstantVelocity`, `KalmanFilter`,
  `MotionModel`, `FixedVector`, `BuildInfo`.
- Functions/methods: `snake_case` — `push_back`, `process_noise`,
  `combined_state`, `predict_ahead`.
- Template parameters: `PascalCase` and descriptive when it aids the call
  site — `ScalarT`/`Scalar`, `SpatialDim`, `Capacity`, `NumModes` — but plain
  `T`/`N` where the surrounding concept already carries the meaning
  (`core/concepts.hpp`).
- Private members: trailing underscore — `size_`, `storage_`, `model_`,
  `q_pos_`.
- Macros: `AUGUR_` prefix, always — `AUGUR_ASSERT`, `AUGUR_SINGLE_THREADED`,
  `AUGUR_HAS_PFR`.
- Namespaces mirror folders exactly: `include/augur/imm/` -> `augur::imm`.
- Files: `snake_case.hpp`/`.cpp`, one primary type per file, filename matches
  the type (`coordinated_turn.hpp` -> `CoordinatedTurn`).

## Concepts, not inheritance

`models::MotionModel` and `filters::Filter` (`models/model_concept.hpp`,
`filters/filter_concept.hpp`, refining the small structural concepts in
`core/concepts.hpp`) ARE the plugin system. A new model/filter needs no base
class, no registration call, no vtable — just the right member functions
(`transition()`/`jacobian()`/`process_noise()` for a model;
`predict()`/`update()`/`state()`/`covariance()`/`last_likelihood()` for a
filter) plus a `static_assert(augur::models::MotionModel<YourType>)` to prove
it compiles against the concept. Write new models/filters the same way:
satisfy the concept structurally, don't inherit from anything, don't add a
registration call. The one exception is the deliberately separate, opt-in
`plugin/registry.hpp` type-erased `FilterRegistry`, for genuine runtime model
selection (`docs/PLUGIN_GUIDE.md`, "When you need runtime selection
instead") — reach for that only when compile-time selection truly isn't an
option, never as a default.

## Template `<>` vs namespace `::`

`<>` carries real, load-bearing configuration: `ConstantVelocity<float, 3>`
and `ConstantVelocity<double, 2>` are different, unrelated, independently
compiled types with zero-overhead abstraction and no vtable in the hot path.
Prefer a template parameter over a runtime flag/enum for anything decided
once at compile time. `::` carries logical grouping into distinctly named,
constrainable concepts — `augur::filters::Filter` and `augur::models::
MotionModel` exist as separate named things you constrain templates
against, not one flat namespace of unrelated names.

## The three-tier convention

Every non-trivial file is honest, in its own top comment, about its tier:

- **Solid** (most of `core/`, `math/`, `models/{constant_velocity,
  constant_acceleration,coordinated_turn,singer}.hpp`, `filters/{kalman,
  extended_kalman,unscented_kalman,particle_filter}.hpp`, all of `imm/`
  except the heterogeneous-mixing files below, `predict/{query,
  latency_compensation}.hpp`, `track/{association,track_manager,
  out_of_sequence,fusion,gm_phd}.hpp`, `reflect/`, `plugin/`, all of
  `utils/`): no special marker — comments explain rationale, not caveats.
- **Flagged sketch** (`models/current_statistical.hpp`
  + `filters/current_statistical_filter.hpp`, `imm/{augmented_layout,
  heterogeneous_mixing,heterogeneous_estimator}.hpp`,
  `filters/sage_husa.hpp`): compiles and
  satisfies its concept, but a named part is a known simplification. Marked
  with a `ROADMAP MODEL --` line at the top of the file comment, naming
  exactly which part is simplified and citing the source for the real
  derivation.
- **Roadmap stub**: not implemented. As of `docs/ROADMAP.md`'s 14 items (plus
  the different-order-IMM-mixing structural item) all being done, nothing
  in the tree is currently in this tier — it exists for genuinely new,
  not-yet-implemented work. Marked with `ROADMAP STUB (see docs/ROADMAP.md,
  "<item name>")` at the top, then prose (what/why/references/intended
  integration), then a **commented-out** sketch of the intended API —
  never a real, compiling declaration that just throws or returns a dummy
  value.

Mark new code the same way: don't let a stub masquerade as solid, and don't
leave a flagged simplification undocumented. `augur.hpp` (the umbrella
header) only `#include`s the solid tier on purpose — pulling in a stub
requires an explicit, visible `#include` of that specific file.

## Citations

Algorithm-backing references go in the file's top comment, in "Author,
Title, Venue, Year, pp." form regardless of source type:

```cpp
// Reference: R. A. Singer, "Estimating Optimal Tracking Filter Performance
// for Manned Maneuvering Targets," IEEE Transactions on Aerospace and
// Electronic Systems, AES-6(4), 1970, pp. 473-483.
```

Cite the specific chapter for a book (`Bar-Shalom, Li & Kirubarajan ...
Wiley, 2001, ch. 11.6`), and add a one-line note on *why* this reference
over alternatives when that isn't obvious from context.

## `utils/` reuse convention

`utils/` holds only domain-agnostic helpers with zero knowledge of augur's
own types — `FixedVector<T, Capacity>`, `Stopwatch<Scalar>`,
`is_specialization_of`, `AUGUR_ASSERT`. The test: the moment a "util" needs
to know what a `MotionModel` or `Filter` is, it has drifted out of `utils/`
and belongs in `core/` (if foundational) or the specific module that owns
the concept. Don't create a second home for cross-cutting helpers.

## CPM / CMake

- Dependencies go through `CPMAddPackage`, never `FetchContent` (source
  caching via `CPM_SOURCE_CACHE`, uniform across GitHub/GitLab/URL sources).
- Header-only deps (Eigen, Boost.PFR) use `DOWNLOAD_ONLY YES` plus a
  hand-rolled `INTERFACE` target + `ALIAS` — never let a header-only dep's
  own (often heavier, test/BLAS-seeking) CMakeLists run.
- New CMake options are named `AUGUR_<THING>` (`AUGUR_WITH_GLM`,
  `AUGUR_BUILD_EXAMPLES`, `AUGUR_BUILD_TESTS`) and default to whatever keeps
  a consumer who "just wants the library" paying for nothing extra
  (`AUGUR_BUILD_TESTS` defaults `OFF`; feature/example options default `ON`).
- Never pull in threading (OpenMP, `std::thread`) inside the library itself
  — `EIGEN_DONT_PARALLELIZE` is set explicitly rather than relying on a
  toolchain default; keep it that way for anything new.
