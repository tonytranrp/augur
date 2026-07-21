# augur

A C++20 motion-prediction / target-tracking library for game and mod
developers, built around an Interacting Multiple Model (IMM) estimator —
mix several motion models (constant velocity, constant acceleration,
coordinated turn, ...) so the tracker itself notices when a target stops
moving in a straight line, instead of you switching models by hand.

`augur` is a suggested/placeholder name — rename freely.

```cpp
using CT = augur::models::CoordinatedTurn<float>;
using KF = augur::filters::KalmanFilter<CT, /*MeasDim=*/2>;

augur::imm::Estimator<KF, KF, KF> tracker{
    KF{CT{/*q_pos=*/1, /*q_turn=*/0.02f}, x0, P0, H, R}, // calm
    KF{CT{/*q_pos=*/1, /*q_turn=*/0.5f},  x0, P0, H, R}, // juking
    KF{CT{/*q_pos=*/1, /*q_turn=*/3.0f},  x0, P0, H, R}, // sharp turn
    augur::imm::ModeMatrix<3, float>::uniform(0.95f)
};

tracker.predict(dt);
tracker.update(detection_xy);
auto [state, covariance] = tracker.combined_state();
auto lead = tracker.predict_ahead(render_latency_seconds);
```

## Why

Plenty of prior art exists for individual pieces (a Kalman filter header
here, a physics-engine integrator there), but there's no small, ergonomic,
header-only C++ library that specifically does IMM-based maneuvering-target
tracking with a GLM/EnTT-caliber API — see `docs/ARCHITECTURE.md` for the
full reasoning and the literature this is built on.

## Design in one paragraph

Header-only. Template-heavy, concepts-constrained API (`models::MotionModel`,
`filters::Filter`) — that's the entire plugin mechanism: any type with the
right member functions works everywhere a built-in model does, no base class,
no registration call. No forced multithreading, anywhere, ever — every
algorithm is a plain reentrant function of its inputs, so the same code runs
identically on Android and desktop without imposing a threading model on
either. Reflection uses Boost.PFR today (works on Android NDK right now) with
a feature-detected seam for C++26 static reflection once compilers actually
ship it. Full reasoning: `docs/ARCHITECTURE.md`.

## Getting started

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/examples/basic_cv_tracking
```

Full build instructions (including Android via the NDK toolchain) in
`docs/GETTING_STARTED.md`.

## Documentation

| Doc | Covers |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Full design rationale, dependency choices, folder structure, known limitations, solid-vs-roadmap tier breakdown |
| [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) | Build commands (desktop + Android), CMake options, five-minute usage walkthrough |
| [`docs/PLUGIN_GUIDE.md`](docs/PLUGIN_GUIDE.md) | Writing your own motion model or filter from scratch |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | 14 prioritized extension ideas with citations and effort estimates |

## What's implemented today

Every item in `docs/ROADMAP.md` (14 extension ideas, plus the different-order
IMM mixing structural improvement) is implemented, tested, and has a working
example under `examples/`.

**Solid**: constant-velocity, constant-acceleration, and coordinated-turn
motion models; linear, extended, unscented, and particle-filter backends; the
full IMM mixing/combination cycle (same state dimension); position/velocity/
2D-and-3D-uncertainty-ellipse queries; multi-target data association (GNN +
JPDA), track lifecycle management, GM-PHD variable-target-count tracking,
out-of-sequence measurement handling, latency compensation, and sensor
fusion; a reflection layer (Boost.PFR for plain aggregates, a hand-written
backend for Eigen vectors, since PFR can't reflect those) driving binary
save-state/network (de)serialization; the plugin layer. All covered by
`tests/unit/`.

**Flagged sketch** (works and is tested, but a named part is a documented
simplification — see the file's own top comment and `docs/ROADMAP.md` for
specifics): the Singer maneuvering-target model; the Current Statistical
adaptive-acceleration model; different-order IMM mixing
(`imm::HeterogeneousEstimator`, mixing e.g. constant-velocity and
coordinated-turn despite their different state dimensions); Sage-Husa
adaptive process-noise estimation; the GM-PHD filter's `extract_targets()`
(doesn't yet split an above-weight-1 component into multiple targets).

Full status, effort estimates, and citations for everything above:
`docs/ROADMAP.md`.

## License

Not yet chosen — add one (MIT and Apache-2.0 are the common choices for a
library like this) before publishing anywhere.
