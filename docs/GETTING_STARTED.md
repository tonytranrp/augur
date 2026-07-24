# augur — Getting Started

## Requirements

- CMake 3.21+
- A C++20 compiler with working concepts support (recent Clang, GCC, or MSVC)
- Internet access on first configure (CPM fetches Eigen, and optionally glm /
  Boost.PFR / Catch2) — set `CPM_SOURCE_CACHE` to a persistent directory so
  repeat builds and CI don't re-download every time

## Build — desktop (Linux / macOS / Windows)

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/examples/basic_cv_tracking
```

Want x86 SIMD (AVX2 etc.)? Eigen doesn't assume an instruction set — pass it
yourself:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native"
```

### Windows note: "ambiguous argument 'HEAD0'" during configure

If a fresh `cmake -B build` fails partway through fetching a git-tag-pinned
dependency (Eigen, Boost.PFR) with an error containing `ambiguous argument
'HEAD0'`, this is very likely a **PATH collision**, not a CPM/CMake bug: some
other tool has put its own `git` shim earlier on `PATH` than your real Git
for Windows install. npm is one common, concrete example — it ships a
`git.cmd` shim that mangles `^` characters CPM's git commands rely on
through `cmd.exe`'s own argument re-expansion, producing this exact symptom.

Check which `git` CMake is actually finding:

```sh
where git
```

If the first hit isn't your real `Git\cmd\git.exe` or `Git\bin\git.exe`,
point CMake at the real one explicitly rather than fighting your `PATH`:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGIT_EXECUTABLE="C:/Program Files/Git/cmd/git.exe"
```

(adjust the path to wherever Git for Windows is actually installed). See
`cmake/get_cpm.cmake`'s own comment for the full history of this error —
it also has a separate, unrelated cause (an old CPM.cmake version against a
new CMake release) that the pinned version in this repo already avoids.

## Build — Android (via the NDK's own CMake toolchain)

```sh
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DAUGUR_BUILD_EXAMPLES=ON
cmake --build build-android
```

NEON is on by default for `arm64-v8a` on current NDKs, so Eigen vectorizes
without any extra flags. Nothing in the solid tier needs anything past a
conforming C++20 front end (see `docs/ARCHITECTURE.md` §6 for the current
tier list, and §2.4 for why this library deliberately doesn't reach for
C++23/26-only features) — CI cross-compiles every example for
`arm64-v8a` on every push to keep that true.

## CMake options

| Option | Default | Effect |
|---|---|---|
| `AUGUR_WITH_GLM` | `ON` | Fetches glm, enables `math/interop_glm.hpp` |
| `AUGUR_BUILD_EXAMPLES` | `ON` | Builds the 17 `examples/` programs |
| `AUGUR_BUILD_TESTS` | `OFF` | Fetches Catch2, builds `tests/` |
| `AUGUR_WARNINGS_AS_ERRORS` | `OFF` | `-Werror` / `/WX` on augur's own targets (CI turns this on) |
| `AUGUR_BUILD_BENCHMARKS` | `OFF` | Fetches Google Benchmark, builds `benchmarks/` |

Turn tests on with:

```sh
cmake -B build -DAUGUR_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

## Using augur from your own CMake project

Since `augur` is header-only (an `INTERFACE` target), the simplest path is
CPM again, from your own project:

```cmake
include(cmake/get_cpm.cmake)  # or your own copy of the same bootstrap
CPMAddPackage("gh:yourname/augur#main")
target_link_libraries(your_game PRIVATE augur::augur)
```

## Five-minute usage walkthrough

Standalone filter, no IMM — see `examples/01_basic_cv_tracking/main.cpp` in
full, condensed here:

```cpp
#include "augur/augur.hpp"

using CV = augur::models::ConstantVelocity<float, 2>;   // 2D, float precision
using KF = augur::filters::KalmanFilter<CV, /*MeasDim=*/2>;

KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
H(0, 0) = 1; H(1, 1) = 1; // measure position directly

KF filter{
    CV{/*accel_noise_density=*/1.0f},
    KF::StateVector::Zero(),
    KF::StateCovariance::Identity() * 10.0f,
    H,
    KF::MeasurementCovariance::Identity() * 0.5f
};

filter.predict(dt);
filter.update(KF::Measurement{measured_x, measured_y});
const auto& estimated_state = filter.state(); // [px, py, vx, vy]
```

Add IMM once one model stops being enough — see
`examples/02_imm_maneuvering_target/main.cpp` for the full mode-probability
walkthrough, and `docs/ARCHITECTURE.md` §5 for the current same-dimension
limitation on what can be mixed together.

Writing your own model — see `docs/PLUGIN_GUIDE.md` and
`examples/03_custom_plugin_model/main.cpp`.
