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
without any extra flags. Nothing in the solid tier (`core/`, `math/`,
`models/{constant_velocity,constant_acceleration,coordinated_turn}`,
`filters/{kalman,extended_kalman}`, `imm/`) needs anything past a conforming
C++20 front end — see `docs/ARCHITECTURE.md` §2.4 for why this library
deliberately doesn't reach for C++23/26-only features.

## CMake options

| Option | Default | Effect |
|---|---|---|
| `AUGUR_WITH_GLM` | `ON` | Fetches glm, enables `math/interop_glm.hpp` |
| `AUGUR_BUILD_EXAMPLES` | `ON` | Builds the three `examples/` programs |
| `AUGUR_BUILD_TESTS` | `OFF` | Fetches Catch2, builds `tests/` |

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
    CV{/*noise_spectral_density=*/1.0f},
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
