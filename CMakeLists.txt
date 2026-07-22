cmake_minimum_required(VERSION 3.21)
project(augur VERSION 0.1.0 LANGUAGES CXX)

# --------------------------------------------------------------------
# Language standard
#
# C++20 is the floor (concepts are load-bearing throughout this
# library). C++23 is used opportunistically where available
# (<numbers>, some ranges usage in roadmap modules) but nothing here
# REQUIRES 23, let alone 26 -- see include/augur/core/config.hpp for
# why: Android NDK and MSVC both still lag C++23/26 feature completeness
# as of this writing, and a "universal" library can't assume otherwise.
# --------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(AUGUR_WITH_GLM       "Enable the glm interop layer (math/interop_glm.hpp)" ON)
option(AUGUR_BUILD_EXAMPLES "Build the example programs under examples/"          ON)
option(AUGUR_BUILD_TESTS    "Build unit tests under tests/"                       OFF)

include(cmake/get_cpm.cmake)

# --------------------------------------------------------------------
# Eigen: header-only, so DOWNLOAD_ONLY -- we never want Eigen's own
# (fairly heavy) CMakeLists driving tests/blas/lapack subprojects into
# our build. This also sidesteps any Fortran-toolchain detection Eigen's
# full build would otherwise attempt, which is a common cross-compile
# footgun (see the CPM discussion linked in docs/ARCHITECTURE.md).
# --------------------------------------------------------------------
CPMAddPackage(
  NAME Eigen3
  GITHUB_REPOSITORY eigen-mirror/eigen
  GIT_TAG 3.4.0
  DOWNLOAD_ONLY YES
)
if(Eigen3_ADDED)
  add_library(Eigen3_Eigen INTERFACE)
  add_library(Eigen3::Eigen ALIAS Eigen3_Eigen)
  target_include_directories(Eigen3_Eigen SYSTEM INTERFACE ${Eigen3_SOURCE_DIR})
endif()

if(AUGUR_WITH_GLM)
  CPMAddPackage("gh:g-truc/glm#1.0.1")
endif()

if(AUGUR_BUILD_TESTS)
  CPMAddPackage("gh:catchorg/Catch2@3.7.1")
endif()

# Optional: Boost.PFR for the portable reflection backend
# (reflect/backends/pfr_backend.hpp). Header-only, no Boost dependency
# despite the name -- see the comment in that header for details.
#
# Pinned to a post-2.2.0 tag deliberately: the standalone 2.2.0 release
# (2023-11-23) predates boostorg/pfr#148 by five days, a real bug where
# clang-cl (which defines both __clang__ and _MSC_VER) gets misrouted
# into config.hpp's MSVC-pattern branch for BOOST_PFR_CORE_NAME_PARSING
# even though it actually captures a __PRETTY_FUNCTION__-shaped string
# (Clang never exposes __FUNCSIG__ to #if defined(), even under
# -fms-extensions -- see llvm/llvm-project#30933) -- boost::pfr::get_name()
# then fails its own internal self-consistency static_assert outright.
# Confirmed via GitHub Actions (windows-latest, `cmake -T ClangCL`):
# fails at 2.2.0, fixed by this tag. PFR stopped cutting standalone
# version tags after 2.2.0 and now only ships bundled with Boost's own
# boost-1.XX.0 releases, all of which contain the one-line fix
# (`defined(_MSC_VER) && !defined(__clang__)`).
CPMAddPackage(
  NAME boost_pfr
  GITHUB_REPOSITORY boostorg/pfr
  GIT_TAG boost-1.85.0
  DOWNLOAD_ONLY YES
)
if(boost_pfr_ADDED)
  add_library(boost_pfr INTERFACE)
  target_include_directories(boost_pfr SYSTEM INTERFACE ${boost_pfr_SOURCE_DIR}/include)
endif()

# --------------------------------------------------------------------
# The library target itself: header-only (INTERFACE), so consuming
# augur never requires building anything beyond your own translation
# units.
# --------------------------------------------------------------------
add_library(augur INTERFACE)
add_library(augur::augur ALIAS augur)

target_include_directories(augur INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(augur INTERFACE cxx_std_20)
target_link_libraries(augur INTERFACE Eigen3::Eigen)

# Explicit, not just relying on Eigen's own default: augur never wants
# OpenMP-driven multithreading pulled in by a transitive dependency,
# on ANY platform, full stop (see core/config.hpp for the rationale).
target_compile_definitions(augur INTERFACE EIGEN_DONT_PARALLELIZE)

if(boost_pfr_ADDED)
  target_link_libraries(augur INTERFACE boost_pfr)
endif()

if(AUGUR_WITH_GLM)
  target_link_libraries(augur INTERFACE glm::glm)
  target_compile_definitions(augur INTERFACE AUGUR_WITH_GLM=1)
endif()

# --------------------------------------------------------------------
# Cross-platform build notes
#
# Desktop (Linux/macOS/Windows):
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   x86 SIMD (AVX2 etc.) is NOT enabled by default by most compilers --
#   pass -DCMAKE_CXX_FLAGS="-march=native" yourself if you want it.
#   Eigen auto-detects whatever instruction set you told the compiler
#   to target; augur never assumes a particular one.
#
# Android (via the NDK's own CMake toolchain file):
#   cmake -B build-android \
#     -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
#     -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24
#   cmake --build build-android
#   NEON is on by default for arm64-v8a in current NDKs, so Eigen
#   vectorizes automatically with no extra flags. C++20 concepts and
#   everything in the "solid" tier of this library (core/, math/,
#   models/{constant_velocity,constant_acceleration,coordinated_turn},
#   filters/{kalman,extended_kalman}, imm/) only need a conforming
#   C++20 front end, which recent NDKs provide -- nothing here reaches
#   for C++23/26-only features. AUGUR_HAS_STD_REFLECTION will simply
#   stay 0 on Android for the foreseeable future (see core/config.hpp);
#   the PFR-backed reflection path is what you get instead, and it
#   works identically to the desktop build.
#
# iOS / other Clang-based embedded targets: the same header-only,
# no-forced-threading design should carry over directly, but this has
# only been reasoned through for desktop + Android specifically -- test
# before assuming.
# --------------------------------------------------------------------

if(AUGUR_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(AUGUR_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
