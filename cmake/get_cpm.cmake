# cmake/get_cpm.cmake
#
# Standard CPM.cmake bootstrap: downloads CPM.cmake itself into the
# build tree if it isn't already there, then includes it. This is the
# documented pattern from the CPM.cmake project itself (cpm-cmake/CPM.cmake)
# rather than something bespoke to this project.
#
# IMPORTANT: the version/hash below (v0.43.1) is the last release this
# was actually verified against during research -- CPM.cmake has
# almost certainly shipped newer releases since. Check
# https://github.com/cpm-cmake/CPM.cmake/releases before you ship, pin
# to whatever is current there, and copy the SHA256 CPM's own release
# page lists for that tag. Do not bump the version number without also
# updating the hash, or this download will fail closed (which is the
# point of pinning a hash at all).
#
# v0.38.3 was the original pin here but its bundled ExternalProject
# git-update step is incompatible with newer CMake (4.x) releases -- it
# fails with "ambiguous argument 'HEAD0'" during the FetchContent update
# step for any git-tag-pinned package (Eigen, Boost.PFR). v0.43.1 fixes
# this; if this error resurfaces on some future CMake release, it's the
# same class of problem and the fix is the same: bump this pin.
#
# CORRECTION (docs/IMPROVEMENT_PLAN.md): that "ambiguous argument
# 'HEAD0'" error has a SECOND, unrelated cause this pin does not fix --
# a real Git for Windows install shadowed on PATH by a different tool's
# own `git` shim (npm's `git.cmd` is one concrete, plausible-at-scale
# example, given how widely npm is installed on Windows dev machines).
# The shim mangles `^` characters CPM's git commands rely on through
# cmd.exe's own argument re-expansion, producing the identical symptom
# with the CPM version pin being irrelevant either way. If bumping this
# pin doesn't fix a fresh "ambiguous argument 'HEAD0'" error, check
# `where git` (Windows) for a shadowing shim ahead of the real
# `git.exe` on PATH, and pass CMake the real one explicitly:
# `-DGIT_EXECUTABLE="C:/Program Files/Git/cmd/git.exe"` (adjust the path
# to your own Git for Windows install). See
# docs/GETTING_STARTED.md's Windows note for the full writeup.

set(CPM_DOWNLOAD_VERSION 0.43.1)
set(CPM_DOWNLOAD_HASH SHA256=1c40fc102ce9625d7de7eb14f541cab30cc3138dca627f0b0ec40293ce6c2934)

if(CPM_SOURCE_CACHE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

# Normalize the path across OSes (mirrors CPM's own get_cpm.cmake).
file(TO_CMAKE_PATH "${CPM_DOWNLOAD_LOCATION}" CPM_DOWNLOAD_LOCATION)

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "augur: downloading CPM.cmake v${CPM_DOWNLOAD_VERSION} to ${CPM_DOWNLOAD_LOCATION}")
  file(
    DOWNLOAD
    https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
    ${CPM_DOWNLOAD_LOCATION}
    EXPECTED_HASH ${CPM_DOWNLOAD_HASH}
  )
endif()

include(${CPM_DOWNLOAD_LOCATION})
