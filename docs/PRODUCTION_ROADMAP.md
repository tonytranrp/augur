# augur — Path to Production

This is a full pass over the repo as it stands today (commit `969177b`,
94 files, all 14 roadmap items + the structural item marked done in
`docs/ROADMAP.md`). Everything in **Problems** below is something I
personally verified by cloning the repo and building/running/testing it —
not a guess from reading code. Everything in **Goals** is a phased plan to
get from "works and is well-tested" (true today) to "production grade"
(not yet true — mainly an infrastructure and process gap, not a code
quality one).

## Where it actually stands right now

The core engineering is genuinely solid: a clean clone builds with zero
errors (library + 15 examples + the full test suite), and the test suite
itself passes **2379 assertions across 66 test cases**. I independently
re-derived (via mpmath, not just re-read) the two trickiest fixes in the
history — the `coordinated_turn.hpp` small-omega Jacobian bug and the
`pfr_backend.hpp` `field_name()` fix — and both are mathematically
correct. The "fixes found along the way" section of `docs/ROADMAP.md` is
genuinely good engineering writing. **This is not a "make the code better"
document.** It's a "the code is good, now make the *project* production
grade" document — CI, packaging, process, and a handful of concrete loose
ends.

## Problems

### P0 — Concrete, cheap, fix these first

**Phase 1 status: all five items resolved**, one commit each
(`14d5f55`..`90c4459`), full test suite passing after every commit. See
docs/ROADMAP.md for two of the five that turned into deeper findings
(items 0 and, indirectly, the association.hpp performance note) beyond
the original one-line description below.

1. **`CLAUDE.md` is stale.** It still describes `filters/adaptive/`, all of
   `track/`, and `predict/latency_compensation.hpp` as roadmap stubs.
   They're not — every one is implemented and tested. This is the exact
   failure mode the file's own "read the source first" rule exists to
   guard against; it just hasn't been re-synced since the `/goal` run
   finished. Fix: rewrite the "Folder map" section against the actual
   `docs/ROADMAP.md` status table.
   **✅ Done** (`14d5f55`) — rewrote the folder map against `docs/ROADMAP.md`'s
   actual per-item status, using the same "Solid"/"Flagged sketch"
   terminology `docs/ARCHITECTURE.md` already uses so the two docs stay
   consistent with each other, not just individually accurate.
2. **`CMakeLists.txt` fetches Eigen from `gitlab.com`.** `GITLAB_REPOSITORY
   libeigen/eigen` fails closed on any network allowlist that includes
   `github.com` but not `gitlab.com` — which is a common posture for CI
   runners, sandboxes, and corporate networks. I confirmed
   `GITHUB_REPOSITORY eigen-mirror/eigen` (the official mirror, active as
   of April 2026) builds cleanly as a drop-in replacement. Given
   "universal" is a stated design goal, this is worth fixing regardless of
   whether you've personally hit it.
   **✅ Done** (`5d47daf`) — switched, and specifically re-verified with a
   genuinely fresh fetch (temporarily moved `.cpmcache/eigen3` aside so
   CPM couldn't cache-hit past the source-URL change): clones cleanly,
   lands at the correct `3.4.0` tag, full build + test suite unaffected.
3. **`model_concept.hpp:26` is missing a `requires`.** `core::Scalar<typename
   T::Scalar>;` inside the `requires{}` block only checks that the
   expression is *well-formed*, not that the concept actually *holds* —
   GCC flags this directly (`-Wmissing-requires`). I tried to construct a
   concrete case where a non-floating `Scalar` slips through undetected and
   didn't manage to (something else in the concept still catches it in
   this codebase today), so I can't say it's currently exploitable — but
   it's a one-word fix (`requires core::Scalar<typename T::Scalar>;`) and
   there's no reason to leave an imprecise constraint in the one file that
   defines the library's entire plugin contract.
   **✅ Done** (`29fb502`) — one-word fix, applied exactly as scoped.
4. **A `-Warray-bounds=` warning in `track/association.hpp`'s
   `joint_probabilistic_data_association()`, lines ~220-228 — chased down,
   looks like a false positive, but silence it properly rather than leaving
   it.** GCC's static analysis flags `current_assignment[track_idx]`
   (a `std::array<int, MaxTracks>`) with subscripts 5/6/7 against a
   4-element array, inside the deeply-inlined recursive `enumerate()`
   lambda. I wrote a standalone repro at full `MaxTracks=4`/`MaxDetections=4`
   capacity (the worst case) and ran it under AddressSanitizer +
   UndefinedBehaviorSanitizer — it completed cleanly, no report from
   either tool. This is consistent with a known GCC false-positive pattern
   (value-range propagation losing precision through several levels of
   inlined recursive generic lambdas) rather than a real bug, but "I ran it
   once under ASan and it was fine" isn't the same as "provably safe" —
   see Goal 3 below for making this a permanent, automated check instead
   of something I verified once by hand.
   **✅ Done** (`ce793c6`) — narrow `#pragma GCC diagnostic` suppression
   (GCC-only, this block only), with the structural correctness argument
   and the ASan/UBSan repro's exact scenario written directly into the
   code comment. Caveat stated honestly: no real GCC toolchain was
   available in the environment this was done in (its `gcc`/`g++` are
   clang aliases — `clang++ --version` and `gcc --version` report the same
   binary), so the GCC-specific warning itself was never re-reproduced,
   only reasoned about and cross-checked via what *was* available
   (ASan/UBSan under clang). Worth a real GCC pass once Phase 2's CI
   matrix exists.
5. **`heterogeneous_imm`'s mode probabilities lean CV-heavy (up to 0.95)
   even while the ground truth is visibly curving.** Not confirmed wrong —
   could be legitimate (CV fits a gentle short-horizon arc reasonably well)
   — but worth a deliberate tuning pass rather than leaving it as an
   unexamined observation.
   **✅ Done** (`90c4459`) — confirmed legitimate (per-step curvature
   deviation is ~6x smaller than measurement noise at this scenario's
   parameters, verified numerically) *and* found a real bug the
   investigation wasn't originally looking for: `big_unknown_variance()`
   (1e4) was large enough to nearly erase a recovering model's own prior
   in one mixing cycle at a mode-probability crossover, briefly producing
   a wrong-signed combined-position estimate — outside the original
   20-step demo's window, only caught by deliberately extending the
   scenario to actually reach a crossover. Fixed (swept to 100) and
   `docs/ROADMAP.md` item 0's prior "stays accurate throughout" claim
   corrected to state what's actually verified. Full details:
   `docs/ROADMAP.md` item 0's "Update" paragraph.

**Warnings note** (relevant to the "zero warnings under
`-Wall -Wextra -Wpedantic`" bar this phase set for itself): confirmed via
a clean rebuild with those flags that **`include/augur/` itself is 0
warnings**. The full build (library + tests) is not literally
warning-free — Catch2's `TEST_CASE` macro uses `__COUNTER__`, which this
environment's clang (a very recent 22.1.4) flags under `-Wc2y-extensions`
at every macro expansion site (135 occurrences, one per `TEST_CASE`/
`SECTION` call, all inside `tests/unit/*.cpp` at the point Catch2's macro
expands, none inside `include/augur/`). That's Catch2's own internal
implementation detail on a pinned third-party dependency, not augur code
— `-isystem` (already used for Catch2's include path) doesn't suppress it
because the diagnostic is attributed to the macro's expansion site in the
including file, not Catch2's header. Not fixed here since there's nothing
in this repo to fix; noted for Phase 2 so the CI warnings-as-errors job
scopes `-Werror` to augur's own sources rather than tripping on this.

### P1 — Real gaps between "well-tested" and "production"

6. **No CI at all.** There is no `.github/workflows/` directory — every
   verification in this document and the previous review happened because
   *I* cloned the repo and ran it by hand. Nothing currently stops a future
   change from silently breaking the build, a platform, or a test.
7. **No warnings-as-errors, anywhere.** The two P0 warnings above exist
   precisely because nothing enforces a clean build. Once they're fixed,
   lock the door.
8. **No sanitizer or static-analysis integration.** The ASan/UBSan run that
   resolved P0-4 was a one-off manual invocation, not something the repo
   can run for itself. Same for `clang-tidy`/`cppcheck` — nothing here
   today beyond compiler warnings.
9. **No performance regression tracking.** `tests/unit/test_helpers.hpp`
   (per `docs/ROADMAP.md` item 13) validates *accuracy* via finite-difference
   Jacobian checks — genuinely good — but nothing validates that a change
   doesn't quietly make `imm::Estimator::predict()` twice as slow. For a
   library whose entire pitch is "high-performance, game-loop-safe," that's
   a real gap, not a nice-to-have.
10. **Android is claimed, never verified.** `docs/GETTING_STARTED.md` and
    `CMakeLists.txt`'s comments describe the NDK toolchain invocation in
    detail and reasoned carefully about why the design should work there
    (no forced threading, PFR over P2996, NEON defaults) — but as far as I
    can tell from the repo, nobody has actually run an NDK cross-compile
    even once. Reasoning about portability and verifying it are different
    things; right now this project has only done the first.
11. **No package manager presence.** Consuming this today means
    clone-and-CPM. No vcpkg port, no Conan recipe, no `find_package(augur)`
    support via CMake's `install()`/`export()`. Fine for a personal project;
    not fine for something other people are meant to depend on.
12. **Eigen is pinned to 3.4.0; upstream is at 5.0.0** (stable since
    September 2025). Possibly a deliberate, reasonable choice — 3.4.0 is a
    long-lived, widely-used version — but right now it reads as
    unexamined rather than decided. Pick one on purpose.

### P2 — Process and longer-horizon

13. **Everything landed in one commit.** Not wrong for how this was built
    (one continuous `/goal` run), but there's no reviewable history —
    "what changed when and why" is only answerable by reading
    `docs/ROADMAP.md` prose, not `git log`/`git blame`.
14. **No `CONTRIBUTING.md`, no `CHANGELOG.md`, no versioning policy.** There's
    a `LICENSE` (Apache-2.0 — a real, reasonable choice), but nothing else
    a third-party contributor or consumer would look for.
15. **No real-world usage yet.** Every check in this document and the
    previous review is internal (does it build, do the tests pass, does
    the math check out). Nobody has actually depended on this from a real
    game or mod project. That's the check no amount of internal review
    substitutes for.

## Goals — phased path to production

Each phase assumes the previous one is done; skipping ahead is possible but
riskier. None of this needs to happen via another open-ended `/goal` run —
several phases (packaging, CI YAML, CONTRIBUTING.md) are exactly the kind
of bounded, checkable task `/goal` handles well individually, but doing
all of them in one pass the way the original roadmap was implemented would
be a much bigger, much harder-to-review diff than last time.

### Phase 1 — Close the loop on known issues (P0 items 1-5)
Fix the five P0 items above. This is small, concrete, and every item has
either a specific fix or a specific next diagnostic step already spelled
out. Do this before anything else — building CI on top of a repo with a
known stale doc and two live (if probably harmless) warnings just means
CI immediately has known-acceptable red items, which defeats the point of
CI.

### Phase 2 — CI and quality gates ✅ Done
- GitHub Actions matrix: Linux + macOS + Windows, at minimum two compilers
  (GCC and Clang), Debug and Release.
- A real Android job: actually invoke the NDK toolchain file and build
  (even if it can't *run* tests on-device in CI, a successful
  cross-compile is a real, currently-missing data point).
- Warnings-as-errors (`-Werror`/`/WX`) once Phase 1 lands.
- A dedicated ASan+UBSan job running the full test suite — this turns
  P0-4's one-off manual check into something that runs on every push and
  would catch a *real* regression the same way it would have caught a real
  version of that warning.
- `clang-tidy` (or `cppcheck`) as a non-blocking-at-first, then
  blocking-once-clean CI step.

**Status**: all five pieces exist in `.github/workflows/ci.yml`, each
confirmed actually running (not just written) via real pushes — 10-job
build matrix (Linux GCC+Clang, macOS Clang, Windows MSVC+ClangCL, ×
Debug/Release), `android-ndk-arm64-v8a`, `asan-ubsan`, and non-blocking
`clang-tidy`, all green as of commit `cf8c92c`. Nothing here needed a
GitHub-side setting (no secrets, no branch protection) — everything
runs on the default `GITHUB_TOKEN` and public runners.

Three real bugs found and fixed along the way, each because the *real*
CI run (not the written YAML) surfaced them:
- The initial global `-DCMAKE_CXX_FLAGS` approach to warnings-as-errors
  held CPM-fetched dependencies (Catch2) to augur's own bar and silently
  dropped MSVC's `/EHsc` default — rescoped to `target_compile_options()`
  on augur's own targets only (`CMakeLists.txt`'s new
  `augur_apply_strict_warnings()`), which fixed both at once.
- `boost::pfr::get_name()` (the reflection layer's field-name extraction,
  docs/ROADMAP.md item 12) fails its own internal self-consistency check
  under ClangCL specifically — a real, five-day-old-at-release upstream
  bug (boostorg/pfr#148) the pinned `2.2.0` tag predates. Fixed by
  bumping the pin to `boost-1.85.0`.
- `macos-clang-debug` intermittently took 20-25+ minutes, twice
  (its own Release sibling ranged 7.4-19.5 minutes across those two
  incidents — the wide range means the second one was likely broader
  runner contention that time, not purely this job oversubscribing
  itself) — root-caused via a cancelled run's cleanup log showing dozens
  of simultaneously-live orphaned `clang` processes, consistent with
  `--parallel`'s bare/unbounded job count oversubscribing a runner whose
  reported core count exceeds what it can actually sustain. Fixed by
  capping at `--parallel 3`; confirmed resolved on the very next run and
  has not recurred since (an independent subagent review afterward
  confirmed this precise timing correction and verified the fix's
  effectiveness directly against the raw CI run history).

Independently reviewed by a subagent afterward (see below) before Phase 3
began.

### Phase 3 — Testing depth
- A throughput/latency micro-benchmark suite (Google Benchmark or similar)
  for the hot paths: `KalmanFilter::predict/update`,
  `imm::Estimator::predict/update`, `HeterogeneousEstimator`'s
  expand/restrict transforms. Track results over time so a future change
  can't quietly regress performance the way it currently can't quietly
  regress accuracy.
- Expand `test_helpers.hpp`'s finite-difference approach to cover any
  model/filter that doesn't already use it (spot-check which ones do
  before assuming coverage is uniform across all 14 items).
- ~~Resolve P0-5 (heterogeneous IMM tuning) with an actual investigation
  rather than a guess, and turn whatever's learned into a regression
  test.~~ Done in Phase 1 itself (P0 item 5), not deferred to here —
  `tests/unit/test_heterogeneous_imm.cpp` already locks in the fix.

### Phase 4 — Packaging and distribution
- `install()`/`export()` rules in `CMakeLists.txt` so
  `find_package(augur)` works for consumers who don't want CPM.
- A vcpkg port and/or Conan recipe.
- Decide the Eigen version question (P1-12) explicitly and document why.
- Semantic versioning policy stated in the README, plus a real
  `CHANGELOG.md` starting now (even a single "0.1.0: initial public
  implementation of all roadmap items" entry is better than none).
- Tag an actual `v0.1.0` release.

### Phase 5 — Documentation and contributor experience
- `CONTRIBUTING.md`: build instructions (already in
  `docs/GETTING_STARTED.md` — link to it), the testing policy
  (`.claude/rules/testing.md`'s Python-first approach is genuinely good
  advice for *anyone* touching this code, not just Claude Code — surface
  it), and PR expectations (given Phase 2's CI, "must pass CI" is now a
  real, checkable bar).
- Generated API reference (Doxygen or similar) alongside the existing
  hand-written `docs/`.
- README badges: build status, license, maybe a version badge once
  Phase 4 tags a release.

### Phase 6 — Prove it under real use
- Actually depend on this from a real project — your MinerBedrock pathfinding
  mod is the obvious candidate, given it could plausibly use
  `predict::query`-style prediction for target tracking, or any other
  project where a real consumer would surface integration friction no
  internal review catches. This is the phase that actually validates
  "production grade" rather than asserting it.

### Phase 7 — 1.0 (only after Phase 6 has real mileage on it)
- An explicit API-stability commitment (what "1.0" promises won't
  silently break).
- Decide `augur` the placeholder name is either final or gets changed
  *before* 1.0, not after — a rename after real consumers exist is a much
  bigger deal than one before.

## Appendix — how the Problems section was verified (so it's checkable, not just asserted)

```sh
# Clone + full build + full test run
git clone https://github.com/tonytranrp/augur.git && cd augur
cmake -B build -DCMAKE_BUILD_TYPE=Release -DAUGUR_BUILD_TESTS=ON -DAUGUR_BUILD_EXAMPLES=ON
cmake --build build
./build/tests/augur_unit_tests          # 2379 assertions, 66 test cases, all passing

# Full warning surface (this is how P0-3 and P0-4 were found)
cmake -B build_strict -DCMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic" -DAUGUR_BUILD_TESTS=ON -DAUGUR_BUILD_EXAMPLES=ON
cmake --build build_strict 2>&1 | grep "warning:"   # exactly 2 unique root causes, ~200 lines from repetition across TUs

# CI/process gaps (this is how P1-6, P1-11, P2-13/14 were found)
find .github -type f              # nothing
find . -iname "*.yml" -o -iname "*.yaml"   # nothing outside build/
git log --oneline                  # 2 commits total
ls CONTRIBUTING.md CHANGELOG.md    # both missing
```

The ASan/UBSan check for P0-4 was a standalone repro (not checked into the
repo) instantiating `joint_probabilistic_data_association<float, 2, 4, 4>`
at full 4-track/4-detection capacity and running it under
`-fsanitize=address,undefined` — clean exit, no report from either tool.
