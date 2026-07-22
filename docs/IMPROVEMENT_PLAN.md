# augur — Improvement Plan

This is a full-codebase investigation run as 9 parallel, independent
subagents against commit `577996b` (Phases 1-2 of `docs/PRODUCTION_ROADMAP.md`
complete): 6 covering every module (`core`+`utils`+`math`, `models`,
`filters`, `imm`, `predict`+`track`, `reflect`+`plugin`+consumer API), plus
3 specifically evaluating external-library opportunities (serialization,
RNG, and a broader scan). Each agent independently read this project's own
conventions first (`CLAUDE.md`, `.claude/rules/*.md`, `docs/ARCHITECTURE.md`)
and was required to verify any real claim empirically — ad hoc `python3 -c`
math checks, real compiled C++ probes against the actual headers, measured
benchmarks (not cited marketing numbers), and in several cases real external
datasets (a live OpenSky Network ADS-B feed, the ETH BIWI pedestrian
trajectory dataset) — rather than reasoning abstractly. Every finding below
states what was actually verified and how; several negative/inconclusive
results are reported too, on purpose, matching this project's own
established practice of reporting honest findings rather than only
convenient ones (see `docs/ROADMAP.md`'s "Fixes found along the way").

This was originally a **planning document**, not a change log — at the time
it was written, nothing below had been implemented yet, and ranking/
sequencing was left as a judgment call for whoever prioritized it next.

## Implementation status (update)

Almost everything below has since been implemented, verified, and pushed,
following this document's own "Suggested sequencing" section at the bottom
in order. Each item's own section still reads as it did when only
investigated, not yet fixed — that's deliberate, so the original finding
stays intact as a historical record; this status section is the current
truth.

**Done** (top findings table, in order): #1 `safe_inverse()`'s two bugs
(fixed relative-scale, not the originally-suggested absolute floor — an
absolute 1e-9 floor turned out numerically inert in float32 for ordinary
covariance magnitudes, found and corrected during implementation); #2/#3
`filters::Filter`'s `MeasDim` bug and `plugin/registry.hpp`'s identical bug
(fixed together, plus `set_state()` added to the concept); #4
`reflect/serialize.hpp`'s Release-mode bounds check (made unconditional,
not just documented); #5 `coordinated_turn.hpp`'s second jacobian bug (three
more related bugs found via the same widened test, all fixed — see that
commit for the full account); #6 track reacquisition (minimal fix, as
scoped); #7 PDAF (`track::pdaf_update()`, new); #8 JPDA clustering (union-
find, provably lossless as this document claimed); #9
`OutOfSequenceEstimator` `Filter` conformance; #10 `imm/` padding heuristic
(cheap override-parameter fix, as scoped — the deeper BLUE-based fix
remains genuinely future work, per this document's own sequencing); #11
CTAD comments (corrected, not just hedged further); #12 `FixedVector`
(rewritten on raw storage + placement-new — notably, this document's
own suggested `std::optional`-per-slot-shaped approach was tried first and
found to make construction SLOWER, not faster, on this toolchain; see the
commit for the measured before/after); #13 `std::function` overhead (EKF/
UKF/ParticleFilter's measurement callback is now a defaulted template
parameter); #14 RNG determinism (xoshiro256++ + Box-Muller vendored,
`std::normal_distribution` replaced). The "Infrastructure / build" section's
`FilterRegistry::names()`/`unregister()` and the git.cmd PATH-shadowing
documentation are also done. The `out_of_sequence.hpp`/
`latency_compensation.hpp` ring buffer is also done (`utils/ring_buffer.hpp`,
measured 6.4x-106x faster than the shift-based code it replaced depending on
`MaxHistory`, not just the negligible-at-current-scale case this document
originally measured). The linear-drag ballistic model is also done
(`models/linear_drag_ballistic.hpp`) — its own derivation caught a real
hand-derivation algebra error via RK4 cross-check before any C++ was written.

**Deliberately not done**, with reasoning (explicitly "no urgency, land
whenever convenient" in this document's own sequencing, revisited and
reaffirmed rather than silently dropped): the quasi-3D coordinated-turn
motion model is a pure addition, not a fix to anything existing, and was
the lowest-priority single item in this document's own sequencing — the
other three items in that same original tier (ring buffer, linear-drag
ballistic, and everything above) are now done.

Every implemented item has its own commit with a from-scratch verification
methodology (ad hoc Python/numpy math checks before any C++, standalone
ASan/UBSan probes for the higher-risk changes, and/or a direct before/after
measurement for every performance claim) — see `git log` for the individual
commits, each of which explains its own verification in detail.

One correction made during synthesis, stated plainly: three of the nine
agents independently hit CMake's "ambiguous argument 'HEAD0'" error during a
fresh `CPMAddPackage` fetch and (reasonably, since they had no way to know
otherwise) attributed it to a genuine CPM v0.43.1 vs. CMake 4.3.2
incompatibility. All three ran on this same machine, which has a **known,
already-diagnosed** cause from earlier this session: npm's own `git.cmd`
shim shadows the real Git for Windows `git.exe` on this machine's `PATH`,
mangling `^` characters through `cmd.exe`'s argument re-expansion — worked
around all session via an explicit `-DGIT_EXECUTABLE="C:/Program
Files/Git/cmd/git.exe"` flag the agents were never told about. So this is
one root cause manifesting identically three times on one machine, not three
independent confirmations of a CPM bug. The still-real, still-worth-fixing
gap underneath: that workaround exists only in this session's own command
history, not written down anywhere in the repo — `cmake/get_cpm.cmake`'s own
comment claims v0.43.1 already fixed "this class of problem," which isn't
true for PATH-shadowing specifically, and any other user with a similar
npm/Git PATH collision (plausibly not rare, given how widely npm is
installed on Windows dev machines) would hit this with zero guidance. See
**P0-New-1** below.

## Top findings — the "if you only fix N things" list

Ranked by a combination of confidence (was it actually reproduced/measured,
not just reasoned about) and severity. Full detail for each is in its
module section below; this table is a navigation aid, not the complete
picture.

| # | Finding | Module | Confidence | Effort |
|---|---|---|---|---|
| 1 | `safe_inverse()` accepts singular PSD matrices via the fast path (`isPositive()` means semi-, not strictly-, definite), silently returning a zero "inverse" instead of erroring or falling back — traced into `KalmanFilter::update()` silently no-op'ing on `R=Zero()` | Foundation | Highest — reproduced, traced to real downstream effect | Low |
| 2 | `plugin/registry.hpp`'s `FilterBox`/`IFilterBox` conflate state dimension with measurement dimension — hard compile error for every realistic filter shape in the repo (the documented modding mechanism doesn't work at all) | reflect/plugin | Highest — reproduced, root-caused, fix verified compiling+running | Low |
| 3 | `filters::Filter` concept synthesizes its check vector from state dimension, not `MeasDim` — same root cause as #2, independently found from the filters side: `static_assert(Filter<T>)` gives a false positive on a real signature mismatch | filters | Highest — reproduced on 2 independent probes, cross-confirms #2 | Low |
| 4 | `reflect/serialize.hpp`'s bounds checks are `AUGUR_ASSERT`-gated, i.e. no-ops under `NDEBUG` — the build type `CLAUDE.md`/`README.md` document as the *first* example command. A truncated/mismatched-shape buffer is silently mis-parsed, not rejected | reflect/plugin | Highest — reproduced across 3 build configs + ASan proof | Low (docs) / Medium (real fix) |
| 5 | `coordinated_turn.hpp::jacobian()`'s small-omega branch has a **second**, independent bug beyond the one already fixed this session: `F(2,3)`/`F(3,2)` silently stay at 0 instead of `∓sin(ω·dt)` | models | Highest — mpmath-verified, existing test tolerance ~2x too loose to catch it at the test's own `dt` | Low |
| 6 | `TrackManager`'s greedy nearest-neighbor has no reacquisition concept — a track's `id` silently transfers to a completely different real-world object after a coast, demonstrated on real pedestrian trajectory data | predict/track | Highest — real dataset, real compiled `TrackManager`, traced a genuine identity swap | Medium |
| 7 | JPDA's association probabilities (`beta`) are computed but nothing in the codebase turns them into a corrected state estimate — no PDAF update exists anywhere | predict/track | High — derived from first principles, cross-checked 2 independent ways | Medium |
| 8 | JPDA's cost is multiplicative across independent (non-gating) track/detection clusters passed to one call — measured **1283x** blowup for 2 disjoint 5-track clusters vs. 1, on real pedestrian data through the real compiled function; clustering first is proven lossless | predict/track | Highest — real data, real function, independent losslessness proof | Low-Medium |
| 9 | `OutOfSequenceEstimator` doesn't satisfy `filters::Filter` — can't compose with `TrackManager` or `imm::Estimator` despite both being named use cases | predict/track | Highest — compiler-proven | Low-Medium |
| 10 | `big_unknown_variance()` (the padding constant fixed from 1e4→100 this session) is itself unit-scale-dependent — a full 90-step dynamic sim with rescaled units (e.g. meters→mm) shows 3.6x worse error and a 56,000x variance mismatch with the constant untouched | imm | High — real dynamic simulation, not just a static single-mix check | Low (override param) / High (real fix, see below) |
| 11 | `CTAD` deduction guides for `Estimator`/`HeterogeneousEstimator` are completely non-functional (not "may struggle," as documented) — verified on clang 22.1.4 and MSVC 19.50 both | imm | High — 2 independent compilers | Trivial (doc fix) |
| 12 | `FixedVector<T,Capacity>` default-constructs/moves all `Capacity` slots regardless of logical size, and requires `T` default-constructible — independently found by 2 separate agents (foundation-layer review and the external-library scan) via different investigative paths | Foundation + broad-scan | High — convergent, both measured real construction-cost multipliers | Medium-High |
| 13 | `std::function`-typed measurement callbacks in EKF/UKF/particle filter cost real, measured overhead (~18% on a full EKF `update()`, up to ~2.3x isolated) — independently found by 2 agents | filters + broad-scan | High — convergent, both measured | Medium |
| 14 | `std::normal_distribution`'s output for an identical seed provably differs between libc++ and MSVC STL (proven by compiling and diffing real output on both, on this machine) — a real cross-platform determinism hazard for any future replay/lockstep feature, though nothing today depends on it | filters (RNG eval) | Highest — compiled and diffed on 2 real toolchains | Low (vendor xoshiro256++) |

## Correctness bugs (fix regardless of priority — these are wrong today)

### `math/backend.hpp::safe_inverse()` — two distinct bugs in degenerate-matrix handling

Every Kalman-style filter in this library routes its innovation-covariance
inverse through this one function, making it the single most safety-critical
primitive in the codebase.

- **Bug A**: the fast-path guard is `ldlt.info()==Success && ldlt.isPositive()`.
  `Eigen::LDLT::isPositive()` means positive-**semi**-definite (confirmed
  directly against Eigen's own source), not strictly positive-definite — so
  an exactly-singular-but-PSD matrix (`R = Zero()`, a completely plausible
  "no sensor noise configured" input) wrongly takes the fast path and
  `ldlt.solve(Identity())` silently returns an all-zero "inverse," not NaN,
  not an error. Traced into `filters/kalman.hpp::update()`: a zero `S_inv`
  makes the Kalman gain `K` zero too, silently turning `update()` into a
  no-op — the exact opposite of the correct R→0 behavior (full trust in the
  new measurement).
- **Bug B**: even when the fallback branch *is* taken, its fixed
  `epsilon=1e-6` nudge plus a raw `.inverse()` call isn't robust — verified
  with a realistically-scaled indefinite input (`diag(-0.5, 100)`) that comes
  back still indefinite after the nudge, and an adversarial case
  (`m = -1e-6·I`) that produces literal NaN.
- **Verified fix**: add `ldlt.vectorD().minCoeff() > pivot_floor` to the
  fast-path guard, and replace the fallback's epsilon-nudge with
  `project_to_psd(m).inverse()` — reusing the library's own existing, more
  robust primitive already in the same file. Confirmed this resolves every
  pathological case tried while being numerically identical (max diff
  1.86e-9 in float32) to plain `.inverse()` on well-conditioned input.
- No dedicated test exists for `safe_inverse`'s degenerate path today —
  plausibly why this went uncaught despite the project's otherwise strong
  testing discipline.

### `plugin/registry.hpp` and `filters::Filter` — the same root cause, found twice independently

`IFilterBox<Scalar,Dim>`/`FilterBox<F>` (`plugin/registry.hpp`) type the
`update()` measurement argument as `Vector<Scalar,Dim>` where `Dim` is bound
to the filter's **state** dimension. Every real `Filter` (`KalmanFilter<Model,
MeasDim>` etc.) actually wants `Vector<Scalar,MeasDim>` — an independent
template parameter, only equal to `dimension` in the degenerate case of
observing the whole state directly. Reproduced: registering a real
`KalmanFilter<ConstantVelocity<float,2>,MeasDim=2>` (dimension=4) — exactly
the README's own teaser shape — fails with `YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES`.

Root cause of *why nothing else caught this*: `filters::Filter`'s own
concept (`filter_concept.hpp`) synthesizes its own check vector from
`T::dimension` too, so `augur::filters::Filter<KalmanFilter<CV,2>>` itself
evaluates `true` despite the mismatch — Eigen's converting constructor is an
unconstrained template, so unevaluated-context concept checking doesn't
instantiate the body deeply enough to catch it, while `FilterBox`'s ordinary
(fully-evaluated) method body does. This means `static_assert(Filter<YourType>)`
— the exact pattern `docs/PLUGIN_GUIDE.md` recommends for validating a
custom filter — gives false confidence about `update()`'s true signature.

grep confirms zero existing callers of `FilterRegistry`/`FilterBox` anywhere
in `examples/`/`tests/`, and there's no `test_plugin_registry.cpp` despite
`docs/ARCHITECTURE.md` §6 listing this file as "Solid ... Covered by
tests/unit/" — the template had likely never actually been instantiated by a
compiler before this investigation.

**Verified fix** (both sides): give `IFilterBox`/`FilterBox`/`FilterRegistry`
a separate `MeasDim` template parameter mirroring `KalmanFilter`'s own
established pattern; have `filter_concept.hpp` synthesize its check vector
from a `measurement_dimension` constant each filter can trivially expose,
not `T::dimension`. Both compiled and ran correctly. Should ship with a real
`tests/unit/test_plugin_registry.cpp` — its total absence is exactly how
this shipped broken.

**Related, same investigation**: `filter_concept.hpp` also doesn't require
`set_state()`, even though `imm::Estimator`, `imm::HeterogeneousEstimator`,
and `track/out_of_sequence.hpp` all call it unconditionally every cycle —
and `docs/PLUGIN_GUIDE.md`'s stated `Filter` contract doesn't mention it
either. A user who follows the guide and validates with the concept gets a
clean pass, then a confusing deep template error the moment they plug into
`imm::Estimator`. Fix: add it to the concept's requires-clause and the
guide's prose — all 6 built-in filters already implement it correctly, so
this breaks nothing existing.

### `reflect/serialize.hpp` — bounds checks vanish under the documented default build type

`ByteReader::read<T>()`/`ByteWriter::write<T>()` and
`SnapshotBuffer::deserialize()`'s count check all gate their bounds checks
behind `AUGUR_ASSERT`, which is a deliberate, documented no-op under
`NDEBUG` (`utils/assert.hpp`) — and `-DCMAKE_BUILD_TYPE=Release` (which
defines `NDEBUG`) is the first build command shown in `CLAUDE.md`,
`README.md`, and `docs/ARCHITECTURE.md`.

Reproduced with a real scenario (`SnapshotBuffer`'s actual public API): a
game changing tracked state shape between patches (`Vector<float,4>` →
`Vector<float,6>`), naively loading an old save with new code:

- **Debug**: `assert()` fires correctly, process aborts.
- **Release** (`-DNDEBUG -O2`, the documented default): exit code 0. Old
  data's real fields print correctly; the two "new" fields print as
  **another snapshot's actual timestamp and px, byte-for-byte** — silently
  spliced in, not garbage, not an error.
- **Release+ASan**: confirms this is a genuine OOB read
  (`container-overflow ... in ByteReader::read<float>()`), not coincidence.

Generalizes beyond version skew to any truncated/corrupted buffer — a save
cut off by a full disk, a malformed network packet, both realistic for this
library's stated modding/networked-game audience. The wire format's own
comment already, reasonably, disclaims cross-version schema support; it does
not disclaim that Release builds have *zero* protection against malformed
input, which is a materially different and currently-undocumented claim.

Two independent, non-exclusive options: (a) document the gap loudly at the
top of `serialize.hpp`/`SnapshotBuffer` (trivial, doesn't close it), or (b)
make `ByteReader`'s bounds check unconditional specifically in this one file
— unlike the rest of the library's asserts, which document internal
invariants the caller never violates, this is the one path parsing
externally-sourced bytes, a materially different trust boundary (medium
effort — needs a real error-reporting shape, touching one test file and
`SnapshotBuffer`'s two callers).

### `models/coordinated_turn.hpp::jacobian()` — a second bug in the already-once-fixed branch

The small-`|omega|<0.2` branch never sets `F(2,3)`/`F(3,2)` (stay at
`Identity()`'s default 0); their true value is `∓sin(ω·dt)` — a
**first-order** term, not the genuinely-negligible second-order case the
adjacent comment correctly describes for `F(0,4)`/`F(1,4)`. Verified via
50-digit mpmath analytic differentiation (not finite-difference, to rule out
truncation artifacts). Only affects EKF covariance propagation, not
`transition()` itself (which uses a much tighter, unaffected `epsilon=1e-5`).

At the repo's own fixed test parameter (`dt=1/30`), the error is ~0.005 —
under `test_helpers.hpp`'s default `0.01` tolerance by roughly 2x, which is
exactly why `tests/unit/test_coordinated_turn.cpp` doesn't catch it today.
At `omega=0.19, dt=1.0s` (both individually plausible, e.g. during a
track-coasting gap), the error is 0.189 absolute. Verified fix: `F(2,3)=-s;
F(3,2)=s;`, reusing the `s,c` already computed unconditionally two lines
below for `F(2,4)`/`F(3,4)` — the exact same idiom already in the file.
Recommend widening the test's finite-difference sweep to vary `dt`, not just
`omega`, since a fixed `dt=1/30` is what let this slip through.

A related, lower-priority finding in the same investigation: `F(0,4)`/`F(1,4)`'s
existing degree-4 Taylor series degrades outside its `dt` assumption (tested
accurate to <0.01% up to `omega·dt≈0.9`, but the branch guard tests `|omega|`
alone rather than the actual small parameter `|omega·dt|`) — only bites at
multi-second `dt`, much rarer than the bug above.

### Multi-target tracking: three real gaps, validated against real pedestrian trajectory data

Using the ETH BIWI pedestrian dataset (a standard multi-object-tracking
benchmark source) through the real, compiled `track/` code:

- **Track identity is not stable across a coast-then-reacquire.** Running
  real `TrackManager` (GNN) against a dense, real 16-frame window: internal
  track `0` followed real pedestrian 8, coasted one frame (pedestrian 8
  genuinely left frame), then was **silently reassigned to pedestrian 21** —
  a different person who had just entered — with `track.id` unchanged
  throughout. Root cause confirmed by reading `track_manager.hpp`: no
  reacquisition/re-ID concept exists at all; `nearest_neighbor()` can't
  distinguish "continuation of the same thing" from "something new that
  happens to be nearby." Over the real 16-frame window: 8 ID switches, 7
  fragmentations across 14 real pedestrians. Minimal fix (low-medium
  effort, low risk): refuse silent reacquisition after any coast, requiring
  fresh confirmation like a brand-new track. Fuller fix (medium risk, changes
  default lifecycle behavior) needs an appearance/feature hook the `Filter`/
  measurement types don't carry today.
- **JPDA computes real association probabilities nothing consumes.**
  `joint_probabilistic_data_association()`'s `beta`/`beta_missed` outputs are
  only ever printed (by its own example/test) — no PDAF (Probabilistic Data
  Association Filter) update exists anywhere to turn them into an actual
  corrected state estimate. Derived the missing combined-innovation update
  from first principles (Bar-Shalom & Fortmann) as a Gaussian-mixture
  moment-match — the same "weighted mean + spread-of-means" pattern already
  trusted in `imm/mixing.hpp::combine()` and `gm_phd.hpp`'s merge step, not
  a new primitive. Cross-checked against an independent direct-mixture numpy
  computation (agreement to 1e-12) and against hard-assignment on the same
  ambiguous case: state differs by 0.08 on a 0.3-0.7 scale, and critically,
  the hard update's covariance trace (1.0) *understates* the true
  uncertainty PDAF correctly reports (1.076) — pretending there was no
  assignment ambiguity when JPDA's own beta values (0.49/0.44/0.06) say
  there clearly was.
- **JPDA's cost is multiplicative across independent clusters, and
  clustering first is provably free.** Beyond the already-documented 32-bit
  bitmask ceiling: built a real 5-pedestrian mutually-gated cluster from real
  ETH data, replicated it at a disjoint spatial offset, ran the actual
  compiled function — 1 cluster of 5 costs 27.7μs; 2 independent clusters of
  5 (10 tracks total, never competing with each other) costs 72,085μs, a
  **1283x** blowup for adding a cluster that shares nothing with the first.
  Independently proved in Python (brute-force joint-event enumeration) that
  solving disconnected clusters separately gives bit-identical marginals
  (1e-12 agreement to a monolithic solve) — clustering first via union-find
  over the gate check the function already computes is a pure win, zero
  output change, low-medium effort.

Two smaller, cheap, low-risk findings from the same investigation:
`OutOfSequenceEstimator` doesn't satisfy `filters::Filter` (no `dimension`,
no split `predict`/`update`, no `last_likelihood()` — compiler-proven,
blocks composing OOSM with `TrackManager`/`imm::Estimator` despite both
being named use cases for it), and `TrackManager::step()` computes each
match's real Mahalanobis `gate_distance` then discards it rather than
storing it on `Track` (a consumer wanting match-quality-driven confidence,
e.g. for aim-assist, has to recompute it themselves today).

GM-PHD's already-documented `extract_targets()` under-count (a component
with weight >1 still yields exactly one target) has a standard, literature-
confirmed fix: the field's actual convention (Vo & Ma 2006) is
`round(weight)` duplicate-mean emission, not a k-means/covariance-split
scheme. Demonstrated against the real filter (three close weight-0.9 births
merge to weight-2.7; `extract_targets()` reports 1 today where round(2.7)=3).
Honest limitation stated plainly: duplicates share the *identical*
mean/covariance — a known limitation of the technique itself, not a
shortcoming of implementing it.

## Performance findings (all measured, not assumed)

- **`KalmanFilter`/`ExtendedKalmanFilter::update()` compute `P·Hᵀ` twice**
  (once for `S=H·P·Hᵀ`, once for `K`), and materialize a full identity matrix
  for `P_ = (I-K·H)·P_` where `P_ -= K·(H·P_)` (reusing the hoisted `P·Hᵀ`)
  avoids both — exactly the class of bug already fixed in
  `track/association.hpp`/`gm_phd.hpp`, not yet applied to the filters
  themselves. Measured **1.24x** speedup on the complete `update()` at
  StateDim=6/MeasDim=3/float, state output bit-identical, covariance
  differing only at float32 noise floor (≤3.1e-6).
- **`std::function` measurement callbacks cost real overhead** in EKF/UKF/
  particle filter — independently measured by two agents. One found ~18%
  overhead on a full simulated EKF `update()` (111.5ns vs 94.7ns) and 2.09-
  2.31x on isolated calls at UKF/particle-filter call counts (~14μs of pure
  type-erasure tax per `ParticleFilter::update()` at 2000 particles). Fix:
  make the callback an optional template parameter (defaulting to
  `std::function` for source compatibility) — matches this project's own
  "prefer a template parameter over a runtime flag for anything decided
  once" convention and `imm::Estimator`'s own zero-vtable precedent. Real,
  quantified tradeoff acknowledged: `std::function` currently gives every
  instantiation the same nameable type regardless of the lambda passed,
  which matters for listing concrete types in `imm::Estimator<Filters...>`;
  a template-parameter callback needs CTAD-friendly factory functions to
  stay equally ergonomic.
- **`utils::FixedVector<T,Capacity>`'s array-backed storage costs scale with
  `Capacity`, not logical size** — independently found by two agents via
  different paths. `std::array<T,Capacity> storage_{}` default-constructs
  all `Capacity` slots unconditionally (confirmed: requires `T`
  default-constructible — the exact friction `track_manager.hpp`'s own
  comment already documents working around via `std::optional<FilterT>`),
  and `emplace_back` copy/move-*assigns* over an already-constructed slot
  rather than placement-constructing. Measured: constructing+filling 16-of-256
  slots costs 10.5x a placement-new equivalent; `TrackManager::step()`'s
  real per-frame pattern (rebuilding `MaxTracks`-sized `FixedVector`s every
  frame) costs 154x more at MaxTracks=1024/16-active than a properly-
  amortized container. **Honest caveat, checked**: at the repo's own actual
  in-tree `MaxTracks` values (4, 8, per existing tests/examples), the gap is
  negligible or reversed — this only bites a consumer who follows the
  library's own stated intent (size `MaxTracks` generously above typical
  active count). `push_back`/`emplace_back` past capacity is also silently
  undefined in Release (`NDEBUG`) — verified this is *inconsistently*
  either invisible (no crash) or an immediate ASan violation purely
  depending on stack layout, undocumented anywhere in the file. Recommended
  approach: vendor a real fix in place (placement-new/manual lifetime,
  mirroring the soon-to-be-standardized `std::inplace_vector`/P0843, which
  research confirmed is motivated by exactly this problem pair) rather than
  add a Boost.Container dependency — matches this project's own demonstrated
  house style (`safe_inverse`/`project_to_psd` are hand-vendored, cited
  algorithms, not library pulls). Medium-high effort (real internals
  rewrite, wide blast radius since every `track/`/`predict/
  latency_compensation.hpp` consumer uses it, though the public API
  wouldn't change) — but two independent agents converging on it raises
  confidence this is worth doing.
- **`HeterogeneousEstimator::predict()` mixing overhead measured at ~2x**
  the three filters' own `predict()` cost combined (574ns vs 292ns/cycle,
  2M-iteration timing test on the real CV+CA+CT case). Root cause: always
  mixes over the full fixed `kAugmentedDim=7` even when the actually-mixed
  models only populate a subset. Scoped specifically to the opt-in
  `HeterogeneousEstimator` path — `imm::Estimator`'s default zero-overhead
  same-dimension path is unaffected. Medium-high effort (shrinking the
  working dimension to the union of just the mixed models' own layouts is a
  real refactor of already-tested machinery) — ranked below the correctness
  findings above given `HeterogeneousEstimator` is already "flagged sketch"
  tier.
- **`std::mt19937_64`+`std::normal_distribution` measured 1.4-1.8x slower
  per sample and ~200x slower to construct** than a hand-vendored
  xoshiro256++ engine, across every realistic `NumParticles` scenario pulled
  from this repo's own tests/examples (300 to 16000). Construction cost
  specifically matters for frequent spawn/despawn of short-lived
  particle-filtered entities (projectiles, temp AI targets); per-sample cost
  matters when many entities are simultaneously particle-filtered in one
  frame (measured ~7-10% of a 16.6ms frame budget from RNG alone at a
  16000-particle single-frame burst). See "External library evaluation"
  below for the full recommendation.
- **`CoordinatedTurn`'s `jacobian()`+`transition()` pair redundantly compute
  the same sin/cos** (both called every `predict()` per `filters/{kalman,
  extended_kalman}.hpp`'s unconditional call pattern). Measured 66% reduction
  (18.55ns → 6.30ns/call) fusing them, ~14.3% of a full
  `KalmanFilter<CoordinatedTurn>::predict()`. **Important, self-corrected
  finding**: this does NOT generalize — benchmarking `Singer`/`ConstantVelocity`/
  `ConstantAcceleration` under the identical hypothesis found *no* win (the
  compiler already commonizes their straight-line/trivial control flow via
  ordinary inlining+CSE; only `CoordinatedTurn`'s branchier structure
  resists it). A caching fix inside the model would violate this project's
  "reentrant function of its inputs" design principle
  (`docs/ARCHITECTURE.md`); the clean alternative is an optional
  `transition_and_jacobian()` extension point detected via `if constexpr
  (requires{...})`, falling back to today's two-call path — reasoned but not
  implemented/verified end-to-end. Low priority given the modest absolute
  cost (likely low-microseconds/frame even at a few hundred tracked
  entities).

## Design-level findings — `imm/`'s padding heuristic

The single most interesting open question this investigation surfaced.
`big_unknown_variance()` was fixed from 1e4 to 100 earlier this session
(`docs/ROADMAP.md` item 0's "Update" paragraph) after it was found to nearly
erase a recovering model's own prior during mixing. This investigation found
the fix is real and correct, but not the end of the story:

- **The constant is unit-scale-dependent.** A full 90-step dynamic
  simulation using the real, unmodified `HeterogeneousEstimator<KFCV,KFCA>`
  (constant velocity, then real sustained acceleration) with position/
  velocity/acceleration expressed in units 1000x smaller (e.g. meters→mm)
  showed combined-position error 3.6x worse (0.046 vs 0.013), post-onset
  AccelX variance ~56,000x too small relative to a healthy baseline, and the
  mode-probability crossover timing itself shifted non-physically (step 30
  vs. never, same 90-step window) — purely from a units choice, constant
  untouched. This is because the padding constant is applied identically to
  every augmented component regardless of coordinate units, but e.g.
  acceleration variance scales as length² while the constant doesn't. MATLAB's
  Sensor Fusion and Tracking Toolbox (`trackingIMM`) solves this via a
  user-supplied `ModelConversionFcn` callback — external confirmation this
  is standard practice, not a novel problem. **Cheap partial fix** (low
  effort/risk): a defaulted constructor parameter on `HeterogeneousEstimator`
  overriding the constant, fully backward-compatible.
- **A cheaper structural step**: per-augmented-component reference-variance
  padding (borrow real variance from whichever other currently-mixed model
  tracks that specific component, e.g. CT's own turn-rate variance) fixed
  both the CV→CA scale problem *and* a demonstrated failure mode of the
  naive "per-model average" alternative (which overshoots turn-rate variance
  by 26,000,000x at the same rescaled unit) in a static single-mix test using
  the real `imm::mix()`. **Not verified dynamically** — has a real,
  articulated but untested risk (could slow the ensemble's uncertainty
  inflation right as a new maneuver begins, since it partly inherits other
  models' pre-maneuver confidence). Medium effort/risk: needs cross-model
  visibility at expand time (a real signature change) plus genuine dynamic
  validation before trusting it.
- **The deepest answer**: literature search surfaced Granström, Willett &
  Bar-Shalom, "Systematic approach to IMM mixing for unequal dimension
  states," IEEE Trans. Aerospace and Electronic Systems, 51(5), 2015, pp.
  2975-2986 — which names "naive constant-variance padding" (augur's
  current approach) as the crude baseline and proposes Best Linear Unbiased
  Estimation (BLUE): deriving the untracked component's conditional
  mean/covariance from a designed cross-covariance structure with the
  tracked ones, instead of an arbitrary zero-mean/huge-variance placeholder.
  It independently identifies the same "dominant model's padding erases a
  recovering model's confidence" failure mode this codebase already found
  and fixed. **Not implemented or numerically verified** — flagged as the
  strongest long-term research lead, substantially more effort than
  anything else in this document (designing a physically-motivated
  cross-covariance prior per component pair, e.g. the real turn-rate/
  lateral-acceleration relationship for coordinated turn).

Two smaller, cheap `imm/` findings from the same investigation:
`HeterogeneousEstimator` has no `predict_ahead()` (the library's headline
latency-compensation capability, per `predict/latency_compensation.hpp`'s
own comment pointing at `Estimator::predict_ahead()`, is simply unreachable
for the "textbook full IMM" 3+ model case — implemented and verified a fix
using only primitives the library already ships, exact match to hand-
computed expectation); and the `Estimator`/`HeterogeneousEstimator` CTAD
deduction guides are completely non-functional on both clang 22.1.4 and
MSVC 19.50 (not "may struggle on some toolchains," as the code's own comment
currently claims) — every example/test already uses the explicit form, so
this is currently invisible, but the comment overstates what works, in
tension with this project's own three-tier honesty convention. Trivial fix:
correct the comment.

## External library evaluation

Three focused investigations, each required to actually build and run
candidate libraries against augur's real use cases rather than compare
feature lists.

**Serialization** (`reflect/serialize.hpp` vs. zpp::bits, bitsery, cereal;
Cap'n Proto/FlatBuffers ruled out early, same reasoning
`docs/ARCHITECTURE.md` §3 already used to rule out Blaze/Armadillo for
math — both need a schema-compiler codegen step, not a fit for a one-CPM-line
dependency posture). **Recommendation: keep the custom system.** All three
tested candidates, once given the same hand-written "N contiguous Scalars"
Eigen glue augur's own `VectorBackend` already provides, tie augur's own
format exactly on wire size (16 bytes for the `SnapshotBuffer::Snapshot`-
shaped test case) — nobody beats the theoretical minimum, and
`reflect::Descriptor<T>` is a general reflection facade (designed to gain a
P2996 backend later with zero call-site changes), not just a serialization
substrate, which none of the candidates replace. The one genuinely
worthwhile finding: **bitsery's `Growable` extension**, if cross-version
save/replay compatibility (the versioning gap `serialize.hpp`'s own comment
already, deliberately, scopes out) ever becomes a real product need, is the
best-fit *opt-in* answer among the three — purpose-built for games/real-time
networking, field-scoped rather than archive-global, and its forward/
backward-compatibility claim was independently verified bidirectionally
(not just trusted from its README), not added wholesale as a replacement.

**RNG** (`filters/particle_filter.hpp`'s `std::mt19937_64`+
`std::normal_distribution`). **Recommendation: vendor xoshiro256++
(Blackman & Vigna, ACM TOMS 47(4), 2021, art. 36) directly, ~40 lines,
public-domain reference implementation, matching this project's own
`safe_inverse`/`project_to_psd` precedent of hand-writing small cited
numerics over adding a dependency for one file's needs.** Measured
1.4-1.8x faster per sample, ~200x faster to construct, across every
realistic particle count in this repo's own tests/examples (see Performance
section above for the numbers). Separately, and empirically (not just
cited): compiled the identical `std::mt19937_64(42)`+`std::normal_distribution<float>`
code under libc++ (the same stdlib Android NDK uses) and MSVC STL on this
machine and diffed the output — `std::normal_distribution` genuinely
diverges between them (~60% of the first 10 draws differ) even from an
identical engine/seed, while the raw engine output itself is identical
(fully specified by the standard). A hand-vendored xoshiro256++ + Box-Muller
candidate was verified bit-identical across both toolchains. **Honestly
scoped**: this doesn't currently bind on anything in augur — no test
asserts exact RNG output, and `track/out_of_sequence.hpp`'s replay mechanism
rolls back via a Gaussian state summary, never replaying the actual particle
cloud, so it's structurally unaffected either way. This is a real concern
for a *future* feature (lockstep replay/spectator resim), not a bug in
anything shipping today — but the performance case alone (measured, not
assumed) already justifies the vendor.

**Broader scan** (everything else in the codebase). Converged independently
on the `std::function` and `FixedVector` findings above (see Performance
section — cross-referenced there rather than repeated). Additionally found:
a hand-duplicated shift-based "ring buffer" in both `track/out_of_sequence.hpp`
and `predict/latency_compensation.hpp` (both call themselves a ring buffer
in their own comments; neither actually is one — O(MaxHistory) per push via
shift-on-full, when a real head-index ring buffer is O(1)). Measured:
genuinely negligible at this repo's own actual `MaxHistory` values (2, 4, 8,
32 — single-digit nanoseconds difference), but a real, gratuitous
inefficiency if `MaxHistory` is scaled up for a longer lag-compensation
window (121x at MaxHistory=512), which is a plausible ask for this exact
feature. `boost::circular_buffer`/`std::deque` both ruled out (heap-backed,
directly conflicting with why `FixedVector` exists in this codebase at
all) — the right fix is a small (~15-20 line) head-index ring buffer
promoted into `utils/`, used by both call sites, which also fixes the DRY
violation of the same pattern hand-copied twice. Low priority given the
measured negligible cost at current scale.

Several things were checked and explicitly found *not* worth changing,
reported for completeness: `plugin/registry.hpp`'s `std::unordered_map`/
`std::function` use (deliberate, documented, opt-in, non-hot-path); no
`std::regex` or heavy `iostream` formatting anywhere in `include/`; JPDA's
and GM-PHD's actual linear algebra already routes through Eigen, with the
non-Eigen parts being the library's own algorithmic value-add, not something
a library would replace.

## Infrastructure / build

**P0-New-1**: the CPM/git "ambiguous argument 'HEAD0'" issue (see the
correction note at the top of this document) has a real, narrower-than-
originally-reported but still-worth-fixing gap: this session's own working
fix (`-DGIT_EXECUTABLE="C:/Program Files/Git/cmd/git.exe"`) is not captured
anywhere in the repo — not `CMakeLists.txt`, not `docs/GETTING_STARTED.md`,
not `cmake/get_cpm.cmake`'s own comment (which incorrectly claims v0.43.1
already fixed "this class of problem" universally). Any user with a similar
PATH collision between a real Git install and another tool's `git` shim
(npm's is one concrete, plausible-at-scale example, given how widely npm is
installed on Windows dev machines) hits a cryptic error with zero guidance
today. Low effort: at minimum, document the workaround and the actual root
cause (a PATH collision, not a CPM bug) in `docs/GETTING_STARTED.md`'s
Windows section; investigate whether CMake's own `find_program(Git)`
selection can be made robust against this by default as a more complete fix.

`FilterRegistry` (`plugin/registry.hpp`) has no way to enumerate what's
registered or unregister an entry — a real gap given its own stated use case
(a mod DLL registering/retracting models at runtime): no `names()`/iteration,
no `unregister()`. Low effort, purely additive.

Minor: Boost.PFR is fetched unconditionally in `CMakeLists.txt` (no
`AUGUR_WITH_*`-style gate, unlike `glm`/Catch2), a small inconsistency with
this project's own stated CMake convention ("a consumer who just wants the
library pays for nothing extra") — a consumer who will only ever use
`models::`/`filters::`/`imm::` still pays for a PFR fetch. Cheap fix (wrap
in an option), low priority.

## New motion models — both rigorously derived and verified, not just proposed

- **Linear-drag ballistic/projectile model** (`dv/dt = g - k·v`, state
  `[p,v]` per axis, SpatialDim-templated like `ConstantVelocity`). Fills a
  real gap: pure-gravity ballistics is already representable via
  `ConstantAcceleration`, but drag isn't, and tracking a thrown/fired
  projectile (not the shooter) is a real game/mod need. Closed-form
  transition derived via sympy, cross-checked against RK4 (max error
  1.2e-12 over 200 trials); jacobian = the affine part exactly, reusing
  `CurrentStatistical`'s own already-verified reasoning for a constant
  offset; **proactively found and fixed its own version of the numerical-
  cancellation family this project already hit once in `coordinated_turn.hpp`**
  (as k→0, the closed form's 1/k, 1/k² terms blow up and cancel — derived
  and verified a small-k Taylor-series safe branch, mirroring CT's own fix
  pattern, exact down to k=0 against RK4); exact (not simplified-placeholder)
  process noise via the Van Loan integral, independently cross-checked
  against direct numerical integration and confirmed PSD, whose k→0 limit
  exactly reproduces `ConstantVelocity`'s own CWNA formula already in the
  codebase (a nice independent internal cross-check). **Honest scoping,
  verified via Reynolds number**: computed Re for realistic game
  projectiles (rifle bullet ≈207,000, arrow ≈32,000, thrown grenade
  ≈61,000) — all solidly in the quadratic-drag regime, not linear/Stokes.
  This model is a good fit specifically for games whose own physics already
  uses linear drag (a common arcade/stability simplification), explicitly
  *not* "realistic bullet physics" — quadratic drag would need numerical
  integration and a harder Jacobian, flagged as a harder, unverified
  follow-up, not proposed here. Structurally verified: compiled a full
  prototype satisfying `static_assert(MotionModel<...>)`, ran correctly
  through real `KalmanFilter::predict()/update()` and mixed into a real
  `imm::Estimator` alongside a built-in `ConstantVelocity`, zero changes
  needed to any existing header.
- **Quasi-3D ("horizontal turn + decoupled vertical") coordinated turn** —
  exactly the gap `imm/augmented_layout.hpp`'s own file comment names
  explicitly (a full SO(3)/Sophus turn model is out of scope per
  `docs/ARCHITECTURE.md`). State `[px,py,vx,vy,omega,pz,vz]`: the existing,
  unmodified `CoordinatedTurn` xy-block reused via composition (mirroring
  how `CurrentStatistical` already reuses `Singer` internally) plus a
  decoupled CV-style vertical channel — the same design real ATC/aircraft-
  tracking literature uses. Finite-difference-verified (matching
  `test_helpers.hpp`'s own `h=1e-3` methodology) across a 13-value omega
  sweep × 30 trials, confirming zero off-block-diagonal jacobian entries.
  Structurally verified compiling and running against real headers.
  Attempted real-world grounding via a live OpenSky Network ADS-B fetch
  (real aircraft over Switzerland): real, simultaneous aircraft showed
  vertical rates from -8.13 to +6.18 m/s across varied, uncorrelated-looking
  headings, qualitatively supporting the decoupling assumption — honestly
  reported as a cross-sectional plausibility check, not a longitudinal
  proof (a second fetch attempting to track one aircraft over time hit an
  API caching limitation). **Inherits both `coordinated_turn.hpp` bugs
  above**, since it reuses that block verbatim. Structural cost beyond the
  model file itself: `core/state_component.hpp`'s canonical augmented-state
  enum is 2D-only today (7 components, no Z) — participating in
  `HeterogeneousEstimator` would need that enum extended plus a new
  `AugmentedLayoutFor<>` specialization; standalone use (own same-dimension
  IMM ensemble, or a lone `KalmanFilter`) needs neither, verified directly.

A waypoint-following model was considered and explicitly rejected with clear
reasoning (`MotionModel::transition(x,dt)` only ever sees current state +
dt, no map/waypoint context; a fully-known path needs no estimation at all,
a partially-known one has a genuine dynamics discontinuity at each waypoint
switch that breaks the EKF's local-linearization assumption) — reported
honestly as a rejected idea, not silently dropped.

## What was investigated and found already solid (no action needed)

Reported for completeness, matching this project's own practice of stating
negative results plainly: `project_to_psd()` itself; 3D `interop_glm.hpp`
conversions; `core/config.hpp`'s feature-detection; `ParticleFilter::
systematic_resample()`'s O(N) correctness; UKF's documented choice to
regenerate sigma points in both `predict()`/`update()`; Joseph-form
covariance symmetrization (hypothesized as a float32 drift risk, tested
across up to 2 million predict/update cycles, found no meaningful
difference from the simplified form already in use); an Iterated EKF
investigation for bearing-only tracking (found real MAP-convergence but a
genuine, literature-consistent MAP-vs-MMSE caveat for skewed posteriors —
would need that caveat stated plainly if ever added, not presented as a
strict upgrade); `SnapshotBuffer::rewind_to()`'s linear interpolation and
angle-wrapping (checked — `CoordinatedTurn`'s state has no wrapped-angle
component, turn *rate* only, so this plausible-sounding concern doesn't
apply); redundant-computation sweeps of `gm_phd.hpp`'s merge step and
`fusion.hpp` (both already minimal, no further hoisting found beyond the
two fixes already shipped this session).

## Suggested sequencing (a starting point, not a mandate)

This wasn't asked to be prioritized into phases the way
`docs/PRODUCTION_ROADMAP.md` was, but a rough shape, if useful:

1. **Cheap, high-confidence correctness fixes first** — `safe_inverse`'s two
   bugs, the `Filter` concept's `MeasDim` fix (which also fixes
   `plugin/registry.hpp`), the `coordinated_turn.hpp` second jacobian bug,
   the missing `model()`/`set_state()`-in-concept gaps. All low effort, all
   independently reproduced, all narrow blast radius.
2. **`reflect/serialize.hpp`'s Release-mode bounds check** — at minimum the
   documentation fix immediately; the real unconditional-check fix as a
   near-term follow-up given the demonstrated silent-corruption severity.
3. **Multi-target tracking gaps** — the JPDA clustering fix (provably
   lossless, low-medium effort) and the track-reacquisition minimal fix
   (low-medium effort) are the best effort-to-impact ratio here; PDAF and
   `OutOfSequenceEstimator`'s `Filter` conformance are real but larger.
4. **Performance work** — the `KalmanFilter`/EKF redundant computation and
   `FixedVector` fixes are the best-justified (measured, and the latter
   independently convergent); `std::function` and the RNG vendor are real
   but lower urgency until profiling shows they matter for a specific
   consumer's actual entity counts.
5. **`imm/` padding heuristic** — the cheap override-parameter fix can land
   anytime; the deeper BLUE-based research is a substantial, separate
   research effort that deserves its own dedicated pass, not a quick patch.
6. **New models and infrastructure fixes** — genuinely valuable, no urgency;
   land whenever convenient.
