# augur — Roadmap

Fourteen extension ideas beyond the core (models + filters + IMM), grouped by
theme, each with why it matters, the relevant literature, which file it
already has a stub in (if any), and a rough effort estimate. This is a
superset of "at least 10" on purpose — pick the ones that match what you're
actually tracking rather than building all of them.

Also tracked here: the one **structural** improvement to the core itself
(different-order IMM mixing) that isn't a new feature so much as completing
the existing one.

Status markers (`✅ Done` / unmarked = not started) are added to each item's
heading as it's implemented, with a pointer to the code, test, and example
that cover it.

## Fixes found along the way (not originally listed as roadmap items)

- **`models/coordinated_turn.hpp` jacobian() small-omega bug** — the
  degenerate branch (`|omega| < epsilon`) left `F(0,4)`/`F(1,4)`/`F(2,4)`/
  `F(3,4)` at their `Identity` default (0) instead of their true small-omega
  limits. Since a target starting at (or passing through) `omega=0` has no
  other path to a nonzero turn-rate estimate, this made the turn-rate state
  structurally unobservable from position measurements alone — IMM mode
  probabilities could never move, no matter how differently the mixed
  filters were tuned. Root-caused and fixed via a Taylor-series small-angle
  expansion (verified against an mpmath reference, ad hoc, per
  `.claude/rules/testing.md`); `examples/02_imm_maneuvering_target` was
  retuned afterward since its old parameters were too mild to show the
  (now-working) effect. See git history / diff for details; regression
  coverage lives in `tests/unit/`.
- **`track/association.hpp` redundant per-detection covariance inversion**
  — both `nearest_neighbor()` and `joint_probabilistic_data_association()`
  loop tracks-outer, detections-inner, but recomputed `safe_inverse()` (and,
  for JPDA, `.determinant()` too) of the SAME per-track innovation
  covariance on every inner-loop iteration — O(tracks × detections)
  redundant decompositions where only O(tracks) are needed, since a
  track's innovation covariance doesn't depend on which detection it's
  being compared against. Hoisted into a `detail::InnovationGate` computed
  once per track, reused across all detections and across both the
  Mahalanobis-distance and Gaussian-likelihood call sites. Pure reordering
  of the same formulas — verified via `tests/unit/test_association.cpp`'s
  existing hand-computed reference values, unchanged bit-for-bit after the
  refactor.
- **`track/gm_phd.hpp::update()` redundant per-detection Kalman gain** — the
  same class of issue, one level more expensive: `S`, `S_inv`, the Kalman
  gain `K`, and `det_s` were recomputed for every (predicted component,
  detection) pair even though all four depend only on the component's own
  predicted covariance (via the filter's fixed `H`/`R`), never on the
  detection. Precomputed once per component into three `FixedVector<...,
  MaxComponents>` arrays ahead of the detections loop, which otherwise keeps
  its exact original structure (same `denom` accumulation, same posterior
  push order) to minimize risk of changing behavior. Verified via
  `tests/unit/test_gm_phd.cpp`'s existing reference value, unchanged after
  the refactor.

## Structural: complete the core

### 0. Different-order IMM mixing ✅ Done
**What**: let `imm::Estimator<Filters...>` mix models of genuinely different
state dimension (5-state coordinated-turn against 6-state constant-velocity),
via the common-augmented-state mapping matrices in Bar-Shalom's general IMM
formulation, instead of today's same-dimension-only restriction.
**Why**: this is the textbook "full" IMM example (CV + CA + CT together) —
worth having once the same-dimension version has proven itself.
**Reference**: Bar-Shalom, Li & Kirubarajan, *Estimation with Applications to
Tracking and Navigation*, Wiley, 2001, ch. 11.6.
**Where**: would live in `imm/mixing.hpp` alongside the existing `mix()`.
**Effort**: Medium-high — the math is well-documented but fiddly to get
exactly right, and needs a reference trajectory to validate against.
**Status**: implemented as a separate, opt-in mechanism rather than by
relaxing `imm::Estimator` itself — `imm::Estimator` stays the zero-overhead,
same-dimension-only default. New: `core/state_component.hpp` (the canonical
7-component augmented state: 2D position/velocity/accel + turn rate),
`imm/augmented_layout.hpp` (external `AugmentedLayoutFor<Model>` trait
mapping `ConstantVelocity`/`ConstantAcceleration`/`CoordinatedTurn`/`Singer`
into it — deliberately external rather than added to those model classes,
so none of them needed to change), `imm/heterogeneous_mixing.hpp`
(expand/restrict transforms, reusing `imm::mix()`/`imm::combine()` from
`imm/mixing.hpp` directly rather than duplicating the mixing math a second
time), and `imm/heterogeneous_estimator.hpp`
(`imm::HeterogeneousEstimator<Filters...>`). A component a lower-order model
doesn't track (e.g. `ConstantVelocity` has no turn rate) is expanded as
"zero mean, large-but-finite variance" rather than "definitely zero" — one
of several valid approaches in the differing-order IMM literature, chosen
and documented plainly in `imm/heterogeneous_mixing.hpp`. Honest finding
from testing this (ad hoc python3, per `.claude/rules/testing.md`, later
turned into the regression test in `tests/unit/test_heterogeneous_imm.cpp`):
once one model's mode probability comes to dominate, that padding
effectively resets the *other* models' confidence about their own unique
states every mixing cycle, so a lower-probability model's sub-estimate for
a state the others don't share can be noisy rather than smoothly
convergent. `examples/04_heterogeneous_imm` demonstrates the full
CV+CA+CT mix the roadmap item names, with that characteristic called out
directly in its own comment rather than glossed over.

**Update (docs/PRODUCTION_ROADMAP.md P0 item 5)**: a follow-up investigation
into "mode probabilities lean CV-heavy even during a visible turn" found two
things, one confirming existing behavior as correct and one real bug. First,
verified numerically that at this example's own (dt, turn-rate, measurement-
noise) combination, a single tick's curvature-induced deviation between a
straight-line and the true arc (~0.0033) is about 6x *smaller* than the
measurement noise sigma (~0.02) — IMM's mode-probability update is
inherently a per-step likelihood comparison, so it genuinely cannot
statistically distinguish CV from CT quickly in that regime; CV staying
dominant for several steps after a turn starts is correct, not a bug.
Second, extending the demo scenario long enough to actually reach a mode-
probability crossover (not covered by the original 20-step window) surfaced
that `big_unknown_variance()`'s original value (1e4) was large enough to
nearly erase a recovering model's own prior in a single mixing cycle right
at the crossover — mode probabilities snapping close to 1.0/0.0 and the
*combined* position estimate briefly going wrong-signed, which the previous
paragraph's now-corrected "stays accurate throughout" claim missed (true
only within the originally-tested 20-step window, not in general).
Empirically swept, `big_unknown_variance` 100 (down from 1e4) sits
mid-plateau (max combined-position error ~0.009 vs ~0.43 at 1e4 in the
extended scenario) and, as a bonus, makes the *original* 20-step example
markedly more compelling on its own: mode probabilities now shift smoothly
and CT's probability visibly crosses 0.5 partway through the turn, rather
than CV staying pinned near 0.9+ for the whole run. See
`imm/heterogeneous_mixing.hpp`'s file comment for the full sweep data and
`examples/04_heterogeneous_imm`'s comment for the demo-level summary.

## Motion models & filtering

### 1. Current Statistical (CS) model ✅ Done (flagged sketch)
**What**: an adaptive-mean refinement of the Singer model — acceleration mean
adapts online instead of being fixed at zero, tracking targets that commit to
a maneuver rather than just perturbing around straight-line motion.
**Why**: consistently shown to outperform plain Singer for targets undergoing
a sustained maneuver rather than a brief jitter.
**Reference**: cited as an improved Singer variant in the maneuvering-target
tracking literature alongside the base Singer model (see `models/singer.hpp`'s
citations).
**Effort**: Low-medium once Singer itself is on solid footing — it's
Singer plus an online mean estimator.
**Status**: turned out to be considerably harder than "low-medium" in
practice — `models/current_statistical.hpp` (the model) +
`filters/current_statistical_filter.hpp` (a Sage-Husa-style wrapper Filter
that feeds the posterior acceleration back as the next cycle's mean). Two
real instabilities were found via ad hoc python3 verification (per
`.claude/rules/testing.md`) before either got anywhere near final: (1) an
initial design folded the adaptive mean into the Kalman state itself, which
has a provable unit-root eigenvalue regardless of tuning and diverged
without bound; (2) even with the mean kept external (the design actually
shipped), feeding the raw posterior estimate back undamped/unclamped *also*
diverges without bound, despite position/velocity tracking staying
accurate throughout — a latent bug that would only surface if anything
consumed the acceleration state directly. Shipped with damping +
clamping, which bounds the growth but does not eliminate a bias toward the
clamp under a very long sustained maneuver — see both files' comments and
`tests/unit/test_current_statistical.cpp` for the specifics. Tune
`adaptation_rate`/`max_mean_accel` to the actual application; the defaults
are not universal. `examples/05_current_statistical` compares it against
plain Singer.

### 2. Adaptive / self-tuning process noise (Sage-Husa) ✅ Done
**What**: estimate Q/R online from the innovation sequence via covariance
matching instead of requiring hand-tuned noise matrices.
**Why**: every model in this library currently makes you guess a noise
density constant; this removes that guesswork.
**Reference**: Sage, A. P. and Husa, G. W., "Adaptive filtering with unknown
prior statistics," *Joint Automatic Control Conference*, 1969. The known
divergence/non-positive-definiteness failure mode of the vanilla formulation
is well documented — a real implementation should use one of the
credibility-weighted or square-root-decomposition fixes from the modern
adaptive-Kalman literature rather than the 1969 version verbatim.
**Where**: `filters/sage_husa.hpp` (originally stubbed under
`filters/adaptive/`; moved flat so the folder matches the declared
`augur::filters` namespace, per `.claude/rules/code-style.md`'s
namespaces-mirror-folders rule).
**Effort**: Medium — the covariance-matching math is straightforward; making
it not diverge in practice is the actual work.
**Status**: implemented as `filters::SageHusaAdaptive<Inner, MeasDim>`. The
fix used for the known divergence failure mode: eigenvalue-flooring
(`augur::math::project_to_psd()`, new in `math/backend.hpp`, same "regularize
rather than propagate garbage" spirit as the existing `safe_inverse()`)
applied to both Q_hat and R_hat after every update — verified (ad hoc
python3, per `.claude/rules/testing.md`) to stay stable and track a real
mid-run change in the true noise level, both for R and Q independently.
Scope note stated directly in the file: this assumes H observes the state's
first MeasDim components directly (position, no scaling/rotation) — the
convention every example in this library already follows — because
`augur::filters::Filter`'s interface doesn't expose a general H/R to pull
from, so the wrapper reimplements the predict/update recursion itself
(reusing the inner Model's `transition()`/`jacobian()`, not duplicating
those) rather than delegating to `Inner::predict()`/`update()`.
`examples/06_sage_husa_adaptive` demonstrates R adapting through an abrupt
sensor-noise change.

### 3. Unscented Kalman Filter (UKF) backend ✅ Done
**What**: a `Filter`-satisfying UKF, for measurement models that are
genuinely nonlinear in a way linearization handles poorly (bearing-only
detection, screen-space projection of a 3D position).
**Why**: `filters::ExtendedKalmanFilter` already covers "nonlinear but
linearizes fine"; UKF is the next tier when the EKF's first-order Taylor
approximation starts showing.
**Effort**: Medium — sigma-point generation and the unscented transform are
mechanical once written once, but it's a new file, not a stub.
**Status**: `filters::UnscentedKalmanFilter<Model, MeasDim>` implemented,
sigma points regenerated fresh every predict()/update() call rather than
reused across calls (simpler, robust to something else calling set_state()
in between). Strongest available check — for a linear model/measurement
the unscented transform must reduce to plain `KalmanFilter` exactly —
passed to float64 precision immediately, but that same check caught a real
float32 bug: the commonly-cited default `alpha=1e-3` makes the central
sigma-point weight enormous (~1e6 for a 4-state model), and the resulting
catastrophic cancellation in `Scalar=float` produced ~0.1-0.2 errors that
would have shipped unnoticed under a looser test. Default changed to
`alpha=1` (lambda=0, the simpler unscaled Julier-Uhlmann case), verified
stable. A second lesson from testing this: an initial regression test used
a range-only measurement and a loose tolerance that still failed — not a
bug, but confirmation (via the same ad hoc python3 workflow) that
range-only from one stationary sensor against a non-radially-moving target
is only weakly observable and converges very slowly; switched the test and
`examples/10_unscented_kalman` to a range+bearing measurement instead,
which fully determines position and converges to <0.001 position error
within 60 steps.

### 4. Particle filter / Sequential Monte Carlo backend ✅ Done
**What**: a `Filter`-satisfying particle filter for multi-modal, non-Gaussian
uncertainty — e.g. "the target is either behind cover or in the open," which
a Gaussian-based filter structurally can't represent as a single mode.
**Why**: rounds out the filter-backend tier for the genuinely hard cases IMM
across Gaussian filters can't reach.
**Effort**: Medium-high — resampling strategy (systematic vs. multinomial)
and particle-count/performance tuning are real design decisions, not
boilerplate.
**Status**: `filters::ParticleFilter<Model, MeasDim, NumParticles>`
implemented (bootstrap/SIR, systematic resampling triggered when effective
sample size drops below `NumParticles/2`, not every cycle). The first filter
in augur that needs randomness (`<random>`, no new dependency) — every other
filter here is a deterministic function of its inputs. Verified end-to-end
(ad hoc python3 + numpy, per `.claude/rules/testing.md`) against a
`KalmanFilter` baseline on a linear system before being ported: with enough
particles the weighted-mean estimate tracks the KF's mean (the exact
solution there) within a small Monte Carlo tolerance throughout. `state()`/
`covariance()` return the particle set's cached weighted mean/covariance,
recomputed after every `predict()`/`update()` (`filters::Filter`'s interface
requires a reference to something owned, not a computed temporary).
`examples/11_particle_filter` runs the exact same detections as
`examples/01_basic_cv_tracking` for direct comparison — deliberately a
case where a Kalman filter is the better tool, included to make the
particle filter's own correctness easy to check against a known answer,
not to claim it's the right choice for a plain linear-Gaussian problem.

## Multi-target

### 5. Data association (GNN + JPDA) ✅ Done
**What**: decide which detection belongs to which existing track before
calling `update()` on anything, via a cheap Global-Nearest-Neighbor baseline
and, when tracks are close together, Joint Probabilistic Data Association.
**Reference**: Fortmann, T., Bar-Shalom, Y., Scheffe, M., "Sonar Tracking of
Multiple Targets Using Joint Probabilistic Data Association," *IEEE J.
Oceanic Engineering*, 8(3), 1983, pp. 173-184.
**Where**: stubbed in `track/association.hpp`.
**Effort**: Low (GNN) to medium-high (JPDA).
**Status**: both implemented in `track/association.hpp`, using bounded
`utils::FixedVector`/no-heap APIs throughout (matching `track/track_manager.hpp`'s
own stated convention) rather than the `std::vector`-based shape this file
originally sketched. `nearest_neighbor()` is a proper *global* greedy match
(resolves conflicts where two tracks want the same detection), not
independent per-track nearest. `joint_probabilistic_data_association()` is
full JPDA via explicit joint-event enumeration (bitmask over detections,
hence a 32-detection cap) — not the simpler independent-per-track PDA
approximation, which was checked (ad hoc python3, per
`.claude/rules/testing.md`) to give visibly different, less correct
marginals on an ambiguous close-tracks case specifically because it ignores
cross-track exclusivity. `examples/07_data_association` shows both on a
well-separated frame and an ambiguous one.

### 6. Track lifecycle management ✅ Done
**What**: the practical glue every multi-target user needs — spawn a
tentative track, confirm it after N consecutive hits, coast it through a few
missed detections rather than killing it instantly, eventually drop it.
**Why**: without this, multi-target tracking degenerates into "reallocate a
filter every frame," which is both wasteful and jittery.
**Where**: stubbed in `track/track_manager.hpp`, built on `utils::FixedVector`
so it never touches the heap per frame for a bounded max-track-count game.
**Effort**: Medium — mostly state-machine bookkeeping, not novel math.
**Status**: `track::TrackManager<FilterT, MaxTracks>` implemented on top of
`track/association.hpp`'s `nearest_neighbor()`. One real design gap found
while implementing: `FixedVector`'s fixed `std::array` storage
default-constructs every slot up front, but `KalmanFilter` (and any real
`Filter`) has no default constructor by design — fixed by holding
`std::optional<FilterT>` in `Track` rather than `FilterT` directly (always
populated once a track exists; the empty state only ever appears
transiently inside `FixedVector`'s own unused capacity). A tentative track
gets exactly one miss before being dropped (no coast grace until
confirmed) — a deliberate choice to avoid accumulating tentative tracks
from clutter. `examples/08_track_lifecycle` runs a full multi-target
scenario: spawn, confirm, coast through a miss run, recover, second target
appearing mid-run.

### 7. GM-PHD filter for unknown/variable target count ✅ Done
**What**: propagate a Gaussian-mixture "intensity" over state space instead of
explicit per-target tracks, so the **number** of targets doesn't need to be
known or fixed in advance.
**Reference**: Vo, B.-N. and Ma, W.-K., "The Gaussian Mixture Probability
Hypothesis Density Filter," *IEEE Trans. Signal Processing*, 54(11), 2006,
pp. 4091-4104. Theoretical foundation: Mahler, R., "Multitarget Bayes
Filtering via First-Order Multitarget Moments," *IEEE Trans. Aerospace and
Electronic Systems*, 39(4), 2003.
**Where**: stubbed in `track/gm_phd.hpp`, explicitly marked the highest-effort
item on this whole list.
**Effort**: High — closed-form GM-PHD needs linear-Gaussian birth/survival/
detection models plus a pruning/merging step to keep the component count
bounded, or it grows unboundedly frame over frame. Start with #5/#6 first;
only reach for this if the target count is genuinely unknown and changing.
**Status**: `track::GmPhdFilter<Model, MeasDim, MaxComponents>` implements
predict (survival + caller-supplied birth components) / update (missed-
detection terms plus one Kalman-updated component per predicted-component ×
detection pair) / `prune_and_merge()` (weight threshold, then Mahalanobis-
distance moment-matching merge — the same formula as `imm::mixing.hpp`'s
`combine()`) / `extract_targets()`. The update math was verified against an
independent numpy computation before being ported — and that verification
caught a real bug **in the Python reference itself**, not the C++: an
initial quick check used a simplified 2D toy state and omitted the
`detection_probability` factor from the per-detection weight formula,
giving a wrong total-weight reference (2.0171) that the first test-writing
pass asserted; re-deriving carefully against the actual 4D state matching
the real test caught the mismatch and confirmed the C++ was right all
along (correct value 2.00155) — a good example of why "verify test suite
second" (`.claude/rules/testing.md`) matters even when the *first*
verification step was ad hoc Python. `MaxComponents` must be sized for the
worst single `update()` call (`num_predicted * (1 + num_detections)`), not
long-run steady state — stated directly in the file. **Update**
(docs/IMPROVEMENT_PLAN.md): `extract_targets()` originally returned one
target per component above threshold regardless of how far above 1 its
weight was; now emits `round(weight)` duplicate-mean targets per
component, the field's own standard convention (Vo & Ma 2006) for this
exact situation, verified against the worked example three close
weight-0.9 births merging to one weight-2.7 component correctly
reports 3 targets, not 1. `examples/09_gm_phd` tracks two targets
(one appearing partway through) with no explicit data association at all.

## Real-time systems integration

### 8. Out-of-sequence measurement (OOSM) handling ✅ Done
**What**: correctly incorporate a detection that arrives after a later one
already got processed — routine when detections cross an unreliable network
link rather than arriving in a single local sensor stream.
**Why**: networked games are exactly the scenario where this happens
constantly; naively discarding late measurements throws away information,
and naively reprocessing in arrival order corrupts the filter's timeline.
**Effort**: Medium — the "retrodiction" approach (reprocess from the
out-of-sequence point forward) is well understood but requires keeping
enough history to replay from.
**Status**: `track::OutOfSequenceEstimator<Inner, MaxHistory>` implemented —
keeps a bounded, sliding-window history of (timestamp, post-update state/
covariance, measurement) snapshots; a late measurement rolls the filter
back to the snapshot just before its true timestamp and replays every step
since, with the late one inserted at its correct chronological position.
Every replayed step's dt is recomputed from consecutive absolute
timestamps rather than stored separately, which is what avoids needing
special-cased "adjust the first replayed step" logic. Honest finding from
testing this (ad hoc python3, per `.claude/rules/testing.md`; an earlier
version of the check had an elapsed-time inconsistency between scenarios
that overstated the effect, corrected before this shipped): for a gentle,
low-process-noise trajectory, discarding a late measurement outright is
actually fairly benign, while misapplying it as if it were fresh at the
current time is clearly worse (~90x the state error, in the tested
scenario) — retrodiction gets the right answer either way, without having
to reason about which failure mode a given system would hit.
`examples/12_out_of_sequence` demonstrates the full lost-then-recovered
sequence.

### 9. Latency compensation / predict-to-render-time ✅ Done
**What**: the bookkeeping layer on top of `Estimator::predict_ahead()` that
real netcode needs — a per-track snapshot ring buffer for server-side "rewind
to what the shooter saw," and a helper that turns "local render clock + last
update timestamp + latency estimate" into the right `predict_ahead()` call.
**References**: Gambetta, G., "Fast-Paced Multiplayer" series — *Client-Side
Prediction and Server Reconciliation*, *Entity Interpolation*, *Lag
Compensation* (gabrielgambetta.com); Bernier, Y., "Latency Compensating
Methods in Client/Server In-game Protocol Design and Optimization," GDC 2001
(the original Source-engine lag-compensation write-up). The tracking
literature analyzes the same problem from the other direction — steady-state
accuracy of an extrapolated track under a fixed processing/communication
delay — in the papers cited alongside `models/singer.hpp`.
**Where**: stubbed in `predict/latency_compensation.hpp`.
**Effort**: Medium — the prediction math already exists; this is mostly
snapshot-history bookkeeping and picking the right horizon.
**Status**: `predict::SnapshotBuffer<Scalar, State, MaxHistory>` (record +
linear-interpolated `rewind_to()`, clamped rather than extrapolated outside
the retained window) and `predict::predict_to_render_time()` implemented.
One deliberate deviation from this file's own original stub, stated in its
comment: timestamps are plain `Scalar` seconds (caller-defined epoch), not
`std::chrono::steady_clock::time_point` — consistent with every other
timing quantity in this library (`predict(dt)`,
`track/out_of_sequence.hpp`'s history) and the "no forced anything"
posture in `docs/ARCHITECTURE.md`; convert from whatever clock you use at
the call site. Mostly bookkeeping rather than novel estimation math, so
verified directly in C++ (`tests/unit/test_latency_compensation.cpp`)
rather than via a separate ad hoc python3 pass, per
`.claude/rules/testing.md`'s own guidance on when that step isn't needed.
`examples/13_latency_compensation` demonstrates both the server-side
rewind and the client-side render-time prediction.

### 10. Sensor / detection fusion ✅ Done
**What**: combine several noisy per-frame "detections" of the same target
(e.g. multiple raycast/vision-cone samples) into one fused measurement before
it ever reaches a filter's `update()`.
**Why**: a filter's `update()` assumes one measurement per step; real
detection pipelines often produce several candidate observations that should
be reconciled first.
**Effort**: Low-medium — a straightforward inverse-covariance-weighted
combination for the Gaussian case.
**Status**: `track::fuse_measurements()` implemented — information-form
(precision matrices add) combination, cross-checked (ad hoc python3 +
numpy, per `.claude/rules/testing.md`) against applying the same
detections as sequential Kalman updates, which must give an identical
result since it's the same underlying math in closed form. Assumes
conditionally-independent detections given the true state (no shared
sensor bias) — the standard assumption this closed form relies on, stated
in the file. `examples/14_sensor_fusion` fuses three simulated
vision-cone samples with different confidence levels into one measurement
before it reaches a `KalmanFilter`.

## Tooling & confidence

### 11. Debug / visualization export ✅ Done
**Status (2D, pre-existing)**: **already implemented**, not just an idea — `predict/query.hpp`'s
`error_ellipse_2d()` extracts the 1-sigma uncertainty ellipse from any
filter's covariance, ready for a debug-draw gizmo. Extending it to 3D
(an ellipsoid via the same eigen-decomposition approach, one dimension up)
is the natural next step.
**Effort**: Low for the 3D extension.
**Status (3D extension)**: `error_ellipsoid_3d()`/`ErrorEllipsoid3D` added to
`predict/query.hpp`, same eigen-decomposition approach one dimension up.
One correctness detail beyond a direct port: `orientation`'s columns are
forced to a proper (determinant +1) rotation rather than left as whatever
sign `Eigen::SelfAdjointEigenSolver` happens to produce (verified via ad hoc
python3 + numpy, per `.claude/rules/testing.md`, that flipping one axis's
sign preserves the ellipsoid while fixing the determinant) — most
debug-draw APIs and quaternion conversions expect a proper rotation, not an
occasional reflection. `error_ellipse_2d()` itself had zero test coverage
before this either, now covered alongside the 3D case in
`tests/unit/test_predict_query.cpp`. `examples/01_basic_cv_tracking` now
prints the 2D ellipse each step, since no example previously showed this
"already implemented" capability at all.

### 12. Reflection-driven (de)serialization ✅ Done
**What**: use `reflect::Descriptor<T>` to auto-serialize any `State<Dim,
Scalar>` (or a whole track) for save-states, replay systems, or network
snapshots, without hand-writing a serializer per state type.
**Why**: this is exactly what the reflection layer in `reflect/` exists to
enable — right now it's plumbing with no consumer.
**Effort**: Medium — needs a concrete wire format decision (the reflection
layer itself is already in place).
**Status**: Wire format is a fixed-width positional binary layout
(`reflect/serialize.hpp`'s `ByteWriter`/`ByteReader` + `serialize()`/
`deserialize()`), chosen deliberately over anything self-describing since
this is an internal engine format for a single build, not a cross-version
interchange format.

A real, non-obvious finding surfaced immediately on the item's own literal
wording ("auto-serialize any `State<Dim,Scalar>`"): **Boost.PFR cannot
reflect `augur::math::Vector<Scalar,N>` at all.** Verified ad hoc, per
`.claude/rules/testing.md`, by compiling a throwaway
`boost::pfr::tuple_size_v<Eigen::Matrix<float,4,1>>` instantiation before
writing any real code — it fails outright with "Boost.PFR: Type must be
aggregate initializable," since Eigen's matrix type has private storage and
user-provided constructors and is therefore not an aggregate. Since `State`
*is* exactly this Eigen alias, the existing `PfrBackend` alone could never
cover the reflection layer's own primary stated use case.

Fixed by adding a second backend rather than forcing Eigen through PFR:
`reflect/backends/vector_backend.hpp`'s `VectorBackend<T>` reflects a
fixed-size Eigen column vector structurally (N contiguous Scalars via
`operator()`), detected via an `EigenFixedVector<T>` concept over Eigen's
own `RowsAtCompileTime`/`ColsAtCompileTime` members (the `Vector<Scalar,N>`
alias isn't a distinct class template, so `utils::is_specialization_of`
can't detect it the usual way). `reflect::Descriptor<T>` now dispatches
between `VectorBackend` and `PfrBackend` via `std::conditional_t`, and also
forwards `field_name()` for the first time (previously defined on
`PfrBackend` but never exposed by `Descriptor` itself — dead capability).
`serialize()`/`deserialize()` recurse through `Descriptor<T>` down to
arithmetic leaves, so a PFR-aggregate that nests a Vector field (exactly
`SnapshotBuffer::Snapshot`'s `{timestamp, state}` shape) round-trips
correctly through both backends composed together — covered explicitly by
`tests/unit/test_reflection_serialization.cpp`'s
"round-trips an aggregate nesting a Vector field" case.

A second, unrelated non-obvious finding while wiring the example/test:
`boost::pfr::get_name` (what `field_name()` calls) requires its `T` to have
**external linkage** — it forms an `extern const T fake_object` internally,
which cannot be defined for a type declared in an anonymous namespace.
`field_count()`/`for_each_field()` don't need this; only `field_name()`
does. Both the example and test aggregates were moved out of an anonymous
namespace once this was hit at compile time, and the constraint is
documented at each call site so it doesn't surprise the next reader.

`predict/latency_compensation.hpp`'s `SnapshotBuffer` gained real
`serialize()`/`deserialize()` methods built on this machinery (count-prefix
+ per-`Snapshot` recursion), making descriptor.hpp's long-standing "used by
SnapshotBuffer's snapshot/replay support" header comment literally true for
the first time rather than aspirational. `kMaxSerializedBytes` is a
compile-time upper bound (via `sizeof`, which is `>=` the true packed byte
count once Eigen's alignment padding is accounted for) sized off the
existing `MaxHistory` template parameter, so the caller can size a
`std::array`/stack buffer once rather than this layer owning a heap-backed
byte buffer. Example: `examples/15_reflection_serialization/main.cpp`
serializes a live `SnapshotBuffer` populated from a real Kalman filter to
bytes and deserializes into a fresh buffer, confirming
`rewind_to()`/interpolation reproduce bit-identical results post-round-trip
(observed diff `0.00e+00`).

### 13. Benchmark / validation harness ✅ Done
**What**: regression-test filter/model accuracy against synthetic
maneuvering-target trajectories (and optionally standard multi-object-
tracking benchmark conventions), so future changes can't silently regress
tracking quality.
**Why**: `tests/unit/test_kalman_cv.cpp` covers the constant-velocity case
only — `CoordinatedTurn`, `Singer`, and the IMM mixing math have zero
automated coverage today, which is the biggest actual risk in this codebase
as it stands.
**Effort**: Medium — mostly synthetic-trajectory generation plus deciding on
acceptable error bounds; this should honestly be prioritized ahead of several
of the feature ideas above.
**Status**: `tests/unit/test_helpers.hpp` adds a generic finite-difference
jacobian checker (compares any `MotionModel`'s analytic `jacobian()` against
a numerical derivative of its own `transition()` — this class of check is
exactly what would have caught, and now would catch, the coordinated-turn
bug above). `tests/unit/test_coordinated_turn.cpp` and
`tests/unit/test_singer.cpp` cover both models' transition/jacobian/
process_noise, including synthetic sustained-turn convergence for CT.
`tests/unit/test_imm_mixing.cpp` hand-verifies `mix()`/
`update_mode_probability()`/`combine()` against an independently-computed
2-model reference plus a 3-model CoordinatedTurn IMM regression test.
`examples/02_imm_maneuvering_target` was retuned so its own demonstration
actually shows what its comment claims. Not yet covered: a formal
multi-object-tracking benchmark convention (e.g. OSPA) — left for a future
pass if/when `track/` grows real multi-target implementations (items 5-7).
