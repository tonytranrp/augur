// tests/unit/test_out_of_sequence.cpp
//
// Coverage for docs/ROADMAP.md item 8 ("OOSM handling",
// track/out_of_sequence.hpp). The retrodiction scenario reproduces an
// independent numpy computation (ad hoc python3, per
// .claude/rules/testing.md) exactly: three measurements where the
// middle one arrives last, as if lost and retransmitted.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "augur/augur.hpp"
#include "augur/imm/estimator.hpp"
#include "augur/track/out_of_sequence.hpp"

using Catch::Matchers::WithinAbs;

namespace {
using CV = augur::models::ConstantVelocity<float, 2>;
using KF = augur::filters::KalmanFilter<CV, /*MeasDim=*/2>;
using OOSM = augur::track::OutOfSequenceEstimator<KF, /*MaxHistory=*/8>;

static_assert(augur::filters::Filter<OOSM>,
              "docs/IMPROVEMENT_PLAN.md: previously blocked composing OOSM with "
              "TrackManager/imm::Estimator despite both being named use cases for it");

KF make_kf() {
    KF::StateVector x0 = KF::StateVector::Zero();
    x0(2) = 1.0f;
    x0(3) = 0.5f;
    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    return KF{CV{0.01f}, x0, KF::StateCovariance::Identity(), H, KF::MeasurementCovariance::Identity() * 0.05f};
}
} // namespace

TEST_CASE("OutOfSequenceEstimator retrodiction matches an independent numpy reference", "[track][oosm]") {
    const float dt = 1.0f / 30.0f;
    const KF::Measurement z1{0.033f, 0.017f};
    const KF::Measurement z2{0.067f, 0.033f};
    const KF::Measurement z3{0.100f, 0.050f};

    OOSM est{make_kf()};
    est.step(dt, z1);        // t=dt
    est.step(2 * dt, z3);    // t=3dt -- z2 (which truly belongs at t=2dt) hasn't arrived yet (lost in transit)

    const bool accepted = est.insert_out_of_sequence(2 * dt, z2); // z2's TRUE timestamp, strictly between z1 and z3
    REQUIRE(accepted);

    // Reference from an independent numpy computation of the same
    // scenario (reprocess z1, z2, z3 in true chronological order).
    REQUIRE_THAT(est.state()(0), WithinAbs(0.10002291f, 1e-3f));
    REQUIRE_THAT(est.state()(1), WithinAbs(0.04997709f, 1e-3f));
    REQUIRE_THAT(est.state()(2), WithinAbs(1.00017733f, 1e-3f));
    REQUIRE_THAT(est.state()(3), WithinAbs(0.49982267f, 1e-3f));
}

TEST_CASE("OutOfSequenceEstimator differs from discarding or misapplying the late measurement",
          "[track][oosm]") {
    // Honest finding (ad hoc python3, per .claude/rules/testing.md, run
    // AFTER an initial version of this test's naive comparisons used an
    // inconsistent elapsed-time span between scenarios and overstated
    // the gap): for this gentle, low-process-noise, constant-velocity
    // trajectory, DISCARDING z2 is actually fairly benign -- the
    // model's own prediction already interpolates the gap well -- while
    // MISAPPLYING it (treating stale position data as a fresh reading
    // at the wrong time) is clearly worse, about 90x the state error of
    // discarding it outright in this scenario. That's a real,
    // interesting asymmetry, not a reason to expect both wrong
    // approaches to be equally bad.
    const float dt = 1.0f / 30.0f;
    const KF::Measurement z1{0.033f, 0.017f};
    const KF::Measurement z2{0.067f, 0.033f};
    const KF::Measurement z3{0.100f, 0.050f};

    // A) discard z2 entirely.
    KF discard = make_kf();
    discard.predict(dt);
    discard.update(z1);
    discard.predict(2 * dt);
    discard.update(z3);

    // B) misapply z2 as if it were fresh, at the current time.
    KF misapply = make_kf();
    misapply.predict(dt);
    misapply.update(z1);
    misapply.predict(2 * dt);
    misapply.update(z3);
    misapply.predict(0.0f);
    misapply.update(z2);

    // C) proper retrodiction.
    OOSM correct{make_kf()};
    correct.step(dt, z1);
    correct.step(2 * dt, z3);
    correct.insert_out_of_sequence(2 * dt, z2);

    const float diff_from_discard = (discard.state() - correct.state()).norm();
    const float diff_from_misapply = (misapply.state() - correct.state()).norm();
    REQUIRE(diff_from_discard > 0.0001f);   // small but nonzero: retrodiction does incorporate real information
    REQUIRE(diff_from_misapply > 0.01f);    // clearly worse: stale data misapplied as fresh corrupts the estimate
    REQUIRE(diff_from_misapply > 10.0f * diff_from_discard); // the actual point: misapplying is much worse than discarding
}

TEST_CASE("OutOfSequenceEstimator rejects a measurement older than the retained history window", "[track][oosm]") {
    using SmallOOSM = augur::track::OutOfSequenceEstimator<KF, /*MaxHistory=*/2>;
    SmallOOSM est{make_kf()};
    const float dt = 1.0f / 30.0f;
    est.step(dt, KF::Measurement{0.03f, 0.0f});
    est.step(dt, KF::Measurement{0.06f, 0.0f});
    est.step(dt, KF::Measurement{0.09f, 0.0f}); // history window (capacity 2) no longer covers t=dt

    REQUIRE_FALSE(est.insert_out_of_sequence(dt * 0.5f, KF::Measurement{0.0f, 0.0f}));
}

TEST_CASE("OutOfSequenceEstimator rejects a measurement at or after the current time", "[track][oosm]") {
    OOSM est{make_kf()};
    const float dt = 1.0f / 30.0f;
    est.step(dt, KF::Measurement{0.03f, 0.0f});
    REQUIRE_FALSE(est.insert_out_of_sequence(dt, KF::Measurement{0.0f, 0.0f}));
    REQUIRE_FALSE(est.insert_out_of_sequence(dt * 2.0f, KF::Measurement{0.0f, 0.0f}));
}

TEST_CASE("OutOfSequenceEstimator's split predict()/update() matches step() exactly", "[track][oosm][regression]") {
    // The whole point of the Filter-conformance fix: predict() and
    // update() must be independently callable (imm::Estimator/
    // TrackManager both call them separately, never step()) and produce
    // the IDENTICAL net effect as the combined step() convenience
    // wrapper that already existed.
    const float dt = 1.0f / 30.0f;
    OOSM via_step{make_kf()};
    OOSM via_split{make_kf()};

    for (int i = 0; i < 5; ++i) {
        const KF::Measurement z{0.01f * static_cast<float>(i), 0.02f * static_cast<float>(i)};
        via_step.step(dt, z);
        via_split.predict(dt);
        via_split.update(z);
    }

    REQUIRE_THAT((via_step.state() - via_split.state()).norm(), WithinAbs(0.0f, 1e-6f));
    REQUIRE_THAT((via_step.covariance() - via_split.covariance()).norm(), WithinAbs(0.0f, 1e-6f));
    REQUIRE_THAT(via_step.current_time(), WithinAbs(via_split.current_time(), 1e-6f));
    REQUIRE_THAT(via_step.last_likelihood(), WithinAbs(via_split.last_likelihood(), 1e-6f));
}

TEST_CASE("OutOfSequenceEstimator composes inside imm::Estimator", "[track][oosm][regression]") {
    // The actual named use case from docs/IMPROVEMENT_PLAN.md's finding
    // -- previously impossible to even compile.
    using ImmOOSM = augur::imm::Estimator<OOSM, OOSM>;
    ImmOOSM tracker{
        OOSM{make_kf()}, OOSM{make_kf()},
        augur::imm::ModeMatrix<2, float>::uniform(0.9f),
    };
    tracker.predict(1.0f / 30.0f);
    tracker.update(KF::Measurement{0.03f, 0.0f});

    const auto [x, P] = tracker.combined_state();
    REQUIRE(x.allFinite());
    REQUIRE(P.allFinite());
}
