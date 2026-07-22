// benchmarks/bench_imm.cpp
//
// docs/PRODUCTION_ROADMAP.md Phase 3: imm::Estimator::predict()/update()
// throughput -- the library's own "star" (per CLAUDE.md), and the whole
// reason a consumer pays the cost of mixing multiple filters instead of
// running one KalmanFilter directly (see benchmarks/bench_kalman.cpp for
// the base case this adds on top of). Uses the same 3-model CV+CA+CT mix
// examples/02_imm_maneuvering_target and the README's own teaser use, at
// the same dimension (CV/CA differ in native dimension from CT, so this
// exercises imm::Estimator's actual same-dimension-only path with
// realistic, differently-tuned instances of the SAME model rather than
// three copies of one model -- see docs/ARCHITECTURE.md's "Known
// limitation: same-order IMM mixing only").

#include <benchmark/benchmark.h>
#include "augur/augur.hpp"

namespace {
using CT = augur::models::CoordinatedTurn<float>;
using KF = augur::filters::KalmanFilter<CT, /*MeasDim=*/2>;
using Tracker = augur::imm::Estimator<KF, KF, KF>;

KF make_filter(float q_turn) {
    KF::StateVector x0 = KF::StateVector::Zero();
    x0(2) = 1.5f;
    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    return KF{CT{0.1f, q_turn}, x0, KF::StateCovariance::Identity(), H, KF::MeasurementCovariance::Identity() * 0.1f};
}

Tracker make_tracker() {
    return Tracker{
        make_filter(0.02f), make_filter(0.5f), make_filter(3.0f),
        augur::imm::ModeMatrix<3, float>::uniform(0.95f),
    };
}
} // namespace

static void BM_ImmEstimator_Predict(benchmark::State& state) {
    Tracker tracker = make_tracker();
    const float dt = 1.0f / 60.0f;
    for (auto _ : state) {
        tracker.predict(dt);
        auto mode_p = tracker.mode_probability(); // copy: DoNotOptimize's const-ref overload is deprecated
        benchmark::DoNotOptimize(mode_p);
    }
}
BENCHMARK(BM_ImmEstimator_Predict);

static void BM_ImmEstimator_Update(benchmark::State& state) {
    Tracker tracker = make_tracker();
    const Tracker::StateVector z{1.0f, 0.5f, 0.0f, 0.0f, 0.0f};
    for (auto _ : state) {
        tracker.update(KF::Measurement{z(0), z(1)});
        auto mode_p = tracker.mode_probability();
        benchmark::DoNotOptimize(mode_p);
    }
}
BENCHMARK(BM_ImmEstimator_Update);

static void BM_ImmEstimator_PredictUpdate(benchmark::State& state) {
    Tracker tracker = make_tracker();
    const float dt = 1.0f / 60.0f;
    const KF::Measurement z{1.0f, 0.5f};
    for (auto _ : state) {
        tracker.predict(dt);
        tracker.update(z);
        auto mode_p = tracker.mode_probability();
        benchmark::DoNotOptimize(mode_p);
    }
}
BENCHMARK(BM_ImmEstimator_PredictUpdate);

static void BM_ImmEstimator_CombinedState(benchmark::State& state) {
    Tracker tracker = make_tracker();
    tracker.predict(1.0f / 60.0f);
    tracker.update(KF::Measurement{1.0f, 0.5f});
    for (auto _ : state) {
        benchmark::DoNotOptimize(tracker.combined_state());
    }
}
BENCHMARK(BM_ImmEstimator_CombinedState);
