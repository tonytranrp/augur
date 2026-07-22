// benchmarks/bench_kalman.cpp
//
// docs/PRODUCTION_ROADMAP.md Phase 3: KalmanFilter::predict()/update()
// throughput, the base case every other filter/estimator in this library
// is built on top of. predict()/update() are benchmarked separately
// (rather than one combined predict+update loop) since they have
// different cost profiles -- predict() is one F*P*F^T+Q; update() adds a
// safe_inverse() and the Kalman gain -- and a regression in one shouldn't
// hide inside an aggregate number dominated by the other.

#include <benchmark/benchmark.h>
#include "augur/augur.hpp"

namespace {
using CV = augur::models::ConstantVelocity<float, 2>;
using KF = augur::filters::KalmanFilter<CV, /*MeasDim=*/2>;

KF make_filter() {
    KF::StateVector x0 = KF::StateVector::Zero();
    x0(2) = 1.5f;
    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    return KF{CV{0.1f}, x0, KF::StateCovariance::Identity(), H, KF::MeasurementCovariance::Identity() * 0.1f};
}
} // namespace

static void BM_KalmanFilter_Predict(benchmark::State& state) {
    KF filter = make_filter();
    const float dt = 1.0f / 60.0f;
    for (auto _ : state) {
        filter.predict(dt);
        auto x = filter.state(); // copy: DoNotOptimize's const-ref overload is deprecated
        benchmark::DoNotOptimize(x);
    }
}
BENCHMARK(BM_KalmanFilter_Predict);

static void BM_KalmanFilter_Update(benchmark::State& state) {
    KF filter = make_filter();
    const KF::Measurement z{1.0f, 0.5f};
    for (auto _ : state) {
        filter.update(z);
        auto x = filter.state(); // copy: DoNotOptimize's const-ref overload is deprecated
        benchmark::DoNotOptimize(x);
    }
}
BENCHMARK(BM_KalmanFilter_Update);

static void BM_KalmanFilter_PredictUpdate(benchmark::State& state) {
    KF filter = make_filter();
    const float dt = 1.0f / 60.0f;
    const KF::Measurement z{1.0f, 0.5f};
    for (auto _ : state) {
        filter.predict(dt);
        filter.update(z);
        auto x = filter.state(); // copy: DoNotOptimize's const-ref overload is deprecated
        benchmark::DoNotOptimize(x);
    }
}
BENCHMARK(BM_KalmanFilter_PredictUpdate);
