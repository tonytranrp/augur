// benchmarks/bench_heterogeneous.cpp
//
// docs/PRODUCTION_ROADMAP.md Phase 3: HeterogeneousEstimator's
// expand/restrict transforms specifically -- the actual cost
// docs/IMPROVEMENT_PLAN.md's imm/ investigation measured at ~2x the
// three mixed filters' OWN predict() cost combined (root cause there:
// always mixing over the full fixed kAugmentedDim=7 even when the mixed
// models only populate a subset). Benchmarking expand_state/
// expand_covariance/restrict_state/restrict_covariance individually,
// not just the whole HeterogeneousEstimator::predict()/update(), is
// deliberate: it's the only way to see how much of that overhead is the
// transform layer itself versus the filters' own math, which a combined
// number would hide.
//
// BM_ThreeFilters_PredictOnly_CVCACT below is deliberately NOT an
// imm::Estimator comparison: imm::Estimator can only mix same-dimension
// filters (CV=4, CA=6, CT=5 native dimensions all differ), so there's no
// way to construct a same-model imm::Estimator<CV,CA,CT> to compare
// against at all -- any imm::Estimator-vs-HeterogeneousEstimator
// benchmark would necessarily compare different model choices on each
// side, confounding "mixing overhead" with "which models were mixed."
// The clean baseline instead is calling predict() on the same three raw
// filters directly, no estimator wrapping at all -- matching
// docs/IMPROVEMENT_PLAN.md's own methodology exactly, so this benchmark
// suite can track that specific, already-identified overhead over time.

#include <benchmark/benchmark.h>
#include "augur/augur.hpp"
#include "augur/imm/heterogeneous_estimator.hpp"

namespace {
using CV = augur::models::ConstantVelocity<float, 2>;
using CA = augur::models::ConstantAcceleration<float, 2>;
using CT = augur::models::CoordinatedTurn<float>;
using KFCV = augur::filters::KalmanFilter<CV, /*MeasDim=*/2>;
using KFCA = augur::filters::KalmanFilter<CA, /*MeasDim=*/2>;
using KFCT = augur::filters::KalmanFilter<CT, /*MeasDim=*/2>;

template <typename Model>
typename Model::State make_state() {
    typename Model::State x = Model::State::Zero();
    x(2) = 1.5f;
    return x;
}

template <typename Model>
typename Model::Transition make_covariance() {
    return Model::Transition::Identity();
}
} // namespace

static void BM_ExpandState_CV(benchmark::State& state) {
    const auto x = make_state<CV>();
    for (auto _ : state) {
        benchmark::DoNotOptimize(augur::imm::expand_state<CV>(x));
    }
}
BENCHMARK(BM_ExpandState_CV);

static void BM_ExpandCovariance_CV(benchmark::State& state) {
    const auto P = make_covariance<CV>();
    for (auto _ : state) {
        benchmark::DoNotOptimize(augur::imm::expand_covariance<CV>(P));
    }
}
BENCHMARK(BM_ExpandCovariance_CV);

static void BM_RestrictState_CT(benchmark::State& state) {
    const auto x_aug = augur::imm::expand_state<CT>(make_state<CT>());
    for (auto _ : state) {
        benchmark::DoNotOptimize(augur::imm::restrict_state<CT>(x_aug));
    }
}
BENCHMARK(BM_RestrictState_CT);

static void BM_RestrictCovariance_CT(benchmark::State& state) {
    const auto P_aug = augur::imm::expand_covariance<CT>(make_covariance<CT>());
    for (auto _ : state) {
        benchmark::DoNotOptimize(augur::imm::restrict_covariance<CT>(P_aug));
    }
}
BENCHMARK(BM_RestrictCovariance_CT);

// Full context: the same CV+CA+CT "textbook full IMM" mix
// examples/04_heterogeneous_imm demonstrates.
namespace {
using HetTracker = augur::imm::HeterogeneousEstimator<KFCV, KFCA, KFCT>;

HetTracker make_heterogeneous_tracker() {
    KFCV::MeasurementMatrix H_cv = KFCV::MeasurementMatrix::Zero();
    H_cv(0, 0) = 1;
    H_cv(1, 1) = 1;
    KFCV::MeasurementCovariance R = KFCV::MeasurementCovariance::Identity() * 0.1f;
    KFCV kf_cv{CV{0.1f}, make_state<CV>(), make_covariance<CV>(), H_cv, R};

    KFCA::MeasurementMatrix H_ca = KFCA::MeasurementMatrix::Zero();
    H_ca(0, 0) = 1;
    H_ca(1, 1) = 1;
    KFCA kf_ca{CA{0.1f}, make_state<CA>(), make_covariance<CA>(), H_ca, R};

    KFCT::MeasurementMatrix H_ct = KFCT::MeasurementMatrix::Zero();
    H_ct(0, 0) = 1;
    H_ct(1, 1) = 1;
    KFCT kf_ct{CT{0.1f, 2.0f}, make_state<CT>(), make_covariance<CT>(), H_ct, R};

    return HetTracker{
        std::move(kf_cv), std::move(kf_ca), std::move(kf_ct),
        augur::imm::ModeMatrix<3, float>::uniform(0.9f),
    };
}
} // namespace

static void BM_HeterogeneousEstimator_Predict(benchmark::State& state) {
    HetTracker tracker = make_heterogeneous_tracker();
    const float dt = 1.0f / 60.0f;
    for (auto _ : state) {
        tracker.predict(dt);
        auto mode_p = tracker.mode_probability(); // copy: DoNotOptimize's const-ref overload is deprecated
        benchmark::DoNotOptimize(mode_p);
    }
}
BENCHMARK(BM_HeterogeneousEstimator_Predict);

// The clean baseline for the above: the same three filters' own
// predict(), called directly with no estimator/mixing layer at all. The
// difference between this and BM_HeterogeneousEstimator_Predict is the
// actual, isolated cost of HeterogeneousEstimator's mixing/expand/
// restrict machinery -- see the file comment at the top for why this,
// not an imm::Estimator comparison, is the correct baseline.
static void BM_ThreeFilters_PredictOnly_CVCACT(benchmark::State& state) {
    KFCV::MeasurementMatrix H_cv = KFCV::MeasurementMatrix::Zero();
    H_cv(0, 0) = 1;
    H_cv(1, 1) = 1;
    KFCV::MeasurementCovariance R = KFCV::MeasurementCovariance::Identity() * 0.1f;
    KFCV kf_cv{CV{0.1f}, make_state<CV>(), make_covariance<CV>(), H_cv, R};

    KFCA::MeasurementMatrix H_ca = KFCA::MeasurementMatrix::Zero();
    H_ca(0, 0) = 1;
    H_ca(1, 1) = 1;
    KFCA kf_ca{CA{0.1f}, make_state<CA>(), make_covariance<CA>(), H_ca, R};

    KFCT::MeasurementMatrix H_ct = KFCT::MeasurementMatrix::Zero();
    H_ct(0, 0) = 1;
    H_ct(1, 1) = 1;
    KFCT kf_ct{CT{0.1f, 2.0f}, make_state<CT>(), make_covariance<CT>(), H_ct, R};

    const float dt = 1.0f / 60.0f;
    for (auto _ : state) {
        kf_cv.predict(dt);
        kf_ca.predict(dt);
        kf_ct.predict(dt);
        // Copies: DoNotOptimize's const-ref overload is deprecated.
        auto x_cv = kf_cv.state();
        auto x_ca = kf_ca.state();
        auto x_ct = kf_ct.state();
        benchmark::DoNotOptimize(x_cv);
        benchmark::DoNotOptimize(x_ca);
        benchmark::DoNotOptimize(x_ct);
    }
}
BENCHMARK(BM_ThreeFilters_PredictOnly_CVCACT);
