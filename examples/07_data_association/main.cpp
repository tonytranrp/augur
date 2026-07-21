// examples/07_data_association/main.cpp
//
// docs/ROADMAP.md item 5: two independent ConstantVelocity trackers,
// fed detections through track/association.hpp instead of a hardcoded
// "detection i belongs to track i" assumption. Frame 1 is easy (tracks
// well separated); frame 2 has the two tracks close enough together
// that naive per-track-nearest assignment would double-claim a
// detection -- shown resolved two ways: nearest_neighbor()'s global
// greedy match, and joint_probabilistic_data_association()'s full
// association-probability weighting.

#include <cstdio>
#include "augur/augur.hpp"
#include "augur/track/association.hpp"

int main() {
    using Scalar = float;
    using CV = augur::models::ConstantVelocity<Scalar, 2>;
    using KF = augur::filters::KalmanFilter<CV, /*MeasDim=*/2>;
    constexpr std::size_t kMaxTracks = 4, kMaxDetections = 4;

    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    KF::MeasurementCovariance R = KF::MeasurementCovariance::Identity() * Scalar(0.1);

    KF::StateVector x0a = KF::StateVector::Zero();
    x0a(0) = 0.0f;
    KF::StateVector x0b = KF::StateVector::Zero();
    x0b(0) = 5.0f;
    KF track_a{CV{1.0f}, x0a, KF::StateCovariance::Identity(), H, R};
    KF track_b{CV{1.0f}, x0b, KF::StateCovariance::Identity(), H, R};

    const Scalar dt = 1.0f / 30.0f;
    track_a.predict(dt);
    track_b.predict(dt);

    augur::utils::FixedVector<augur::math::Vector<Scalar, 2>, kMaxTracks> predictions;
    predictions.push_back(track_a.state().head<2>());
    predictions.push_back(track_b.state().head<2>());
    augur::utils::FixedVector<augur::math::Matrix<Scalar, 2>, kMaxTracks> covariances;
    covariances.push_back((H * track_a.covariance() * H.transpose()).eval());
    covariances.push_back((H * track_b.covariance() * H.transpose()).eval());

    std::printf("=== Frame 1: well-separated detections ===\n");
    augur::utils::FixedVector<augur::math::Vector<Scalar, 2>, kMaxDetections> detections1;
    detections1.push_back({0.05f, 0.02f});
    detections1.push_back({5.10f, -0.05f});
    const auto gnn1 = augur::track::nearest_neighbor<Scalar, 2, kMaxTracks, kMaxDetections>(
        predictions, covariances, detections1, Scalar(3.0));
    for (const auto& a : gnn1) {
        std::printf("  GNN: track %zu <- detection %zu (Mahalanobis %.3f)\n", a.track_index, a.detection_index, a.gate_distance);
    }

    std::printf("\n=== Frame 2: ambiguous detections between two close tracks ===\n");
    augur::utils::FixedVector<augur::math::Vector<Scalar, 2>, kMaxDetections> detections2;
    detections2.push_back({0.3f, 0.0f});
    detections2.push_back({0.7f, 0.0f});
    augur::utils::FixedVector<augur::math::Vector<Scalar, 2>, kMaxTracks> close_predictions;
    close_predictions.push_back({0.0f, 0.0f});
    close_predictions.push_back({1.0f, 0.0f});
    augur::utils::FixedVector<augur::math::Matrix<Scalar, 2>, kMaxTracks> close_covariances;
    close_covariances.push_back(augur::math::Matrix<Scalar, 2>::Identity());
    close_covariances.push_back(augur::math::Matrix<Scalar, 2>::Identity());

    const auto gnn2 = augur::track::nearest_neighbor<Scalar, 2, kMaxTracks, kMaxDetections>(
        close_predictions, close_covariances, detections2, Scalar(3.0));
    for (const auto& a : gnn2) {
        std::printf("  GNN (hard assignment): track %zu <- detection %zu\n", a.track_index, a.detection_index);
    }

    const auto jpda = augur::track::joint_probabilistic_data_association<Scalar, 2, kMaxTracks, kMaxDetections>(
        close_predictions, close_covariances, detections2, Scalar(3.0), Scalar(0.01));
    for (std::size_t i = 0; i < 2; ++i) {
        std::printf("  JPDA: track %zu -> P(det0)=%.3f P(det1)=%.3f P(missed)=%.3f\n",
                    i, jpda.beta(int(i), 0), jpda.beta(int(i), 1), jpda.beta_missed(int(i)));
    }
    return 0;
}
