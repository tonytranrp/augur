// examples/09_gm_phd/main.cpp
//
// docs/ROADMAP.md item 7, the highest-effort roadmap item: GM-PHD
// tracking two targets without ever explicitly associating a detection
// to a specific track -- unlike track/track_manager.hpp, which assumes
// a roughly-known, identity-preserving set of tracks. A birth model
// keeps proposing "a target could appear near the origin or near
// (10,0)"; the filter's own component weights settle onto however many
// real targets the detections actually support.

#include <cstdio>
#include "augur/augur.hpp"
#include "augur/track/gm_phd.hpp"

int main() {
    using Scalar = float;
    using CV = augur::models::ConstantVelocity<Scalar, 2>;
    using PHD = augur::track::GmPhdFilter<CV, /*MeasDim=*/2, /*MaxComponents=*/64>;

    PHD::MeasurementMatrix H = PHD::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;

    PHD::Config config;
    config.survival_probability = Scalar(0.99);
    config.detection_probability = Scalar(0.9);
    config.clutter_intensity = Scalar(0.001);
    config.prune_weight_threshold = Scalar(1e-3);
    config.merge_mahalanobis_threshold = Scalar(2.0);
    config.max_components_after_prune = 16;

    PHD filter{CV{Scalar(0.1)}, H, PHD::MeasurementCovariance::Identity() * Scalar(0.05), config};

    augur::utils::FixedVector<PHD::GaussianComponent, 4> birth;
    PHD::GaussianComponent b0, b1;
    b0.weight = Scalar(0.3);
    b0.mean = PHD::State::Zero();
    b0.covariance = PHD::StateCovariance::Identity() * Scalar(4.0);
    b1.weight = Scalar(0.3);
    b1.mean = PHD::State::Zero();
    b1.mean(0) = Scalar(10.0);
    b1.covariance = PHD::StateCovariance::Identity() * Scalar(4.0);
    birth.push_back(b0);
    birth.push_back(b1);

    const Scalar dt = Scalar(1.0 / 30.0);
    for (int step = 0; step < 20; ++step) {
        augur::utils::FixedVector<PHD::Measurement, 8> detections;
        detections.push_back({0.0f, 0.0f});             // target near the origin, present the whole time
        if (step >= 8) detections.push_back({10.0f, 0.0f}); // second target appears partway through

        filter.predict(dt, birth);
        filter.update(detections);
        filter.prune_and_merge();

        if (step % 4 == 3 || step == 19) {
            const auto targets = filter.extract_targets<8>(Scalar(0.5));
            std::printf("step %2d: %zu component(s), %zu extracted target(s) above weight 0.5:",
                        step, filter.components().size(), targets.size());
            for (const auto& t : targets) {
                std::printf("  (w=%.2f pos=(%.2f,%.2f))", t.weight, t.mean(0), t.mean(1));
            }
            std::printf("\n");
        }
    }
    return 0;
}
