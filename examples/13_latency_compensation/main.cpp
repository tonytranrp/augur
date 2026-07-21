// examples/13_latency_compensation/main.cpp
//
// docs/ROADMAP.md item 9: SnapshotBuffer for server-side "rewind to
// what the shooter saw" lag compensation, plus predict_to_render_time()
// for extrapolating a track to the local render clock given a one-way
// latency estimate.

#include <cstdio>
#include "augur/augur.hpp"
#include "augur/predict/latency_compensation.hpp"

int main() {
    using Scalar = float;
    using CV = augur::models::ConstantVelocity<Scalar, 2>;
    using KF = augur::filters::KalmanFilter<CV, /*MeasDim=*/2>;

    KF::StateVector x0 = KF::StateVector::Zero();
    x0(2) = 2.0f; // vx
    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    KF filter{CV{0.1f}, x0, KF::StateCovariance::Identity(), H, KF::MeasurementCovariance::Identity() * 0.1f};

    // Server-side: record this track's state every tick, so a shot fired
    // by a laggy client can be checked against where the target REALLY
    // was at the time that client saw it, not where it is now.
    augur::predict::SnapshotBuffer<Scalar, KF::StateVector, 32> history;

    const Scalar dt = Scalar(1.0 / 30.0);
    Scalar t = 0;
    for (int tick = 0; tick < 10; ++tick) {
        filter.predict(dt);
        filter.update(KF::Measurement{2.0f * t, 0.0f});
        t += dt;
        history.record(t, filter.state());
    }

    std::printf("current server time: %.4f, current pos: (%.3f, %.3f)\n", t, filter.state()(0), filter.state()(1));

    // A client with 100ms one-way latency fired at a target it saw
    // 0.1s ago -- rewind the server's own history to that moment.
    const Scalar client_latency = 0.10f;
    const auto rewound = history.rewind_to(t - client_latency);
    std::printf("rewound to t=%.4f (what the shooter actually saw): pos=(%.3f, %.3f)\n",
                t - client_latency, rewound(0), rewound(1));

    // Client-side: extrapolate this track to the local render clock.
    const Scalar last_update_time = t;
    const Scalar render_time = t + Scalar(0.033); // one frame ahead
    const auto predicted = augur::predict::predict_to_render_time(filter, last_update_time, render_time, client_latency);
    std::printf("predicted to render time %.4f: pos=(%.3f, %.3f)\n", render_time, predicted(0), predicted(1));
    return 0;
}
