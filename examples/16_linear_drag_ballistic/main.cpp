// examples/16_linear_drag_ballistic/main.cpp
//
// docs/IMPROVEMENT_PLAN.md's new-model finding: models/linear_drag_ballistic.hpp
// side by side with ConstantAcceleration (pure gravity, no drag), tracking
// a thrown projectile -- the scenario drag actually matters for: a real
// thrown/fired object loses horizontal speed to air resistance in a way
// ConstantAcceleration structurally cannot represent (it has no notion
// of "deceleration proportional to velocity" -- only a directly-tracked,
// unconstrained acceleration state).
//
// SCOPE reminder from linear_drag_ballistic.hpp's own file comment:
// linear drag is a good fit for games whose own physics already uses it
// (a common arcade/stability simplification), not "realistic bullet
// physics" -- real projectiles at game-relevant speeds are solidly in
// the quadratic-drag regime.

#include <cstdio>
#include "augur/augur.hpp"
#include "augur/models/linear_drag_ballistic.hpp"

int main() {
    using Scalar = float;
    using LDB = augur::models::LinearDragBallistic<Scalar, 2>;
    using CA = augur::models::ConstantAcceleration<Scalar, 2>;
    using KFLDB = augur::filters::KalmanFilter<LDB, /*MeasDim=*/2>;
    using KFCA = augur::filters::KalmanFilter<CA, /*MeasDim=*/2>;

    const LDB::GravityVector gravity{0.0f, -9.81f}; // Y-up, Y is "down" when negated

    KFLDB::MeasurementMatrix H_ldb = KFLDB::MeasurementMatrix::Zero();
    H_ldb(0, 0) = 1;
    H_ldb(1, 1) = 1;
    KFLDB::MeasurementCovariance R = KFLDB::MeasurementCovariance::Identity() * 0.05f;
    KFLDB ldb_filter{LDB{gravity, /*drag_coefficient=*/0.3f, /*accel_noise_density=*/1.0f},
                      KFLDB::StateVector::Zero(), KFLDB::StateCovariance::Identity(), H_ldb, R};

    KFCA::MeasurementMatrix H_ca = KFCA::MeasurementMatrix::Zero();
    H_ca(0, 0) = 1;
    H_ca(1, 1) = 1;
    KFCA ca_filter{CA{1.0f}, KFCA::StateVector::Zero(), KFCA::StateCovariance::Identity(), H_ca, R};

    // Ground truth: thrown at 45 degrees, real linear drag (k=0.3) --
    // simulated via the SAME model's own transition(), the honest way
    // to generate "truth" for a model that's meant to represent it
    // exactly (see examples/13_latency_compensation for the identical
    // pattern).
    LDB truth_model{gravity, 0.3f, 0.0f};
    LDB::State truth = LDB::State::Zero();
    truth(2) = 8.0f;  // vx
    truth(3) = 8.0f;  // vy

    const Scalar dt = Scalar(1.0 / 30.0);
    std::printf("step  truth_pos              LDB (drag-aware) est    CA (no-drag) est\n");
    for (int step = 0; step < 45; ++step) {
        truth = truth_model.transition(truth, dt);
        const KFLDB::Measurement z{truth(0), truth(1)}; // noise-free for a clean comparison

        ldb_filter.predict(dt);
        ldb_filter.update(z);
        ca_filter.predict(dt);
        ca_filter.update(z);

        if (step % 10 == 9 || step == 44) {
            std::printf("%3d   (%.3f,%.3f)   (%.3f,%.3f)   (%.3f,%.3f)\n", step, truth(0), truth(1),
                        ldb_filter.state()(0), ldb_filter.state()(1), ca_filter.state()(0), ca_filter.state()(1));
        }
    }

    // Predict 1s further ahead from the last known state -- this is
    // where the two models' different assumptions actually diverge
    // visibly: LDB knows velocity decays toward gravity's own terminal
    // contribution, CA extrapolates whatever acceleration it last
    // inferred (noisy, and with no structural reason to decay).
    const auto ldb_ahead = ldb_filter.model().transition(ldb_filter.state(), 1.0f);
    const auto ca_ahead = ca_filter.model().transition(ca_filter.state(), 1.0f);
    std::printf("\n1s-ahead prediction: LDB=(%.3f,%.3f)  CA=(%.3f,%.3f)\n", ldb_ahead(0), ldb_ahead(1), ca_ahead(0),
                ca_ahead(1));
    return 0;
}
