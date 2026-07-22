// examples/17_coordinated_turn_3d/main.cpp
//
// docs/IMPROVEMENT_PLAN.md's "quasi-3D coordinated turn" finding:
// models/coordinated_turn_3d.hpp tracking a target that both turns
// horizontally (like examples/02_imm_maneuvering_target's own sharp-turn
// scenario) AND climbs at a steady rate -- e.g. an aircraft or a
// rocket-jumping player banking through a turn while gaining altitude.
// A plain 2D CoordinatedTurn has nowhere to put the climb; a plain 3D
// ConstantVelocity has no way to represent the turn. CoordinatedTurn3D
// represents both at once, by design decoupled (see that file's own
// comment for the honest scope of that assumption).

#include <cstdio>
#include "augur/augur.hpp"
#include "augur/models/coordinated_turn_3d.hpp"

int main() {
    using Scalar = float;
    using CT3D = augur::models::CoordinatedTurn3D<Scalar>;
    using KF = augur::filters::KalmanFilter<CT3D, /*MeasDim=*/3>;

    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1; // px
    H(1, 1) = 1; // py
    H(2, 5) = 1; // pz
    KF::MeasurementCovariance R = KF::MeasurementCovariance::Identity() * 0.05f;

    KF::StateVector x0 = KF::StateVector::Zero();
    x0(2) = 8.0f; // initial vx
    KF filter{CT3D{0.1f, 3.0f, 0.2f}, x0, KF::StateCovariance::Identity(), H, R};

    // Ground truth: banks into a 1.5 rad/s turn after 1s while climbing at
    // a steady 2 m/s throughout -- simulated via the SAME model's own
    // transition(), the honest way to generate "truth" for a model that's
    // meant to represent it exactly (see examples/16_linear_drag_ballistic
    // for the identical pattern).
    CT3D truth_model{};
    CT3D::State truth = CT3D::State::Zero();
    truth(2) = 8.0f; // vx
    truth(6) = 2.0f; // vz (climb rate)
    const Scalar dt = Scalar(1.0 / 30.0);
    const Scalar true_omega = 1.5f;

    std::printf("step   truth (px,py,pz)          KF estimate (px,py,pz)        omega_est\n");
    for (int step = 0; step < 60; ++step) {
        truth(4) = (step < 30) ? 0.0f : true_omega; // straight for 1s, then a sustained turn
        truth = truth_model.transition(truth, dt);

        filter.predict(dt);
        filter.update(KF::Measurement{truth(0), truth(1), truth(5)});

        if (step % 15 == 14 || step == 59) {
            std::printf("%3d   (%7.3f,%7.3f,%7.3f)   (%7.3f,%7.3f,%7.3f)   %.3f\n", step, truth(0), truth(1),
                        truth(5), filter.state()(0), filter.state()(1), filter.state()(5), filter.state()(4));
        }
    }

    // The vertical channel never needed to know about the turn at all --
    // vz stayed a clean constant-velocity estimate throughout.
    std::printf("\nfinal vz estimate: %.4f (true: 2.0000)\n", filter.state()(6));
    return 0;
}
