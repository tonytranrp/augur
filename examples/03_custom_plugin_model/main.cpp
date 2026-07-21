// examples/03_custom_plugin_model/main.cpp
//
// The point of this example is what ISN'T here: no augur header
// declares FixedRateCircular, no registration call tells the library
// it exists, and it doesn't live under include/augur/ at all -- it's
// defined right here in user code. Because it satisfies
// augur::models::MotionModel structurally, augur::filters::KalmanFilter
// accepts it exactly like a built-in model. This is what "open/plugin
// architecture" means concretely: the contract is the concept, not a
// base class or a factory you have to register with.
//
// FixedRateCircular tracks a target moving in a circle at a KNOWN,
// fixed angular rate (unlike models::CoordinatedTurn, which estimates
// an unknown turn rate as part of the state) -- a reasonable model for,
// say, a turret-mounted enemy with a known patrol rotation speed.

#include <cstdio>
#include "augur/augur.hpp"

template <typename ScalarT>
class FixedRateCircular {
public:
    using Scalar = ScalarT;
    static constexpr std::size_t dimension = 4; // px, py, vx, vy

    using State = augur::math::Vector<Scalar, dimension>;
    using Transition = augur::math::Matrix<Scalar, dimension>;

    FixedRateCircular(Scalar angular_rate, Scalar process_noise)
        : omega_(angular_rate), q_(process_noise) {}

    [[nodiscard]] State transition(const State& x, Scalar dt) const {
        return build_transition(dt) * x;
    }

    [[nodiscard]] Transition jacobian(const State&, Scalar dt) const {
        return build_transition(dt); // omega is fixed, not part of the state -> truly linear
    }

    [[nodiscard]] Transition process_noise(Scalar dt) const {
        Transition Q = Transition::Identity() * (q_ * dt);
        return Q;
    }

private:
    [[nodiscard]] Transition build_transition(Scalar dt) const {
        const Scalar c = std::cos(omega_ * dt);
        const Scalar s = std::sin(omega_ * dt);
        Transition F = Transition::Identity();
        // position += velocity * dt (small-dt straight-line step)...
        F(0, 2) = dt;
        F(1, 3) = dt;
        // ...and velocity itself rotates at the known fixed rate.
        F(2, 2) = c;  F(2, 3) = -s;
        F(3, 2) = s;  F(3, 3) = c;
        return F;
    }

    Scalar omega_;
    Scalar q_;
};

static_assert(augur::models::MotionModel<FixedRateCircular<float>>,
              "FixedRateCircular satisfies MotionModel with zero augur-side changes");

int main() {
    using Scalar = float;
    using Model = FixedRateCircular<Scalar>;
    using KF = augur::filters::KalmanFilter<Model, /*MeasDim=*/2>;

    KF::StateVector x0{0, 5, 1, 0}; // start at (0,5) moving in +x
    KF::StateCovariance P0 = KF::StateCovariance::Identity() * Scalar(2);
    KF::MeasurementMatrix H = KF::MeasurementMatrix::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    KF::MeasurementCovariance R = KF::MeasurementCovariance::Identity() * Scalar(0.2);

    KF filter{Model{/*angular_rate=*/Scalar(0.8), /*process_noise=*/Scalar(0.1)}, x0, P0, H, R};

    const Scalar dt = Scalar(1.0 / 30.0);
    for (int i = 0; i < 6; ++i) {
        filter.predict(dt);
        // (no detections in this trimmed-down example -- predict-only,
        // just to show the plugin model's transition() driving the filter)
        const auto& x = filter.state();
        std::printf("step %d: pos=(%.3f, %.3f)\n", i, x(0), x(1));
    }
    return 0;
}
