#pragma once
// augur/filters/extended_kalman.hpp
//
// Same predict step as KalmanFilter (it already uses model.jacobian(),
// so a nonlinear *process* model like CoordinatedTurn is fine there
// too) -- this class only differs in update(), which takes a nonlinear
// measurement function h(x) and its Jacobian H(x) instead of a fixed
// linear H matrix. Use this when what you observe isn't a linear
// function of the state: bearing-only detection (atan2), a screen-space
// projection of a 3D position, range-only sonar/radar-style sensors,
// etc.

#include "augur/filters/filter_concept.hpp"
#include "augur/math/backend.hpp"
#include "augur/models/model_concept.hpp"
#include <cmath>
#include <functional>
#include <numbers>

namespace augur::filters {

// MeasurementFn:   StateVector -> Measurement            (h(x))
// MeasurementJac:  StateVector -> MeasurementMatrix (Jacobian of h at x)
template <models::MotionModel ModelT, int MeasDim>
class ExtendedKalmanFilter {
public:
    using Scalar = typename ModelT::Scalar;
    using Model = ModelT;
    static constexpr std::size_t dimension = ModelT::dimension;

    using StateVector = augur::math::Vector<Scalar, dimension>;
    using StateCovariance = augur::math::Matrix<Scalar, dimension>;
    using Measurement = augur::math::Vector<Scalar, MeasDim>;
    using MeasurementMatrix = augur::math::Matrix<Scalar, MeasDim, dimension>;
    using MeasurementCovariance = augur::math::Matrix<Scalar, MeasDim>;
    using MeasurementFn = std::function<Measurement(const StateVector&)>;
    using MeasurementJacobianFn = std::function<MeasurementMatrix(const StateVector&)>;

    ExtendedKalmanFilter(Model model,
                          StateVector initial_state,
                          StateCovariance initial_covariance,
                          MeasurementFn measurement_fn,
                          MeasurementJacobianFn measurement_jacobian_fn,
                          MeasurementCovariance measurement_noise)
        : model_(std::move(model)),
          x_(std::move(initial_state)),
          P_(std::move(initial_covariance)),
          h_(std::move(measurement_fn)),
          H_fn_(std::move(measurement_jacobian_fn)),
          R_(std::move(measurement_noise)) {}

    void predict(Scalar dt) {
        const StateCovariance F = model_.jacobian(x_, dt);
        x_ = model_.transition(x_, dt);
        P_ = F * P_ * F.transpose() + model_.process_noise(dt);
    }

    void update(const Measurement& z) {
        const MeasurementMatrix H = H_fn_(x_);
        const Measurement y = z - h_(x_);
        const MeasurementCovariance S = H * P_ * H.transpose() + R_;
        const MeasurementCovariance S_inv = augur::math::safe_inverse<Scalar, MeasDim>(S);
        const auto K = P_ * H.transpose() * S_inv;

        x_ = x_ + K * y;
        const StateCovariance I = StateCovariance::Identity();
        P_ = (I - K * H) * P_;

        const Scalar exponent = Scalar(-0.5) * (y.transpose() * S_inv * y)(0, 0);
        const Scalar det_s = S.determinant();
        last_likelihood_ = (det_s > Scalar(0))
            ? std::exp(exponent) / std::sqrt(std::pow(Scalar(2) * std::numbers::pi_v<Scalar>, MeasDim) * det_s)
            : Scalar(1e-12);
    }

    [[nodiscard]] const StateVector& state() const { return x_; }
    [[nodiscard]] const StateCovariance& covariance() const { return P_; }
    [[nodiscard]] Scalar last_likelihood() const { return last_likelihood_; }
    [[nodiscard]] const Model& model() const { return model_; }

    void set_state(const StateVector& x, const StateCovariance& P) {
        x_ = x;
        P_ = P;
    }

private:
    Model model_;
    StateVector x_;
    StateCovariance P_;
    MeasurementFn h_;
    MeasurementJacobianFn H_fn_;
    MeasurementCovariance R_;
    Scalar last_likelihood_ = Scalar(1);
};

} // namespace augur::filters
