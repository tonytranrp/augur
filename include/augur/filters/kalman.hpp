#pragma once
// augur/filters/kalman.hpp
//
// A textbook linear Kalman filter, generic over any models::MotionModel.
// This is the default Filter used inside imm::Estimator, and also
// perfectly usable standalone for a single motion model with no IMM
// mixing at all -- not every tracked entity needs three competing
// models.
//
// Measurement model is linear (z = H x + noise), which covers the
// common game case of directly observing position (H selects the
// position sub-block, R is your sensor/detection noise). For a
// genuinely nonlinear measurement (e.g. bearing-only), see
// extended_kalman.hpp instead -- it reuses this same state-propagation
// code and only swaps the update step.

#include "augur/filters/filter_concept.hpp"
#include "augur/math/backend.hpp"
#include "augur/models/model_concept.hpp"
#include <cmath>
#include <numbers>

namespace augur::filters {

template <models::MotionModel ModelT, int MeasDim>
class KalmanFilter {
public:
    using Scalar = typename ModelT::Scalar;
    using Model = ModelT;
    static constexpr std::size_t dimension = ModelT::dimension;

    using StateVector = augur::math::Vector<Scalar, dimension>;
    using StateCovariance = augur::math::Matrix<Scalar, dimension>;
    using Measurement = augur::math::Vector<Scalar, MeasDim>;
    using MeasurementMatrix = augur::math::Matrix<Scalar, MeasDim, dimension>;
    using MeasurementCovariance = augur::math::Matrix<Scalar, MeasDim>;

    KalmanFilter(Model model,
                 StateVector initial_state,
                 StateCovariance initial_covariance,
                 MeasurementMatrix measurement_matrix,
                 MeasurementCovariance measurement_noise)
        : model_(std::move(model)),
          x_(std::move(initial_state)),
          P_(std::move(initial_covariance)),
          H_(std::move(measurement_matrix)),
          R_(std::move(measurement_noise)) {}

    void predict(Scalar dt) {
        const StateCovariance F = model_.jacobian(x_, dt);
        x_ = model_.transition(x_, dt);
        P_ = F * P_ * F.transpose() + model_.process_noise(dt);
    }

    void update(const Measurement& z) {
        const Measurement y = z - H_ * x_; // innovation
        const MeasurementCovariance S = H_ * P_ * H_.transpose() + R_;
        const MeasurementCovariance S_inv = augur::math::safe_inverse<Scalar, MeasDim>(S);
        const auto K = P_ * H_.transpose() * S_inv; // Kalman gain

        x_ = x_ + K * y;
        const StateCovariance I = StateCovariance::Identity();
        P_ = (I - K * H_) * P_;

        last_likelihood_ = gaussian_likelihood(y, S, S_inv);
    }

    [[nodiscard]] const StateVector& state() const { return x_; }
    [[nodiscard]] const StateCovariance& covariance() const { return P_; }
    [[nodiscard]] Scalar last_likelihood() const { return last_likelihood_; }
    [[nodiscard]] const Model& model() const { return model_; }

    // Mutable escape hatch for adaptive-filter wrappers that compose
    // with an existing Filter rather than reimplementing its math (see
    // filters/current_statistical_filter.hpp and the intended shape
    // sketched in filters/adaptive/sage_husa.hpp) -- they need to
    // rewrite the inner model's own tunable parameters between cycles
    // based on the filter's own output.
    [[nodiscard]] Model& model() { return model_; }

    // Escape hatch for imm::Estimator's mixing step, which needs to
    // directly overwrite state/covariance with a blended estimate
    // between prediction cycles.
    void set_state(const StateVector& x, const StateCovariance& P) {
        x_ = x;
        P_ = P;
    }

private:
    [[nodiscard]] Scalar gaussian_likelihood(const Measurement& y,
                                              const MeasurementCovariance& S,
                                              const MeasurementCovariance& S_inv) const {
        const Scalar exponent = Scalar(-0.5) * (y.transpose() * S_inv * y)(0, 0);
        const Scalar det_s = S.determinant();
        if (!(det_s > Scalar(0))) {
            return Scalar(1e-12); // degenerate covariance: treat as maximally surprising
        }
        const Scalar normalizer = std::sqrt(std::pow(Scalar(2) * std::numbers::pi_v<Scalar>, MeasDim) * det_s);
        return std::exp(exponent) / normalizer;
    }

    Model model_;
    StateVector x_;
    StateCovariance P_;
    MeasurementMatrix H_;
    MeasurementCovariance R_;
    Scalar last_likelihood_ = Scalar(1);
};

} // namespace augur::filters
