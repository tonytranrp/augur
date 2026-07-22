#pragma once
// augur/filters/adaptive/sage_husa.hpp
//
// docs/ROADMAP.md, "Adaptive / self-tuning process noise": Sage-Husa
// adaptive filtering estimates the process/measurement noise covariances
// (Q/R) online from the innovation sequence via covariance matching,
// instead of requiring hand-tuned noise matrices.
//
// References:
//   - Sage, A. P. and Husa, G. W., "Adaptive filtering with unknown
//     prior statistics," Joint Automatic Control Conference, 1969.
//   - The known failure mode of that vanilla 1969 formulation --
//     adaptively-estimated Q/R can lose positive-semi-definiteness and
//     diverge -- is well documented; this implementation's fix is
//     eigenvalue-flooring (augur::math::project_to_psd(), math/backend.hpp)
//     after every update, verified stable (ad hoc python3, per
//     .claude/rules/testing.md) through a real mid-run change in the
//     true noise level.
//
// SCOPE, stated plainly: this assumes H selects the state's first
// MeasDim components directly (z observes position, no scaling/rotation)
// -- the convention every example and test in this library already
// follows. That assumption is what lets this wrapper compute H*P*H^T,
// P*H^T, and the innovation itself (augur::filters::Filter has no H/R
// accessors to pull a general H from). If your measurement model isn't
// a direct position read, this wrapper isn't the right fit as-is.
//
// Composes with the INNER filter's Model for transition()/jacobian()
// (not duplicated), the same "wrapper wraps an existing Filter" pattern
// as filters/current_statistical_filter.hpp -- but reimplements the
// predict/update recursion itself rather than delegating to
// Inner::predict()/update(), because adapting Q/R requires intercepting
// quantities (H*P*H^T, the Kalman gain, the pre-adaptation innovation)
// that augur::filters::Filter's interface doesn't expose. Inner is kept
// in sync via set_state() purely so callers can still read state()/
// covariance() through the familiar interface.

#include <algorithm>
#include <cmath>
#include <numbers>
#include "augur/filters/filter_concept.hpp"
#include "augur/math/backend.hpp"

namespace augur::filters {

template <filters::Filter Inner, int MeasDim>
class SageHusaAdaptive {
public:
    using Scalar = typename Inner::Scalar;
    using Model = typename Inner::Model;
    static constexpr std::size_t dimension = Inner::dimension;
    static constexpr int state_dim = static_cast<int>(Inner::dimension);

    using StateVector = augur::math::Vector<Scalar, state_dim>;
    using StateCovariance = augur::math::Matrix<Scalar, state_dim>;
    using Measurement = augur::math::Vector<Scalar, MeasDim>;
    using MeasurementCovariance = augur::math::Matrix<Scalar, MeasDim>;

    explicit SageHusaAdaptive(Inner inner,
                               MeasurementCovariance initial_R,
                               StateCovariance initial_Q,
                               Scalar forgetting_factor = Scalar(0.98),
                               Scalar min_eigenvalue = Scalar(1e-9))
        : inner_(std::move(inner)),
          R_hat_(std::move(initial_R)),
          Q_hat_(std::move(initial_Q)),
          forgetting_factor_(forgetting_factor),
          min_eigenvalue_(min_eigenvalue),
          F_(StateCovariance::Identity()),
          prev_posterior_cov_(inner_.covariance()) {}

    void predict(Scalar dt) {
        prev_posterior_cov_ = inner_.covariance();
        F_ = inner_.model().jacobian(inner_.state(), dt);
        const StateVector x_prior = inner_.model().transition(inner_.state(), dt);
        const StateCovariance P_prior = F_ * inner_.covariance() * F_.transpose() + Q_hat_;
        inner_.set_state(x_prior, P_prior);
    }

    void update(const Measurement& z) {
        const StateVector x_prior = inner_.state();
        const StateCovariance P_prior = inner_.covariance();

        const Measurement y = z - x_prior.template head<MeasDim>();
        const MeasurementCovariance H_P_Ht = P_prior.template topLeftCorner<MeasDim, MeasDim>();
        const MeasurementCovariance S = H_P_Ht + R_hat_;
        const MeasurementCovariance S_inv = augur::math::safe_inverse<Scalar, MeasDim>(S);
        const augur::math::Matrix<Scalar, state_dim, MeasDim> K = P_prior.template leftCols<MeasDim>() * S_inv;

        const StateVector x_post = x_prior + K * y;
        const StateCovariance P_post = P_prior - K * P_prior.template topRows<MeasDim>();

        const Scalar exponent = Scalar(-0.5) * (y.transpose() * S_inv * y)(0, 0);
        const Scalar det_s = S.determinant();
        last_likelihood_ = (det_s > Scalar(0))
            ? std::exp(exponent) / std::sqrt(std::pow(Scalar(2) * std::numbers::pi_v<Scalar>, Scalar(MeasDim)) * det_s)
            : Scalar(1e-12);

        const Scalar d = Scalar(1) - forgetting_factor_;
        const MeasurementCovariance R_update = y * y.transpose() - H_P_Ht;
        R_hat_ = augur::math::project_to_psd<Scalar, MeasDim>(
            (Scalar(1) - d) * R_hat_ + d * R_update, min_eigenvalue_);

        const StateCovariance Q_update = K * y * y.transpose() * K.transpose() + P_post - F_ * prev_posterior_cov_ * F_.transpose();
        Q_hat_ = augur::math::project_to_psd<Scalar, state_dim>(
            (Scalar(1) - d) * Q_hat_ + d * Q_update, min_eigenvalue_);

        inner_.set_state(x_post, P_post);
    }

    [[nodiscard]] const StateVector& state() const { return inner_.state(); }
    [[nodiscard]] const StateCovariance& covariance() const { return inner_.covariance(); }
    [[nodiscard]] Scalar last_likelihood() const { return last_likelihood_; }
    [[nodiscard]] const Model& model() const { return inner_.model(); }

    [[nodiscard]] const MeasurementCovariance& measurement_noise_estimate() const { return R_hat_; }
    [[nodiscard]] const StateCovariance& process_noise_estimate() const { return Q_hat_; }

    void set_state(const StateVector& x, const StateCovariance& P) { inner_.set_state(x, P); }

private:
    Inner inner_;
    MeasurementCovariance R_hat_;
    StateCovariance Q_hat_;
    Scalar forgetting_factor_;
    Scalar min_eigenvalue_;
    Scalar last_likelihood_ = Scalar(1);
    StateCovariance F_;
    StateCovariance prev_posterior_cov_;
};

} // namespace augur::filters
