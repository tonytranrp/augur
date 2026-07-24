#pragma once
// augur/filters/unscented_kalman.hpp
//
// docs/ROADMAP.md item 3: a Filter-satisfying Unscented Kalman Filter,
// for measurement models that are genuinely nonlinear in a way
// ExtendedKalmanFilter's first-order Taylor linearization handles
// poorly (bearing-only detection, a screen-space projection of a 3D
// position, range-only sonar/radar-style sensors -- the same use cases
// extended_kalman.hpp's own file comment names). Sigma points are
// regenerated fresh from the CURRENT (state, covariance) on every
// predict() and update() call, rather than stashing propagated points
// between calls -- simpler, and robust to something else (e.g.
// imm mixing) calling set_state() in between, at the cost of one extra
// Cholesky decomposition per cycle versus reusing predict()'s points.
//
// Reference: Julier, S. J. and Uhlmann, J. K., "A New Extension of the
// Kalman Filter to Nonlinear Systems," 1997; Wan, E. A. and van der
// Merwe, R., "The Unscented Kalman Filter for Nonlinear Estimation,"
// 2000 (source of the alpha/beta/kappa scaled sigma-point formulation
// and weights used here).
//
// Verified (ad hoc python3 + numpy, per .claude/rules/testing.md)
// before being written here: for a LINEAR transition and measurement
// function, this UKF formulation must reduce to standard KalmanFilter
// exactly (the unscented transform is exact for affine functions) --
// confirmed to agree with filters::KalmanFilter to ~1e-10 in float64.
//
// That same check caught a real float32 numerical-robustness bug before
// it shipped: the commonly-cited "textbook" scaled-sigma-point default
// of alpha=1e-3 makes (n+lambda) tiny, which makes the central weight
// Wm[0] enormous (order 1e6 for a 4-state model) -- exactly correct in
// exact arithmetic, but in augur::Scalar=float, the huge weight forces
// a catastrophic-cancellation sum when combined with the other sigma
// points' contributions, producing errors around 0.1-0.2 (verified
// numerically, float32 vs. an mpmath/float64 reference) instead of the
// near-machine-precision agreement KalmanFilter comparison demands.
// Config::alpha defaults to 1 (lambda=0, the simpler *unscaled*
// Julier-Uhlmann transform) instead, verified stable across the same
// checks -- pass a smaller alpha only if you also confirm it stays
// accurate at your chosen Scalar's precision.

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numbers>
#include "augur/filters/observe_position.hpp"
#include "augur/math/backend.hpp"
#include "augur/models/model_concept.hpp"

namespace augur::filters {

// MeasurementFnT is a template parameter, not hardcoded to
// std::function -- defaulted to std::function<...> so every existing
// call site is unaffected. See filters/extended_kalman.hpp's identical
// change for the full reasoning (docs/IMPROVEMENT_PLAN.md measured
// 2.09-2.31x overhead on isolated calls at UKF/particle-filter call
// counts from std::function's type erasure specifically).
template <models::MotionModel ModelT, int MeasDim,
          typename MeasurementFnT = std::function<augur::math::Vector<typename ModelT::Scalar, MeasDim>(
              const augur::math::Vector<typename ModelT::Scalar, ModelT::dimension>&)>>
class UnscentedKalmanFilter {
public:
    using Scalar = typename ModelT::Scalar;
    using Model = ModelT;
    static constexpr std::size_t dimension = ModelT::dimension;
    static constexpr int StateDim = static_cast<int>(dimension);
    static constexpr int NumSigmaPoints = 2 * StateDim + 1;

    using StateVector = augur::math::Vector<Scalar, StateDim>;
    using StateCovariance = augur::math::Matrix<Scalar, StateDim>;
    using Measurement = augur::math::Vector<Scalar, MeasDim>;
    using MeasurementCovariance = augur::math::Matrix<Scalar, MeasDim>;
    using MeasurementFn = MeasurementFnT;

    // RENAMED from this class's own former `Config` (the name now
    // belongs to the construction aggregate below, matching what
    // `::Config` means everywhere else in this library --
    // track/track_manager.hpp's and track/gm_phd.hpp's Configs are both
    // full construction aggregates, not tuning knobs). Semantics and
    // defaults unchanged.
    struct SigmaPointTuning {
        // alpha=1 (not the textbook-common 1e-3 -- see the file comment
        // for the float32 catastrophic-cancellation bug that default
        // caused, found via ad hoc python3 verification before this
        // shipped): lambda=0 exactly, the unscaled Julier-Uhlmann
        // transform. beta=2 (optimal for Gaussian priors, Wan & van der
        // Merwe). kappa=0 (a common, simple default; 3-n is another
        // common choice but can make (n+lambda) non-positive for large
        // n, which breaks the Cholesky step below).
        Scalar alpha = Scalar(1);
        Scalar beta = Scalar(2);
        Scalar kappa = Scalar(0);
    };

    explicit UnscentedKalmanFilter(Model model,
                                   StateVector initial_state,
                                   StateCovariance initial_covariance,
                                   MeasurementFn measurement_fn,
                                   MeasurementCovariance measurement_noise,
                                   SigmaPointTuning tuning = SigmaPointTuning{})
        : model_(std::move(model)),
          x_(std::move(initial_state)),
          P_(std::move(initial_covariance)),
          h_(std::move(measurement_fn)),
          R_(std::move(measurement_noise)),
          tuning_(tuning) {
        const Scalar lambda = tuning_.alpha * tuning_.alpha * (Scalar(StateDim) + tuning_.kappa) - Scalar(StateDim);
        lambda_ = lambda;
        Wm_[0] = lambda / (Scalar(StateDim) + lambda);
        Wc_[0] = Wm_[0] + (Scalar(1) - tuning_.alpha * tuning_.alpha + tuning_.beta);
        for (int i = 1; i < NumSigmaPoints; ++i) {
            Wm_[i] = Scalar(1) / (Scalar(2) * (Scalar(StateDim) + lambda));
            Wc_[i] = Wm_[i];
        }
    }

    // Designated-initializer-friendly construction aggregate -- see
    // filters/kalman.hpp::Config's identical rationale. measurement_fn
    // defaults the same way filters/extended_kalman.hpp::Config's does.
    struct Config {
        Model model;
        StateVector initial_state = StateVector::Zero();
        StateCovariance initial_covariance = StateCovariance::Identity();
        MeasurementFn measurement_fn = ObservePositionFn<Scalar, MeasDim, StateDim>();
        MeasurementCovariance measurement_noise = MeasurementCovariance::Identity();
        SigmaPointTuning sigma_points{};
    };

    explicit UnscentedKalmanFilter(Config config)
        : UnscentedKalmanFilter(std::move(config.model),
                                std::move(config.initial_state),
                                std::move(config.initial_covariance),
                                std::move(config.measurement_fn),
                                std::move(config.measurement_noise),
                                config.sigma_points) {}

    void predict(Scalar dt) {
        const auto sigma = generate_sigma_points(x_, P_);
        std::array<StateVector, NumSigmaPoints> propagated;
        for (int i = 0; i < NumSigmaPoints; ++i) propagated[i] = model_.transition(sigma[i], dt);

        StateVector x_pred = StateVector::Zero();
        for (int i = 0; i < NumSigmaPoints; ++i) x_pred += Wm_[i] * propagated[i];

        StateCovariance P_pred = model_.process_noise(dt);
        for (int i = 0; i < NumSigmaPoints; ++i) {
            const StateVector diff = propagated[i] - x_pred;
            P_pred += Wc_[i] * (diff * diff.transpose());
        }

        x_ = x_pred;
        P_ = P_pred;
    }

    void update(const Measurement& z) {
        const auto sigma = generate_sigma_points(x_, P_);
        std::array<Measurement, NumSigmaPoints> transformed;
        for (int i = 0; i < NumSigmaPoints; ++i) transformed[i] = h_(sigma[i]);

        Measurement z_pred = Measurement::Zero();
        for (int i = 0; i < NumSigmaPoints; ++i) z_pred += Wm_[i] * transformed[i];

        MeasurementCovariance S = R_;
        augur::math::Matrix<Scalar, StateDim, MeasDim> Pxz = augur::math::Matrix<Scalar, StateDim, MeasDim>::Zero();
        for (int i = 0; i < NumSigmaPoints; ++i) {
            const Measurement dz = transformed[i] - z_pred;
            const StateVector dx = sigma[i] - x_;
            S += Wc_[i] * (dz * dz.transpose());
            Pxz += Wc_[i] * (dx * dz.transpose());
        }

        const MeasurementCovariance S_inv = augur::math::safe_inverse<Scalar, MeasDim>(S);
        const auto K = Pxz * S_inv;
        const Measurement innovation = z - z_pred;

        x_ = x_ + K * innovation;
        P_ = P_ - K * S * K.transpose();

        const Scalar exponent = Scalar(-0.5) * (innovation.transpose() * S_inv * innovation)(0, 0);
        const Scalar det_s = S.determinant();
        last_likelihood_ = (det_s > Scalar(0))
            ? std::exp(exponent) / std::sqrt(std::pow(Scalar(2) * std::numbers::pi_v<Scalar>, Scalar(MeasDim)) * det_s)
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
    [[nodiscard]] std::array<StateVector, NumSigmaPoints> generate_sigma_points(const StateVector& mean,
                                                                                 const StateCovariance& cov) const {
        Eigen::LLT<StateCovariance> llt((Scalar(StateDim) + lambda_) * cov);
        const StateCovariance L = llt.matrixL();

        std::array<StateVector, NumSigmaPoints> points;
        points[0] = mean;
        for (int i = 0; i < StateDim; ++i) {
            points[1 + i] = mean + L.col(i);
            points[1 + StateDim + i] = mean - L.col(i);
        }
        return points;
    }

    Model model_;
    StateVector x_;
    StateCovariance P_;
    MeasurementFn h_;
    MeasurementCovariance R_;
    SigmaPointTuning tuning_;
    Scalar lambda_;
    std::array<Scalar, NumSigmaPoints> Wm_;
    std::array<Scalar, NumSigmaPoints> Wc_;
    Scalar last_likelihood_ = Scalar(1);
};

} // namespace augur::filters
