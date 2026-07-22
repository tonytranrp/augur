#pragma once
// augur/track/gm_phd.hpp
//
// docs/ROADMAP.md, "GM-PHD filter for variable/unknown target count" --
// explicitly the highest-effort, most-advanced roadmap item. Everything
// else in track/ assumes detections associate to a roughly-known set of
// tracks. The Gaussian Mixture Probability Hypothesis Density filter
// instead propagates a Gaussian mixture representing "expected density
// of targets in state space" and never explicitly solves the
// association problem -- what makes it viable when the NUMBER of
// targets is itself unknown and changing.
//
// References: Vo, B.-N. and Ma, W.-K., "The Gaussian Mixture Probability
// Hypothesis Density Filter," IEEE Transactions on Signal Processing,
// 54(11), 2006, pp. 4091-4104. Theoretical foundation: Mahler, R.,
// "Multitarget Bayes Filtering via First-Order Multitarget Moments,"
// IEEE Trans. Aerospace and Electronic Systems, 39(4), 2003.
//
// The core update equations (missed-detection terms plus one updated
// component per (predicted component, detection) pair, weighted by
// relative likelihood against a clutter hypothesis) were verified
// against a hand-computable 2-component, 2-detection example (ad hoc
// python3 + numpy, per .claude/rules/testing.md) before being written
// here -- total posterior weight correctly estimated ~2 targets, with
// cross-matched (wrong pairing) components correctly collapsing to
// ~0 weight. The merge step reuses the exact same weighted moment-
// matching formula as imm::mixing.hpp's combine() (same math, same
// justification), not a new derivation.
//
// CAPACITY, stated plainly: update()'s posterior has (1 + num_detections)
// times as many components as went into it (a missed-detection term
// plus one updated term per detection, for every predicted component) --
// this is the "silently grows unbounded frame over frame" problem the
// roadmap item's own comment warns about. prune_and_merge() must be
// called every cycle to keep the component count bounded; MaxComponents
// must be sized for the worst single update() call
// (num_predicted_components * (1 + max_detections_per_frame)), not for
// the long-run steady state.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <utility>
#include "augur/math/backend.hpp"
#include "augur/models/model_concept.hpp"
#include "augur/utils/assert.hpp"
#include "augur/utils/fixed_vector.hpp"

namespace augur::track {

template <augur::models::MotionModel Model, int MeasDim, std::size_t MaxComponents>
class GmPhdFilter {
public:
    using Scalar = typename Model::Scalar;
    static constexpr std::size_t dimension = Model::dimension;
    using State = typename Model::State;
    using StateCovariance = typename Model::Transition;
    using Measurement = augur::math::Vector<Scalar, MeasDim>;
    using MeasurementMatrix = augur::math::Matrix<Scalar, MeasDim, static_cast<int>(dimension)>;
    using MeasurementCovariance = augur::math::Matrix<Scalar, MeasDim>;

    struct GaussianComponent {
        Scalar weight = Scalar(0);
        State mean = State::Zero();
        StateCovariance covariance = StateCovariance::Identity();
    };

    struct Config {
        Scalar survival_probability = Scalar(0.99);
        Scalar detection_probability = Scalar(0.9);
        Scalar clutter_intensity = Scalar(0.01);       // assumed false-alarm density, same units as the Gaussian likelihood
        Scalar prune_weight_threshold = Scalar(1e-3);  // components below this are discarded outright
        Scalar merge_mahalanobis_threshold = Scalar(4.0);
        std::size_t max_components_after_prune = MaxComponents / 4; // headroom for next cycle's (1+num_detections)x growth
    };

    explicit GmPhdFilter(Model model, MeasurementMatrix H, MeasurementCovariance R, Config config = Config{})
        : model_(std::move(model)), H_(std::move(H)), R_(std::move(R)), config_(config) {}

    // birth_components: caller-supplied prior over where NEW targets are
    // expected to appear this cycle (e.g. spawn regions), added to the
    // mixture at their given (typically small) weight.
    template <std::size_t MaxBirth>
    void predict(Scalar dt, const augur::utils::FixedVector<GaussianComponent, MaxBirth>& birth_components) {
        for (auto& c : components_) {
            c.weight *= config_.survival_probability;
            const StateCovariance F = model_.jacobian(c.mean, dt);
            c.mean = model_.transition(c.mean, dt);
            c.covariance = F * c.covariance * F.transpose() + model_.process_noise(dt);
        }
        for (const auto& b : birth_components) {
            AUGUR_ASSERT(!components_.full(), "GmPhdFilter::predict: component capacity exceeded by birth components");
            components_.push_back(b);
        }
    }

    template <std::size_t MaxDetections>
    void update(const augur::utils::FixedVector<Measurement, MaxDetections>& detections) {
        const std::size_t num_predicted = components_.size();
        augur::utils::FixedVector<GaussianComponent, MaxComponents> posterior;

        for (std::size_t i = 0; i < num_predicted; ++i) {
            posterior.push_back(GaussianComponent{
                (Scalar(1) - config_.detection_probability) * components_[i].weight,
                components_[i].mean, components_[i].covariance});
        }

        // S, S_inv, K, and det_s depend only on component i's own
        // predicted covariance (via the fixed H_/R_) -- never on which
        // detection z it's later combined with. Precomputing them once
        // per component here, rather than inside the detections loop
        // below, turns what was num_predicted * num_detections redundant
        // safe_inverse/determinant/gain calls into num_predicted (the
        // same class of hoist as track/association.hpp's
        // detail::InnovationGate, just also carrying the Kalman gain,
        // which association.hpp has no use for). Purely a reordering of
        // the same formulas -- verified against
        // tests/unit/test_gm_phd.cpp's existing reference value.
        augur::utils::FixedVector<MeasurementCovariance, MaxComponents> S_inv_by_component;
        augur::utils::FixedVector<augur::math::Matrix<Scalar, static_cast<int>(dimension), MeasDim>, MaxComponents> K_by_component;
        augur::utils::FixedVector<Scalar, MaxComponents> det_s_by_component;
        for (std::size_t i = 0; i < num_predicted; ++i) {
            const auto& c = components_[i];
            const MeasurementCovariance S = H_ * c.covariance * H_.transpose() + R_;
            const MeasurementCovariance S_inv = augur::math::safe_inverse<Scalar, MeasDim>(S);
            S_inv_by_component.push_back(S_inv);
            K_by_component.push_back(c.covariance * H_.transpose() * S_inv);
            det_s_by_component.push_back(S.determinant());
        }

        for (const auto& z : detections) {
            augur::utils::FixedVector<Scalar, MaxComponents> unnormalized;
            augur::utils::FixedVector<State, MaxComponents> updated_mean;
            augur::utils::FixedVector<StateCovariance, MaxComponents> updated_cov;
            Scalar denom = config_.clutter_intensity;

            for (std::size_t i = 0; i < num_predicted; ++i) {
                const auto& c = components_[i];
                const auto& S_inv = S_inv_by_component[i];
                const auto& K = K_by_component[i];
                const Scalar det_s = det_s_by_component[i];
                const Measurement innovation = z - H_ * c.mean;

                const Scalar exponent = Scalar(-0.5) * (innovation.transpose() * S_inv * innovation)(0, 0);
                const Scalar likelihood = (det_s > Scalar(0))
                    ? std::exp(exponent) / std::sqrt(std::pow(Scalar(2) * std::numbers::pi_v<Scalar>, Scalar(MeasDim)) * det_s)
                    : Scalar(0);
                const Scalar qi = config_.detection_probability * c.weight * likelihood;

                unnormalized.push_back(qi);
                denom += qi;
                updated_mean.push_back(State(c.mean + K * innovation));
                updated_cov.push_back(StateCovariance((StateCovariance::Identity() - K * H_) * c.covariance));
            }

            for (std::size_t i = 0; i < num_predicted; ++i) {
                AUGUR_ASSERT(!posterior.full(),
                             "GmPhdFilter::update: posterior capacity exceeded -- call prune_and_merge() every "
                             "cycle, or raise MaxComponents to fit num_predicted_components * (1 + max_detections)");
                posterior.push_back(GaussianComponent{unnormalized[i] / denom, updated_mean[i], updated_cov[i]});
            }
        }
        components_ = std::move(posterior);
    }

    // Discards low-weight components, then repeatedly merges the
    // highest-weight remaining component with every other remaining
    // component within merge_mahalanobis_threshold of it (moment
    // matching -- the same weighted-mean + spread-of-means covariance
    // formula as imm::mixing.hpp's combine()), and finally caps the
    // total count at max_components_after_prune by keeping only the
    // highest-weight survivors.
    void prune_and_merge() {
        augur::utils::FixedVector<GaussianComponent, MaxComponents> candidates;
        for (const auto& c : components_) {
            if (c.weight >= config_.prune_weight_threshold) candidates.push_back(c);
        }

        augur::utils::FixedVector<bool, MaxComponents> used;
        for (std::size_t i = 0; i < candidates.size(); ++i) used.push_back(false);

        augur::utils::FixedVector<GaussianComponent, MaxComponents> merged;
        while (true) {
            std::size_t best = candidates.size();
            for (std::size_t i = 0; i < candidates.size(); ++i) {
                if (used[i]) continue;
                if (best == candidates.size() || candidates[i].weight > candidates[best].weight) best = i;
            }
            if (best == candidates.size()) break;

            const auto& anchor = candidates[best];
            const StateCovariance anchor_cov_inv = augur::math::safe_inverse<Scalar, static_cast<int>(dimension)>(anchor.covariance);

            Scalar total_weight = Scalar(0);
            State weighted_mean = State::Zero();
            augur::utils::FixedVector<std::size_t, MaxComponents> group;
            for (std::size_t i = 0; i < candidates.size(); ++i) {
                if (used[i]) continue;
                const State diff = candidates[i].mean - anchor.mean;
                const Scalar maha2 = (diff.transpose() * anchor_cov_inv * diff)(0, 0);
                if (maha2 > config_.merge_mahalanobis_threshold * config_.merge_mahalanobis_threshold) continue;
                group.push_back(i);
                total_weight += candidates[i].weight;
                weighted_mean += candidates[i].weight * candidates[i].mean;
            }
            weighted_mean /= total_weight;

            StateCovariance weighted_cov = StateCovariance::Zero();
            for (std::size_t idx : group) {
                const State diff = candidates[idx].mean - weighted_mean;
                weighted_cov += candidates[idx].weight * (candidates[idx].covariance + diff * diff.transpose());
                used[idx] = true;
            }
            weighted_cov /= total_weight;

            merged.push_back(GaussianComponent{total_weight, weighted_mean, weighted_cov});
        }

        if (merged.size() > config_.max_components_after_prune) {
            augur::utils::FixedVector<std::size_t, MaxComponents> order;
            for (std::size_t i = 0; i < merged.size(); ++i) order.push_back(i);
            std::sort(order.begin(), order.end(),
                      [&](std::size_t a, std::size_t b) { return merged[a].weight > merged[b].weight; });
            augur::utils::FixedVector<GaussianComponent, MaxComponents> capped;
            for (std::size_t i = 0; i < config_.max_components_after_prune; ++i) capped.push_back(merged[order[i]]);
            components_ = std::move(capped);
        } else {
            components_ = std::move(merged);
        }
    }

    // Components at or above weight_threshold each represent an
    // estimated real target (a weight near 1 means "very likely exactly
    // one target here"). This returns one target per qualifying
    // component; it does not split a component whose weight is well
    // above 1 into multiple targets (a documented refinement, not
    // implemented -- see docs/ROADMAP.md item 7's status note).
    template <std::size_t MaxOut>
    [[nodiscard]] augur::utils::FixedVector<GaussianComponent, MaxOut> extract_targets(Scalar weight_threshold) const {
        augur::utils::FixedVector<GaussianComponent, MaxOut> out;
        for (const auto& c : components_) {
            if (c.weight < weight_threshold) continue;
            AUGUR_ASSERT(!out.full(), "GmPhdFilter::extract_targets: output capacity exceeded");
            out.push_back(c);
        }
        return out;
    }

    [[nodiscard]] const augur::utils::FixedVector<GaussianComponent, MaxComponents>& components() const { return components_; }

private:
    Model model_;
    MeasurementMatrix H_;
    MeasurementCovariance R_;
    Config config_;
    augur::utils::FixedVector<GaussianComponent, MaxComponents> components_;
};

} // namespace augur::track
