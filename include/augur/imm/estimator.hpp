#pragma once
// augur/imm/estimator.hpp
//
// augur::imm::Estimator<Filters...> -- an Interacting Multiple Model
// estimator over an arbitrary, caller-chosen pack of Filter types. This
// is the library's centerpiece: it's what turns "I have a Kalman filter"
// into "I have a tracker that notices when a target stops moving in a
// straight line."
//
// Reference: Bar-Shalom, Li & Kirubarajan, "Estimation with Applications
// to Tracking and Navigation," Wiley-Interscience, 2001, ch. 11.
//
// Usage (both `<>` and `::` doing real work, not decoration -- the
// template pack IS the set of models you're choosing to mix, chosen at
// compile time with zero virtual dispatch):
//
//   using CT = augur::models::CoordinatedTurn<float>;
//   using KF = augur::filters::KalmanFilter<CT, /*MeasDim=*/2>;
//
//   augur::imm::Estimator<KF, KF, KF> tracker{
//       KF{CT{/*q_pos=*/1, /*q_turn=*/0.02f}, x0, P0, H, R},  // "calm"
//       KF{CT{/*q_pos=*/1, /*q_turn=*/0.5f},  x0, P0, H, R},  // "juking"
//       KF{CT{/*q_pos=*/1, /*q_turn=*/3.0f},  x0, P0, H, R},  // "sharp turn"
//       augur::imm::ModeMatrix<3, float>::uniform(0.95f)
//   };
//   tracker.predict(dt);
//   tracker.update(detection_xy);
//   auto [x, P] = tracker.combined_state();
//
// KNOWN LIMITATION: every Filters... must share Scalar, dimension, and
// Measurement type (enforced below with static_assert). Mixing models
// of genuinely different *order* (e.g. a 5-state CT against a 6-state
// CV) needs the common-space mapping-matrix machinery from Bar-Shalom
// ch. 11.6 that this v0.1 does not implement -- see
// docs/ARCHITECTURE.md and docs/ROADMAP.md. Mixing several
// differently-tuned instances of the same model (as above) sidesteps
// the issue entirely and is often enough in practice.

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "augur/filters/filter_concept.hpp"
#include "augur/imm/mixing.hpp"
#include "augur/imm/mode_matrix.hpp"
#include "augur/math/backend.hpp"

namespace augur::imm {

template <filters::Filter... Filters>
class Estimator {
    static_assert(sizeof...(Filters) >= 2, "IMM needs at least two filters to mix between -- for one model, just use the Filter directly.");

    using FilterTuple = std::tuple<Filters...>;
    using First = std::tuple_element_t<0, FilterTuple>;

public:
    using Scalar = typename First::Scalar;
    static constexpr std::size_t dimension = First::dimension;
    static constexpr std::size_t model_count = sizeof...(Filters);

    static_assert((... && (Filters::dimension == dimension)),
                  "augur::imm::Estimator requires every mixed filter to share a state dimension. "
                  "See docs/ARCHITECTURE.md 'Known limitation: same-order mixing only'.");
    static_assert((std::is_same_v<typename Filters::Scalar, Scalar> && ...),
                  "all mixed filters must share a Scalar type");

    using StateVector = augur::math::Vector<Scalar, static_cast<int>(dimension)>;
    using StateCovariance = augur::math::Matrix<Scalar, static_cast<int>(dimension)>;

    Estimator(Filters... filters, ModeMatrix<model_count, Scalar> transition)
        : filters_(std::move(filters)...), transition_(std::move(transition)) {
        mode_probability_.fill(Scalar(1) / Scalar(model_count));
    }

    void predict(Scalar dt) {
        const auto prev_state = gather_state(std::make_index_sequence<model_count>{});
        const auto prev_cov = gather_covariance(std::make_index_sequence<model_count>{});
        const auto mixed = imm::mix<model_count, static_cast<int>(dimension), Scalar>(
            prev_state, prev_cov, mode_probability_, transition_);
        scatter_and_predict(mixed, dt, std::make_index_sequence<model_count>{});
    }

    template <typename Measurement>
    void update(const Measurement& z) {
        std::apply([&z](auto&... f) { (f.update(z), ...); }, filters_);
        const auto likelihood = gather_likelihood(std::make_index_sequence<model_count>{});
        mode_probability_ = imm::update_mode_probability<model_count, Scalar>(mode_probability_, transition_, likelihood);
    }

    [[nodiscard]] std::pair<StateVector, StateCovariance> combined_state() const {
        const auto state = gather_state(std::make_index_sequence<model_count>{});
        const auto cov = gather_covariance(std::make_index_sequence<model_count>{});
        return imm::combine<model_count, static_cast<int>(dimension), Scalar>(state, cov, mode_probability_);
    }

    // Propagates the CURRENT combined estimate through every model's
    // own transition() for `horizon` seconds and re-blends by the
    // current mode probabilities. This is a pragmatic one-shot
    // extrapolation (e.g. "predict to render time" / lead-shot queries)
    // rather than a full re-run of the mixing cycle over many small
    // dt steps -- good enough for horizons of a few hundred ms; for
    // longer horizons prefer stepping predict() repeatedly instead.
    [[nodiscard]] StateVector predict_ahead(Scalar horizon) const {
        const auto [x_combined, p_combined] = combined_state();
        (void)p_combined;
        return predict_ahead_impl(x_combined, horizon, std::make_index_sequence<model_count>{});
    }

    template <std::size_t I>
    [[nodiscard]] const auto& filter() const { return std::get<I>(filters_); }

    [[nodiscard]] const std::array<Scalar, model_count>& mode_probability() const { return mode_probability_; }

private:
    template <std::size_t... Is>
    [[nodiscard]] std::array<StateVector, model_count> gather_state(std::index_sequence<Is...>) const {
        return {std::get<Is>(filters_).state()...};
    }

    template <std::size_t... Is>
    [[nodiscard]] std::array<StateCovariance, model_count> gather_covariance(std::index_sequence<Is...>) const {
        return {std::get<Is>(filters_).covariance()...};
    }

    template <std::size_t... Is>
    [[nodiscard]] std::array<Scalar, model_count> gather_likelihood(std::index_sequence<Is...>) const {
        return {std::get<Is>(filters_).last_likelihood()...};
    }

    template <std::size_t... Is>
    void scatter_and_predict(const MixedInputs<model_count, static_cast<int>(dimension), Scalar>& mixed,
                              Scalar dt, std::index_sequence<Is...>) {
        ((std::get<Is>(filters_).set_state(mixed.state[Is], mixed.covariance[Is]), std::get<Is>(filters_).predict(dt)), ...);
    }

    template <std::size_t... Is>
    [[nodiscard]] StateVector predict_ahead_impl(const StateVector& x, Scalar horizon, std::index_sequence<Is...>) const {
        StateVector out = StateVector::Zero();
        ((out += mode_probability_[Is] * std::get<Is>(filters_).model().transition(x, horizon)), ...);
        return out;
    }

    FilterTuple filters_;
    ModeMatrix<model_count, Scalar> transition_;
    std::array<Scalar, model_count> mode_probability_;
};

// NOT functional -- confirmed, not hedged: verified (docs/IMPROVEMENT_PLAN.md)
// on both clang 22.1.4 and MSVC 19.50 that this deduces an EMPTY
// Filters pack instead of the intended one (a parameter pack followed
// by a fixed trailing argument isn't a deduction context either
// compiler resolves the way this guide intends), which then fails
// Estimator's own `sizeof...(Filters) >= 2` static_assert before you
// even reach a real error about your actual filter types. Always use
// the explicit form instead: `Estimator<A, B, C> tracker{a, b, c,
// transition};` -- every example and test in this library already does.
// Left in place as a documented non-solution rather than deleted, so a
// future attempt to fix CTAD here isn't starting from nothing.
template <filters::Filter... Filters>
Estimator(Filters..., ModeMatrix<sizeof...(Filters), typename std::tuple_element_t<0, std::tuple<Filters...>>::Scalar>)
    -> Estimator<Filters...>;

} // namespace augur::imm
