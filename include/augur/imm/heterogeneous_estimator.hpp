#pragma once
// augur/imm/heterogeneous_estimator.hpp
//
// augur::imm::HeterogeneousEstimator<Filters...> -- an IMM estimator
// like imm::Estimator, but for the case where the mixed Filters... have
// genuinely DIFFERENT state dimension (docs/ROADMAP.md item 0,
// "different-order IMM mixing" -- the "textbook full IMM example" of
// mixing constant velocity + constant acceleration + coordinated turn
// together). imm::Estimator itself is untouched and remains the
// default, zero-overhead, same-dimension-only path; this is the
// separate, heavier, opt-in mechanism for when that restriction is the
// actual problem (see docs/ARCHITECTURE.md §5, "Known limitation:
// same-order IMM mixing only").
//
// Every Filter::Model here must satisfy imm::AugmentedMappable
// (imm/augmented_layout.hpp) -- i.e. have a known mapping into the
// canonical augmented state (core/state_component.hpp). Internally,
// mixing happens by expanding every filter's native state/covariance
// into that shared augmented space, running the exact same mixing math
// imm::Estimator uses (imm::mix()/imm::combine(), reused directly, not
// reimplemented -- see imm/heterogeneous_mixing.hpp), then restricting
// each target filter's mixed result back down to its own native
// dimension before its own predict().
//
// combined_state() necessarily returns the AUGMENTED state/covariance
// rather than a per-model native one -- there's no single "native"
// dimension to return when the mixed models don't share one. Use
// core::StateComponent / core::augmented_index_of() to pull out the
// specific component you want (see examples/04_heterogeneous_imm).

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "augur/filters/filter_concept.hpp"
#include "augur/imm/heterogeneous_mixing.hpp"
#include "augur/imm/mixing.hpp"
#include "augur/imm/mode_matrix.hpp"

namespace augur::imm {

template <filters::Filter... Filters>
    requires (AugmentedMappable<typename Filters::Model> && ...)
class HeterogeneousEstimator {
    static_assert(sizeof...(Filters) >= 2, "IMM needs at least two filters to mix between -- for one model, just use the Filter directly.");

    using FilterTuple = std::tuple<Filters...>;
    using First = std::tuple_element_t<0, FilterTuple>;

public:
    using Scalar = typename First::Scalar;
    static constexpr std::size_t model_count = sizeof...(Filters);

    static_assert((std::is_same_v<typename Filters::Scalar, Scalar> && ...),
                  "all mixed filters must share a Scalar type");

    // padding_variance overrides big_unknown_variance() (imm/
    // heterogeneous_mixing.hpp) for every expand_covariance() call this
    // estimator makes -- see that function's own file comment for why
    // the default (100) is a fixed absolute constant that can be the
    // wrong scale for a caller whose state is expressed in different
    // units. Defaulted for source compatibility: every existing caller
    // gets identical behavior to before this parameter existed.
    HeterogeneousEstimator(Filters... filters, ModeMatrix<model_count, Scalar> transition,
                            Scalar padding_variance = big_unknown_variance<Scalar>())
        : filters_(std::move(filters)...), transition_(std::move(transition)), padding_variance_(padding_variance) {
        mode_probability_.fill(Scalar(1) / Scalar(model_count));
    }

    void predict(Scalar dt) {
        const auto prev_state_aug = gather_augmented_state(std::make_index_sequence<model_count>{});
        const auto prev_cov_aug = gather_augmented_covariance(std::make_index_sequence<model_count>{});
        const auto mixed = imm::mix<model_count, kAugmentedDim, Scalar>(
            prev_state_aug, prev_cov_aug, mode_probability_, transition_);
        scatter_and_predict(mixed, dt, std::make_index_sequence<model_count>{});
    }

    template <typename Measurement>
    void update(const Measurement& z) {
        std::apply([&z](auto&... f) { (f.update(z), ...); }, filters_);
        const auto likelihood = gather_likelihood(std::make_index_sequence<model_count>{});
        mode_probability_ = imm::update_mode_probability<model_count, Scalar>(mode_probability_, transition_, likelihood);
    }

    [[nodiscard]] std::pair<AugmentedVector<Scalar>, AugmentedMatrix<Scalar>> combined_state() const {
        const auto state = gather_augmented_state(std::make_index_sequence<model_count>{});
        const auto cov = gather_augmented_covariance(std::make_index_sequence<model_count>{});
        return imm::combine<model_count, kAugmentedDim, Scalar>(state, cov, mode_probability_);
    }

    template <std::size_t I>
    [[nodiscard]] const auto& filter() const { return std::get<I>(filters_); }

    [[nodiscard]] const std::array<Scalar, model_count>& mode_probability() const { return mode_probability_; }

private:
    template <std::size_t... Is>
    [[nodiscard]] std::array<AugmentedVector<Scalar>, model_count> gather_augmented_state(std::index_sequence<Is...>) const {
        return {expand_state<typename std::tuple_element_t<Is, FilterTuple>::Model>(std::get<Is>(filters_).state())...};
    }

    template <std::size_t... Is>
    [[nodiscard]] std::array<AugmentedMatrix<Scalar>, model_count> gather_augmented_covariance(std::index_sequence<Is...>) const {
        return {expand_covariance<typename std::tuple_element_t<Is, FilterTuple>::Model>(
            std::get<Is>(filters_).covariance(), padding_variance_)...};
    }

    template <std::size_t... Is>
    [[nodiscard]] std::array<Scalar, model_count> gather_likelihood(std::index_sequence<Is...>) const {
        return {std::get<Is>(filters_).last_likelihood()...};
    }

    template <std::size_t... Is>
    void scatter_and_predict(const MixedInputs<model_count, kAugmentedDim, Scalar>& mixed, Scalar dt, std::index_sequence<Is...>) {
        ((std::get<Is>(filters_).set_state(
              restrict_state<typename std::tuple_element_t<Is, FilterTuple>::Model>(mixed.state[Is]),
              restrict_covariance<typename std::tuple_element_t<Is, FilterTuple>::Model>(mixed.covariance[Is])),
          std::get<Is>(filters_).predict(dt)),
         ...);
    }

    FilterTuple filters_;
    ModeMatrix<model_count, Scalar> transition_;
    std::array<Scalar, model_count> mode_probability_;
    Scalar padding_variance_;
};

// NOT functional -- see imm::Estimator's own deduction guide for the
// identical, verified (not hedged) failure mode on both clang 22.1.4 and
// MSVC 19.50. Always use the explicit form:
// `HeterogeneousEstimator<A, B, C> tracker{a, b, c, transition};`.
template <filters::Filter... Filters>
HeterogeneousEstimator(Filters..., ModeMatrix<sizeof...(Filters), typename std::tuple_element_t<0, std::tuple<Filters...>>::Scalar>)
    -> HeterogeneousEstimator<Filters...>;

} // namespace augur::imm
