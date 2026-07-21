#pragma once
// augur/imm/mixing.hpp
//
// The actual IMM math (Blom & Bar-Shalom's "interacting" step, plus the
// mode-probability update and final combination), lifted out of
// Estimator into free functions so it's independently testable
// (see tests/unit/) without spinning up a full Estimator<...>.
//
// Follows the standard four-step IMM cycle as given in Bar-Shalom, Li &
// Kirubarajan, "Estimation with Applications to Tracking and
// Navigation" (Wiley, 2001), ch. 11:
//   1. mixing probabilities  mu_{i|j}
//   2. mixed initial conditions per filter (x0_j, P0_j)
//   3. (caller runs each filter's own predict+update)
//   4. mode probability update from each filter's likelihood, then
//      combination into one output estimate
//
// IMPORTANT SIMPLIFICATION: this implementation assumes every model in
// the mix shares the same state dimension. Bar-Shalom's general
// formulation allows mixing models of *different* order (e.g. a 5-state
// coordinated-turn model against a 6-state constant-velocity model) via
// linear mapping matrices onto a common augmented state space -- that
// is real, useful, and NOT implemented here. See docs/ARCHITECTURE.md
// ("Known limitation: same-order mixing only") and docs/ROADMAP.md for
// what adding it properly would take. Practically: mix several
// same-order model instances with different tunings (e.g. three
// CoordinatedTurn models representing "calm", "juking", "sharp turn"
// regimes) until that lands.

#include <array>
#include <cstddef>
#include "augur/imm/mode_matrix.hpp"
#include "augur/math/backend.hpp"

namespace augur::imm {

template <std::size_t NumModels, int Dim, typename Scalar>
struct MixedInputs {
    std::array<augur::math::Vector<Scalar, Dim>, NumModels> state;
    std::array<augur::math::Matrix<Scalar, Dim>, NumModels> covariance;
};

// Step 1 + 2: compute the mixed (x0_j, P0_j) each filter should be
// re-initialized with before its own predict+update this cycle.
template <std::size_t NumModels, int Dim, typename Scalar>
[[nodiscard]] inline MixedInputs<NumModels, Dim, Scalar> mix(
    const std::array<augur::math::Vector<Scalar, Dim>, NumModels>& prev_state,
    const std::array<augur::math::Matrix<Scalar, Dim>, NumModels>& prev_covariance,
    const std::array<Scalar, NumModels>& mode_probability,
    const ModeMatrix<NumModels, Scalar>& transition) {

    std::array<Scalar, NumModels> normalizer{}; // c_j
    for (std::size_t j = 0; j < NumModels; ++j) {
        Scalar c = Scalar(0);
        for (std::size_t i = 0; i < NumModels; ++i) {
            c += transition(i, j) * mode_probability[i];
        }
        normalizer[j] = c;
    }

    MixedInputs<NumModels, Dim, Scalar> out{};
    for (std::size_t j = 0; j < NumModels; ++j) {
        auto& x0 = out.state[j];
        x0 = augur::math::Vector<Scalar, Dim>::Zero();
        const Scalar c_j = normalizer[j] > Scalar(0) ? normalizer[j] : Scalar(1e-12);

        std::array<Scalar, NumModels> mix_weight{};
        for (std::size_t i = 0; i < NumModels; ++i) {
            mix_weight[i] = (transition(i, j) * mode_probability[i]) / c_j;
            x0 += mix_weight[i] * prev_state[i];
        }

        auto& P0 = out.covariance[j];
        P0 = augur::math::Matrix<Scalar, Dim>::Zero();
        for (std::size_t i = 0; i < NumModels; ++i) {
            const auto diff = prev_state[i] - x0;
            P0 += mix_weight[i] * (prev_covariance[i] + diff * diff.transpose());
        }
    }
    return out;
}

// Step 4a: mode probability update given each filter's post-update
// likelihood and the same per-mode normalizers c_j computed in mix().
template <std::size_t NumModels, typename Scalar>
[[nodiscard]] inline std::array<Scalar, NumModels> update_mode_probability(
    const std::array<Scalar, NumModels>& mode_probability,
    const ModeMatrix<NumModels, Scalar>& transition,
    const std::array<Scalar, NumModels>& likelihood) {

    std::array<Scalar, NumModels> c{};
    for (std::size_t j = 0; j < NumModels; ++j) {
        Scalar sum = Scalar(0);
        for (std::size_t i = 0; i < NumModels; ++i) {
            sum += transition(i, j) * mode_probability[i];
        }
        c[j] = sum;
    }

    std::array<Scalar, NumModels> unnormalized{};
    Scalar total = Scalar(0);
    for (std::size_t j = 0; j < NumModels; ++j) {
        unnormalized[j] = c[j] * likelihood[j];
        total += unnormalized[j];
    }
    total = total > Scalar(0) ? total : Scalar(1e-12);

    std::array<Scalar, NumModels> out{};
    for (std::size_t j = 0; j < NumModels; ++j) {
        out[j] = unnormalized[j] / total;
    }
    return out;
}

// Step 4b: combine each filter's post-update (x_j, P_j) into one
// output estimate, weighted by the freshly updated mode probabilities.
template <std::size_t NumModels, int Dim, typename Scalar>
[[nodiscard]] inline std::pair<augur::math::Vector<Scalar, Dim>, augur::math::Matrix<Scalar, Dim>> combine(
    const std::array<augur::math::Vector<Scalar, Dim>, NumModels>& state,
    const std::array<augur::math::Matrix<Scalar, Dim>, NumModels>& covariance,
    const std::array<Scalar, NumModels>& mode_probability) {

    auto x = augur::math::Vector<Scalar, Dim>::Zero().eval();
    for (std::size_t j = 0; j < NumModels; ++j) {
        x += mode_probability[j] * state[j];
    }
    auto P = augur::math::Matrix<Scalar, Dim>::Zero().eval();
    for (std::size_t j = 0; j < NumModels; ++j) {
        const auto diff = state[j] - x;
        P += mode_probability[j] * (covariance[j] + diff * diff.transpose());
    }
    return {x, P};
}

} // namespace augur::imm
