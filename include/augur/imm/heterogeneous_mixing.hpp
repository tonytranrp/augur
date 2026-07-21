#pragma once
// augur/imm/heterogeneous_mixing.hpp
//
// The expand/restrict transform that makes different-order IMM mixing
// (docs/ROADMAP.md item 0) possible: once every mixed filter's native
// state/covariance is expanded into the shared augmented space
// (core/state_component.hpp, imm/augmented_layout.hpp), the actual
// mixing math is IDENTICAL to the same-dimension case -- this file
// reuses imm::mix()/imm::combine() from imm/mixing.hpp directly rather
// than duplicating the Blom & Bar-Shalom formulas a second time.
// imm::update_mode_probability() needs no change at all: it was already
// dimension-independent.
//
// DESIGN CHOICE, stated plainly: a lower-order model's state has no
// opinion at all about a component it doesn't track (e.g.
// ConstantVelocity has no turn-rate state). Rather than treating that
// as "definitely zero" (overconfident and wrong), expand_covariance()
// below fills it in as "zero mean, kBigUnknownVariance variance" -- a
// large but finite placeholder standing for "no information," not a
// claim about the true value. Bar-Shalom, Li & Kirubarajan (ch. 11.6)
// give the general mapping-matrix framework without mandating one
// specific padding scheme; this is the specific, documented choice made
// here. Practically: a model's mixed-in prior for a component it
// doesn't itself track gets quickly overridden by its own next
// update() against a real measurement rather than staying stuck at a
// wrong value, because the huge prior variance makes the Kalman gain
// trust the new measurement almost completely. Verified numerically
// (ad hoc python3, per .claude/rules/testing.md) against a
// hand-computable 2-model example before being ported here -- see
// tests/unit/test_heterogeneous_imm.cpp for the permanent record of
// that check.

#include <array>
#include <cstddef>
#include "augur/core/state_component.hpp"
#include "augur/imm/augmented_layout.hpp"
#include "augur/imm/mixing.hpp"
#include "augur/imm/mode_matrix.hpp"
#include "augur/math/backend.hpp"

namespace augur::imm {

inline constexpr int kAugmentedDim = static_cast<int>(core::kStateComponentCount);

template <typename Scalar>
using AugmentedVector = augur::math::Vector<Scalar, kAugmentedDim>;

template <typename Scalar>
using AugmentedMatrix = augur::math::Matrix<Scalar, kAugmentedDim>;

template <typename Scalar>
[[nodiscard]] constexpr Scalar big_unknown_variance() {
    // See file comment: large enough that any real measurement quickly
    // dominates it, far larger than any covariance a well-tuned filter
    // in this library would produce on its own.
    return static_cast<Scalar>(1e4);
}

template <AugmentedMappable Model>
[[nodiscard]] AugmentedVector<typename Model::Scalar> expand_state(const typename Model::State& x) {
    AugmentedVector<typename Model::Scalar> out = AugmentedVector<typename Model::Scalar>::Zero();
    constexpr auto map = augmented_index_map<Model>();
    for (std::size_t i = 0; i < Model::dimension; ++i) {
        out(static_cast<int>(map[i])) = x(static_cast<int>(i));
    }
    return out;
}

template <AugmentedMappable Model>
[[nodiscard]] AugmentedMatrix<typename Model::Scalar> expand_covariance(const typename Model::Transition& P) {
    using Scalar = typename Model::Scalar;
    AugmentedMatrix<Scalar> out = AugmentedMatrix<Scalar>::Identity() * big_unknown_variance<Scalar>();
    constexpr auto map = augmented_index_map<Model>();
    for (std::size_t i = 0; i < Model::dimension; ++i) {
        for (std::size_t j = 0; j < Model::dimension; ++j) {
            out(static_cast<int>(map[i]), static_cast<int>(map[j])) = P(static_cast<int>(i), static_cast<int>(j));
        }
    }
    return out;
}

template <AugmentedMappable Model>
[[nodiscard]] typename Model::State restrict_state(const AugmentedVector<typename Model::Scalar>& x_aug) {
    typename Model::State out;
    constexpr auto map = augmented_index_map<Model>();
    for (std::size_t i = 0; i < Model::dimension; ++i) {
        out(static_cast<int>(i)) = x_aug(static_cast<int>(map[i]));
    }
    return out;
}

template <AugmentedMappable Model>
[[nodiscard]] typename Model::Transition restrict_covariance(const AugmentedMatrix<typename Model::Scalar>& P_aug) {
    typename Model::Transition out;
    constexpr auto map = augmented_index_map<Model>();
    for (std::size_t i = 0; i < Model::dimension; ++i) {
        for (std::size_t j = 0; j < Model::dimension; ++j) {
            out(static_cast<int>(i), static_cast<int>(j)) = P_aug(static_cast<int>(map[i]), static_cast<int>(map[j]));
        }
    }
    return out;
}

} // namespace augur::imm
