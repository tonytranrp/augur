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
// below fills it in as "zero mean, big_unknown_variance() variance" -- a
// large but finite placeholder standing for "no information," not a
// claim about the true value. Bar-Shalom, Li & Kirubarajan (ch. 11.6)
// give the general mapping-matrix framework without mandating one
// specific padding scheme; this is the specific, documented choice made
// here. Practically: a model's mixed-in prior for a component it
// doesn't itself track gets quickly overridden by its own next
// update() against a real measurement rather than staying stuck at a
// wrong value, because the large prior variance makes the Kalman gain
// trust the new measurement almost completely. Verified numerically
// (ad hoc python3, per .claude/rules/testing.md) against a
// hand-computable 2-model example before being ported here -- see
// tests/unit/test_heterogeneous_imm.cpp for the permanent record of
// that check.
//
// MAGNITUDE (docs/PRODUCTION_ROADMAP.md P0 item 5): originally 1e4.
// That was too large -- when a dominant model's expanded state mixes
// INTO a non-dominant model (weighted by mix_weight = transition(i,j) *
// mode_probability[i] / c_j), the padding contributes mix_weight * 1e4
// to the receiving model's mixed covariance, which for any non-trivial
// mix_weight overwhelms that model's own prior so completely that its
// very next update() effectively discards all memory of its own state
// (Kalman gain saturates when prior covariance vastly exceeds
// measurement-implied information) -- worst right at a mode-probability
// crossover, exactly where mix_weight is largest for the recovering
// model. Empirically swept (ad hoc python3/C++, per
// .claude/rules/testing.md) against examples/04_heterogeneous_imm's own
// CV+CA+CT sustained-turn scenario extended long enough to observe a
// real crossover: max combined-position error over the run was ~0.43
// at 1e4 (visibly wrong-signed at points) vs. a stable ~0.009-0.013
// plateau for every value from 50 to 1000, degrading again above ~5000.
// 100 sits in the middle of that plateau -- still ~65x a typical
// well-tuned filter covariance in this library (order 1-2), comfortably
// "large" without being so extreme it erases a model's own information
// in one mixing cycle.
//
// UNIT-SCALE DEPENDENCE (docs/IMPROVEMENT_PLAN.md, found in a later
// investigation): 100 is an ABSOLUTE constant, but a covariance's own
// scale depends on the caller's chosen units (position in meters vs.
// millimeters changes variance by 1e6). A full 90-step dynamic
// simulation using real position/velocity/acceleration rescaled 1000x
// smaller (meters -> mm) showed combined-position error 3.6x worse
// (0.046 vs. 0.013) and post-onset variance ~56,000x too small relative
// to a healthy baseline, purely from the units choice -- this constant
// untouched. expand_covariance() below takes an optional
// padding_variance parameter (defaulting to this function, so every
// EXISTING caller's behavior is unchanged) specifically so a caller
// whose units make 100 the wrong absolute scale can override it --
// HeterogeneousEstimator threads this through as its own optional,
// defaulted constructor parameter. This is the CHEAP partial fix the
// plan scoped; the deeper fix (a real per-unit-system-aware or
// BLUE-based padding scheme, see Granström, Willett & Bar-Shalom,
// "Systematic approach to IMM mixing for unequal dimension states,"
// IEEE Trans. Aerospace and Electronic Systems, 51(5), 2015, pp.
// 2975-2986) is a substantial, separate research effort the plan
// deliberately did not fold into this change.

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
    // in this library would produce on its own -- but not so large that
    // mixing it into a non-dominant model erases that model's own prior
    // in a single cycle (empirically swept; 100 sits mid-plateau).
    return static_cast<Scalar>(100);
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
[[nodiscard]] AugmentedMatrix<typename Model::Scalar> expand_covariance(
    const typename Model::Transition& P,
    typename Model::Scalar padding_variance = big_unknown_variance<typename Model::Scalar>()) {
    using Scalar = typename Model::Scalar;
    AugmentedMatrix<Scalar> out = AugmentedMatrix<Scalar>::Identity() * padding_variance;
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
