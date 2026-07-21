#pragma once
// augur/core/state_component.hpp
//
// Support for mixing MotionModels of genuinely different state
// *dimension* in one IMM (docs/ROADMAP.md item 0, "different-order IMM
// mixing" -- see docs/ARCHITECTURE.md §5, "Known limitation: same-order
// IMM mixing only"). augur::imm::Estimator<Filters...> still requires
// same dimension by design (that's the zero-overhead default path);
// augur::imm::HeterogeneousEstimator<Filters...> (imm/heterogeneous_estimator.hpp)
// is the separate, opt-in mechanism for the different-order case, and
// this file is its foundation.
//
// The approach (Bar-Shalom, Li & Kirubarajan, "Estimation with
// Applications to Tracking and Navigation," Wiley, 2001, ch. 11.6):
// define ONE fixed, canonical "augmented" state space big enough to
// hold every quantity any mixed model might track, and give each
// mixable model a `layout` declaring which augmented-space slots its
// own native state vector corresponds to, in order. Mixing then
// happens by expanding every model's native state into this shared
// augmented space, doing the usual IMM mixing math there, and
// restricting the result back down to each model's own native space
// before its own predict()/update(). This file only defines the shared
// vocabulary (the slot enum and the canonical layout); the actual
// expand/mix/restrict math lives in imm/heterogeneous_mixing.hpp.
//
// This is deliberately scoped to the "textbook full IMM example"
// docs/ROADMAP.md names as the target (constant velocity + constant
// acceleration + coordinated turn, mixed together) rather than a fully
// generic arbitrary-component-union system -- extend kAugmentedLayout
// if a future model needs a slot that isn't here yet.

#include <array>
#include <cstddef>

namespace augur::core {

enum class StateComponent : std::size_t {
    PositionX,
    PositionY,
    VelocityX,
    VelocityY,
    AccelX,
    AccelY,
    TurnRate,
};

inline constexpr std::size_t kStateComponentCount = 7;

// The canonical augmented state, in a fixed order every model's
// `layout` maps into. Every built-in mixable model (ConstantVelocity,
// ConstantAcceleration, CoordinatedTurn, Singer) fits inside this.
inline constexpr std::array<StateComponent, kStateComponentCount> kAugmentedLayout = {
    StateComponent::PositionX, StateComponent::PositionY,
    StateComponent::VelocityX, StateComponent::VelocityY,
    StateComponent::AccelX,    StateComponent::AccelY,
    StateComponent::TurnRate,
};

// Where a given component sits in kAugmentedLayout. constexpr, so this
// is resolved entirely at compile time for any model's layout.
[[nodiscard]] constexpr std::size_t augmented_index_of(StateComponent c) {
    for (std::size_t i = 0; i < kStateComponentCount; ++i) {
        if (kAugmentedLayout[i] == c) return i;
    }
    return kStateComponentCount; // unreachable for any component actually in the layout
}

} // namespace augur::core
