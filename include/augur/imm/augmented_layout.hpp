#pragma once
// augur/imm/augmented_layout.hpp
//
// Declares, for each built-in model that can participate in
// different-order IMM mixing (docs/ROADMAP.md item 0), which slots of
// the canonical augmented state (core/state_component.hpp) its own
// native state vector corresponds to, in order.
//
// This is an external trait specialization (`AugmentedLayoutFor<Model>`),
// the same pattern utils/type_traits.hpp's is_specialization_of uses,
// rather than a member added to each model class -- the mapping is an
// IMM-mixing concern, not something ConstantVelocity/ConstantAcceleration/
// CoordinatedTurn need to know about themselves, and this way none of
// those (already solid, already tested) files need to change at all.
//
// The primary template is intentionally a plain, complete (but empty)
// struct rather than an incomplete/undeclared one: accessing a member of
// an incomplete type inside a requires-expression is murky territory in
// the standard, whereas "does this complete type have a `value` member"
// is unambiguous SFINAE. A model with no specialization here simply
// doesn't satisfy AugmentedMappable and can't be used with
// HeterogeneousEstimator -- nothing else about it changes.
//
// Scoped deliberately to 2D: CoordinatedTurn is inherently planar (see
// its own file comment), so a 3D ConstantVelocity/ConstantAcceleration
// has no CT counterpart to mix with anyway. Extend
// core/kAugmentedLayout plus the specializations below if a future 3D
// coordinated-turn model is ever added.

#include <array>
#include "augur/core/state_component.hpp"
#include "augur/models/constant_acceleration.hpp"
#include "augur/models/constant_velocity.hpp"
#include "augur/models/coordinated_turn.hpp"
#include "augur/models/model_concept.hpp"
#include "augur/models/singer.hpp"

namespace augur::imm {

template <typename Model>
struct AugmentedLayoutFor {};

template <typename Scalar>
struct AugmentedLayoutFor<augur::models::ConstantVelocity<Scalar, 2>> {
    static constexpr std::array<core::StateComponent, 4> value = {
        core::StateComponent::PositionX, core::StateComponent::PositionY,
        core::StateComponent::VelocityX, core::StateComponent::VelocityY,
    };
};

template <typename Scalar>
struct AugmentedLayoutFor<augur::models::ConstantAcceleration<Scalar, 2>> {
    static constexpr std::array<core::StateComponent, 6> value = {
        core::StateComponent::PositionX, core::StateComponent::PositionY,
        core::StateComponent::VelocityX, core::StateComponent::VelocityY,
        core::StateComponent::AccelX,    core::StateComponent::AccelY,
    };
};

template <typename Scalar>
struct AugmentedLayoutFor<augur::models::CoordinatedTurn<Scalar>> {
    static constexpr std::array<core::StateComponent, 5> value = {
        core::StateComponent::PositionX, core::StateComponent::PositionY,
        core::StateComponent::VelocityX, core::StateComponent::VelocityY,
        core::StateComponent::TurnRate,
    };
};

template <typename Scalar>
struct AugmentedLayoutFor<augur::models::Singer<Scalar, 2>> {
    static constexpr std::array<core::StateComponent, 6> value = {
        core::StateComponent::PositionX, core::StateComponent::PositionY,
        core::StateComponent::VelocityX, core::StateComponent::VelocityY,
        core::StateComponent::AccelX,    core::StateComponent::AccelY,
    };
};

template <typename Model>
concept AugmentedMappable = augur::models::MotionModel<Model> && requires {
    { AugmentedLayoutFor<Model>::value } -> std::convertible_to<const std::array<core::StateComponent, Model::dimension>&>;
};

// Per-native-index -> augmented-space-index map, resolved entirely at
// compile time from a model's AugmentedLayoutFor specialization.
template <AugmentedMappable Model>
[[nodiscard]] constexpr std::array<std::size_t, Model::dimension> augmented_index_map() {
    std::array<std::size_t, Model::dimension> map{};
    for (std::size_t i = 0; i < Model::dimension; ++i) {
        map[i] = core::augmented_index_of(AugmentedLayoutFor<Model>::value[i]);
    }
    return map;
}

} // namespace augur::imm
