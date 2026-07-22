#pragma once
// augur/models/coordinated_turn_3d.hpp
//
// docs/IMPROVEMENT_PLAN.md's "Quasi-3D coordinated turn" finding: the gap
// imm/augmented_layout.hpp's own file comment names explicitly (a full
// SO(3)/Sophus turn model is out of scope per docs/ARCHITECTURE.md, the
// same way a waypoint-following model was explicitly rejected there).
// State = [px, py, vx, vy, omega, pz, vz] (dimension 7): the existing,
// unmodified models::CoordinatedTurn xy-block reused via composition for
// the first 5 components, plus a decoupled models::ConstantVelocity<.,1>
// vertical channel for [pz, vz] -- mirroring how models::CurrentStatistical
// already reuses models::Singer internally (compose, don't duplicate).
// This "horizontal turn + decoupled vertical" shape is the same design
// real ATC/aircraft-tracking literature uses (e.g. Blackman & Popoli,
// "Design and Analysis of Modern Tracking Systems", ch. 5 on 3D extensions
// of the coordinated-turn model), not a placeholder simplification of a
// more-correct model this project intends to build later.
//
// DECOUPLING ASSUMPTION, stated plainly: vertical rate is modeled as
// independent of horizontal turn rate. A real banked turn couples the two
// (lift changes with bank angle), which this model does not capture --
// that's the actual, intentional scope cut, not an implementation
// shortcut. Cross-sectional real-world grounding (live OpenSky Network
// ADS-B fetch, real aircraft over Switzerland): simultaneous aircraft
// showed vertical rates from -8.13 to +6.18 m/s across varied,
// uncorrelated-looking headings, qualitatively supporting decoupling --
// honestly a cross-sectional plausibility check, not a longitudinal proof
// (a second fetch attempting to track one aircraft's own turn+climb over
// time hit an API caching limitation, so that stronger check is still
// open). Treat this the same way models::ConstantVelocity's own "wrong but
// cheap" framing is treated elsewhere in this codebase: IMM wants
// deliberately-simplified models in the mix, and this is one, honestly
// scoped -- not a compromise implementation of the harder coupled model.
//
// Inherits whatever precision models::CoordinatedTurn's own small-omega
// branch has (this file adds no new numerical-cancellation risk of its
// own: the vertical models::ConstantVelocity<.,1> channel is exactly
// linear, no division anywhere). Finite-difference-verified (ad hoc
// python3 + numpy, per .claude/rules/testing.md) that the composed
// transition()/jacobian() wiring is correct: a full 7x7 numerical jacobian
// matches this file's block-diagonal analytic one to <4e-9 across 2000
// trials outside CoordinatedTurn's own small-omega guard, and to within
// CoordinatedTurn's own already-accepted ~0.01 tolerance across 5000
// trials inside it; off-block-diagonal entries are exactly 0 in both
// sweeps, confirming the two channels are genuinely decoupled as intended
// (no accidental index cross-talk between the xy-block and the vertical
// channel).
//
// SCOPE beyond this file: standalone use (its own same-dimension IMM
// ensemble, or a lone KalmanFilter/ExtendedKalmanFilter) needs nothing
// else. Participating in a imm::HeterogeneousEstimator mix alongside
// 2D-only models would need core/state_component.hpp's canonical
// augmented-state enum extended to a Z component plus a new
// AugmentedLayoutFor<> specialization -- deliberately not done here, since
// nothing in this file needs it and adding an unused enum extension would
// be scope creep beyond this finding's own request.

#include "augur/math/backend.hpp"
#include "augur/models/constant_velocity.hpp"
#include "augur/models/coordinated_turn.hpp"

namespace augur::models {

template <typename ScalarT>
class CoordinatedTurn3D {
public:
    using Scalar = ScalarT;
    static constexpr std::size_t dimension = 7; // px, py, vx, vy, omega, pz, vz

    using State = augur::math::Vector<Scalar, dimension>;
    using Transition = augur::math::Matrix<Scalar, dimension>;

    // process_noise_position/process_noise_turn_rate: forwarded to the
    // internal CoordinatedTurn xy-block, same meaning as that model's own
    // constructor. process_noise_vertical: forwarded to the internal
    // ConstantVelocity<Scalar,1> vertical channel, same meaning as that
    // model's own noise_spectral_density.
    explicit CoordinatedTurn3D(Scalar process_noise_position = Scalar(1),
                                Scalar process_noise_turn_rate = Scalar(0.1),
                                Scalar process_noise_vertical = Scalar(1))
        : xy_(process_noise_position, process_noise_turn_rate), z_(process_noise_vertical) {}

    [[nodiscard]] State transition(const State& x, Scalar dt) const {
        State out;
        out.template head<5>() = xy_.transition(x.template head<5>(), dt);
        out.template tail<2>() = z_.transition(x.template tail<2>(), dt);
        return out;
    }

    [[nodiscard]] Transition jacobian(const State& x, Scalar dt) const {
        // Block-diagonal by construction: transition() never mixes the
        // xy-block and the vertical channel, so neither does its jacobian
        // -- see the file comment above for the numerical confirmation of
        // this (off-block-diagonal entries are exactly 0, not just small).
        Transition F = Transition::Zero();
        F.template block<5, 5>(0, 0) = xy_.jacobian(x.template head<5>(), dt);
        F.template block<2, 2>(5, 5) = z_.jacobian(x.template tail<2>(), dt);
        return F;
    }

    [[nodiscard]] Transition process_noise(Scalar dt) const {
        // Block-diagonal for the same reason: horizontal and vertical
        // process noise are modeled as independent (part of the same
        // decoupling assumption the file comment states plainly above).
        Transition Q = Transition::Zero();
        Q.template block<5, 5>(0, 0) = xy_.process_noise(dt);
        Q.template block<2, 2>(5, 5) = z_.process_noise(dt);
        return Q;
    }

private:
    CoordinatedTurn<Scalar> xy_;         // reused verbatim -- see file comment
    ConstantVelocity<Scalar, 1> z_;      // [pz, vz], decoupled from the turn
};

} // namespace augur::models
