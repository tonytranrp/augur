#pragma once
// augur/math/interop_glm.hpp
//
// Game code already has positions/velocities as glm::vec3 -- forcing
// callers to hand-roll Eigen conversions at every call site is exactly
// the kind of friction that makes a "correct" library go unused. This
// header is the ONLY place that ever mentions glm, and it only exists
// at all when the consumer opted in via CMake (AUGUR_WITH_GLM). If you
// don't use glm, this header is never even parsed.

#include "augur/core/config.hpp"

#if AUGUR_WITH_GLM

#include <glm/glm.hpp>
#include "augur/math/backend.hpp"

namespace augur::math::glm_interop {

template <typename Scalar>
[[nodiscard]] inline Vector<Scalar, 3> to_augur(const glm::vec<3, Scalar>& v) {
    return Vector<Scalar, 3>{v.x, v.y, v.z};
}

template <typename Scalar>
[[nodiscard]] inline glm::vec<3, Scalar> to_glm(const Vector<Scalar, 3>& v) {
    return glm::vec<3, Scalar>{v(0), v(1), v(2)};
}

} // namespace augur::math::glm_interop

#endif // AUGUR_WITH_GLM
