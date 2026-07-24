#pragma once
// augur/augur.hpp
//
// Convenience "include everything solid" header. Pulls in the core +
// tested tier (config, concepts, math, the three built-in motion
// models, the linear/extended Kalman filters, and the IMM estimator)
// plus the plugin concept and reflection facade. Deliberately does NOT
// include the roadmap-tier stubs under track/, filters/sage_husa.hpp, or
// predict/latency_compensation.hpp -- pull those in explicitly once
// you've implemented them, so an accidental #include of an
// unimplemented stub isn't hiding behind "I just included augur.hpp".
//
// Prefer including only the specific headers you need in real code;
// this file is mainly for examples/ and quick prototyping.

#include "augur/core/concepts.hpp"
#include "augur/core/config.hpp"

#include "augur/math/backend.hpp"
#include "augur/math/interop_glm.hpp"

#include "augur/models/coordinated_turn.hpp"
#include "augur/models/constant_acceleration.hpp"
#include "augur/models/constant_velocity.hpp"
#include "augur/models/model_concept.hpp"

#include "augur/filters/extended_kalman.hpp"
#include "augur/filters/filter_concept.hpp"
#include "augur/filters/kalman.hpp"

#include "augur/imm/estimator.hpp"
#include "augur/imm/mixing.hpp"
#include "augur/imm/mode_matrix.hpp"

#include "augur/predict/query.hpp"

#include "augur/plugin/concepts.hpp"

#include "augur/reflect/descriptor.hpp"
#include "augur/reflect/has_reflection.hpp"
