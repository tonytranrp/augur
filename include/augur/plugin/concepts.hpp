#pragma once
// augur/plugin/concepts.hpp
//
// The PRIMARY plugin mechanism in augur is structural typing via
// concepts, not this file: any type satisfying models::MotionModel or
// filters::Filter already works as a plugin, full stop -- no
// registration call, no base class, no vtable, no dependency on augur
// having ever heard of your type. That's what "open instead of
// hardcoded" means at zero runtime cost.
//
// This concept exists purely as a convenience label for generic code
// (docs, static_asserts, template constraints) that wants to say "any
// kind of augur plugin" without spelling out MotionModel-or-Filter
// every time. It adds no capability beyond the two concepts it unions.
//
// If you need plugins selected at RUNTIME (e.g. choosing which motion
// model to use from a config file, or a third-party mod injecting a
// new model into a host application it wasn't compiled against) --
// concepts can't help you there, since the compiler needs to know the
// type at compile time. See plugin/registry.hpp's FilterRegistry for
// that heavier, opt-in, pay-only-if-you-use-it mechanism.

#include "augur/filters/filter_concept.hpp"
#include "augur/models/model_concept.hpp"

namespace augur::plugin {

template <typename T>
concept PredictionPlugin = models::MotionModel<T> || filters::Filter<T>;

} // namespace augur::plugin
