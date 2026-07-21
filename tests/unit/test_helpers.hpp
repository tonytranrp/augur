#pragma once
// tests/unit/test_helpers.hpp
//
// Shared test-only utilities. Deliberately NOT under include/augur/ --
// per .claude/rules/code-style.md's utils/ convention, this is test
// infrastructure with Catch2 and MotionModel-shaped assumptions baked
// in, not a domain-agnostic library helper.
//
// The main thing here is a generic finite-difference jacobian checker:
// comparing a model's analytic jacobian() against a numerical
// derivative of its own transition() is a strong, general regression
// test that would have caught the class of bug found in
// models/coordinated_turn.hpp's small-omega branch (see
// docs/ROADMAP.md, "Fixes found along the way") -- apply this to every
// MotionModel, not just the one that already had a bug found this way.

#include <cmath>
#include <cstddef>
#include <catch2/catch_test_macros.hpp>
#include "augur/models/model_concept.hpp"

namespace augur_test {

// eps and the default tolerance were chosen empirically for
// augur::Scalar=float via an ad hoc python3 sweep (per
// .claude/rules/testing.md, not saved as a project file): eps=1e-3
// keeps float32 rounding error well below the truncation-error
// crossover, and a correct analytic jacobian matches the resulting
// numerical one to within about 0.002 in practice across several
// MotionModels. The default tolerance below has several times that
// margin, while staying far below the ~0.03 error the original
// coordinated_turn.hpp bug produced at the exact state that exposed
// it -- loose enough not to be flaky, tight enough to still catch a
// real missing/wrong jacobian term.
template <augur::models::MotionModel Model>
void check_jacobian_matches_finite_difference(const Model& model,
                                               const typename Model::State& x,
                                               typename Model::Scalar dt,
                                               typename Model::Scalar tolerance = typename Model::Scalar(0.01)) {
    using Scalar = typename Model::Scalar;
    constexpr auto Dim = Model::dimension;
    constexpr Scalar eps = Scalar(1e-3);

    const auto F_analytic = model.jacobian(x, dt);
    for (std::size_t j = 0; j < Dim; ++j) {
        typename Model::State x_plus = x;
        typename Model::State x_minus = x;
        x_plus(static_cast<int>(j)) += eps;
        x_minus(static_cast<int>(j)) -= eps;
        // Explicit State type (not auto) is load-bearing here: transition()
        // returns by value, and assigning the difference-of-temporaries
        // expression to `auto` deduces an Eigen lazy expression template
        // holding references to those already-destroyed temporaries --
        // this was caught the hard way (nan/garbage values) while writing
        // this very check.
        const typename Model::State column =
            (model.transition(x_plus, dt) - model.transition(x_minus, dt)) / (Scalar(2) * eps);

        for (std::size_t i = 0; i < Dim; ++i) {
            const Scalar analytic = F_analytic(static_cast<int>(i), static_cast<int>(j));
            const Scalar numeric = column(static_cast<int>(i));
            INFO("jacobian(" << i << "," << j << "): analytic=" << analytic << " finite-diff=" << numeric);
            REQUIRE(std::abs(analytic - numeric) <= tolerance);
        }
    }
}

} // namespace augur_test
