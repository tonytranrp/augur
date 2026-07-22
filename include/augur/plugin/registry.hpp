#pragma once
// augur/plugin/registry.hpp
//
// OPT-IN, pay-only-if-you-use-it runtime plugin mechanism. Most users
// never need this file: augur::imm::Estimator<Filters...> and standalone
// filters::Filter usage are chosen at COMPILE time via templates, with
// zero indirection. This file exists for the harder case where the set
// of active models genuinely isn't known until runtime -- e.g. a config
// file says which models to mix, or (given this is meant for modding
// contexts) a separately-compiled mod DLL wants to hand a host
// application a new motion model without the host being recompiled
// against it. That requires type erasure, which costs a vtable
// indirection per call -- a deliberate, opt-in tradeoff, never the
// library's default path.

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "augur/filters/filter_concept.hpp"
#include "augur/math/backend.hpp"

namespace augur::plugin {

// Type-erased interface any runtime-registered filter is wrapped in.
// Deliberately minimal: predict/update/state/covariance only, matching
// filters::Filter's core surface (last_likelihood() omitted here since
// mode-mixing across a *runtime* set of models is a further roadmap
// item -- see docs/ROADMAP.md; this registry is for standalone runtime-
// selected filters today, not yet a runtime IMM).
//
// Dim and MeasDim are separate template parameters, not one reused for
// both -- update() takes a MeasDim-sized measurement, state()/
// covariance() return Dim-sized state, and those two are only equal in
// the degenerate case of observing the whole state directly. An earlier
// version of this file used Dim for both, which is a real, reproduced
// bug (docs/IMPROVEMENT_PLAN.md): registering a real
// KalmanFilter<ConstantVelocity<float,2>, MeasDim=2> (dimension=4) --
// the README's own teaser shape -- failed to compile at all. Mirrors
// filters::KalmanFilter<Model, MeasDim>'s own established pattern.
template <typename Scalar, std::size_t Dim, std::size_t MeasDim>
class IFilterBox {
public:
    virtual ~IFilterBox() = default;
    virtual void predict(Scalar dt) = 0;
    virtual void update(const augur::math::Vector<Scalar, static_cast<int>(MeasDim)>& z) = 0;
    [[nodiscard]] virtual const augur::math::Vector<Scalar, static_cast<int>(Dim)>& state() const = 0;
    [[nodiscard]] virtual const augur::math::Matrix<Scalar, static_cast<int>(Dim)>& covariance() const = 0;
};

// Wraps any concrete filters::Filter as an IFilterBox. This adapter is
// the only place a vtable enters the picture; the wrapped filter itself
// is still the same zero-overhead type used everywhere else in the
// library.
//
// measurement_dimension is DERIVED from F::Measurement's own compile-time
// row count, not a second template parameter the caller has to supply
// and keep in sync by hand -- F::Measurement is now a required part of
// filters::Filter itself (see filter_concept.hpp), so this can never
// mismatch what F::update() actually accepts.
template <filters::Filter F>
class FilterBox final : public IFilterBox<typename F::Scalar, F::dimension,
                                           static_cast<std::size_t>(F::Measurement::RowsAtCompileTime)> {
public:
    static constexpr std::size_t measurement_dimension = static_cast<std::size_t>(F::Measurement::RowsAtCompileTime);

    explicit FilterBox(F filter) : filter_(std::move(filter)) {}

    void predict(typename F::Scalar dt) override { filter_.predict(dt); }
    void update(const augur::math::Vector<typename F::Scalar, static_cast<int>(measurement_dimension)>& z) override {
        filter_.update(z);
    }
    [[nodiscard]] const augur::math::Vector<typename F::Scalar, static_cast<int>(F::dimension)>& state() const override {
        return filter_.state();
    }
    [[nodiscard]] const augur::math::Matrix<typename F::Scalar, static_cast<int>(F::dimension)>& covariance() const override {
        return filter_.covariance();
    }

private:
    F filter_;
};

// Name -> factory registry, so a runtime config ("use_model: coordinated_turn")
// or a mod DLL can instantiate a filter it names by string rather than
// by compile-time type.
template <typename Scalar, std::size_t Dim, std::size_t MeasDim>
class FilterRegistry {
public:
    using Box = IFilterBox<Scalar, Dim, MeasDim>;
    using Factory = std::function<std::unique_ptr<Box>()>;

    void register_factory(std::string name, Factory factory) {
        factories_[std::move(name)] = std::move(factory);
    }

    [[nodiscard]] std::unique_ptr<Box> create(const std::string& name) const {
        const auto it = factories_.find(name);
        return it != factories_.end() ? it->second() : nullptr;
    }

    [[nodiscard]] bool contains(const std::string& name) const {
        return factories_.contains(name);
    }

    // Enumeration/retraction: a real gap given this registry's own
    // stated use case (a mod DLL registering/retracting models at
    // runtime) -- docs/IMPROVEMENT_PLAN.md. Purely additive.
    [[nodiscard]] std::vector<std::string> names() const {
        std::vector<std::string> result;
        result.reserve(factories_.size());
        for (const auto& [name, factory] : factories_) result.push_back(name);
        return result;
    }

    bool unregister(const std::string& name) { return factories_.erase(name) > 0; }

private:
    std::unordered_map<std::string, Factory> factories_;
};

} // namespace augur::plugin
