# augur — Writing Your Own Model (Plugin Guide)

The plugin mechanism is a concept, not an API you call to "register"
anything. If your type has the right member functions, `augur` accepts it —
that's the entire mechanism. This guide walks through writing one from
scratch; the finished version is `examples/03_custom_plugin_model/main.cpp`.

## The contract

Your type needs to satisfy `augur::models::MotionModel`
(`include/augur/models/model_concept.hpp`), which requires:

```cpp
struct YourModel {
    using Scalar = /* float or double */;
    static constexpr std::size_t dimension = /* your state size */;

    // x_{k+1} = f(x_k, dt) — the deterministic dynamics
    auto transition(const State& x, Scalar dt) const -> State;

    // Jacobian of transition() w.r.t. x, evaluated at x. For a linear
    // model this is just your constant transition matrix (x is unused);
    // for a nonlinear model it genuinely depends on x.
    auto jacobian(const State& x, Scalar dt) const -> Transition;

    // Process noise covariance Q for a step of length dt.
    auto process_noise(Scalar dt) const -> Transition;
};
```

where `State = augur::math::Vector<Scalar, dimension>` and
`Transition = augur::math::Matrix<Scalar, dimension>`.

That's it. No inheritance, no header of augur's you need to modify, no
factory to register with. Check it compiles against the concept directly:

```cpp
static_assert(augur::models::MotionModel<YourModel>);
```

## Worked example: fixed-rate circular motion

A target moving in a circle at a **known**, fixed angular rate (as opposed to
`models::CoordinatedTurn`, which *estimates* an unknown turn rate as part of
the state) — a reasonable model for, say, a turret with a known patrol
rotation speed. State = `[px, py, vx, vy]`.

```cpp
template <typename ScalarT>
class FixedRateCircular {
public:
    using Scalar = ScalarT;
    static constexpr std::size_t dimension = 4;
    using State = augur::math::Vector<Scalar, dimension>;
    using Transition = augur::math::Matrix<Scalar, dimension>;

    FixedRateCircular(Scalar angular_rate, Scalar process_noise)
        : omega_(angular_rate), q_(process_noise) {}

    State transition(const State& x, Scalar dt) const { return build_transition(dt) * x; }
    Transition jacobian(const State&, Scalar dt) const { return build_transition(dt); }
    Transition process_noise(Scalar dt) const { return Transition::Identity() * (q_ * dt); }

private:
    Transition build_transition(Scalar dt) const {
        const Scalar c = std::cos(omega_ * dt), s = std::sin(omega_ * dt);
        Transition F = Transition::Identity();
        F(0, 2) = dt; F(1, 3) = dt;               // position += velocity * dt
        F(2, 2) = c;  F(2, 3) = -s;                // velocity rotates at the known fixed rate
        F(3, 2) = s;  F(3, 3) = c;
        return F;
    }
    Scalar omega_, q_;
};
```

Because `jacobian()` doesn't depend on `x` here (omega is a fixed constructor
parameter, not part of the state), this model is truly linear — the Jacobian
*is* the transition matrix. Drop it straight into `filters::KalmanFilter`:

```cpp
using KF = augur::filters::KalmanFilter<FixedRateCircular<float>, /*MeasDim=*/2>;
KF filter{FixedRateCircular<float>{0.8f, 0.1f}, x0, P0, H, R};
```

No augur header changed. No registration call happened.

## Mixing your model into an IMM

`imm::Estimator<Filters...>` requires every mixed filter to share `Scalar`
and state `dimension` (see `docs/ARCHITECTURE.md` §5 for why) — as long as
your model's dimension matches the others you want to mix it with, wrap it in
a `Filter` (e.g. `filters::KalmanFilter<YourModel, MeasDim>`) and hand it to
`Estimator` exactly like a built-in one:

```cpp
augur::imm::Estimator<KF, KF, BuiltInKF> tracker{
    your_filter, another_your_filter, built_in_filter,
    augur::imm::ModeMatrix<3, float>::uniform(0.95f)
};
```

## Writing your own `Filter` instead of just a model

If a Kalman-style recursive estimator isn't the right fit (say, you want a
custom smoother, or a domain-specific closed-form update), implement
`augur::filters::Filter` directly (`include/augur/filters/filter_concept.hpp`)
instead of going through `models::MotionModel` + `KalmanFilter` at all — the
required surface is `predict(dt)`, `update(z)`, `state()`, `covariance()`,
and `last_likelihood()` (the last one is what IMM mode-mixing uses to reweight
your filter against the others in the pack). Same rule applies: satisfy the
concept, and `imm::Estimator` accepts it.

## When you need runtime selection instead

Everything above is chosen at **compile time** via templates. If you
genuinely need to choose a model at runtime — a config file names which model
to use, or (relevant if you're building on top of this for a modding context)
a separately-compiled mod wants to hand a host application a new model
without the host being recompiled against it — see
`include/augur/plugin/registry.hpp`'s `FilterRegistry` instead. It's a
type-erased, name-keyed factory registry; it costs a vtable indirection per
call, which is why it's a separate, explicitly opt-in file rather than how
`imm::Estimator` works by default.
