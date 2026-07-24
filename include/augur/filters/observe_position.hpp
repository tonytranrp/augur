#pragma once
// augur/filters/observe_position.hpp
//
// The "observe position directly" measurement model, packaged once.
// Every example and test used to hand-build the same H matrix with the
// same three-line ritual (Zero(), then ones on the position diagonal);
// this header is that ritual as a named function, plus function-object
// forms of the same model for the nonlinear filters' measurement-
// callable slots.
//
// Why this is safe as a default: every built-in model lays out its
// position components FIRST in the state vector -- the same layout
// convention predict/query.hpp's position()/velocity() accessors and
// filters/sage_husa.hpp's direct-position assumption already
// rely on. One honest exception, stated here rather than discovered
// the hard way: CoordinatedTurn3D's state is
// [px, py, vx, vy, omega, pz, vz], so its pz lives at index 5 -- to
// observe 3D position from that model, hand-build H with H(2,5)=1
// (examples/17_coordinated_turn_3d does exactly this) instead of using
// this helper, which would select [px, py, omega].
//
// The matrix form feeds filters::KalmanFilter (linear measurement,
// z = H x + noise); the function-object forms feed the nonlinear
// filters (ExtendedKalmanFilter / UnscentedKalmanFilter /
// ParticleFilter), whose measurement slots take callables. The functor
// types are deliberately named structs rather than lambdas so they can
// also be spelled as the MeasurementFnT template argument -- the
// zero-std::function fast path those filters document.

#include "augur/math/backend.hpp"

namespace augur::filters {

// H selecting the first MeasDim state components: Zero with ones on
// the leading diagonal. The matrix every "measure position directly"
// example previously built by hand.
template <typename Scalar, int MeasDim, int StateDim>
[[nodiscard]] inline augur::math::Matrix<Scalar, MeasDim, StateDim> observe_position() {
    static_assert(MeasDim >= 1, "a measurement must have at least one component");
    static_assert(MeasDim <= StateDim,
                  "cannot observe more components than the state has");
    augur::math::Matrix<Scalar, MeasDim, StateDim> H =
        augur::math::Matrix<Scalar, MeasDim, StateDim>::Zero();
    for (int i = 0; i < MeasDim; ++i) {
        H(i, i) = Scalar(1);
    }
    return H;
}

// Same matrix, deduced from a filter alias so call sites don't repeat
// dimensions the filter type already knows:
//   using KF = filters::KalmanFilter<CV, 2>;
//   KF filter{model, x0, P0, filters::observe_position<KF>(), R};
template <typename FilterT>
    requires requires { typename FilterT::MeasurementMatrix; }
[[nodiscard]] inline typename FilterT::MeasurementMatrix observe_position() {
    return observe_position<typename FilterT::Scalar,
                            FilterT::MeasurementMatrix::RowsAtCompileTime,
                            FilterT::MeasurementMatrix::ColsAtCompileTime>();
}

// h(x) = first MeasDim components of x -- the callable twin of the
// matrix above, for ExtendedKalmanFilter / UnscentedKalmanFilter /
// ParticleFilter measurement slots.
template <typename Scalar, int MeasDim, int StateDim>
struct ObservePositionFn {
    [[nodiscard]] augur::math::Vector<Scalar, MeasDim>
    operator()(const augur::math::Vector<Scalar, StateDim>& x) const {
        return x.template head<MeasDim>();
    }
};

// dh/dx for the function above -- constant, exactly the
// observe_position() matrix (h is linear). ExtendedKalmanFilter's
// jacobian slot wants a callable of x, so this wraps the constant in
// one.
template <typename Scalar, int MeasDim, int StateDim>
struct ObservePositionJacobianFn {
    [[nodiscard]] augur::math::Matrix<Scalar, MeasDim, StateDim>
    operator()(const augur::math::Vector<Scalar, StateDim>&) const {
        return observe_position<Scalar, MeasDim, StateDim>();
    }
};

} // namespace augur::filters
