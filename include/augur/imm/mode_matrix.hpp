#pragma once
// augur/imm/mode_matrix.hpp
//
// The Markov mode-transition matrix Pi (N x N: Pi(i,j) = P(mode j at
// k | mode i at k-1)) that governs how quickly the IMM believes the
// target can switch between the models in the mix. Kept as its own
// small type rather than a raw Eigen::Matrix so `uniform()` can express
// "how sticky should mode membership be" as a single, tunable number.

#include <cstddef>
#include "augur/math/backend.hpp"

namespace augur::imm {

template <std::size_t NumModes, typename Scalar>
class ModeMatrix {
public:
    using Matrix = augur::math::Matrix<Scalar, static_cast<int>(NumModes)>;

    explicit ModeMatrix(Matrix transition) : transition_(std::move(transition)) {}

    // Every mode has `stay_probability` chance of persisting into the
    // next step, with the remainder split evenly across the other
    // (NumModes - 1) modes. This is the matrix most users reach for
    // first -- tune stay_probability down (e.g. 0.90 instead of 0.98)
    // if your target switches maneuvers faster than the default mix
    // seems to notice.
    [[nodiscard]] static ModeMatrix uniform(Scalar stay_probability) {
        static_assert(NumModes >= 1, "IMM requires at least one model");
        Matrix m = Matrix::Constant((Scalar(1) - stay_probability) / Scalar(NumModes - 1 == 0 ? 1 : NumModes - 1));
        for (std::size_t i = 0; i < NumModes; ++i) {
            m(static_cast<int>(i), static_cast<int>(i)) = stay_probability;
        }
        return ModeMatrix{m};
    }

    [[nodiscard]] const Matrix& matrix() const { return transition_; }
    [[nodiscard]] Scalar operator()(std::size_t from, std::size_t to) const {
        return transition_(static_cast<int>(from), static_cast<int>(to));
    }

private:
    Matrix transition_;
};

} // namespace augur::imm
