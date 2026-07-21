#pragma once
// augur/utils/timing.hpp
//
// Small std::chrono wrapper for the two things augur and its examples
// keep needing: "how much wall time since last frame" and "how far
// ahead do I need to predict to compensate for X seconds of latency".
// Nothing platform-specific -- std::chrono::steady_clock is available
// and monotonic on every target this library cares about, desktop and
// Android alike.

#include <chrono>

namespace augur::utils {

template <typename Scalar>
class Stopwatch {
public:
    Stopwatch() : last_(Clock::now()) {}

    // Returns elapsed seconds since the previous call (or construction,
    // for the first call) and resets the internal marker. This is the
    // `dt` you feed straight into Estimator::predict(dt).
    [[nodiscard]] Scalar tick() {
        const auto now = Clock::now();
        const Scalar dt = std::chrono::duration<Scalar>(now - last_).count();
        last_ = now;
        return dt;
    }

    // Elapsed seconds since the last tick(), without resetting -- for
    // read-only "how stale is this" checks (e.g. deciding whether to
    // coast a track versus mark it lost; see track/track_manager.hpp).
    [[nodiscard]] Scalar peek() const {
        return std::chrono::duration<Scalar>(Clock::now() - last_).count();
    }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point last_;
};

} // namespace augur::utils
