#pragma once
// augur/track/out_of_sequence.hpp
//
// docs/ROADMAP.md item 8, "Out-of-sequence measurement (OOSM)
// handling": correctly incorporate a detection that arrives after a
// later one has already been processed -- routine when detections
// cross an unreliable network link. Naively discarding the late
// measurement throws away real information; naively applying it at the
// CURRENT time (as if it were fresh) actively corrupts the filter's
// timeline. The standard fix is retrodiction: keep a bounded history of
// (timestamp, post-update state/covariance, measurement) snapshots,
// and when a late measurement arrives, roll the filter back to the
// snapshot just before its true timestamp and replay every step since
// -- including the late one, inserted at its correct chronological
// position -- forward to the present.
//
// Verified (ad hoc python3 + numpy, per .claude/rules/testing.md)
// before being written here: on a simple 3-measurement scenario where
// the middle measurement arrives last (as if lost and retransmitted),
// retrodiction differs meaningfully from both discarding it (~0.015
// state error) and misapplying it at the current time (~0.018 state
// error, plus a visibly worse velocity estimate) -- confirming this is
// worth implementing properly rather than a nicety.
//
// dt for every replayed step is recomputed from consecutive ABSOLUTE
// timestamps (this_entry.timestamp - previous_entry.timestamp) rather
// than stored separately -- this is what avoids needing special-cased
// "adjust the first replayed step's dt" logic: whether the "previous"
// entry is an original snapshot or the newly-inserted one, the
// subtraction is correct either way.
//
// Composes with an existing Filter (predict()/update()/set_state()),
// the same pattern filters/current_statistical_filter.hpp and
// filters/adaptive/sage_husa.hpp use, rather than duplicating filter
// math.

#include <cstddef>
#include "augur/filters/filter_concept.hpp"
#include "augur/utils/fixed_vector.hpp"

namespace augur::track {

template <filters::Filter Inner, std::size_t MaxHistory>
class OutOfSequenceEstimator {
public:
    using Scalar = typename Inner::Scalar;
    using Measurement = typename Inner::Measurement;
    using StateVector = typename Inner::StateVector;
    using StateCovariance = typename Inner::StateCovariance;

    explicit OutOfSequenceEstimator(Inner filter, Scalar initial_time = Scalar(0))
        : filter_(std::move(filter)), current_time_(initial_time) {
        history_.push_back(Entry{initial_time, filter_.state(), filter_.covariance(), Measurement::Zero()});
    }

    // Normal, in-sequence step.
    void step(Scalar dt, const Measurement& z) {
        filter_.predict(dt);
        filter_.update(z);
        current_time_ += dt;
        push_history(Entry{current_time_, filter_.state(), filter_.covariance(), z});
    }

    // z was actually measured at measurement_time (< current_time_).
    // Returns false (no-op) if measurement_time predates the oldest
    // retained snapshot -- MaxHistory must be sized for the worst
    // network lag this needs to tolerate, stated plainly rather than
    // silently doing the wrong thing on an out-of-range request.
    bool insert_out_of_sequence(Scalar measurement_time, const Measurement& z) {
        if (history_.empty() || measurement_time < history_[0].timestamp || measurement_time >= current_time_) {
            return false;
        }

        std::size_t insert_after = 0;
        for (std::size_t i = 0; i < history_.size(); ++i) {
            if (history_[i].timestamp <= measurement_time) insert_after = i; else break;
        }

        filter_.set_state(history_[insert_after].state, history_[insert_after].covariance);

        augur::utils::FixedVector<Entry, MaxHistory> to_replay;
        for (std::size_t i = insert_after + 1; i < history_.size(); ++i) to_replay.push_back(history_[i]);
        while (history_.size() > insert_after + 1) history_.swap_remove(history_.size() - 1);

        Scalar prev_t = history_[insert_after].timestamp;
        replay_one(measurement_time, z, prev_t);
        for (const auto& entry : to_replay) {
            replay_one(entry.timestamp, entry.measurement, prev_t);
        }
        return true;
    }

    [[nodiscard]] const StateVector& state() const { return filter_.state(); }
    [[nodiscard]] const StateCovariance& covariance() const { return filter_.covariance(); }
    [[nodiscard]] Scalar current_time() const { return current_time_; }
    [[nodiscard]] const Inner& filter() const { return filter_; }

private:
    struct Entry {
        Scalar timestamp = Scalar(0);
        StateVector state = StateVector::Zero();
        StateCovariance covariance = StateCovariance::Identity();
        Measurement measurement = Measurement::Zero(); // the measurement applied to reach this entry
    };

    void replay_one(Scalar timestamp, const Measurement& z, Scalar& prev_t) {
        filter_.predict(timestamp - prev_t);
        filter_.update(z);
        push_history(Entry{timestamp, filter_.state(), filter_.covariance(), z});
        prev_t = timestamp;
    }

    void push_history(const Entry& entry) {
        if (history_.full()) {
            for (std::size_t i = 1; i < history_.size(); ++i) history_[i - 1] = history_[i];
            history_[history_.size() - 1] = entry;
        } else {
            history_.push_back(entry);
        }
    }

    Inner filter_;
    Scalar current_time_;
    augur::utils::FixedVector<Entry, MaxHistory> history_;
};

} // namespace augur::track
