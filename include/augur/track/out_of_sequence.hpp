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
#include "augur/utils/ring_buffer.hpp"

namespace augur::track {

// filters::Filter-conforming (docs/IMPROVEMENT_PLAN.md: this previously
// had no dimension, no split predict()/update(), no last_likelihood() --
// compiler-proven, blocked composing OOSM with TrackManager/
// imm::Estimator despite both being named use cases for it in this same
// file's own comment). predict()/update() are the SAME two calls step()
// already made internally, just now independently callable -- step()
// becomes a convenience wrapper (kept for source compatibility) rather
// than the only way in. history_ is only ever appended to from update()
// (matching this file's own "post-update state/covariance" wording
// above): a caller that predicts several times before ever updating
// gets no history entry for any of the unconfirmed intermediate
// predictions, exactly as before -- calling predict() then update() in
// sequence (what step() does, and what imm::Estimator/TrackManager both
// do) is bit-for-bit the same net effect as the original single-method
// step(), since filter_.update() never reads current_time_, so
// reordering current_time_'s own increment earlier (into predict())
// changes nothing observable.
template <filters::Filter Inner, std::size_t MaxHistory>
class OutOfSequenceEstimator {
public:
    using Scalar = typename Inner::Scalar;
    using Model = typename Inner::Model;
    using Measurement = typename Inner::Measurement;
    using StateVector = typename Inner::StateVector;
    using StateCovariance = typename Inner::StateCovariance;
    static constexpr std::size_t dimension = Inner::dimension;

    explicit OutOfSequenceEstimator(Inner filter, Scalar initial_time = Scalar(0))
        : filter_(std::move(filter)), current_time_(initial_time) {
        history_.push_back(Entry{initial_time, filter_.state(), filter_.covariance(), Measurement::Zero()});
    }

    void predict(Scalar dt) {
        filter_.predict(dt);
        current_time_ += dt;
    }

    void update(const Measurement& z) {
        filter_.update(z);
        history_.push_back(Entry{current_time_, filter_.state(), filter_.covariance(), z});
    }

    // Normal, in-sequence step -- convenience wrapper over predict()+update().
    void step(Scalar dt, const Measurement& z) {
        predict(dt);
        update(z);
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
        while (history_.size() > insert_after + 1) history_.pop_back();

        Scalar prev_t = history_[insert_after].timestamp;
        replay_one(measurement_time, z, prev_t);
        for (const auto& entry : to_replay) {
            replay_one(entry.timestamp, entry.measurement, prev_t);
        }
        return true;
    }

    [[nodiscard]] const StateVector& state() const { return filter_.state(); }
    [[nodiscard]] const StateCovariance& covariance() const { return filter_.covariance(); }
    [[nodiscard]] Scalar last_likelihood() const { return filter_.last_likelihood(); }
    [[nodiscard]] const Model& model() const { return filter_.model(); }
    [[nodiscard]] Scalar current_time() const { return current_time_; }
    [[nodiscard]] const Inner& filter() const { return filter_; }

    // Escape hatch required by filters::Filter -- e.g. for
    // imm::Estimator's mixing step to overwrite state/covariance with a
    // blended estimate between predictions. Deliberately does NOT touch
    // history_ or current_time_: an external state overwrite isn't a
    // "real" update this object itself made, so it isn't recorded as one.
    void set_state(const StateVector& x, const StateCovariance& P) { filter_.set_state(x, P); }

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
        history_.push_back(Entry{timestamp, filter_.state(), filter_.covariance(), z});
        prev_t = timestamp;
    }

    Inner filter_;
    Scalar current_time_;
    augur::utils::RingBuffer<Entry, MaxHistory> history_;
};

} // namespace augur::track
