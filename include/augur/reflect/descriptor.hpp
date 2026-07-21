#pragma once
// augur/reflect/descriptor.hpp
//
// augur::reflect::Descriptor<T> is the ONE reflection API the rest of
// the library (and your code) should ever call. Underneath, it
// dispatches to whichever backend actually fits T:
//
//   T is a fixed-size augur::math::Vector<Scalar,N> (Eigen column vector)
//     -> reflect/backends/vector_backend.hpp, unconditionally. Boost.PFR
//        cannot reflect this T at all (verified ad hoc, per
//        .claude/rules/testing.md -- Eigen's matrix type isn't an
//        aggregate, so PFR's own static_assert rejects it outright), so
//        this path exists whether or not PFR is even available.
//   otherwise, AUGUR_HAS_STD_REFLECTION (C++26 P2996, when your toolchain has it)
//     -> not yet implemented here (no shipping compiler to test
//        against as of this writing -- see std_backend note below)
//   otherwise, AUGUR_HAS_PFR (Boost.PFR, works today incl. Android NDK)
//     -> reflect/backends/pfr_backend.hpp, for plain user-defined
//        aggregates (a snapshot struct, a track-summary struct, etc.)
//   neither
//     -> Descriptor<T> is still a valid type, but for_each_field/etc.
//        are compile errors with a clear static_assert message rather
//        than silently doing nothing.
//
// This mirrors the approach the Glaze JSON library ships today: P2996
// support is opt-in and, per Glaze's own docs, "as C++26 compilers
// mature and P2996 becomes widely available, it will become the
// preferred reflection mechanism" -- with the existing portable path
// continuing to work unchanged in the meantime. augur adopts the same
// stance rather than gating any feature on reflection support that
// doesn't exist on Android NDK yet.
//
// Used by: reflect/serialize.hpp's generic binary (de)serialization, in
// turn used by predict/latency_compensation.hpp's SnapshotBuffer -- see
// docs/ROADMAP.md, "Reflection-driven (de)serialization".

#include <string_view>
#include <type_traits>

#include "augur/reflect/backends/vector_backend.hpp"
#include "augur/reflect/has_reflection.hpp"

#if AUGUR_HAS_PFR
  #include "augur/reflect/backends/pfr_backend.hpp"
#endif

namespace augur::reflect {

template <typename T>
class Descriptor {
public:
#if AUGUR_HAS_PFR
    using Backend = std::conditional_t<backends::EigenFixedVector<T>,
                                        backends::VectorBackend<T>,
                                        backends::PfrBackend<T>>;
#else
    struct FallbackBackend {
        static constexpr std::size_t field_count() {
            static_assert(has_any_reflection_backend,
                          "No reflection backend available: add Boost.PFR via CPM "
                          "(see CMakeLists.txt) or build with a compiler that has "
                          "C++26 static reflection (P2996).");
            return 0;
        }
        static std::string_view field_name(std::size_t) {
            static_assert(has_any_reflection_backend,
                          "No reflection backend available: add Boost.PFR via CPM "
                          "(see CMakeLists.txt) or build with a compiler that has "
                          "C++26 static reflection (P2996).");
            return {};
        }
    };
    using Backend = std::conditional_t<backends::EigenFixedVector<T>,
                                        backends::VectorBackend<T>,
                                        FallbackBackend>;
#endif

    [[nodiscard]] static constexpr std::size_t field_count() { return Backend::field_count(); }

    // Only meaningful for a PFR-backed aggregate (returns the real member
    // name); for a Vector, VectorBackend synthesizes "0", "1", ... since
    // components have no real names to report.
    [[nodiscard]] static std::string_view field_name(std::size_t index) {
        return Backend::field_name(index);
    }

    template <typename Visitor>
    static void for_each_field(T& value, Visitor&& visitor) {
        Backend::for_each_field(value, std::forward<Visitor>(visitor));
    }

    template <typename Visitor>
    static void for_each_field(const T& value, Visitor&& visitor) {
        Backend::for_each_field(value, std::forward<Visitor>(visitor));
    }
};

} // namespace augur::reflect
