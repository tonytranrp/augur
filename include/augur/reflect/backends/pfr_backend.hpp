#pragma once
// augur/reflect/backends/pfr_backend.hpp
//
// Boost.PFR-backed implementation of the Descriptor operations. This is
// pure C++14/17 aggregate-destructuring trickery -- no compiler
// extension, no code generation step -- so it works identically on
// desktop and on Android NDK today. This is the backend descriptor.hpp
// falls back to whenever AUGUR_HAS_STD_REFLECTION is 0 but
// AUGUR_HAS_PFR is 1.
//
// Only ever included from descriptor.hpp, and only compiled when PFR is
// actually available (checked via augur/core/config.hpp) -- never
// include this header directly.

#include "augur/core/config.hpp"

#if AUGUR_HAS_PFR

#include <array>
#include <boost/pfr.hpp>
#include <cstddef>
#include <string_view>
#include <utility>

namespace augur::reflect::backends {

namespace detail {
// boost::pfr::get_name<I, T>() needs I as a compile-time template
// parameter, not a runtime value -- this builds the full name table
// once, at compile time (one get_name<Is, T>() per field, expanded via
// the index_sequence pack), so field_name() below can do a plain
// runtime array index into an already-resolved table instead of
// needing a runtime-to-compile-time dispatch. This was a real bug in
// an earlier version of this file (see docs/ROADMAP.md item 12): it
// called get_name<0, T>() unconditionally regardless of the requested
// index, silently wrong for any index != 0.
template <typename T, std::size_t... Is>
constexpr std::array<std::string_view, sizeof...(Is)> make_field_names(std::index_sequence<Is...>) {
    return {boost::pfr::get_name<Is, T>()...};
}
} // namespace detail

template <typename T>
struct PfrBackend {
    [[nodiscard]] static constexpr std::size_t field_count() {
        return boost::pfr::tuple_size_v<T>;
    }

    [[nodiscard]] static std::string_view field_name(std::size_t index) {
        static constexpr auto names = detail::make_field_names<T>(std::make_index_sequence<boost::pfr::tuple_size_v<T>>{});
        return names[index];
    }

    template <typename Visitor>
    static void for_each_field(T& value, Visitor&& visitor) {
        boost::pfr::for_each_field(value, std::forward<Visitor>(visitor));
    }

    template <typename Visitor>
    static void for_each_field(const T& value, Visitor&& visitor) {
        boost::pfr::for_each_field(value, std::forward<Visitor>(visitor));
    }
};

} // namespace augur::reflect::backends

#endif // AUGUR_HAS_PFR
