#pragma once
// augur/reflect/backends/vector_backend.hpp
//
// Boost.PFR cannot reflect augur::math::Vector<Scalar,N> (== Eigen::Matrix
// <Scalar,N,1>) at all -- Eigen's matrix type has private storage and
// user-provided constructors, so it fails PFR's own aggregate-initializable
// static_assert outright. Verified ad hoc (per .claude/rules/testing.md) by
// compiling a throwaway boost::pfr::tuple_size_v<Eigen::Matrix<float,4,1>>
// instantiation: it fails to compile with "Boost.PFR: Type must be
// aggregate initializable." Since augur::reflect exists specifically to
// (de)serialize State<Dim,Scalar> (docs/ROADMAP.md, "Reflection-driven
// (de)serialization"), and State IS exactly this Eigen vector alias, PFR
// alone can never cover the library's own primary use case -- this backend
// closes that gap by reflecting a fixed-size Eigen column vector
// structurally (N contiguous Scalars), without going through PFR or
// requiring T to be an aggregate at all.
//
// Only ever included from descriptor.hpp, which picks this backend over
// PfrBackend whenever T matches EigenFixedVector below -- never include
// this header directly.

#include <array>
#include <concepts>
#include <cstddef>
#include <string_view>
#include <utility>

namespace augur::reflect::backends {

// Structural match for augur::math::Vector<Scalar,N>. That alias is a
// plain `using`, not a distinct class template (it just names
// Eigen::Matrix<Scalar,N,1>), so detection can't go through
// utils/type_traits.hpp's is_specialization_of the way augmented_layout.hpp
// does for actual template names -- instead this checks Eigen's own
// compile-time introspection members, which every Eigen::Matrix
// instantiation exposes regardless of alias. Excludes dynamic-size vectors
// (RowsAtCompileTime == Eigen::Dynamic == -1, so `> 0` is false) since
// those have no compile-time field_count, and excludes anything with more
// than one column (a Matrix, not a Vector).
template <typename T>
concept EigenFixedVector = requires {
    { T::RowsAtCompileTime } -> std::convertible_to<int>;
    { T::ColsAtCompileTime } -> std::convertible_to<int>;
} && T::ColsAtCompileTime == 1 && T::RowsAtCompileTime > 0;

namespace detail {
// Synthesizes "0", "1", ... field names for a Vector's N components,
// which have no real names to reflect (unlike a PFR-aggregate's actual
// member names) -- built once at compile time, same table-then-index
// shape as pfr_backend.hpp's make_field_names, so Descriptor::field_name()
// stays a plain array lookup regardless of which backend is active. 4
// chars covers up to 999 components, far beyond any realistic tracking
// state (the largest in this library, kAugmentedLayout, has 7).
constexpr std::array<char, 4> format_index(std::size_t index) {
    std::array<char, 4> buffer{'0', '\0', '\0', '\0'};
    if (index == 0) return buffer;
    std::array<char, 3> digits{};
    int count = 0;
    while (index > 0 && count < 3) {
        digits[static_cast<std::size_t>(count++)] = static_cast<char>('0' + (index % 10));
        index /= 10;
    }
    for (int i = 0; i < count; ++i) {
        buffer[static_cast<std::size_t>(i)] = digits[static_cast<std::size_t>(count - 1 - i)];
    }
    buffer[static_cast<std::size_t>(count)] = '\0';
    return buffer;
}

template <std::size_t... Is>
constexpr std::array<std::array<char, 4>, sizeof...(Is)> make_index_buffers(std::index_sequence<Is...>) {
    return {format_index(Is)...};
}
} // namespace detail

template <typename T>
struct VectorBackend {
    [[nodiscard]] static constexpr std::size_t field_count() {
        return static_cast<std::size_t>(T::RowsAtCompileTime);
    }

    [[nodiscard]] static std::string_view field_name(std::size_t index) {
        static constexpr auto buffers = detail::make_index_buffers(std::make_index_sequence<field_count()>{});
        return std::string_view(buffers[index].data());
    }

    template <typename Visitor>
    static void for_each_field(T& value, Visitor&& visitor) {
        for (std::size_t i = 0; i < field_count(); ++i) visitor(value(static_cast<int>(i)));
    }

    template <typename Visitor>
    static void for_each_field(const T& value, Visitor&& visitor) {
        for (std::size_t i = 0; i < field_count(); ++i) visitor(value(static_cast<int>(i)));
    }
};

} // namespace augur::reflect::backends
