// tests/unit/test_predict_query.cpp
//
// predict/query.hpp had zero automated coverage before this file,
// despite docs/README.md and docs/ROADMAP.md both describing
// error_ellipse_2d() as "already implemented, not just an idea." Covers
// both that function and its docs/ROADMAP.md item 11 3D extension,
// error_ellipsoid_3d().

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "augur/augur.hpp"
#include "augur/predict/query.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("position/velocity extract the right sub-vectors", "[predict]") {
    augur::math::Vector<float, 6> state;
    state << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f; // e.g. a ConstantAcceleration<float,3> state
    const auto pos = augur::predict::position<float, 3, 6>(state);
    const auto vel = augur::predict::velocity<float, 3, 6>(state);
    REQUIRE_THAT(pos(0), WithinAbs(1.0f, 1e-6));
    REQUIRE_THAT(pos(2), WithinAbs(3.0f, 1e-6));
    REQUIRE_THAT(vel(0), WithinAbs(4.0f, 1e-6));
    REQUIRE_THAT(vel(2), WithinAbs(6.0f, 1e-6));
}

TEST_CASE("error_ellipse_2d matches hand-computed axes and orientation for a diagonal covariance", "[predict]") {
    augur::math::Matrix<float, 4> cov = augur::math::Matrix<float, 4>::Identity();
    cov(0, 0) = 1.0f; // x variance
    cov(1, 1) = 4.0f; // y variance -- bigger, so the major axis should point along +/-Y

    const auto ellipse = augur::predict::error_ellipse_2d<float, 4>(cov);

    REQUIRE_THAT(ellipse.semi_major, WithinAbs(2.0f, 1e-5)); // sqrt(4)
    REQUIRE_THAT(ellipse.semi_minor, WithinAbs(1.0f, 1e-5)); // sqrt(1)
    // Axis orientation has a 180-degree ambiguity (v and -v are the same
    // axis) -- check it's aligned with +/-Y (rotation of +/-pi/2 from +X),
    // not a specific sign.
    const float normalized = std::fmod(std::abs(ellipse.rotation_radians), 3.14159265f);
    REQUIRE_THAT(normalized, WithinAbs(3.14159265f / 2.0f, 1e-3f));
}

TEST_CASE("error_ellipse_2d reconstructs the original covariance block", "[predict]") {
    // A correlated (non-axis-aligned) 2x2 block -- checks the general
    // case, not just the diagonal one above.
    augur::math::Matrix<float, 4> cov = augur::math::Matrix<float, 4>::Identity() * 0.1f;
    cov(0, 0) = 3.0f;
    cov(1, 1) = 1.5f;
    cov(0, 1) = cov(1, 0) = 0.8f;

    const auto ellipse = augur::predict::error_ellipse_2d<float, 4>(cov);

    const float c = std::cos(ellipse.rotation_radians);
    const float s = std::sin(ellipse.rotation_radians);
    augur::math::Matrix<float, 2> R;
    R << c, -s, s, c;
    augur::math::Matrix<float, 2> D = augur::math::Matrix<float, 2>::Zero();
    D(0, 0) = ellipse.semi_major * ellipse.semi_major;
    D(1, 1) = ellipse.semi_minor * ellipse.semi_minor;
    const augur::math::Matrix<float, 2> reconstructed = R * D * R.transpose();
    const augur::math::Matrix<float, 2> block = cov.topLeftCorner<2, 2>();

    REQUIRE_THAT((reconstructed - block).norm(), WithinAbs(0.0f, 1e-4f));
}

TEST_CASE("error_ellipsoid_3d matches hand-computed axes for a diagonal covariance", "[predict]") {
    augur::math::Matrix<float, 6> cov = augur::math::Matrix<float, 6>::Identity();
    cov(0, 0) = 1.0f;
    cov(1, 1) = 4.0f;
    cov(2, 2) = 9.0f;

    const auto ellipsoid = augur::predict::error_ellipsoid_3d<float, 6>(cov);

    REQUIRE_THAT(ellipsoid.semi_axis_major, WithinAbs(3.0f, 1e-5)); // sqrt(9)
    REQUIRE_THAT(ellipsoid.semi_axis_mid, WithinAbs(2.0f, 1e-5));   // sqrt(4)
    REQUIRE_THAT(ellipsoid.semi_axis_minor, WithinAbs(1.0f, 1e-5)); // sqrt(1)
}

TEST_CASE("error_ellipsoid_3d's orientation is a proper rotation that reconstructs the covariance", "[predict]") {
    // A general (correlated, non-axis-aligned) SPD 3x3 block.
    augur::math::Matrix<float, 6> cov = augur::math::Matrix<float, 6>::Identity() * 0.05f;
    cov(0, 0) = 5.0f;
    cov(1, 1) = 2.0f;
    cov(2, 2) = 1.0f;
    cov(0, 1) = cov(1, 0) = 0.9f;
    cov(0, 2) = cov(2, 0) = -0.4f;
    cov(1, 2) = cov(2, 1) = 0.3f;

    const auto ellipsoid = augur::predict::error_ellipsoid_3d<float, 6>(cov);
    const auto& R = ellipsoid.orientation;

    // Orthonormal.
    REQUIRE_THAT((R.transpose() * R - augur::math::Matrix<float, 3>::Identity()).norm(), WithinAbs(0.0f, 1e-4f));
    // Proper rotation, not a reflection -- most debug-draw/quaternion
    // consumers expect determinant +1.
    REQUIRE_THAT(R.determinant(), WithinAbs(1.0f, 1e-3f));

    augur::math::Matrix<float, 3> D = augur::math::Matrix<float, 3>::Zero();
    D(0, 0) = ellipsoid.semi_axis_major * ellipsoid.semi_axis_major;
    D(1, 1) = ellipsoid.semi_axis_mid * ellipsoid.semi_axis_mid;
    D(2, 2) = ellipsoid.semi_axis_minor * ellipsoid.semi_axis_minor;
    const augur::math::Matrix<float, 3> reconstructed = R * D * R.transpose();
    const augur::math::Matrix<float, 3> block = cov.topLeftCorner<3, 3>();

    REQUIRE_THAT((reconstructed - block).norm(), WithinAbs(0.0f, 1e-3f));
}
