# tests/CMakeLists.txt
# Only configured when AUGUR_BUILD_TESTS=ON (see root CMakeLists.txt),
# which is also the only time Catch2 gets fetched via CPM -- consumers
# who just want the library never pay for a test framework dependency.

add_executable(augur_unit_tests
    unit/test_safe_inverse.cpp
    unit/test_plugin_registry.cpp
    unit/test_kalman_cv.cpp
    unit/test_constant_acceleration.cpp
    unit/test_coordinated_turn.cpp
    unit/test_singer.cpp
    unit/test_imm_mixing.cpp
    unit/test_heterogeneous_imm.cpp
    unit/test_predict_query.cpp
    unit/test_current_statistical.cpp
    unit/test_sage_husa.cpp
    unit/test_association.cpp
    unit/test_track_manager.cpp
    unit/test_gm_phd.cpp
    unit/test_unscented_kalman.cpp
    unit/test_particle_filter.cpp
    unit/test_out_of_sequence.cpp
    unit/test_latency_compensation.cpp
    unit/test_fusion.cpp
    unit/test_reflection_serialization.cpp
)
target_link_libraries(augur_unit_tests PRIVATE augur::augur Catch2::Catch2WithMain)
target_include_directories(augur_unit_tests PRIVATE unit)
augur_apply_strict_warnings(augur_unit_tests)

include(CTest)
add_test(NAME augur_unit_tests COMMAND augur_unit_tests)
