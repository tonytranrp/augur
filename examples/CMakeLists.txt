# examples/CMakeLists.txt
# Each example is a tiny standalone executable linked only against
# `augur` -- no example depends on another, so `cmake --build . --target
# imm_maneuvering_target` builds just the one you're looking at.

add_executable(basic_cv_tracking 01_basic_cv_tracking/main.cpp)
target_link_libraries(basic_cv_tracking PRIVATE augur::augur)

add_executable(imm_maneuvering_target 02_imm_maneuvering_target/main.cpp)
target_link_libraries(imm_maneuvering_target PRIVATE augur::augur)

add_executable(custom_plugin_model 03_custom_plugin_model/main.cpp)
target_link_libraries(custom_plugin_model PRIVATE augur::augur)

add_executable(heterogeneous_imm 04_heterogeneous_imm/main.cpp)
target_link_libraries(heterogeneous_imm PRIVATE augur::augur)

add_executable(current_statistical 05_current_statistical/main.cpp)
target_link_libraries(current_statistical PRIVATE augur::augur)

add_executable(sage_husa_adaptive 06_sage_husa_adaptive/main.cpp)
target_link_libraries(sage_husa_adaptive PRIVATE augur::augur)

add_executable(data_association 07_data_association/main.cpp)
target_link_libraries(data_association PRIVATE augur::augur)

add_executable(track_lifecycle 08_track_lifecycle/main.cpp)
target_link_libraries(track_lifecycle PRIVATE augur::augur)

add_executable(gm_phd 09_gm_phd/main.cpp)
target_link_libraries(gm_phd PRIVATE augur::augur)

add_executable(unscented_kalman 10_unscented_kalman/main.cpp)
target_link_libraries(unscented_kalman PRIVATE augur::augur)

add_executable(particle_filter 11_particle_filter/main.cpp)
target_link_libraries(particle_filter PRIVATE augur::augur)

add_executable(out_of_sequence 12_out_of_sequence/main.cpp)
target_link_libraries(out_of_sequence PRIVATE augur::augur)

add_executable(latency_compensation 13_latency_compensation/main.cpp)
target_link_libraries(latency_compensation PRIVATE augur::augur)

add_executable(sensor_fusion 14_sensor_fusion/main.cpp)
target_link_libraries(sensor_fusion PRIVATE augur::augur)

add_executable(reflection_serialization 15_reflection_serialization/main.cpp)
target_link_libraries(reflection_serialization PRIVATE augur::augur)
