# benchmarks/CMakeLists.txt
# Only configured when AUGUR_BUILD_BENCHMARKS=ON (see root CMakeLists.txt),
# which is also the only time Google Benchmark gets fetched via CPM --
# consumers who just want the library never pay for a benchmarking
# framework dependency. One binary, like tests/'s own augur_unit_tests,
# rather than one executable per hot path -- Google Benchmark's own
# convention is a single binary registering many benchmarks, and
# `--benchmark_filter=` selects a subset at run time without needing
# separate build targets.

add_executable(augur_benchmarks
    bench_kalman.cpp
    bench_imm.cpp
    bench_heterogeneous.cpp
)
target_link_libraries(augur_benchmarks PRIVATE augur::augur benchmark::benchmark_main)
augur_apply_strict_warnings(augur_benchmarks)
