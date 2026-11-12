# Kernels of the spinor CCSD amplitude equations (cc/fast/amps.py) and of its
# (T) correction (cc/fast/triples.py), from zccsdt (csrc/zccsd_kernels.c,
# csrc/zccsd_t.c).  Thin subdir CMake; flags, OpenMP and RPATH from lib/.
option(ZCCSD_NATIVE "Compile the CCSD kernels with -march=native" OFF)
add_library(zc_zccsd SHARED zccsd_kernels.c zccsd_t.c)
target_compile_options(zc_zccsd PRIVATE -O3 -fcx-limited-range)
if(ZCCSD_NATIVE)
  target_compile_options(zc_zccsd PRIVATE -march=native)
endif()
target_link_libraries(zc_zccsd PRIVATE ${OPENMP_C_PROPERTIES} m)
set_target_properties(zc_zccsd PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
