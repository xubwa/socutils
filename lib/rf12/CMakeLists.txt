# rf12: quadruple precision (__float128) real symmetric (generalized)
# eigensolver for socutils.rf12 (relativistic F12 reference code).
# Thin subdir CMake (pyscf/lib style); requires the GCC quadmath library.
add_library(rf12 SHARED qlinalg.c)
set_target_properties(rf12 PROPERTIES
  PREFIX "lib" OUTPUT_NAME "rf12"
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
target_link_libraries(rf12 quadmath m ${OPENMP_C_PROPERTIES})
