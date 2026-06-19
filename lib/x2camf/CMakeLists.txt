# x2camf_c: pure-C reimplementation of the X2CAMF SOC integrals.
# Thin subdir CMake (pyscf/lib style); the common configuration (build type,
# flags, BLAS/LAPACK, OpenMP, RPATH) comes from the parent lib/CMakeLists.txt.
# Numerically-unsafe flags (e.g. -ffast-math) are avoided to preserve agreement
# with the reference Eigen implementation.
set(X2CAMF_C_SRC
  csrc/x2c_mat.c
  csrc/x2c_general.c
  csrc/x2c_int_sph_basic.c
  csrc/x2c_int_sph.c
  csrc/x2c_int_sph_gaunt.c
  csrc/x2c_int_sph_gauge.c
  csrc/x2c_dhf_sph.c
  csrc/x2c_dhf_sph_ca.c
  csrc/x2c_dhf_sph_pcc.c
  c_api/x2camf_c.c
)

add_library(x2camf_c SHARED ${X2CAMF_C_SRC})
target_include_directories(x2camf_c PRIVATE csrc c_api)
set_target_properties(x2camf_c PROPERTIES
  PUBLIC_HEADER c_api/x2camf_c.h
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
target_link_libraries(x2camf_c PRIVATE
  ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES} m)
