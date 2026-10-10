# qdhf: quadruple precision (__float128) Gaussian integrals (McMurchie-
# Davidson), gemm and Hermitian Jacobi eigensolver for socutils.qdhf.
# Thin subdir CMake (pyscf/lib style); requires the GCC quadmath library.
add_library(qdhf SHARED qmath.c qint.c)
set_target_properties(qdhf PROPERTIES
  PREFIX "lib" OUTPUT_NAME "qdhf"
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
target_link_libraries(qdhf quadmath m ${OPENMP_C_PROPERTIES})
