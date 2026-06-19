# ccsdt_clib: signed antisymmetric pack/unpack + permutation kernels for the
# spinor CCSDT residual (socutils.cc).  Thin subdir CMake (pyscf/lib style);
# common configuration comes from the parent lib/CMakeLists.txt.
add_library(ccsdt_clib SHARED ccsdt_pack.c ccsdt_asym.c ccsdt_asym_pack.c)
set_target_properties(ccsdt_clib PROPERTIES
  PREFIX "lib" OUTPUT_NAME "ccsdt_clib"
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
target_link_libraries(ccsdt_clib PUBLIC ${OPENMP_C_PROPERTIES})
