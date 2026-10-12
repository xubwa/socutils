# zgasci: full / GAS CI kernels over spinors (socutils.hci.gasci).  Plain C99
# with OpenMP and nothing else; thin subdir CMake (pyscf/lib style), the
# common configuration comes from the parent lib/CMakeLists.txt.
add_library(zgasci SHARED zgasci.c)
set_target_properties(zgasci PROPERTIES
  PREFIX "lib" OUTPUT_NAME "zgasci"
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
target_link_libraries(zgasci PUBLIC ${OPENMP_C_PROPERTIES} m)
