# zgasci: full / GAS CI kernels over spinors (socutils.hci.gasci).  Plain C99
# with OpenMP and BLAS (GEMM and AXPY of the BLAS kernel); thin subdir CMake
# (pyscf/lib style), the common configuration comes from the parent
# lib/CMakeLists.txt.
add_library(zgasci SHARED zgasci.c)
set_target_properties(zgasci PROPERTIES
  PREFIX "lib" OUTPUT_NAME "zgasci"
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
target_link_libraries(zgasci PUBLIC ${OPENMP_C_PROPERTIES} ${BLAS_LIBRARIES} m)
# Complex products without the NaN/Inf recovery of C99 Annex G (__muldc3).
include(CheckCCompilerFlag)
check_c_compiler_flag(-fcx-limited-range ZGASCI_HAS_CX_LIMITED_RANGE)
if(ZGASCI_HAS_CX_LIMITED_RANGE)
  target_compile_options(zgasci PRIVATE -fcx-limited-range)
endif()
