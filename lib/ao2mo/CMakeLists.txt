# nrr_opt: optimized spinor (j-spinor) AO->MO e1/e2 kernels (nrr_fast.py).
# Unlike the other subdirs, this target does not stand alone: it reuses
# AO2MOfill_nr_s4 / AO2MOmmm_r_iltj etc. straight from an already-installed
# pyscf (libao2mo/libnp_helper/libcvhf/libcint) instead of recompiling them,
# so it links against *that* pyscf rather than a bundled/self-contained
# library -- see NOTES.md.  The pyscf in use is whichever the configuring
# Python interpreter resolves, found the same way nrr_fast.py used to at
# runtime (``from pyscf import lib``), just moved to configure time.
#
# The parent lib/CMakeLists.txt's CMAKE_C_STANDARD 99 (inherited here) is
# what keeps nrr_ao2mo_opt.c's K&R-style `(*fmmm)()` declarations valid --
# their old meaning ("unspecified arguments") was only redefined by C23,
# which gcc >= 14 defaults to.

find_program(PYTHON_EXECUTABLE NAMES python3 python)
if(NOT PYTHON_EXECUTABLE)
  message(FATAL_ERROR "ao2mo/nrr_opt needs a Python interpreter (to locate pyscf)")
endif()

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c
          "from pyscf import lib; import os; print(os.path.dirname(os.path.abspath(lib.__file__)))"
  OUTPUT_VARIABLE PYSCF_LIB_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _pyscf_lib_rc)
if(NOT _pyscf_lib_rc EQUAL 0 OR NOT PYSCF_LIB_DIR)
  message(FATAL_ERROR "ao2mo/nrr_opt: could not import pyscf via ${PYTHON_EXECUTABLE} "
                       "(needed to link against its libao2mo/libnp_helper/libcvhf/libcint)")
endif()
message(STATUS "ao2mo/nrr_opt: linking against pyscf at ${PYSCF_LIB_DIR}")

set(PYSCF_DEPS_LIB_DIR ${PYSCF_LIB_DIR}/deps/lib)
find_library(PYSCF_AO2MO_LIB     NAMES ao2mo     PATHS ${PYSCF_LIB_DIR} NO_DEFAULT_PATH)
find_library(PYSCF_NP_HELPER_LIB NAMES np_helper PATHS ${PYSCF_LIB_DIR} NO_DEFAULT_PATH)
find_library(PYSCF_CVHF_LIB      NAMES cvhf      PATHS ${PYSCF_LIB_DIR} NO_DEFAULT_PATH)
find_library(PYSCF_CINT_LIB      NAMES cint      PATHS ${PYSCF_DEPS_LIB_DIR} NO_DEFAULT_PATH)
foreach(_var PYSCF_AO2MO_LIB PYSCF_NP_HELPER_LIB PYSCF_CVHF_LIB PYSCF_CINT_LIB)
  if(NOT ${_var})
    message(FATAL_ERROR "ao2mo/nrr_opt: could not find lib for ${_var} under "
                         "${PYSCF_LIB_DIR} / ${PYSCF_DEPS_LIB_DIR}")
  endif()
endforeach()
file(GLOB PYSCF_OPENBLAS_LIB
  "${PYSCF_LIB_DIR}/libopenblas*.so"
  "${PYSCF_DEPS_LIB_DIR}/libopenblas*.so")

add_library(nrr_opt SHARED nrr_ao2mo_opt.c)
target_include_directories(nrr_opt PRIVATE
  ${PYSCF_LIB_DIR} ${PYSCF_LIB_DIR}/deps/include)
target_link_libraries(nrr_opt PRIVATE
  ${PYSCF_AO2MO_LIB} ${PYSCF_NP_HELPER_LIB} ${PYSCF_CVHF_LIB} ${PYSCF_CINT_LIB}
  ${PYSCF_OPENBLAS_LIB} ${OPENMP_C_PROPERTIES})
set_target_properties(nrr_opt PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  INSTALL_RPATH "$ORIGIN:${PYSCF_LIB_DIR}:${PYSCF_DEPS_LIB_DIR}")

# The spinor AO->MO kernels as rewritten in zccsdt (used by cc/fast/nrr_fast.py)
add_library(zc_nrr_opt SHARED zc_nrr_ao2mo_opt.c)
target_include_directories(zc_nrr_opt PRIVATE
  ${PYSCF_LIB_DIR} ${PYSCF_LIB_DIR}/deps/include)
target_link_libraries(zc_nrr_opt PRIVATE
  ${PYSCF_AO2MO_LIB} ${PYSCF_NP_HELPER_LIB} ${PYSCF_CVHF_LIB} ${PYSCF_CINT_LIB}
  ${PYSCF_OPENBLAS_LIB} ${OPENMP_C_PROPERTIES})
set_target_properties(zc_nrr_opt PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  INSTALL_RPATH "$ORIGIN:${PYSCF_LIB_DIR}:${PYSCF_DEPS_LIB_DIR}")
