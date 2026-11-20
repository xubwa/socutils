# pbcdf: short-range (erfc) 2- and 3-centre integrals over explicit lattice
# images with per-shell-triple screening (socutils.pbc.rsdf).  Like ao2mo and
# vhf it links the libcint of an installed pyscf, located through the
# configuring Python interpreter.

find_program(PYTHON_EXECUTABLE NAMES python3 python)
if(NOT PYTHON_EXECUTABLE)
  message(FATAL_ERROR "pbcdf needs a Python interpreter (to locate pyscf)")
endif()
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c
          "from pyscf import lib; import os; print(os.path.dirname(os.path.abspath(lib.__file__)))"
  OUTPUT_VARIABLE PYSCF_LIB_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _pyscf_lib_rc)
if(NOT _pyscf_lib_rc EQUAL 0 OR NOT PYSCF_LIB_DIR)
  message(FATAL_ERROR "pbcdf: could not import pyscf via ${PYTHON_EXECUTABLE}")
endif()
set(PYSCF_DEPS_LIB_DIR ${PYSCF_LIB_DIR}/deps/lib)
find_library(PYSCF_CINT_LIB NAMES cint PATHS ${PYSCF_DEPS_LIB_DIR} NO_DEFAULT_PATH)
if(NOT PYSCF_CINT_LIB)
  message(FATAL_ERROR "pbcdf: could not find libcint under ${PYSCF_DEPS_LIB_DIR}")
endif()

add_library(socpbcdf SHARED sr_int.c)
target_include_directories(socpbcdf PRIVATE ${PYSCF_LIB_DIR}/deps/include)
target_link_libraries(socpbcdf PRIVATE ${PYSCF_CINT_LIB} ${OPENMP_C_PROPERTIES} m)
set_target_properties(socpbcdf PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  INSTALL_RPATH "$ORIGIN:${PYSCF_LIB_DIR}:${PYSCF_DEPS_LIB_DIR}")
