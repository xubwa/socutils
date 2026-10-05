# socvhf: prescreen functions for pyscf's relativistic direct-SCF drivers that
# skip the one-centre shell quartets (somf/x2cmp.py).  Like ao2mo/nrr_opt it
# wraps functions of an already-installed pyscf (libcvhf) instead of standing
# alone, so it locates that pyscf through the configuring Python interpreter.

find_program(PYTHON_EXECUTABLE NAMES python3 python)
if(NOT PYTHON_EXECUTABLE)
  message(FATAL_ERROR "vhf/socvhf needs a Python interpreter (to locate pyscf)")
endif()

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c
          "from pyscf import lib; import os; print(os.path.dirname(os.path.abspath(lib.__file__)))"
  OUTPUT_VARIABLE PYSCF_LIB_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _pyscf_lib_rc)
if(NOT _pyscf_lib_rc EQUAL 0 OR NOT PYSCF_LIB_DIR)
  message(FATAL_ERROR "vhf/socvhf: could not import pyscf via ${PYTHON_EXECUTABLE} "
                       "(needed to link against its libcvhf)")
endif()

find_library(PYSCF_CVHF_LIB NAMES cvhf PATHS ${PYSCF_LIB_DIR} NO_DEFAULT_PATH)
if(NOT PYSCF_CVHF_LIB)
  message(FATAL_ERROR "vhf/socvhf: could not find libcvhf under ${PYSCF_LIB_DIR}")
endif()

add_library(socvhf SHARED mc_screen.c)
target_include_directories(socvhf PRIVATE
  ${PYSCF_LIB_DIR} ${PYSCF_LIB_DIR}/deps/include)
target_link_libraries(socvhf PRIVATE ${PYSCF_CVHF_LIB})
set_target_properties(socvhf PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  INSTALL_RPATH "$ORIGIN:${PYSCF_LIB_DIR}:${PYSCF_LIB_DIR}/deps/lib")
