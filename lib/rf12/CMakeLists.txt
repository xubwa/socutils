# rf12: quadruple precision (__float128) real symmetric (generalized)
# eigensolver for socutils.rf12 (relativistic F12 reference code).
# Thin subdir CMake (pyscf/lib style); requires the GCC quadmath library.
add_library(rf12 SHARED qlinalg.c)
set_target_properties(rf12 PROPERTIES
  PREFIX "lib" OUTPUT_NAME "rf12"
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
target_link_libraries(rf12 quadmath m ${OPENMP_C_PROPERTIES})

# rf12_f12: F12 derivative integral classes (stg/yp x ipvip1ipvip2, ipip1ipip2)
# and the bra-contraction driver.  Links the libcint that pyscf uses; that
# libcint must be built with -DWITH_F12=1 (see rf12/notes/stage2_derivation.md).
find_program(PYTHON_EXECUTABLE NAMES python3 python)
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c
          "from pyscf import lib; import os; print(os.path.join(os.path.dirname(os.path.abspath(lib.__file__)), 'deps', 'lib'))"
  OUTPUT_VARIABLE RF12_CINT_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_library(RF12_CINT_LIB NAMES cint PATHS ${RF12_CINT_DIR} NO_DEFAULT_PATH)
if(RF12_CINT_LIB)
  add_library(rf12_f12 SHARED f12ints.c)
  target_include_directories(rf12_f12 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(rf12_f12 PRIVATE ${RF12_CINT_LIB} ${OPENMP_C_PROPERTIES})
  set_target_properties(rf12_f12 PROPERTIES
    PREFIX "lib" OUTPUT_NAME "rf12_f12"
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    BUILD_RPATH ${RF12_CINT_DIR} INSTALL_RPATH ${RF12_CINT_DIR})
else()
  message(WARNING "rf12_f12: libcint of pyscf not found; socutils.rf12.ints4c unavailable")
endif()
