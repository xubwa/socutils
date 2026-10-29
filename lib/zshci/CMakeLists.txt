# zshci: selected CI for spinor Hamiltonians.
#
# Unlike its siblings this is a program, not a ctypes library; it is run by
# socutils.hci.shci.  The executables go to lib/zshci/bin/:
#   zshci       threads only
#   zshci_mpi   built as well when an MPI C++ compiler is found
#   zshci_test  self-checks (lib/zshci/README.md)
#
# Needs C++17, OpenMP, HDF5 (the C library; a serial build is enough) and
# Eigen 3 (found through CMake, or -DEIGEN_ROOT=<directory containing Eigen/>).
# Like the other optional pieces it is skipped, with a notice, when a
# dependency is missing.  The stand-alone Makefile next to this file builds
# the same programs without CMake.

find_package(HDF5 COMPONENTS C)
find_package(OpenMP COMPONENTS CXX)
find_package(Eigen3 3.3 NO_MODULE QUIET)
if(NOT EIGEN3_FOUND AND EIGEN_ROOT)
  add_library(zshci_eigen INTERFACE)
  target_include_directories(zshci_eigen SYSTEM INTERFACE ${EIGEN_ROOT})
  add_library(Eigen3::Eigen ALIAS zshci_eigen)
  set(EIGEN3_FOUND TRUE)
endif()

if(NOT HDF5_FOUND OR NOT OpenMP_CXX_FOUND OR NOT EIGEN3_FOUND)
  message(STATUS "zshci is not built: it needs HDF5, OpenMP and Eigen 3 "
                 "(HDF5: ${HDF5_FOUND}, OpenMP: ${OpenMP_CXX_FOUND}, "
                 "Eigen: ${EIGEN3_FOUND})")
  return()
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fcx-limited-range ZSHCI_HAS_CX_LIMITED_RANGE)

# The FCI kernel (src/zgasci, C99) with BLAS if CMake finds one.
enable_language(C)
find_package(OpenMP COMPONENTS C)
find_package(BLAS)
add_library(zshci_zgasci OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/src/zgasci/zgasci.c
            ${CMAKE_CURRENT_SOURCE_DIR}/src/gas_dist.c)
target_include_directories(zshci_zgasci PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(zshci_zgasci PROPERTIES C_STANDARD 99)
target_compile_options(zshci_zgasci PRIVATE -O3 -funroll-loops)
if(ZSHCI_HAS_CX_LIMITED_RANGE)
  target_compile_options(zshci_zgasci PRIVATE -fcx-limited-range)
endif()
target_link_libraries(zshci_zgasci PRIVATE OpenMP::OpenMP_C)
if(NOT BLAS_FOUND)
  target_compile_definitions(zshci_zgasci PRIVATE ZGASCI_NO_BLAS)
endif()

file(GLOB ZSHCI_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)
list(REMOVE_ITEM ZSHCI_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/run_modes.cc)
set(ZSHCI_LIBRARY_SOURCES ${ZSHCI_SOURCES})
list(REMOVE_ITEM ZSHCI_LIBRARY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc)

# The drivers of the diag and select modes, one file for each mode, kind of
# integrals and width (src/run_modes.h), so that they compile in parallel.
set(ZSHCI_MAX_DET_WORDS 4 CACHE STRING
    "largest determinant width in 64-bit words")
set(ZSHCI_RUN_SOURCES)
foreach(mode select diag)
  foreach(kind complexspinor realspinor unrestricted spatial)
    foreach(width RANGE 1 ${ZSHCI_MAX_DET_WORDS})
      set(file ${CMAKE_CURRENT_BINARY_DIR}/run/${mode}_${kind}_${width}.cc)
      file(CONFIGURE OUTPUT ${file} CONTENT
           "#define ZSHCI_RUN_MODE_${mode}\n#define ZSHCI_RUN_KIND_${kind}\n#define ZSHCI_RUN_WIDTH ${width}\n#include \"run_modes.cc\"\n")
      list(APPEND ZSHCI_RUN_SOURCES ${file})
    endforeach()
  endforeach()
endforeach()
list(APPEND ZSHCI_SOURCES ${ZSHCI_RUN_SOURCES})

function(zshci_program name)
  add_executable(${name} ${ARGN} $<TARGET_OBJECTS:zshci_zgasci>)
  set_target_properties(${name} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
    INSTALL_RPATH_USE_LINK_PATH TRUE)
  # Complex products as the textbook formula; see the Makefile.
  if(ZSHCI_HAS_CX_LIMITED_RANGE)
    target_compile_options(${name} PRIVATE -fcx-limited-range)
  endif()
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_include_directories(${name} SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party ${HDF5_INCLUDE_DIRS})
  target_compile_definitions(${name} PRIVATE
    ZSHCI_MAX_DET_WORDS=${ZSHCI_MAX_DET_WORDS})
  target_link_libraries(${name} PRIVATE
    OpenMP::OpenMP_CXX Eigen3::Eigen ${HDF5_LIBRARIES} ${BLAS_LIBRARIES} m)
endfunction()

zshci_program(zshci ${ZSHCI_SOURCES})
zshci_program(zshci_test ${ZSHCI_LIBRARY_SOURCES}
              ${CMAKE_CURRENT_SOURCE_DIR}/test/zshci_test.cc)

find_package(MPI COMPONENTS CXX QUIET)
if(MPI_CXX_FOUND)
  zshci_program(zshci_mpi ${ZSHCI_SOURCES})
  target_compile_definitions(zshci_mpi PRIVATE ZSHCI_USE_MPI)
  target_link_libraries(zshci_mpi PRIVATE MPI::MPI_CXX)
else()
  message(STATUS "zshci_mpi is not built: no MPI C++ compiler found")
endif()
