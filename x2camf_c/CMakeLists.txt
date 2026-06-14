cmake_minimum_required (VERSION 3.5)
project (x2camf_c C)

# CMake build modelled on pyscf/lib/CMakeLists.txt so that this library
# slots into the PySCF build conventions when contributed upstream:
#   * Release/RELWITHDEBINFO default with -O3 -funroll-loops
#   * BLAS/LAPACK with optional vendor selection (-DBLA_VENDOR=OpenBLAS|MKL|...)
#   * OpenMP guarded by ENABLE_OPENMP (on by default)
#   * the shared object is placed next to its Python ctypes loader and uses
#     an $ORIGIN RPATH so it locates its dependencies at runtime
# Numerically-unsafe flags (e.g. -ffast-math) are deliberately avoided to
# preserve agreement with the reference Eigen implementation.

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()
set(CMAKE_VERBOSE_MAKEFILE OFF)

if (CMAKE_COMPILER_IS_GNUCC)
  if (UNIX AND NOT APPLE AND NOT DEFINED ENV{CMAKE_C_COMPILER})
    set(CMAKE_C_FLAGS "-Wl,--no-as-needed ${CMAKE_C_FLAGS}")
  endif()
endif()
set(CMAKE_C_FLAGS "-Wall ${CMAKE_C_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "-O3 -funroll-loops")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -funroll-loops -g")

# Architecture-specific overrides, as in pyscf (optional cmake.arch.inc)
if(EXISTS "${PROJECT_SOURCE_DIR}/cmake.arch.inc")
  include("${PROJECT_SOURCE_DIR}/cmake.arch.inc")
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# BLAS / LAPACK. Honour a user-provided BLAS_LIBRARIES (e.g. the common
# MKL single-dynamic-library form  -DBLAS_LIBRARIES=mkl_rt , or OpenBLAS),
# otherwise let find_package locate it; -DBLA_VENDOR=OpenBLAS (or
# Intel10_64lp, ...) selects a threaded vendor library, which also
# parallelizes the dense linear algebra (dgemm/dsyev/...) on top of the
# OpenMP integral loops.
set(_blas_user_provided TRUE)
if (NOT BLAS_LIBRARIES)
  set(_blas_user_provided FALSE)
  find_package(BLAS REQUIRED)
  message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
endif()
if (NOT LAPACK_LIBRARIES)
  if (_blas_user_provided)
    # An explicitly supplied BLAS is assumed self-contained: mkl_rt and
    # OpenBLAS both provide the LAPACK routines (dsyev/dgetrf/dgetri/dgesv)
    # we need, so do not drag in a separate reference LAPACK.
    set(LAPACK_LIBRARIES "")
    message(STATUS "Using user BLAS_LIBRARIES for LAPACK too: ${BLAS_LIBRARIES}")
  else()
    find_package(LAPACK REQUIRED)
    message(STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}")
  endif()
endif()

# OpenMP (parallelizes the 2e-integral and Fock-build loops)
option(ENABLE_OPENMP "Compile with OpenMP" ON)
set(OPENMP_C_PROPERTIES "")
if(ENABLE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(HAVE_OPENMP 1)
    set(OPENMP_C_PROPERTIES OpenMP::OpenMP_C)
  endif()
endif()

# Put the compiled library next to its ctypes loader (the x2camf package),
# mirroring pyscf placing its libraries in pyscf/lib. May be overridden on
# the command line (the repository Makefile does so).
if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/x2camf")
endif()

# RPATH so the loaded .so finds its (possibly bundled) dependencies, as in
# pyscf. $ORIGIN = the directory of libx2camf_c.so itself. These must be set
# before the target is created.
if (WIN32)
  #
elseif (APPLE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/deps/lib")
else ()
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/deps/lib")
endif()

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
set_target_properties(x2camf_c PROPERTIES PUBLIC_HEADER c_api/x2camf_c.h)
target_link_libraries(x2camf_c PRIVATE
  ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES} m)

install(TARGETS x2camf_c
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include
)
