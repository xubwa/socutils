# Top-level CMake for the socutils C/C++ libraries, modelled on
# pyscf/lib/CMakeLists.txt.
#
# It centralises the common build configuration -- build type, warning and
# optimisation flags, BLAS/LAPACK and OpenMP discovery, and an $ORIGIN RPATH --
# then delegates each shared library to its own subdirectory through
# add_subdirectory().  Following the PySCF convention, every library is emitted
# flat into this directory (socutils/lib/) and loaded from there by
# ``socutils.lib.load_library``:
#
#   x2camf  ->  lib/libx2camf_c.so    (import x2camf, the SOC integral backend)
#   ccsdt   ->  lib/libccsdt_clib.so  (socutils.cc CCSDT kernels)
#   zquatev ->  lib/libzquatev.so     (Kramers-restricted spinor SCF)
#
# The subdirectory CMakeLists.txt files are thin (just the target plus its
# link/output), exactly as in pyscf/lib, so they slot into pyscf/lib unchanged
# if contributed upstream.
cmake_minimum_required(VERSION 3.5)
project(socutils C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()
set(CMAKE_VERBOSE_MAKEFILE OFF)

if(CMAKE_COMPILER_IS_GNUCC)
  if(UNIX AND NOT APPLE AND NOT DEFINED ENV{LDFLAGS})
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed ${CMAKE_SHARED_LINKER_FLAGS}")
  endif()
endif()
set(CMAKE_C_FLAGS "-Wall ${CMAKE_C_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "-O3 -funroll-loops")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -funroll-loops -g")
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Architecture-specific overrides (optional), as in pyscf.
if(EXISTS "${PROJECT_SOURCE_DIR}/cmake.arch.inc")
  include("${PROJECT_SOURCE_DIR}/cmake.arch.inc")
endif()

# --- BLAS / LAPACK -----------------------------------------------------------
# Honour a user-supplied BLAS_LIBRARIES (e.g. -DBLAS_LIBRARIES=mkl_rt or
# OpenBLAS); otherwise discover one.  -DBLA_VENDOR=OpenBLAS|Intel10_64lp|...
# selects a threaded vendor library.
set(_blas_user_provided TRUE)
if(NOT BLAS_LIBRARIES)
  set(_blas_user_provided FALSE)
  find_package(BLAS REQUIRED)
  message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
endif()
if(NOT LAPACK_LIBRARIES)
  if(_blas_user_provided)
    # An explicitly supplied BLAS (mkl_rt, OpenBLAS) already provides LAPACK;
    # do not drag in a separate reference LAPACK.
    set(LAPACK_LIBRARIES "")
    message(STATUS "Using user BLAS_LIBRARIES for LAPACK too: ${BLAS_LIBRARIES}")
  else()
    find_package(LAPACK REQUIRED)
    message(STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}")
  endif()
endif()

# --- OpenMP ------------------------------------------------------------------
option(ENABLE_OPENMP "Compile with OpenMP" ON)
set(OPENMP_C_PROPERTIES "")
if(ENABLE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(HAVE_OPENMP 1)
    set(OPENMP_C_PROPERTIES OpenMP::OpenMP_C)
  endif()
endif()

# --- RPATH: each .so locates its (possibly bundled) deps next to itself ------
if(WIN32)
  # nothing
elseif(APPLE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/deps/lib")
else()
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/deps/lib")
endif()

add_subdirectory(x2camf)
add_subdirectory(ccsdt)
add_subdirectory(zquatev)
