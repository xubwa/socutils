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
#   ccsdt   ->  lib/libccsdt_clib.so  (socutils.cc CCSDT kernels)
#   ao2mo   ->  lib/libnrr_opt.so     (socutils.lib.ao2mo.nrr_fast spinor AO2MO)
#   vhf     ->  lib/libsocvhf.so      (socutils.somf.x2cmp multi-centre J/K prescreen)
#   qdhf    ->  lib/libqdhf.so        (socutils.qdhf, quad precision HF; needs libquadmath)
#
# x2camf (the C implementation, import x2camf) and zquatev (Kramers-restricted
# spinor SCF) are separate repositories, built with ExternalProject into
# lib/deps/lib (libx2camf_c.so, libzquatev.so); see the end of this file.
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

add_subdirectory(ccsdt)
add_subdirectory(ao2mo)
add_subdirectory(vhf)

# --- external libraries: x2camf (C implementation) and zquatev ----------------
# Built from their own repositories with ExternalProject, as pyscf/lib does for
# libcint and libxc, with the BLAS/LAPACK found above, and installed into
# lib/deps/lib.  X2CAMF_SOURCE_DIR / ZQUATEV_SOURCE_DIR use a local checkout
# instead of cloning.
include(ExternalProject)
string(REPLACE ";" "|" _blas_pipe "${BLAS_LIBRARIES}")
string(REPLACE ";" "|" _lapack_pipe "${LAPACK_LIBRARIES}")
set(_ext_cmake_args
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
  "-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}"
  -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
  -DCMAKE_INSTALL_LIBDIR:PATH=lib
  -DBLAS_LIBRARIES=${_blas_pipe})
if(LAPACK_LIBRARIES)
  list(APPEND _ext_cmake_args -DLAPACK_LIBRARIES=${_lapack_pipe})
endif()

option(BUILD_X2CAMF "Download and build the x2camf C library" ON)
set(X2CAMF_GIT "https://github.com/xubwa/x2camf-c.git" CACHE STRING "x2camf (C) repository")
set(X2CAMF_VERSION "fcfbd1372a053f61da93b325606b384b1432eac0" CACHE STRING "x2camf (C) git tag or commit")
set(X2CAMF_SOURCE_DIR "" CACHE PATH "local x2camf (C) source tree instead of cloning")
option(BUILD_ZQUATEV "Download and build zquatev" ON)
# sunqm/zquatev with the CMake change of xubwa/zquatev cmake-blas-install (PR pending)
set(ZQUATEV_GIT "https://github.com/xubwa/zquatev.git" CACHE STRING "zquatev repository")
set(ZQUATEV_VERSION "f715cc3d7e11780291224a8f66ef067469318f7c" CACHE STRING "zquatev git tag or commit")
set(ZQUATEV_SOURCE_DIR "" CACHE PATH "local zquatev source tree instead of cloning")

foreach(_ext X2CAMF ZQUATEV)
  if(BUILD_${_ext})
    string(TOLOWER ${_ext} _name)
    if(${_ext}_SOURCE_DIR)
      set(_src SOURCE_DIR ${${_ext}_SOURCE_DIR})
    else()
      set(_src GIT_REPOSITORY ${${_ext}_GIT} GIT_TAG ${${_ext}_VERSION})
    endif()
    ExternalProject_Add(${_name}_ext
      ${_src}
      PREFIX ${PROJECT_BINARY_DIR}/deps
      INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
      LIST_SEPARATOR |
      CMAKE_ARGS ${_ext_cmake_args})
  endif()
endforeach()

option(ENABLE_QDHF "Quadruple precision HF kernels (requires libquadmath)" ON)
if(ENABLE_QDHF)
  include(CheckCSourceCompiles)
  set(CMAKE_REQUIRED_LIBRARIES quadmath)
  check_c_source_compiles("#include <quadmath.h>
int main() { __float128 x = sqrtq(2.Q); return x > 1 ? 0 : 1; }" HAVE_QUADMATH)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(HAVE_QUADMATH)
    add_subdirectory(qdhf)
  else()
    message(WARNING "libquadmath not found; socutils.qdhf is not built")
  endif()
endif()
