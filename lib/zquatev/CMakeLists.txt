cmake_minimum_required (VERSION 3.5)
project (zquatev CXX)

# Builds libzquatev as a ctypes-loadable shared object, modelled on
# x2camf_c/CMakeLists.txt (and pyscf/lib conventions) so it slots into the
# PySCF build conventions if contributed upstream:
#   * Release/RELWITHDEBINFO default with -O3
#   * BLAS/LAPACK with optional vendor selection (-DBLA_VENDOR=OpenBLAS|MKL|...)
#   * the shared object is placed next to its Python ctypes loader and uses
#     an $ORIGIN RPATH so it locates its dependencies at runtime
# Numerically-unsafe flags (e.g. -ffast-math) are deliberately avoided.
#
# The csrc/ sources are a verbatim copy of Toru Shiozaki's zquatev and remain
# under BSD-2-Clause (see LICENSE and the per-file headers); the capi/ shim is
# part of socutils (Apache-2.0).

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()
set(CMAKE_VERBOSE_MAKEFILE OFF)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# BLAS / LAPACK. Honour a user-provided BLAS_LIBRARIES (MKL single dynamic
# library -DBLAS_LIBRARIES=mkl_rt, OpenBLAS, ...); otherwise locate them.
# zquatev needs both BLAS (zgemm/zaxpy/...) and LAPACK (zheev/zlartg/...).
set(_blas_user_provided TRUE)
if (NOT BLAS_LIBRARIES)
  set(_blas_user_provided FALSE)
  find_package(BLAS REQUIRED)
  message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
endif()
if (NOT LAPACK_LIBRARIES)
  if (_blas_user_provided)
    # An explicitly supplied BLAS is assumed self-contained (mkl_rt / OpenBLAS
    # provide the LAPACK routines too).
    set(LAPACK_LIBRARIES "")
    message(STATUS "Using user BLAS_LIBRARIES for LAPACK too: ${BLAS_LIBRARIES}")
  else()
    find_package(LAPACK REQUIRED)
    message(STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}")
  endif()
endif()

# Put the compiled library next to its ctypes loader (the zquatev package),
# mirroring pyscf placing its libraries in pyscf/lib.
if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}")
endif()

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

set(ZQUATEV_SRC
  csrc/zquatev.cc
  csrc/blocked.cc
  csrc/unblocked.cc
  csrc/transpose.cc
  capi/zquatev_capi.cc
)

add_library(zquatev SHARED ${ZQUATEV_SRC})
target_include_directories(zquatev PRIVATE csrc capi)
target_link_libraries(zquatev PUBLIC ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
set_target_properties(zquatev PROPERTIES PREFIX "lib" OUTPUT_NAME "zquatev")
