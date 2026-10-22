cmake_minimum_required(VERSION 3.12)
project(zmr C)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_STANDARD 99)
# -fcx-limited-range: plain formulas for complex products (no NaN recovery
# calls), which lets the loops vectorize
set(CMAKE_C_FLAGS_RELEASE "-O3 -funroll-loops -fcx-limited-range")
set(ZMR_ARCH "native" CACHE STRING "value of -march (empty: compiler default)")
if(ZMR_ARCH)
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=${ZMR_ARCH}")
endif()

find_package(OpenMP REQUIRED)

# the MKL single dynamic library that PySCF links to; threaded through the
# same OpenMP runtime as the kernels (GNU threading layer)
if(NOT MKL_RT_LIBRARY)
  find_library(MKL_RT_LIBRARY
    NAMES mkl_rt libmkl_rt.so.3 libmkl_rt.so.2 libmkl_rt.so.1
    HINTS ${PYTHON_PREFIX}/lib $ENV{CONDA_PREFIX}/lib $ENV{MKLROOT}/lib/intel64 $ENV{MKLROOT}/lib)
endif()
if(NOT MKL_RT_LIBRARY)
  message(FATAL_ERROR "libmkl_rt not found; pass -DMKL_RT_LIBRARY=/path/to/libmkl_rt.so")
endif()
message(STATUS "MKL: ${MKL_RT_LIBRARY}")
get_filename_component(MKL_LIB_DIR ${MKL_RT_LIBRARY} DIRECTORY)

add_library(zmr SHARED src/zmr.c)
target_link_libraries(zmr PRIVATE OpenMP::OpenMP_C ${MKL_RT_LIBRARY} m)
# resolve libmkl_rt and the OpenMP runtime from the same directory as PySCF does
set_target_properties(zmr PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  BUILD_RPATH "${MKL_LIB_DIR}"
  INSTALL_RPATH "${MKL_LIB_DIR}"
  LINK_FLAGS "-Wl,--disable-new-dtags")
