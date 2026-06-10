cmake_minimum_required(VERSION 3.9)
project(x2camf_c CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(X2CAMF_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/x2camf"
    CACHE PATH "Location of the upstream x2camf C++ sources")

if(NOT EXISTS "${X2CAMF_SOURCE_DIR}/src/dhf_sph.cpp")
  message(FATAL_ERROR
    "Upstream x2camf sources not found at ${X2CAMF_SOURCE_DIR}. "
    "Run: git submodule update --init external/x2camf")
endif()

# Eigen: prefer the submodule shipped with upstream x2camf, fall back to a
# user-provided or system installation.
if(EXISTS "${X2CAMF_SOURCE_DIR}/eigen/Eigen/Core")
  set(EIGEN3_INCLUDE_DIR "${X2CAMF_SOURCE_DIR}/eigen")
elseif(NOT EIGEN3_INCLUDE_DIR)
  find_package(Eigen3 3.3 QUIET NO_MODULE)
  if(Eigen3_FOUND)
    get_target_property(EIGEN3_INCLUDE_DIR Eigen3::Eigen
                        INTERFACE_INCLUDE_DIRECTORIES)
  else()
    message(FATAL_ERROR
      "Eigen headers not found. Either initialize the eigen submodule "
      "inside external/x2camf or pass -DEIGEN3_INCLUDE_DIR=/path/to/eigen")
  endif()
endif()
message(STATUS "Using Eigen headers from ${EIGEN3_INCLUDE_DIR}")

set(X2CAMF_CORE_SRC
  ${X2CAMF_SOURCE_DIR}/src/dhf_sph_ca.cpp
  ${X2CAMF_SOURCE_DIR}/src/dhf_sph_pcc.cpp
  ${X2CAMF_SOURCE_DIR}/src/dhf_sph.cpp
  ${X2CAMF_SOURCE_DIR}/src/general.cpp
  ${X2CAMF_SOURCE_DIR}/src/int_sph_basic.cpp
  ${X2CAMF_SOURCE_DIR}/src/int_sph_gauge.cpp
  ${X2CAMF_SOURCE_DIR}/src/int_sph_gaunt.cpp
  ${X2CAMF_SOURCE_DIR}/src/int_sph.cpp
)

add_library(x2camf_c SHARED ${X2CAMF_CORE_SRC} c_api/x2camf_c.cpp)
target_include_directories(x2camf_c PRIVATE
  ${X2CAMF_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  c_api
)
target_compile_features(x2camf_c PRIVATE cxx_std_11)
set_target_properties(x2camf_c PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  PUBLIC_HEADER c_api/x2camf_c.h
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(x2camf_c PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS x2camf_c
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include
)
