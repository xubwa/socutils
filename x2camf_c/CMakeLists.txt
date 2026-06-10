cmake_minimum_required(VERSION 3.9)
project(x2camf_c C)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Pure C translation of the X2CAMF core (no C++, no Eigen, no pybind11).
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
set_target_properties(x2camf_c PROPERTIES
  C_STANDARD 99
  POSITION_INDEPENDENT_CODE ON
  PUBLIC_HEADER c_api/x2camf_c.h
)

find_package(LAPACK REQUIRED)
target_link_libraries(x2camf_c PRIVATE ${LAPACK_LIBRARIES} m)

find_package(OpenMP)
if(OpenMP_C_FOUND)
  target_link_libraries(x2camf_c PRIVATE OpenMP::OpenMP_C)
endif()

install(TARGETS x2camf_c
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include
)
