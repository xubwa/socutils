# socdft: multi-collinear XC kernel (socutils.dft.noncollinear).  Standalone C,
# OpenMP-parallel over grid points; it consumes the eval_xc1 output of pyscf's
# libxc interface, so it does not link against pyscf.

add_library(socdft SHARED mcol.c xc_hess.c)
target_link_libraries(socdft PRIVATE ${OPENMP_C_PROPERTIES} m)
set_target_properties(socdft PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
