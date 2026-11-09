# socdft: multi-collinear XC kernel (socutils.dft.noncollinear).  Standalone C,
# OpenMP-parallel over grid points; it consumes the eval_xc1 output of pyscf's
# libxc interface, so it does not link against pyscf.

# vxc_rks.c calls dgemm through a pointer to the BLAS already in the process
# (pyscf's), so no BLAS is linked here
add_library(socdft SHARED mcol.c xc_hess.c fxc_multi.c vxc_rks.c gtoval.c)
# gtoval.c: -ffast-math so that the exp of its SIMD loops maps to glibc's libmvec
set_source_files_properties(gtoval.c PROPERTIES COMPILE_OPTIONS "-ffast-math;-mprefer-vector-width=512")
target_link_libraries(socdft PRIVATE ${OPENMP_C_PROPERTIES} mvec m)
set_target_properties(socdft PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
