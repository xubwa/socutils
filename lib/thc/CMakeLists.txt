# thc: OpenMP kernels for the memory-bound steps of the route-1 THC exchange (socutils.pbc.thc): pair-indexed
# accumulation and packed-pair unpacking.  Standalone (no pyscf libraries).
add_library(socthc SHARED thc_k.c)
target_link_libraries(socthc PRIVATE ${OPENMP_C_PROPERTIES})
set_target_properties(socthc PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
