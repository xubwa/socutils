#!/bin/bash
# Build the pure-C libx2camf_c.so directly (no cmake needed).
set -e
cd "$(dirname "$0")"
OUT=build_c
mkdir -p $OUT
CFLAGS="-std=gnu99 -fopenmp -O2 -fPIC -Wall -Icsrc -Ic_api"
SRC="csrc/x2c_mat.c csrc/x2c_general.c csrc/x2c_int_sph_basic.c csrc/x2c_int_sph.c \
     csrc/x2c_int_sph_gaunt.c csrc/x2c_int_sph_gauge.c csrc/x2c_dhf_sph.c \
     csrc/x2c_dhf_sph_ca.c csrc/x2c_dhf_sph_pcc.c c_api/x2camf_c.c"
for f in $SRC; do
  o=$OUT/$(basename ${f%.c}).o
  gcc $CFLAGS -c "$f" -o "$o"
done
gcc -shared -fopenmp $OUT/*.o -llapack -lblas -lm -o $OUT/libx2camf_c.so
echo "built $OUT/libx2camf_c.so"
