#!/bin/sh
# spin-orbit counterparts of run35.sh (same lattice constants and k-meshes);
# restartable through the .rows files.  One process per volume, with a private
# scratch directory removed afterwards (the GDF files are ~10 GB each).
run() {   # A B a_min a_max npts nk
    i=0
    while [ $i -lt $5 ]; do
        a=$(python3 -c "import numpy; print('%.6f' % numpy.linspace($3, $4, $5)[$i])")
        export PYSCF_TMPDIR=$(mktemp -d /tmp/eos42.XXXXXX)
        python3 42-rocksalt_eos_so.py $1 $2 $a $a 1 $6 > /dev/null 2>> 42-rocksalt_eos_so.$1$2.err
        rm -rf "$PYSCF_TMPDIR"
        i=$((i + 1))
    done
    python3 42-rocksalt_eos_so.py $1 $2 $3 $4 $5 $6 > 42-rocksalt_eos_so.$1$2.k$6.out 2>&1   # all volumes from .rows
}
# NaCl (3^3) was run directly; K Cl 3^3 is kept in *.KCl.k3.*.  The meshes below
# are those of the spin-free runs (run35b.sh).
# run K Cl 6.10 6.80 8 2   (done: *.KCl.k2.*)
run Ag Cl 5.40 6.10 8 2
# run Rb I 7.10 7.90 8 2   (not needed for Table 8)
