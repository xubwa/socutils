#!/bin/sh
# spin-orbit counterparts of run35.sh (same lattice constants and k-meshes);
# restartable through the .rows files
python3 42-rocksalt_eos_so.py K Cl 6.10 6.80 8 3 > 42-rocksalt_eos_so.KCl.out 2>&1
python3 42-rocksalt_eos_so.py Rb I 7.10 7.90 8 2 > 42-rocksalt_eos_so.RbI.out 2>&1
python3 42-rocksalt_eos_so.py Ag Cl 5.40 6.10 8 3 > 42-rocksalt_eos_so.AgCl.out 2>&1
