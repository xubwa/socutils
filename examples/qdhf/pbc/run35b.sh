#!/bin/sh
export PYTHONPATH=/home/user/socenv:/home/user/pyscf
python3 35-rocksalt_eos.py K Cl 6.10 6.80 8 3 > 35-rocksalt_eos.KCl.out 2>&1
python3 35-rocksalt_eos.py Ag Cl 5.40 6.10 8 3 > 35-rocksalt_eos.AgCl.out 2>&1
python3 35-rocksalt_eos.py Rb I 7.10 7.90 8 2 > 35-rocksalt_eos.RbI.out 2>&1
