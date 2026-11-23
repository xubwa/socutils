#!/usr/bin/env python
'''
THCJK in parallel with MPI (mpi4py).

    mpirun -np 4 python -m mpi4py 05-thc_mpi.py

Every rank runs the same script (SPMD).  Setting jk.comm before jk.build()
distributes the build and every J/K over the ranks: the uniform grid by rows,
the local fits by atom, the reciprocal-space sums by G vectors.  The ranks
then follow the same SCF; rank 0's orbitals are broadcast, so only rank 0
needs to print.

cell.max_memory is the budget of each rank.  Set the number of OpenMP threads
per rank with OMP_NUM_THREADS, and MKL_DYNAMIC=FALSE for reproducible runs.
'''
import os
import sys
import numpy
from mpi4py import MPI
from pyscf.pbc import gto
from pyscf.pbc.tools import pbc as pbctools
from socutils.scf import spinor_hf
from socutils.pbc import spinor_scf, split_direct
from socutils.pbc.thc import THCJK
from socutils.cd import accd

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

prim = gto.Cell()
prim.a = numpy.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) * 4.083 / 2
prim.atom = 'Li 0 0 0; H 2.0415 2.0415 2.0415'
prim.basis = 'cc-pvdz'
prim.exp_to_discard = 0.1
prim.nucmod = 'G'
prim.precision = 1e-10
prim.verbose = 0
prim.build()
cell = pbctools.super_cell(prim, [2, 2, 2])
cell.max_memory = 8000           # MB per rank

auxbasis = accd.accd_basis(prim, verbose=0)

jk = THCJK(cell, auxbasis, alpha_c=4., nmu_factor=16., nuc_zeta=spinor_scf.nuclear_zeta(cell), local_rcut=6.)
jk.comm = comm                   # before build()
jk.stream = True                 # block-wise build within the memory budget (the default path under MPI)
jk.verbose = rank == 0
jk.build()

mf = split_direct.periodic(spinor_hf.KRHF(cell), jk).x2camf()
mf.verbose = 4 if rank == 0 else 0
mf.stdout = sys.stdout if rank == 0 else open(os.devnull, 'w')
e = mf.kernel()

# every rank has the same energy; gather to check
energies = comm.gather(e)
if rank == 0:
    print('E(THC, %d ranks) = %.8f   (spread over the ranks %.1e)' % (comm.Get_size(), e, numpy.ptp(energies)))
