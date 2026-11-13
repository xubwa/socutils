# quartz THC route-1 SCF under MPI; env SC (supercell), EXX (ewald|stc), SYMM, SYMMPIV, NMU, MAXMEM
'''quartz supercells: E/cell with exxdiv ewald vs smoothed truncated Coulomb exchange (THC route 1, MPI)'''
import os, sys, time, runpy, numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import RULES
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc import spinor_scf, split_direct
from socutils.pbc.thc import THCJK
from socutils.scf import spinor_hf
sc = [int(x) for x in os.environ.get('SC', '1,1,1').split(',')]
exx = os.environ.get('EXX', 'ewald')
symm = os.environ.get('SYMM')
B = dict(runpy.run_path(RULES)['BASIS'])
sysm = cells.SYSTEMS['quartz']
if sc != [1, 1, 1]:
    sysm = cells.supercell(sysm, sc)
cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=B, max_memory=int(os.environ.get('MAXMEM', 25000)))
aux = {e: auxgen_from_basis({e: B[e]}, contract=False, prune_lmax=False, threshold=1e-8)[e] for e in ('O', 'Si')}
jk = THCJK(cell, aux, alpha_c=4., nmu_factor=float(os.environ.get('NMU', 16)), nuc_zeta=spinor_scf.nuclear_zeta(cell), local_rcut=6.)
jk.stream = True; jk.verbose = rank == 0; jk.comm = comm if size > 1 else None
if os.environ.get('SYMMPIV'):
    jk.symm_pivots = os.environ['SYMMPIV'] == '1'
if symm is not None:
    jk.symm_grid = symm == '1'
if exx == 'stc':
    jk.exx_trunc = 'auto'
t = time.time(); jk.build(); tb = time.time() - t
mf = split_direct.periodic(spinor_hf.SpinorSCF(cell), jk)
mf.exxdiv = exx
mf.conv_tol = 1e-9
mf.verbose = 4 if rank == 0 else 0
e = mf.kernel()
ncell = int(np.prod(sc))
if rank == 0:
    print('RESULT sc %s exxdiv %s: R_c %s bohr, E = %.10f, E/cell = %.10f, nu_G0 %.6f (build %.0f s), converged %s'
          % (sc, exx, jk.exx_trunc, e, e / ncell, mf.madelung(), tb, mf.converged), flush=True)
