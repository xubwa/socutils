'''SPMD THC (route 1) spinor HF of a production cell: every rank runs this script; the THC build (mu blocks) and the
J/K of every SCF cycle are split over the ranks (mpi4py).  Without mpirun it is the serial program.

usage: mpirun -np N python -m mpi4py thc_mpi_scf.py CIF NX,NY,NZ BASIS_PY [ACTIVE_LABEL]
       env MAXMEM (MB per rank, default 100000), MAXCYC, CONV'''
import os, sys, time, runpy, numpy as np
from mpi4py import MPI
from pyscf import gto
from socutils.pbc.bench import cells
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc import spinor_scf, split_direct
from socutils.pbc.thc import THCJK
from socutils.scf import spinor_hf
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
cif, sc, basis_py = sys.argv[1], [int(x) for x in sys.argv[2].split(',')], sys.argv[3]
lab = sys.argv[4] if len(sys.argv) > 4 else None
sysm = cells.from_cif(cif)
if sc != [1, 1, 1]:
    sysm = cells.supercell(sysm, sc)
if lab:
    k, at = 0, []
    for e, x in sysm[1]:
        if e == 'Eu':
            e = lab if k == 0 else e
            k += 1
        at.append((e, x))
    sysm = (sysm[0], at)
B = dict(runpy.run_path(basis_py)['BASIS'])
cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=B, max_memory=int(os.environ.get('MAXMEM', 100000)))
aux = {}
for l_ in sorted(set(cell.atom_symbol(i) for i in range(cell.natm))):
    el = gto.mole._std_symbol(l_)
    aux[l_] = auxgen_from_basis({el: B[l_]}, contract=False, prune_lmax=False, threshold=1e-8)[el]
jk = THCJK(cell, aux, alpha_c=4., nmu_factor=16., nuc_zeta=spinor_scf.nuclear_zeta(cell), local_rcut=6.)
jk.stream = True
jk.verbose = rank == 0
jk.comm = comm if size > 1 else None
comm.Barrier(); t = time.time(); jk.build(); comm.Barrier(); tb = time.time() - t
if rank == 0:
    print('%d ranks x %s threads: %d atoms, nao %d; build %.0f s (%s)' % (size, os.environ.get('OMP_NUM_THREADS'), cell.natm,
          cell.nao, tb, ', '.join('%s %.0f' % kv for kv in jk.th.timings.items())), flush=True)
mf = split_direct.periodic(spinor_hf.SpinorSCF(cell), jk)
mf.conv_tol = float(os.environ.get('CONV', 1e-8))
mf.max_cycle = int(os.environ.get('MAXCYC', 50))
mf.verbose = 4 if rank == 0 else 0
times = []
gv = mf.get_veff
def timed(*a, **k):
    t0 = time.time(); v = gv(*a, **k); times.append(time.time() - t0)
    if rank == 0:
        print('  get_veff %d: %.0f s' % (len(times), times[-1]), flush=True)
    return v
mf.get_veff = timed
t = time.time(); e = mf.kernel(); ts = time.time() - t
es = comm.allgather(e)
if rank == 0:
    print('%d ranks: SCF %.0f s, %d get_veff %s s, E = %.10f (ranks agree to %.1e), converged %s'
          % (size, ts, len(times), [round(x) for x in times], e, max(es) - min(es), mf.converged), flush=True)
