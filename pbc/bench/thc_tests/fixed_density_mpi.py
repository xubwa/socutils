'''fixed density E_J, E_K under MPI with fixed pivots (piv file): np1 stored phi vs npN grid split'''
import os, time, runpy, numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import RULES
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc import spinor_scf, split_direct
from socutils.pbc.thc import THCJK
from socutils.scf import spinor_hf
B = dict(runpy.run_path(RULES)['BASIS'])
cell = cells.make_cell(cells.SYSTEMS['quartz'], 1e-10, nucmod='G', basis=B, max_memory=30000)
aux = {e: auxgen_from_basis({e: B[e]}, contract=False, prune_lmax=False, threshold=1e-8)[e] for e in ('O', 'Si')}
jk = THCJK(cell, aux, alpha_c=4., nmu_factor=16., nuc_zeta=spinor_scf.nuclear_zeta(cell), local_rcut=6.)
jk.stream = True; jk.verbose = rank == 0; jk.comm = comm if size > 1 else None
jk.piv_file = 'fixdm_piv.npy'
if os.environ.get('ISDF_CUT'):
    jk.isdf_cut = float(os.environ['ISDF_CUT'])
if os.environ.get('GB'):
    jk.stream_gb = float(os.environ['GB'])
jk.build()
mf = split_direct.periodic(spinor_hf.SpinorSCF(cell), jk)
dm = mf.get_init_guess()
dm = comm.bcast(dm if rank == 0 else None, root=0)
vj, vk = mf.get_jk(cell, dm)
ej, ek = np.einsum('ij,ji', dm, vj).real * .5, np.einsum('ij,ji', dm, vk).real * .5
th = jk.th
if rank == 0:
    print('RESULT np %d: EJ %.10f EK %.10f |V| %.12e |A| %.12e |Alr| %.12e' % (size, ej, ek, np.linalg.norm(th.V),
          np.linalg.norm(th.A), np.linalg.norm(th.A_lr)), flush=True)
    np.savez('fixdm_np%d_%s.npz' % (size, os.environ.get('TAG', '')), V=th.V, A=th.A, A_lr=th.A_lr, Xi=getattr(th, 'Xi_lr', 0))
