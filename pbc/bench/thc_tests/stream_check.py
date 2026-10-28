'''Streamed THC build (no xi storage) vs the full build on quartz, route 1: V, A, A_lr, Xi_lr and K.'''
import time, runpy, numpy as np
from pyscf.pbc import scf as pscf
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import RULES
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc import spinor_scf
from socutils.pbc.thc import THCJK
B = dict(runpy.run_path(RULES)['BASIS'])
cell = cells.make_cell(cells.SYSTEMS['quartz'], 1e-10, nucmod='G', basis=B, max_memory=150000)
aux = {e: auxgen_from_basis({e: B[e]}, contract=False, prune_lmax=False, threshold=1e-8)[e] for e in ('O', 'Si')}
jk = THCJK(cell, aux, alpha_c=4., nmu_factor=16., nuc_zeta=spinor_scf.nuclear_zeta(cell), local_rcut=6.)
jk.build()
th = jk.th
dm = pscf.RHF(cell).get_init_guess(key='minao')
ref = {k: getattr(th, k).copy() for k in ('V', 'A', 'A_lr', 'Xi_lr')}
K0 = jk.get_jk(dm)[1]
nmu = len(th.pts)
for nb in (None,):
    t = time.time(); th._build_thc_stream(th.pts, nb=nb); ts = time.time() - t
    print('stream nb %s: %.0f s (full build %s)' % (nb, ts, {k: round(v) for k, v in th.timings.items()}))
    for k, v in ref.items():
        print('  %-6s max %.2e (|ref| %.2e)' % (k, abs(getattr(th, k) - v).max(), abs(v).max()))
    K1 = jk.get_jk(dm)[1]
    print('  K max diff %.2e, E_x diff %.2e' % (abs(K1 - K0).max(), -.25 * np.einsum('ij,ji', dm, K1 - K0)), flush=True)
