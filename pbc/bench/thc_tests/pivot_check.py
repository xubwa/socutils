'''Coarse point selection: column-by-column vs blocked pivoted Cholesky (and kmax), route 1 on quartz: pivot time,
K and E_x vs the default.'''
import time, runpy, numpy as np
from pyscf.pbc import scf as pscf
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import RULES
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc import spinor_scf
from socutils.pbc import thc
B = dict(runpy.run_path(RULES)['BASIS'])
cell = cells.make_cell(cells.SYSTEMS['quartz'], 1e-10, nucmod='G', basis=B, max_memory=150000)
aux = {e: auxgen_from_basis({e: B[e]}, contract=False, prune_lmax=False, threshold=1e-8)[e] for e in ('O', 'Si')}
dm = pscf.RHF(cell).get_init_guess(key='minao')
ref = None
for blocked, kf in ((False, 20), (True, 20), (True, 16)):
    orig = thc.THCExchange.build_common
    def bc(self, *a, **k):
        self.blocked_pivots, self.kmax_factor = blocked, kf
        return orig(self, *a, **k)
    thc.THCExchange.build_common = bc
    jk = thc.THCJK(cell, aux, alpha_c=4., nmu_factor=16., nuc_zeta=spinor_scf.nuclear_zeta(cell), local_rcut=6.)
    jk.stream = True
    jk.build()
    thc.THCExchange.build_common = orig
    K = jk.get_jk(dm)[1]
    if ref is None:
        ref = K
    print('blocked %s kmax %d: pivots %.0f s; K vs column-by-column max %.2e, E_x %.2e'
          % (blocked, kf, jk.th.timings['pivots'], abs(K - ref).max(), -.25 * np.einsum('ij,ji', dm, K - ref)), flush=True)
