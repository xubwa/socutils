'''Coarse point selection: column-by-column vs blocked pivoted Cholesky (and kmax), route 1 on quartz: pivot time,
K and E_x vs the default.'''
import time, runpy, numpy as np
from pyscf.pbc import scf as pscf
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import RULES
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc import spinor_scf
from socutils.pbc import thc
import sys
from pyscf import gto
# usage: pivot_check.py [CIF BASIS_PY]   (default quartz with the RULES basis)
if len(sys.argv) > 2:
    B = dict(runpy.run_path(sys.argv[2])['BASIS'])
    cell = cells.make_cell(cells.from_cif(sys.argv[1]), 1e-10, nucmod='G', basis=B, max_memory=150000)
else:
    B = dict(runpy.run_path(RULES)['BASIS'])
    cell = cells.make_cell(cells.SYSTEMS['quartz'], 1e-10, nucmod='G', basis=B, max_memory=150000)
els = sorted(set(cell.atom_symbol(i) for i in range(cell.natm)))
aux = {e: auxgen_from_basis({gto.mole._std_symbol(e): B[e]}, contract=False, prune_lmax=False,
                            threshold=1e-8)[gto.mole._std_symbol(e)] for e in els}
dm = pscf.RHF(cell).get_init_guess(key='minao')
ref = None
for blocked, co in ((False, 1), (False, 2), (True, 2), (False, 3), (True, 1)):
    orig = thc.THCExchange.build_common
    def bc(self, *a, **k):
        self.blocked_pivots, self.coarsen = blocked, co
        return orig(self, *a, **k)
    thc.THCExchange.build_common = bc
    jk = thc.THCJK(cell, aux, alpha_c=4., nmu_factor=16., nuc_zeta=spinor_scf.nuclear_zeta(cell), local_rcut=6.)
    jk.stream = True
    jk.build()
    thc.THCExchange.build_common = orig
    K = jk.get_jk(dm)[1]
    if ref is None:
        ref = K
    print('blocked %s coarsen %d: pivots %.0f s; K vs first (column-by-column, full grid) max %.2e, E_x %.2e'
          % (blocked, co, jk.th.timings['pivots'], abs(K - ref).max(), -.25 * np.einsum('ij,ji', dm, K - ref)), flush=True)
