import cProfile, pstats, runpy, numpy as np
from pyscf.pbc import scf as pscf, gto as pgto
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import split_basis, RULES
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc.thc import THCExchange
from socutils.pbc.thc_gapw import GAPWJ
B = dict(runpy.run_path(RULES)['BASIS'])
ac = 4.
full = cells.make_cell(cells.SYSTEMS['quartz'], 1e-10, nucmod='G', basis=B, max_memory=150000)
cell = cells.make_cell(cells.SYSTEMS['quartz'], 1e-10, nucmod='G', basis=split_basis(B, ac), max_memory=150000)
aux = {e: auxgen_from_basis({e: B[e]}, contract=False, prune_lmax=False, threshold=1e-8)[e] for e in ('O', 'Si')}
dm0 = pscf.RHF(full).get_init_guess(key='minao')
T = np.linalg.lstsq(cell.pbc_intor('int1e_ovlp'), pgto.intor_cross('int1e_ovlp', cell, full), rcond=1e-12)[0]
dm = T @ dm0 @ T.T
th = THCExchange(cell, aux, ac, selector='kmeans', local_rcut=6.)
th.build_common()
g = GAPWJ(th, nrad=600)
import time
t = time.time(); J1 = g.get_j(dm); t1 = time.time() - t
t = time.time(); J2 = g.get_j(dm); t2 = time.time() - t
print('get_j first %.1f s, second (cached geometry) %.1f s, |J1 - J2| %.1e, cache %.2f GB'
      % (t1, t2, abs(J1 - J2).max(), getattr(g, '_geo_bytes', 0) / 1e9), flush=True)
pr = cProfile.Profile(); pr.enable(); g.get_j(dm); pr.disable()
pstats.Stats(pr).sort_stats('cumulative').print_stats(25)
