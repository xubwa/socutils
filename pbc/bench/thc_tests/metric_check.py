import runpy, numpy as np, time
from pyscf.df import addons as df_addons
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import split_basis, RULES
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc.bench.hybrid_k import build as rsdf_build
B = dict(runpy.run_path(RULES)['BASIS'])
cell = cells.make_cell(cells.SYSTEMS['quartz'], 1e-10, nucmod='G', basis=split_basis(B, 4.), max_memory=150000)
aux = {e: auxgen_from_basis({e: B[e]}, contract=False, prune_lmax=False, threshold=1e-8)[e] for e in ('O', 'Si')}
r = rsdf_build(cell, aux)
auxcell = df_addons.make_auxmol(cell, aux)
t = time.time()
with auxcell.with_range_coulomb(-0.5):
    M = auxcell.pbc_intor('int2c2e', hermi=1)
print('pbc_intor SR metric vs RSDF metric_sr: max %.2e (max %.2e), %.1f s' % (abs(M - r.metric_sr).max(), abs(M).max(), time.time() - t))
