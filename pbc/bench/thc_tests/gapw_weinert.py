'''GAPW J with the compensation charges matching every multipole of the local expansion (L = lmax, Weinert) and no
neighbour local potentials, vs the default (L = 4, neighbours on); quartz, J vs RSDF.'''
import time, runpy, numpy as np
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
th = THCExchange(cell, aux, ac, selector='kmeans').build_common()
Jr = th.r.get_jk(dm, hermi=1, with_k=False)[0]
for L, eta, nb in ((4, 2., True), (4, 2., False), (8, 2., False), (12, 2., False), (12, 3., False), (12, 1.5, False)):
    t = time.time(); g = GAPWJ(th, L=L, eta=eta, nrad=600); g.neighbours = nb; ts = time.time() - t
    g.get_j(dm)
    t = time.time(); J = g.get_j(dm); tj = time.time() - t
    print('L %2d eta %.1f neighbours %-5s: setup %.0f s, get_j %.1f s; vs RSDF max %.2e, E_J %.2e'
          % (L, eta, nb, ts, tj, abs(J - Jr).max(), .5 * np.einsum('ij,ji', dm, J - Jr)), flush=True)
