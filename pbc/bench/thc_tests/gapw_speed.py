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
th = THCExchange(cell, aux, ac, selector='kmeans').build_common()       # Bloch build: RSDF J reference
Jr = th.r.get_jk(dm, hermi=1, with_k=False)[0]
for nrad in (600,):
    t = time.time(); g = GAPWJ(th, nrad=nrad); ts = time.time() - t
    t = time.time(); J = g.get_j(dm); tj = time.time() - t
    t = time.time(); J2 = g.get_j(dm); print('second get_j (cached) %.1f s, |J - J2| %.1e' % (time.time() - t, abs(J - J2).max()))
    print('nrad %d: setup %.0f s, get_j %.0f s; vs RSDF max %.2e, E_J %.2e' % (nrad, ts, tj, abs(J - Jr).max(), .5 * np.einsum('ij,ji', dm, J - Jr)), flush=True)
