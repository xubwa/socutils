'''GAPW-style J (pbc/thc_gapw.py) against the exact J (SplitDirectJK: exact SR + exact LR rows) and RSDF, minao dm.'''
import os, sys, time, runpy, numpy as np
from pyscf.pbc import scf as pscf, gto as pgto
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import split_basis, RULES
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc.thc import THCExchange
from socutils.pbc.thc_gapw import GAPWJ
from socutils.pbc import split_direct, spinor_scf
from socutils.pbc.rsdf import RSDF
name, ac = sys.argv[1], float(sys.argv[2])
params = [tuple(float(v) for v in p.split(':')) for p in sys.argv[3].split(',')]    # L:eta:lmax[:nrad:nang]
sysm = cells.from_cif(name) if name.endswith('.cif') else cells.SYSTEMS[name]
B = dict(runpy.run_path(RULES)['BASIS'])
full = cells.make_cell(sysm, 1e-10, nucmod='G', basis=B, max_memory=150000)
cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=split_basis(B, ac), max_memory=150000)
nz = spinor_scf.nuclear_zeta(full)
els = sorted(set(full.atom_symbol(i) for i in range(full.natm)))
aux = {e: auxgen_from_basis({e: B[e]}, contract=False, prune_lmax=False, threshold=1e-8)[e] for e in els}
dm0 = pscf.RHF(full).get_init_guess(key='minao')
T = np.linalg.lstsq(cell.pbc_intor('int1e_ovlp'), pgto.intor_cross('int1e_ovlp', cell, full), rcond=1e-12)[0]
dm = T @ dm0 @ T.T
t = time.time()
sd = split_direct.SplitDirectJK(full, cuts=8., nuc_zeta=nz).build()
J0 = sd.get_jk(dm0, hermi=1, with_k=False)[0]
r = RSDF(full, aux, omega=0.5, precision=1e-10, nuc_zeta=nz, lr='gspace'); r.cderi_in_memory = True; r.verbose = 0; r.build()
J1 = r.get_jk(dm0, hermi=1, with_k=False)[0]
ej = lambda J: .5 * np.einsum('ij,ji', dm0, J)
print('exact J %.0f s; RSDF J vs exact: E_J %.2e Eh, max %.2e' % (time.time() - t, ej(J1 - J0), abs(J1 - J0).max()), flush=True)
th = THCExchange(cell, aux, ac, selector='kmeans').build_common()
for p in params:
    L, eta, lmax = int(p[0]), p[1], int(p[2])
    nrad, nang = (int(p[3]), int(p[4])) if len(p) > 4 else (150, 590)
    t = time.time()
    g = GAPWJ(th, L=L, eta=eta, lmax=lmax, nrad=nrad, nang=nang)
    if p is params[0]:
        Yl = g.Y
        print('Ylm orthonormality on Lebedev: %.1e' % max(abs((y * g.wang) @ y.T - np.eye(len(y))).max() for y in Yl), flush=True)
    t1 = time.time()
    Jg = T.T @ g.get_j(dm) @ T
    print('GAPW L %d eta %.1f lmax %d grid %dx%d: E_J %.2e Eh, max %.2e (setup %.0f s, J %.0f s)'
          % (L, eta, lmax, nrad, nang, ej(Jg - J0), abs(Jg - J0).max(), t1 - t, time.time() - t1), flush=True)
