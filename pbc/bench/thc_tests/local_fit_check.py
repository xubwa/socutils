'''Local (two-atom cluster, per image pair) fit of the CX pairs (pbc/thc_local.py) against the Gamma-Bloch-domain
fit of THCExchange (RSDF keep_raw): non-robust SR K of the CX blocks vs the robust level 2, for both.'''
import os, sys, time, runpy, numpy as np
from pyscf.pbc import scf as pscf, gto as pgto
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import split_basis, RULES
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc.bench.hybrid_k import k_from_factors
from socutils.pbc.thc import THCExchange
from socutils.pbc.thc_local import local_cx_fit
name, ac = sys.argv[1], float(sys.argv[2])
rcuts = [float(x) for x in sys.argv[3].split(',')]
rdoms = [float(x) for x in (sys.argv[4] if len(sys.argv) > 4 else '0').split(',')]
sysm = cells.from_cif(name) if name.endswith('.cif') else cells.SYSTEMS[name]
if os.environ.get('SUPERCELL'):
    sysm = cells.supercell(sysm, [int(x) for x in os.environ['SUPERCELL'].split(',')])
B = dict(runpy.run_path(RULES)['BASIS'])
full = cells.make_cell(sysm, 1e-10, nucmod='G', basis=B, max_memory=150000)
cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=split_basis(B, ac), max_memory=150000)
els = sorted(set(full.atom_symbol(i) for i in range(full.natm)))
aux = {e: auxgen_from_basis({e: B[e]}, contract=False, prune_lmax=False, threshold=1e-8)[e] for e in els}
dm0 = pscf.RHF(full).get_init_guess(key='minao')
T = np.linalg.lstsq(cell.pbc_intor('int1e_ovlp'), pgto.intor_cross('int1e_ovlp', cell, full), rcond=1e-12)[0]
dm = T @ dm0 @ T.T
th = THCExchange(cell, aux, ac, omega=float(os.environ.get('OMEGA', 0.5)), selector='kmeans').build_common()
th.phi = None
nao, J3, M = cell.nao, th.J, th.M
compact = np.ones(nao, bool); compact[th.smooth] = False
Jss = np.zeros_like(J3); Jss[:, th.ss] = J3[:, th.ss]
ew, U = np.linalg.eigh(M); Mh = (U * np.sqrt(np.maximum(ew, 0))) @ U.T
Y0 = Mh @ th.C
K_rob = k_from_factors(th.C, J3, dm, nao) + k_from_factors(J3, th.C, dm, nao) - k_from_factors(Y0, Y0, dm, nao)
f = lambda K: T.T @ K @ T
def k_non(C):
    Y = Mh @ C
    return k_from_factors(C, Jss, dm, nao) + k_from_factors(Jss, C, dm, nao) + k_from_factors(Y, Y, dm, nao)
d = f(k_non(th.C) - K_rob)
print('Bloch-domain fit, non-robust - robust: K max %.2e, E_x %.2e Eh' % (abs(d).max(), -.25 * np.einsum('ij,ji', dm0, d)), flush=True)
for rc, rd in [(a, b) for a in rcuts for b in rdoms]:
    C, st = local_cx_fit(cell, aux, compact, omega=th.omega, r_cut=rc, r_dom=rd)
    d = f(k_non(C) - K_rob)
    zero = ~np.any(C[:, th.cx] != 0, axis=0)
    print('local fit r_cut %.1f A r_dom %.1f A: non-robust - robust: K max %.2e, E_x %.2e Eh; CX columns without coefficients %d of %d; '
          '|C - C_Bloch| max %.1e' % (rc, rd, abs(d).max(), -.25 * np.einsum('ij,ji', dm0, d), zero.sum(), th.cx.sum(),
                                      abs(C - th.C).max()), flush=True)
