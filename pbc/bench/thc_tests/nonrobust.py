'''K with the CX-CX block non-robust (C^T M C) and CX-SS one-sided (C^T J_SS), against the robust level 2, and the
exact SR K where cached (exact_<name>.npy in the cell basis).'''
import os, sys, runpy, numpy as np
from pyscf.pbc import scf as pscf, gto as pgto
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import split_basis, RULES
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc.bench.hybrid_k import _inv_metric, k_from_factors
from socutils.pbc.thc import THCExchange
name, ac = sys.argv[1], float(sys.argv[2])
sysm = cells.from_cif(name) if name.endswith('.cif') else cells.SYSTEMS[name]
B = dict(runpy.run_path(RULES)['BASIS'])
full = cells.make_cell(sysm, 1e-10, nucmod='G', basis=B, max_memory=150000)
cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=split_basis(B, ac), max_memory=150000)
els = sorted(set(full.atom_symbol(i) for i in range(full.natm)))
aux = {e: auxgen_from_basis({e: B[e]}, contract=False, prune_lmax=False, threshold=1e-8)[e] for e in els}
dm0 = pscf.RHF(full).get_init_guess(key='minao')
T = np.linalg.lstsq(cell.pbc_intor('int1e_ovlp'), pgto.intor_cross('int1e_ovlp', cell, full), rcond=1e-12)[0]
dm = T @ dm0 @ T.T
th = THCExchange(cell, aux, ac, selector='kmeans').build_common()
th.phi = None
nao, J3, C, M = cell.nao, th.J, th.C, th.M
v, e = _inv_metric(M)
W = (v / np.sqrt(e)).T
L = W @ J3
Ls = np.zeros_like(L); Ls[:, th.ss] = L[:, th.ss]
Kss = k_from_factors(Ls, Ls, dm, nao)
Jss = np.zeros_like(J3); Jss[:, th.ss] = J3[:, th.ss]
ew, U = np.linalg.eigh(M)
Mh = (U * np.sqrt(np.maximum(ew, 0))) @ U.T
Y = Mh @ C
K_rob = Kss + k_from_factors(C, J3, dm, nao) + k_from_factors(J3, C, dm, nao) - k_from_factors(Y, Y, dm, nao)
K_non = Kss + k_from_factors(C, Jss, dm, nao) + k_from_factors(Jss, C, dm, nao) + k_from_factors(Y, Y, dm, nao)
f = lambda K: T.T @ K @ T
d = f(K_non - K_rob)
print('%s alpha_c %g: non-robust CX-CX - robust level 2: K max %.2e, E_x %.2e Eh (max |K| %.1e)'
      % (name, ac, abs(d).max(), -.25 * np.einsum('ij,ji', dm0, d), abs(f(K_rob)).max()), flush=True)
ex = os.environ.get('EXACT')
if ex and os.path.exists(ex):
    K0 = np.load(ex)
    for lab, K in (('robust', K_rob), ('non-robust', K_non)):
        dd = f(K) - K0
        print('   %s vs exact SR (SR part, no LR): K max %.2e, E_x %.2e Eh' % (lab, abs(dd).max(), -.25 * np.einsum('ij,ji', dm0, dd)))
