'''Charge-constrained local fit of the CX pairs: C_p = M^-1 J_p + M^-1 q lam,  q^T C_p = S_p (the pair's charge).
Reports, for the unconstrained and the constrained fit: SR non-robust K vs robust level 2, and the LR of the CX blocks
(fitted: C^T M_LR C, C^T A_LR X) vs the exact LR rows.'''
import sys, runpy, numpy as np
from pyscf.pbc import scf as pscf, gto as pgto
from pyscf.gto.ft_ao import ft_ao
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import split_basis, RULES
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc.bench.hybrid_k import k_from_factors, _inv_metric
from socutils.pbc.thc import THCExchange
name, ac, fac = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
sysm = cells.from_cif(name) if name.endswith('.cif') else cells.SYSTEMS[name]
B = dict(runpy.run_path(RULES)['BASIS'])
full = cells.make_cell(sysm, 1e-10, nucmod='G', basis=B, max_memory=150000)
cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=split_basis(B, ac), max_memory=150000)
els = sorted(set(full.atom_symbol(i) for i in range(full.natm)))
aux = {e: auxgen_from_basis({e: B[e]}, contract=False, prune_lmax=False, threshold=1e-8)[e] for e in els}
dm0 = pscf.RHF(full).get_init_guess(key='minao')
T = np.linalg.lstsq(cell.pbc_intor('int1e_ovlp'), pgto.intor_cross('int1e_ovlp', cell, full), rcond=1e-12)[0]
dm = T @ dm0 @ T.T
th = THCExchange(cell, aux, ac, selector='coarse').build_common()
th.build_thc(int(round(fac * len(th.smooth))))
nao, J3, M, r = cell.nao, th.J, th.M, th.r
ti, tj = th.ti, th.tj
q = ft_ao(th.auxmol, np.zeros((1, 3)))[0].real                           # aux charges
Spair = cell.pbc_intor('int1e_ovlp')[ti, tj]                            # pair charges (Gamma)
M_lr = r.metric + r.c_g0 * np.outer(q, q) - r.metric_sr
M_lr = .5 * (M_lr + M_lr.T)
pos = -np.ones(nao, dtype=int); pos[th.smooth] = np.arange(len(th.smooth))
X = th.phimu[:, pos[ti[th.ss]]] * th.phimu[:, pos[tj[th.ss]]]
Jss = np.zeros_like(J3); Jss[:, th.ss] = J3[:, th.ss]
Jlr_ss = np.zeros_like(J3); Jlr_ss[:, th.ss] = th.A_lr @ X
ew, U = np.linalg.eigh(M); Mh = (U * np.sqrt(np.maximum(ew, 0))) @ U.T
ewl, Ul = np.linalg.eigh(M_lr); Mlh = (Ul * np.sqrt(np.maximum(ewl, 0))) @ Ul.T
Lss = th.Llr.copy(); Lss[:, th.cx] = 0.
K_rows = k_from_factors(th.Llr, th.Llr, dm, nao) - k_from_factors(Lss, Lss, dm, nao)
Y0 = Mh @ th.C
K_rob = k_from_factors(th.C, J3, dm, nao) + k_from_factors(J3, th.C, dm, nao) - k_from_factors(Y0, Y0, dm, nao)
# constrained C
pat = np.repeat([th.auxmol.bas_atom(i) for i in range(th.auxmol.nbas)], np.diff(th.auxmol.ao_loc_nr()))
pa, pb = th.aoat[ti], th.aoat[tj]
Cq = np.zeros_like(th.C)
for A in range(cell.natm):
    for Bt in range(A + 1):
        cols = np.where(th.cx & (((pa == A) & (pb == Bt)) | ((pa == Bt) & (pb == A))))[0]
        if len(cols) == 0:
            continue
        D = np.where((pat == A) | (pat == Bt))[0]
        v, e = _inv_metric(M[np.ix_(D, D)])
        Mi = (v / e) @ v.T
        c0 = Mi @ J3[np.ix_(D, cols)]
        mq = Mi @ q[D]
        lam = (Spair[cols] - q[D] @ c0) / (q[D] @ mq)
        Cq[np.ix_(D, cols)] = c0 + np.outer(mq, lam)
print('max |q^T C - S|: unconstrained %.2e, constrained %.2e' % (abs(q @ th.C - Spair)[th.cx].max(), abs(q @ Cq - Spair)[th.cx].max()))
f = lambda K: T.T @ K @ T
ex = lambda K: -.25 * np.einsum('ij,ji', dm0, f(K))
for lab, C in (('unconstrained', th.C), ('charge-constrained', Cq)):
    Y = Mh @ C
    K_non = k_from_factors(C, Jss, dm, nao) + k_from_factors(Jss, C, dm, nao) + k_from_factors(Y, Y, dm, nao)
    Yl = Mlh @ C
    K_lr = k_from_factors(C, Jlr_ss, dm, nao) + k_from_factors(Jlr_ss, C, dm, nao) + k_from_factors(Yl, Yl, dm, nao)
    print('%s: SR non-robust - robust: K max %.2e E_x %.2e | LR fitted - rows: K max %.2e E_x %.2e Eh'
          % (lab, abs(f(K_non - K_rob)).max(), ex(K_non - K_rob), abs(f(K_lr - K_rows)).max(), ex(K_lr - K_rows)), flush=True)
