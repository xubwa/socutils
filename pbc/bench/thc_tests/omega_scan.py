'''Total K (SR + LR, vector convention) of level 3 with the non-robust CX SR part, against the exact total K
(exact SR at this omega by RSJK + all exact LR rows), for several omega; and the number of LR rows.'''
import os, sys, time, runpy, numpy as np
from pyscf.pbc import scf as pscf, gto as pgto
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import split_basis, RULES
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc.bench.hybrid_k import k_from_factors
from socutils.pbc import thc as thcmod
from socutils.pbc.thc import THCExchange
name, ac, fac = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
omegas = [float(x) for x in sys.argv[4].split(',')]
exact = bool(os.environ.get('EXACT'))
sysm = cells.from_cif(name) if name.endswith('.cif') else cells.SYSTEMS[name]
B = dict(runpy.run_path(RULES)['BASIS'])
full = cells.make_cell(sysm, 1e-10, nucmod='G', basis=B, max_memory=150000)
cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=split_basis(B, ac), max_memory=150000)
els = sorted(set(full.atom_symbol(i) for i in range(full.natm)))
aux = {e: auxgen_from_basis({e: B[e]}, contract=False, prune_lmax=False, threshold=1e-8)[e] for e in els}
dm0 = pscf.RHF(full).get_init_guess(key='minao')
T = np.linalg.lstsq(cell.pbc_intor('int1e_ovlp'), pgto.intor_cross('int1e_ovlp', cell, full), rcond=1e-12)[0]
dm = T @ dm0 @ T.T
nao = cell.nao
f = lambda K: T.T @ K @ T
ex = lambda K: -.25 * np.einsum('ij,ji', dm0, f(K))
for w in omegas:
    t = time.time()
    th = THCExchange(cell, aux, ac, omega=w, selector='coarse').build_common()
    th.build_thc(int(round(fac * len(th.smooth))))
    J3, C, Llr = th.J, th.C, th.Llr
    Pm = th.phimu
    sm = th.smooth
    K_ss = np.zeros((nao, nao))
    K_ss[np.ix_(sm, sm)] = Pm.T @ ((Pm @ dm[np.ix_(sm, sm)] @ Pm.T) * th.V) @ Pm
    pos = -np.ones(nao, dtype=int); pos[sm] = np.arange(len(sm))
    X = Pm[:, pos[th.ti[th.ss]]] * Pm[:, pos[th.tj[th.ss]]]
    Jt = np.zeros_like(J3); Jt[:, th.ss] = th.A @ X
    ew, U = np.linalg.eigh(th.M); Y = ((U * np.sqrt(np.maximum(ew, 0))) @ U.T) @ C
    Lss = Llr.copy(); Lss[:, th.cx] = 0.
    K_cx = k_from_factors(C, Jt, dm, nao) + k_from_factors(Jt, C, dm, nao) + k_from_factors(Y, Y, dm, nao)
    K_lr = k_from_factors(Llr, Llr, dm, nao) - k_from_factors(Lss, Lss, dm, nao)
    K3 = K_ss + K_cx + K_lr
    line = 'omega %.2f: LR rows %d, naux %d; level 3 (non-robust CX)' % (w, len(Llr), len(th.M))
    if exact:
        from socutils.pbc.bench.hybrid_k import exact_sr
        import socutils.pbc.bench.hybrid_k as hk
        hk.OMEGA = w
        K0 = exact_sr(full, dm0, 1e-8) + f(k_from_factors(Llr, Llr, dm, nao))
        d = f(K3) - K0
        line += ' vs exact total K: max %.2e, E_x %.2e Eh' % (abs(d).max(), -.25 * np.einsum('ij,ji', dm0, d))
    print(line + '  (%.0f s)' % (time.time() - t), flush=True)
