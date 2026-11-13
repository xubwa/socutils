'''LR part of the CX-involving exchange from the fitted CX densities instead of the exact LR rows:
   CX-CX  C^T M_LR C,   CX-SS  C^T A_LR X (THC SS side),   against the exact LR rows (non-robust SR in both).'''
import sys, runpy, numpy as np
from pyscf.pbc import scf as pscf, gto as pgto
from pyscf.gto.ft_ao import ft_ao
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import split_basis, RULES
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc.bench.hybrid_k import k_from_factors
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
nao, C, r = cell.nao, th.C, th.r
q = ft_ao(th.auxmol, np.zeros((1, 3)))[0].real
M_lr = r.metric + r.c_g0 * np.outer(q, q) - r.metric_sr                  # aux LR metric (G != 0)
M_lr = .5 * (M_lr + M_lr.T)
pos = -np.ones(nao, dtype=int); pos[th.smooth] = np.arange(len(th.smooth))
X = th.phimu[:, pos[th.ti[th.ss]]] * th.phimu[:, pos[th.tj[th.ss]]]
Jlr_ss = np.zeros_like(th.J); Jlr_ss[:, th.ss] = th.A_lr @ X               # (P|q)_LR for SS q from THC
ew, U = np.linalg.eigh(M_lr)
Yl = ((U * np.sqrt(np.maximum(ew, 0))) @ U.T) @ C
K_fit = k_from_factors(C, Jlr_ss, dm, nao) + k_from_factors(Jlr_ss, C, dm, nao) + k_from_factors(Yl, Yl, dm, nao)
Lss = th.Llr.copy(); Lss[:, th.cx] = 0.
K_rows = k_from_factors(th.Llr, th.Llr, dm, nao) - k_from_factors(Lss, Lss, dm, nao)
d = T.T @ (K_fit - K_rows) @ T
print('%s alpha_c %g N_mu %d: LR of the CX blocks, fitted - exact rows: K max %.2e, E_x %.2e Eh (max |K_LR,CX| %.1e; M_LR eig min %.1e)'
      % (name, ac, len(th.pts), abs(d).max(), -.25 * np.einsum('ij,ji', dm0, d), abs(T.T @ K_rows @ T).max(), ew.min()), flush=True)
