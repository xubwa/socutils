import time, runpy, numpy as np
from pyscf.pbc import scf as pscf
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import RULES, split_basis
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc.bench.hybrid_k import _inv_metric
from socutils.pbc import split_direct, spinor_scf
from socutils.pbc.thc import THCJK
B = dict(runpy.run_path(RULES)['BASIS'])
cell = cells.make_cell(cells.SYSTEMS['quartz'], 1e-10, nucmod='G', basis=B, max_memory=150000)
nz = spinor_scf.nuclear_zeta(cell)
aux = {e: auxgen_from_basis({e: B[e]}, contract=False, prune_lmax=False, threshold=1e-8)[e] for e in ('O', 'Si')}
dm0 = pscf.RHF(cell).get_init_guess(key='minao')
jk = THCJK(cell, aux, alpha_c=4., nmu_factor=12., nuc_zeta=nz).build()
th, T = jk.th, jk.T
dm = T @ dm0 @ T.T
sd = split_direct.SplitDirectJK(cell, cuts=4., nuc_zeta=nz).build()       # same split -> exact SR in this basis
ds = (sd.T @ dm0 @ sd.T.T)[None, None]
vj_sr = sd.rsjk._get_jk_sr(ds, hermi=1, kpts=np.zeros((1, 3)), with_j=True, with_k=False)[0][0, 0].real
J0 = sd.T.T @ vj_sr @ sd.T                                                # exact SR J, cell basis
ti, tj = th.ti, th.tj
d = (dm + dm.T)[ti, tj]; d[ti == tj] *= .5
def unpack(jp):
    J = np.zeros((th.nao, th.nao)); J[ti, tj] = jp; J[tj, ti] = jp; return T.T @ J @ T
def ej(J):
    return .5 * np.einsum('ij,ji', dm0, J)
Jr, M, C, Y = th.J, th.M, th.C, th.Y
v, e = _inv_metric(M)
L1 = (v / np.sqrt(e)).T @ Jr                                              # global SR fit vectors
J1 = unpack(L1.T @ (L1 @ d))
# level 2 SR: SS-SS global + CX robust (exact 3c)
ss = th.ss
Lss = np.zeros_like(L1); Lss[:, ss] = L1[:, ss]
J2 = unpack(Lss.T @ (Lss @ d) + C.T @ (Jr @ d) + Jr.T @ (C @ d) - Y.T @ (Y @ d))
# global CX in the same robust form (= level 1 check)
cxi = np.where(th.cx)[0]
Cg = np.zeros_like(C); Cg[:, cxi] = (v / e) @ (v.T @ Jr[:, cxi])
ew, U = np.linalg.eigh(M); Yg = ((U * np.sqrt(np.maximum(ew, 0))) @ U.T) @ Cg
J2g = unpack(Lss.T @ (Lss @ d) + Cg.T @ (Jr @ d) + Jr.T @ (Cg @ d) - Yg.T @ (Yg @ d))
# THC grid J (SR part): get_j minus the LR rows
Llr = th.Llr
Jlr = unpack(Llr.T @ (Llr @ d))
J3 = T.T @ th.get_j(dm) @ T - Jlr
J3t = T.T @ th.get_j_thc(dm) @ T - Jlr
for lab, J in (('level 1 (global fit)', J1), ('level 2 global-CX check', J2g), ('level 2 (local CX)', J2),
               ('level 3 grid J', J3), ('level 3 THC J', J3t)):
    print('%-26s SR J vs exact: E_J %.2e Eh, max %.2e' % (lab, ej(J - J0), abs(J - J0).max()), flush=True)
