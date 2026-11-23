"""U- 6d state at QZ, started from the converged TZ solution: the TZ (contracted) U- 6d spinors are
projected onto the QZ basis (contracted or uncontracted), orthonormalised, and used both as the
initial density and as the maximum-overlap reference per (l, j, m_j) block; damping 0.5.  The
neutral U at QZ by aufbau as before.  Masked constrained SCF as in ea_sph.py / mom_6d.py."""
import re, sys, os
import numpy as np, scipy.linalg
from pyscf import gto
from socutils.tools.basisopt import genbas
sys.argv = [sys.argv[0], 'Q']
exec(open('mom_6d.py').read().split("for name, src in (")[0].replace("OUT = open('mom_6d_%s.res' % z, 'w')", "OUT = None"))        # setup, mom, info, CLOSED, OPEN, OUT='mom_6d_Q.res'
OUT = open('mom2_6d_Q.res', 'w')
def say(*a):
    OUT.write(' '.join(str(x) for x in a) + '\n'); OUT.flush()
EXTRA = {**OPEN, ('d3/2', '1/2'): 1}
# 1. TZ contracted U- 6d (reference solution)
tz = genbas.read(G + 'GENBAS.thuof_aug', 'U:ADYVTZ-SOG')
molt0, mft0 = setup(genbas.to_pyscf(tz, kappa=True), 0, OPEN, 'mom2_TZ_U.log'); mft0.kernel()
tags = np.asarray(mft0.irrep_mo); o = abs(mft0.mo_occ) > 0.5
ref, nblock = {}, {}
for key in mft0.irrep_ao:
    idx = np.where(tags == key)[0]; occ_idx = idx[o[idx]]
    if key == 'd3/2,1/2':
        virt = idx[~o[idx]]; occ_idx = np.concatenate([occ_idx, [virt[np.argmin(mft0.mo_energy[virt].real)]]])
    if len(occ_idx): ref[key] = mft0.mo_coeff[:, occ_idx]; nblock[key] = len(occ_idx)
molt, mft = setup(genbas.to_pyscf(tz, kappa=True), -1, EXTRA, 'mom2_TZ_Um2.log')
mft = mom(mft, ref, nblock); et = mft.kernel(mft0.make_rdm1())
say('TZ contracted U-2 (6d) E %.8f converged %s  %s' % (et, mft.converged, info(mft, ['d3/2,1/2', 'd3/2,3/2'])))
tags_t = np.asarray(mft.irrep_mo); ot = abs(mft.mo_occ) > 0.5
qz = genbas.read(G + 'GENBAS.thuof_aug', 'U:ADYVQZ-SOG')
res = {}
for lab, b in (('contracted', genbas.to_pyscf(qz, kappa=True)),
               ('uncontracted', [[l, [e, 1.0]] for l, (ex, c) in sorted(qz.shells.items()) for e in sorted(ex, reverse=True)])):
    mol0, mf0 = setup(b, 0, OPEN, 'mom2_QZ_%s_U.log' % lab); e0 = mf0.kernel()
    mol1, mf1 = setup(b, -1, EXTRA, 'mom2_QZ_%s_Um2.log' % lab)
    s22 = mol1.intor('int1e_ovlp_spinor'); s21 = gto.intor_cross('int1e_ovlp_spinor', mol1, molt)
    # project the TZ occupied spinors block by block, orthonormalise inside each block
    refq, nq, cols = {}, {}, []
    for key in mft.irrep_ao:
        idx = np.where((tags_t == key) & ot)[0]
        if not len(idx): continue
        c = np.linalg.solve(s22, s21 @ mft.mo_coeff[:, idx])
        m = c.conj().T @ s22 @ c
        c = c @ scipy.linalg.inv(scipy.linalg.sqrtm(m))
        refq[key] = c; nq[key] = len(idx); cols.append(c)
    call = np.hstack(cols)
    dm0 = call @ call.conj().T
    mf1 = mom(mf1, refq, nq); mf1.damp = 0.5
    e1 = mf1.kernel(dm0)
    say('QZ %-12s U E %.8f converged %s | U-2 (6d) E %.8f converged %s  %s' % (lab, e0, mf0.converged, e1, mf1.converged, info(mf1, ['d3/2,1/2', 'd3/2,3/2', 'f5/2,1/2'])))
    say('QZ %-12s EA_SCF(6d) %+.4f eV' % (lab, (e0 - e1) * EV))
    res[lab] = (e0, e1)
d = ((res['contracted'][0] - res['contracted'][1]) - (res['uncontracted'][0] - res['uncontracted'][1])) * EV * 1e3
say('QZ contraction error of EA(6d) %+.2f meV' % d)
