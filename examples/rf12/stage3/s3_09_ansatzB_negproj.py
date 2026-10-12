import numpy, sys, time
from pyscf import gto, scf
from socutils.rf12 import fourc, dc_mp2f12_kb
from socutils.rf12.nr import gbs, mp2f12
c0 = 137.035999084
mol = gto.M(atom='He 0 0 0', unit='bohr', basis=gbs.basis('aug-cc-pvdz', ['He']), verbose=0)
rhf = scf.RHF(mol); rhf.conv_tol = 1e-12; rhf.kernel()
nr = mp2f12.MP2F12(rhf, gbs.basis('aug-cc-pvdz-optri', ['He'])); nr.kernel()
for f in (10., 20.):
    with fourc.light_speed(c0 * f):
        mf = scf.DHF(mol); mf.conv_tol = 1e-11; mf.kernel()
    for pn in (True, False):
        m = dc_mp2f12_kb.DCMP2F12KB(mf, gbs.basis('aug-cc-pvdz-optri', ['He']), c0 * f)
        m.project_negative = pn
        m.kernel()
        print('c=%4.0f c0 project_negative=%s: F12(B) - stage2 = %.4f uEh   c*max|<g|N J>| = %.3e' % (
            f, pn, (m.e_f12 - nr.e_f12) * 1e6, m.ov_neg), flush=True)
