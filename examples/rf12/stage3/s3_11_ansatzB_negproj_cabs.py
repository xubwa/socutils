import sys
from pyscf import gto, scf
from socutils.rf12 import fourc, dc_mp2f12_kb
from socutils.rf12.nr import gbs, mp2f12
c0 = 137.035999084
obs, cab = sys.argv[1], sys.argv[2]
mol = gto.M(atom='He 0 0 0', unit='bohr', basis=gbs.basis(obs, ['He']), verbose=0)
rhf = scf.RHF(mol); rhf.conv_tol = 1e-12; rhf.kernel()
cb = gbs.basis(cab, ['He'])
nr = mp2f12.MP2F12(rhf, cb); nr.kernel()
with fourc.light_speed(c0 * 20):
    mf = scf.DHF(mol); mf.conv_tol = 1e-11; mf.kernel()
for pn in (True, False):
    m = dc_mp2f12_kb.DCMP2F12KB(mf, cb, c0 * 20); m.project_negative = pn; m.kernel()
    print('%s/%s 20c0 projN=%s  B-stage2 = %.3f uEh  c*maxOvN = %.3e' % (obs, cab, pn, (m.e_f12 - nr.e_f12) * 1e6, m.ov_neg), flush=True)
