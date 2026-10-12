import numpy, sys, time
from pyscf import gto, scf
from socutils.rf12 import fourc, dc_mp2f12_kb
from socutils.rf12.nr import gbs, mp2f12
c0 = 137.035999084
name = sys.argv[1]; fz = int(sys.argv[2])
mol = gto.M(atom=name + ' 0 0 0', unit='bohr', basis=gbs.basis('aug-cc-pvdz', [name]), verbose=0)
rhf = scf.RHF(mol); rhf.conv_tol = 1e-12; rhf.kernel()
nr = mp2f12.MP2F12(rhf, gbs.basis('aug-cc-pvdz-optri', [name]), frozen=fz // 2); nr.kernel()
print('stage-2: F12 %.12f' % nr.e_f12)
res = {}
for f in [float(x) for x in sys.argv[3:]]:
    t = time.time()
    with fourc.light_speed(c0 * f):
        mf = scf.DHF(mol); mf.conv_tol = 1e-11; mf.kernel()
    m = dc_mp2f12_kb.DCMP2F12KB(mf, gbs.basis('aug-cc-pvdz-optri', [name]), c0 * f, frozen=fz)
    m.kernel(); res[f] = m.e_f12
    print('c = %6.1f c0: MP2 %.12f  F12(B) %.12f  diff vs stage 2 %.4f uEh  [kin RI-exact %.1e]  (%.0fs)' % (
        f, m.e_mp2, m.e_f12, (m.e_f12 - nr.e_f12) * 1e6, m._kin_ri_vs_exact, time.time() - t), flush=True)
if 10 in res and 20 in res:
    ex = (4 * res[20] - res[10]) / 3
    print('c->inf (10,20): F12(B) %.12f  diff vs stage 2 %.4f uEh' % (ex, (ex - nr.e_f12) * 1e6))
