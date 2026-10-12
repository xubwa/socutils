import numpy, sys, json, time
from pyscf import gto, scf
from socutils.rf12 import fourc, dc_mp2f12
from socutils.rf12.nr import gbs, mp2f12
c0 = 137.035999084
name = sys.argv[1]; fz = int(sys.argv[2]) if len(sys.argv) > 2 else 0
mol = gto.M(atom=name + ' 0 0 0', unit='bohr', basis=gbs.basis('aug-cc-pvdz', [name]), verbose=0)
rhf = scf.RHF(mol); rhf.conv_tol = 1e-12; rhf.kernel()
nr = mp2f12.MP2F12(rhf, gbs.basis('aug-cc-pvdz-optri', [name]), frozen=fz // 2); nr.kernel()
print('stage-2 NR: MP2 %.12f F12 %.12f (U part of B) ' % (nr.e_mp2, nr.e_f12))
res = {}
for f in [float(x) for x in sys.argv[3:]] or [1, 10, 20]:
    t = time.time()
    with fourc.light_speed(c0 * f):
        mf = scf.DHF(mol); mf.conv_tol = 1e-11; mf.kernel()
    m = dc_mp2f12.DCMP2F12(mf, gbs.basis('aug-cc-pvdz-optri', [name]), c0 * f, frozen=fz)
    m.kernel()
    res[f] = (m.e_mp2, m.e_f12)
    print('c = %5.1f c0: MP2 %.12f  F12(A) %.12f  (%.0fs)' % (f, m.e_mp2, m.e_f12, time.time() - t), flush=True)
if 10 in res and 20 in res:
    ex = [(4 * res[20][k] - res[10][k]) / 3 for k in (0, 1)]
    print('c->inf: MP2 %.12f (diff %.4f uEh)  F12(A) %.12f  diff vs stage 2 %.3f uEh' % (
        ex[0], (ex[0] - nr.e_mp2) * 1e6, ex[1], (ex[1] - nr.e_f12) * 1e6))
