import numpy, sys, time
from pyscf import gto, scf
from socutils.rf12 import fourc, dc_mp2f12
from socutils.rf12.nr import gbs, mp2f12
c0 = 137.035999084
mol = gto.M(atom='He 0 0 0', unit='bohr', basis=gbs.basis('aug-cc-pvdz', ['He']), verbose=0)
rhf = scf.RHF(mol); rhf.conv_tol = 1e-12; rhf.kernel()
for cabs in sys.argv[1:]:
    cb = gbs.basis(cabs, ['He'])
    nr = mp2f12.MP2F12(rhf, cb); nr.kernel()
    res = {}
    for f in (10, 20):
        with fourc.light_speed(c0 * f):
            mf = scf.DHF(mol); mf.conv_tol = 1e-11; mf.kernel()
        m = dc_mp2f12.DCMP2F12(mf, cb, c0 * f); m.kernel(); res[f] = m.e_f12
    ex = (4 * res[20] - res[10]) / 3
    print('%-20s ncabs(NR) %3d | NR F12 %.9f | A(c->inf) %.9f | A - NR %9.1f uEh' % (cabs, nr.dims['ncabs'], nr.e_f12, ex, (ex - nr.e_f12) * 1e6), flush=True)
