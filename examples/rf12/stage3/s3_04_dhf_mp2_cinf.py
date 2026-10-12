import numpy, json, sys
from pyscf import gto, scf, mp
from socutils.rf12 import fourc
from socutils.rf12.nr import gbs
geoms = {'He': ('He 0 0 0', ['He']), 'Ne': ('Ne 0 0 0', ['Ne']),
         'H2O': ('O 0 0 -0.124038860300; H 0 -1.431430901356 0.984293362719; H 0 1.431430901356 0.984293362719', ['O', 'H'])}
c0 = 137.035999084
for name in sys.argv[1:]:
    atom, els = geoms[name]
    mol = gto.M(atom=atom, unit='bohr', basis=gbs.basis('aug-cc-pvdz', els), verbose=0)
    rhf = scf.RHF(mol); rhf.conv_tol = 1e-12; rhf.kernel()
    emp2 = mp.MP2(rhf).kernel()[0]
    res = {}
    for f in (1, 10, 20, 40):
        with fourc.light_speed(c0 * f):
            mf = scf.DHF(mol); mf.conv_tol = 1e-11; mf.kernel()
            res[f] = (mf.e_tot, fourc.mp2(mf, c0 * f))
        print(name, 'c = %3d c0: E_DHF %.12f  E_MP2 %.12f' % (f, res[f][0], res[f][1]), flush=True)
    ex = lambda a, b: (4 * b - a) / 3          # 1/c^2 Richardson: E(c), E(2c)
    for lab, i, ref in (('HF', 0, rhf.e_tot), ('MP2', 1, emp2)):
        r1 = ex(res[10][i], res[20][i]); r2 = ex(res[20][i], res[40][i])
        print('  %-3s NR %.12f | c->inf (10,20) %.12f diff %.3f uEh | (20,40) %.12f diff %.3f uEh | rel. effect at c0: %.6f' % (
            lab, ref, r1, (r1 - ref) * 1e6, r2, (r2 - ref) * 1e6, res[1][i] - ref))
