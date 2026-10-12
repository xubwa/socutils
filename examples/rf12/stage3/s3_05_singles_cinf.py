import numpy, sys, json, os
import socutils.rf12
from pyscf import gto, scf
from socutils.rf12 import fourc, dc_space
from socutils.rf12.nr import gbs
ref = json.load(open(os.path.join(os.path.dirname(socutils.rf12.__file__), 'test', 'data', 'psi4_aug-cc-pvdz.json')))
geoms = {'He': ('He 0 0 0', ['He']), 'Ne': ('Ne 0 0 0', ['Ne'])}
c0 = 137.035999084
for name in sys.argv[1:]:
    atom, els = geoms[name]
    mol = gto.M(atom=atom, unit='bohr', basis=gbs.basis('aug-cc-pvdz', els), verbose=0)
    es = {}
    for f in (1, 10, 20):
        c = c0 * f
        with fourc.light_speed(c):
            mf = scf.DHF(mol); mf.conv_tol = 1e-11; mf.kernel()
        sp = dc_space.DCSpace(mf, gbs.basis('aug-cc-pvdz-optri', els), c)
        o = sp.slices['o']
        es[f] = sp.cabs_singles()
        print(name, 'c=%2d c0' % f, sp.dims, 'metric err %.1e' % sp.metric_error,
              '| F_oo diag - eps_DHF %.1e' % abs(numpy.diag(sp.fock)[o].real - mf.mo_energy[mol.nao_2c():mol.nao_2c() + mol.nelectron]).max(),
              '| singles %.12f' % es[f], flush=True)
    ext = (4 * es[20] - es[10]) / 3
    r = ref[name + '_ae']['F12 CABS CORRECTION ENERGY']
    print('  singles c->inf %.12f  stage-2 %.12f  diff %.4f uEh   rel. effect at c0 %.3e' % (ext, r, (ext - r) * 1e6, es[1] - r))
