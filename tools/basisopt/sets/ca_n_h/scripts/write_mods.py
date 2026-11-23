"""PySCF modules of the Ca/N/H X2C sets: aug-cc-pV{T,Q,5}Z, scalar (sf) and j-adapted (sog, kappa format)."""
import numpy as np
from socutils.tools.basisopt import genbas
from pyscf import gto
HDR = {'SF': 'SF-X2C-1e atomic HF contraction (X2CAMF atomic program, Gaussian nucleus).',
       'SOG': 'j-adapted contraction, X2CAMF (DC-Gaunt AMF) atomic HF, Gaussian nucleus; kappa format.'}
for z in 'TQ5':
    for k in ('SF', 'SOG'):
        b = {el: genbas.to_pyscf(genbas.read('GENBAS.canh', '%s:APV%sZ-%s' % (el.upper(), z, k)), kappa=(k == 'SOG'))
             for el in ('Ca', 'N', 'H')}
        path = 'pyscf/apv%sz_x2c_%s.py' % (z.lower(), k.lower())
        genbas.write_pyscf(path, b, header='aug-cc-pV%sZ, %s\n%s\nSource: canh_x2c/build_ca_val.py, GENBAS.canh' % (z, k, HDR[k]))
        mol = gto.M(atom='Ca 0 0 0; N 0 0 2.00; H 0 0 3.01', basis=b, verbose=0, spin=0)
        e = np.linalg.eigvalsh(mol.intor('int1e_ovlp'))
        es = np.linalg.eigvalsh(mol.intor('int1e_ovlp_spinor'))
        print('%-24s nao_nr %4d nao_2c %4d  min eig S(sph) %.1e  S(spinor) %.1e' % (path, mol.nao_nr(), mol.nao_2c(), e[0], es[0]))
