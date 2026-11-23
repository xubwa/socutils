"""Y:DYVDZ-SOG: j-adapted counterpart of Y:DYVDZ-SF (user request 2026-11-16).  Same recipe as make_y.py:
dyall.v2z primitives, average of 4d1 5s2, Gaussian nucleus, one contracted function per occupied
shell (per kappa for l >= 1), uncontracted 2nd s, 2nd p, 1st+2nd d (Dyall 4d DZ), f uncontracted;
contraction from the X2CAMF (DC-Gaunt AMF) atomic SCF."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'audit'))
import dirac_basis
from socutils.tools.basisopt import contract, genbas
prims = {l: sorted(v, reverse=True) for l, v in dirac_basis.read('/home/xwang405/apps/DIRAC-23.0-Source/basis/dyall.v2z')[39].items()}
pattern = contract.pattern_occupied('Y', prims, {0: [2], 1: [2], 2: [1, 2]}, gau_nuc=True)
e = contract.contract('Y', prims, pattern, kind='sog', gau_nuc=True, name='DYVDZ-SOG', conv=1e-9, max_iter=300)
print('Y:DYVDZ-SOG %s -> %s, E_SCF(prims) %.8f' % (''.join('%d%s' % (len(x), 'spdfg'[l]) for l, (x, c) in sorted(e.shells.items())),
      ''.join('%d%s' % (c.shape[1], 'spdfg'[l]) for l, (x, c) in sorted(e.shells.items())), e.scf.e_scf))
contract.print_contraction_error(e, 'Y', gau_nuc=True)
e.name = 'Y:DYVDZ-SOG'
out = os.path.join(HERE, 'GENBAS.y')
if e.name not in [s.strip() for s in open(out)]:
    genbas.append(out, e)
genbas.write_pyscf(os.path.join(HERE, 'dyvdz_sog.py'), {'Y': genbas.to_pyscf(e, kappa=True)},
                   header='Y:DYVDZ-SOG: dyall.v2z, j-adapted X2CAMF (DC-Gaunt AMF) contraction (4d1 5s2 average), Gaussian nucleus, kappa format. Source: y_sf/make_y_sog.py')
