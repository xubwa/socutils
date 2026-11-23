"""Molecular overlap of linear CaNH (Ca-N 2.00 A, N-H 1.01 A) for combinations of Ca/N/H sets
from BSE: smallest eigenvalues of S (contracted and uncontracted)."""
import numpy as np, basis_set_exchange as bse
from pyscf import gto
def nw(name, el):
    try: return gto.basis.parse(bse.get_basis(name, elements=[el], fmt='nwchem', header=False))
    except Exception: return None
combos = [('cc-pVTZ', 'aug-cc-pVTZ'), ('cc-pVQZ', 'aug-cc-pVQZ'), ('cc-pV5Z', 'aug-cc-pV5Z'),
          ('aug-cc-pVTZ-X2C', 'aug-cc-pVTZ-X2C'), ('aug-cc-pVQZ-X2C', 'aug-cc-pVQZ-X2C'),
          ('aug-cc-pVTZ-X2C', 'aug-cc-pVTZ'), ('aug-cc-pVQZ-X2C', 'aug-cc-pVQZ')]
for ca, nh in combos:
    b = {'Ca': nw(ca, 'Ca'), 'N': nw(nh, 'N'), 'H': nw(nh, 'H')}
    if None in b.values():
        print('%-16s + %-16s : missing %s' % (ca, nh, [k for k, v in b.items() if v is None])); continue
    for unc in (False, True):
        bb = {k: gto.uncontract(v) if unc else v for k, v in b.items()}
        mol = gto.M(atom='Ca 0 0 0; N 0 0 2.00; H 0 0 3.01', basis=bb, verbose=0)
        e = np.linalg.eigvalsh(mol.intor('int1e_ovlp'))
        print('%-16s + %-16s %-5s nao %4d  eig %.1e %.1e %.1e' % (ca, nh, 'unc' if unc else 'con', mol.nao, *e[:3]))
