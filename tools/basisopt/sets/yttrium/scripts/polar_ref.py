"""Finite-field dipole polarizability of Y+ (5s2) and Y3+ ([Kr]), closed-shell RHF with SF-X2C-1e
and Gaussian nucleus, for a contracted Y basis module and its uncontracted primitives.
alpha_zz from the energy: E(F) = E0 - alpha F^2 / 2 - gamma F^4 / 24, fields 0.001/0.002 au
(Richardson-type fit of E(+-F) and E(+-2F))."""
import sys, numpy as np
from pyscf import gto, scf
from socutils.tools.basisopt import genbas
basis = genbas.read_pyscf(sys.argv[1])['Y']
ref = genbas.read_pyscf(sys.argv[2])['Y'] if len(sys.argv) > 2 else gto.uncontract(basis)
def energy(b, charge, f):
    mol = gto.M(atom='Y 0 0 0', basis={'Y': b}, charge=charge, spin=0, nucmod={'Y': 'G'}, verbose=0)
    mf = scf.RHF(mol).sfx2c1e(); mf.conv_tol = 1e-12
    h = mf.get_hcore()
    with mol.with_common_orig((0, 0, 0)):
        dz = mol.intor('int1e_r', comp=3)[2]
    mf.get_hcore = lambda *a: h + f * dz
    e = mf.kernel()
    assert mf.converged
    return e, mol.nao
for charge in (1, 3):
    out = []
    for lab, b in (('contracted', basis), ('reference', ref)):
        F = 0.001
        e0, nao = energy(b, charge, 0.0)
        e1 = (energy(b, charge, F)[0] + energy(b, charge, -F)[0]) / 2
        e2 = (energy(b, charge, 2 * F)[0] + energy(b, charge, -2 * F)[0]) / 2
        alpha = -(16 * (e1 - e0) - (e2 - e0)) / (6 * F ** 2)
        out.append(alpha)
        print('Y%d+ %-12s nao %4d  E0 %.8f  alpha %.4f au' % (charge, lab, nao, e0, alpha), flush=True)
    print('Y%d+ contraction error in alpha: %.4f au (%.2f %%)' % (charge, out[0] - out[1], 100 * (out[0] - out[1]) / out[1]))
