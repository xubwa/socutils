'''
Perturbative treatment of spin-orbit integrals.
X2CAMF scheme and MMF-BP operators are available.
'''
from pyscf import gto, scf, mcscf
from pyscf.x2c import x2c
import x2camf
import numpy
import somf
mol = gto.M(
    verbose = 4,
    atom = [[36,[0.0,0.0,0.0]]],
    basis = 'cc-pvtz',
    symmetry = False,
    spin = 0)
mfx2c = scf.RHF(mol).x2c()
mfx2c.kernel()

mfbp = scf.RHF(mol)
mfbp.kernel()


# X2CAMF scheme
x2cints = somf.get_soc_integrals(mfx2c, pc1e='x2c1', pc2e='x2c', unc=True, atomic=True)

# Molecular mean-field Breit-Pauli integrals
bpints = somf.get_soc_integrals(mfbp, pc1e='bp', pc2e='bp', unc=True, atomic=False)


