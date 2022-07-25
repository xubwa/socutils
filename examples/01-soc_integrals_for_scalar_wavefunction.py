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
mf = scf.RHF(mol).x2c()
mf.kernel()
x2cints = somf.get_soc_integrals(mf, pc1e='x2c1', pc2e='x2c', unc=True, atomic=True)

mf = scf.RHF(mol)
mf.kernel()
bpints = somf.get_soc_integrals(mf, pc1e='bp', pc2e='bp', unc=False, atomic=False)


