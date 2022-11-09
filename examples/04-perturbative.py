'''
Obtain soc integrals for given molecule.
We refer to this method as perturbative treatment of spin-orbit coupling within X2CAMF scheme.
The SOC contributions are evaluated in a consistent way to include only first-order terms.

Ref.
Mol. Phys. 118, e1768313 (2020); DOI:10.1080/00268976.2020.1768313
'''
from pyscf import gto, scf, mcscf
from pyscf.x2c import x2c
import x2camf
import numpy
import somf
import somf_pt
mol = gto.M(
    verbose = 4,
    atom = [[10,[0.0,0.0,0.0]]],
    basis = 'cc-pvtz',
    symmetry = False,
    spin = 0)

# X2CAMF-PT scheme
x2cints = somf_pt.x2camf_pt(mol)


