'''
Obtain soc integrals for given molecule.
'''
from pyscf import gto
from socutils.somf import somf_pt, somf

mol = gto.M(
    verbose = 4,
    atom = [[10,[0.0,0.0,0.0]]],
    basis = 'sto-3g',
    unit = "bohr",
    symmetry = False,
    spin = 0)

# Gaunt and gauge terms are included by default

# X2CAMF-PT scheme (perturbational)
x2cints = somf_pt.get_psoc_x2camf(mol)
print(x2cints)
#The only non-zero values in x,y,z components are +-6.53530640e-04

# X2CAMF scheme (variational)
x2cints = somf.get_soc_x2camf(mol)
print(x2cints)
#The only non-zero values in x,y,z components are +-6.53437798e-04
