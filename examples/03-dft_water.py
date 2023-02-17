'''
Variational treatment of spin-orbit integrals using spinor Kohn-Sham calculations.
'''
import pyscf
from pyscf import scf, gto, x2c, dft
import x2camf_hf

mol = gto.M(verbose=3,
            atom=[["O", (0.,          0., -0.12390941)],
                  [  1, (0., -1.42993701,  0.98326612)],
                  [  1, (0.,  1.42993701,  0.98326612)]],
            basis='unc-ccpvdz',
            unit='Bohr')
mf = x2c.UKS(mol)
mf.xc = 'b3lyp'
mf.with_x2c = x2camf_hf.X2CAMF(mol, with_gaunt=True, with_breit=True)
euks = mf.kernel()
mfr = x2c.RKS(mol)
mfr.xc = 'b3lyp'
mfr.with_x2c = x2camf_hf.X2CAMF(mol, with_gaunt=True, with_breit=True)
erks = mfr.kernel()

# Coulomb:   -76.4307008465042
print("Energy from spinor UKS(Coulomb):    %16.10g" % euks)
print("Energy from spinor RKS(Coulomb):    %16.10g" % erks)
