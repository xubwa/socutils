'''
Variational treatment of spin-orbit integrals using spinor Hartree-Fock calculations.
'''
import pyscf
from pyscf import scf, gto, x2c
import x2camf_hf

mol = gto.M(verbose=3,
                atom=[["O", (0., 0., -0.12390941)], 
		          [1, (0., -1.42993701, 0.98326612)],
                      [1, (0.,  1.42993701, 0.98326612)]],
                basis='unc-ccpvdz',
                unit = 'Bohr')
mf = x2camf_hf.X2CAMF_RHF(mol, with_gaunt=False, with_breit=False)
e_spinor = mf.scf()
mf = x2camf_hf.X2CAMF_RHF(mol, with_gaunt=True, with_breit=False)
e_gaunt = mf.scf()
mf = x2camf_hf.X2CAMF_RHF(mol, with_gaunt=True, with_breit=True)
e_breit = mf.scf()
gmf = x2camf_hf.x2camf_ghf(scf.GHF(mol), with_gaunt=True, with_breit=True)
e_ghf = gmf.kernel()
gmf = x2camf_hf.x2camf_ghf(scf.GHF(mol), with_gaunt=False, with_breit=False)
e_ghf = gmf.kernel()
gmf = x2camf_hf.x2camf_ghf(scf.GHF(mol), with_gaunt=True, with_breit=False)
e_ghf_gaunt = gmf.kernel()
gmf = x2camf_hf.x2camf_ghf(scf.GHF(mol), with_gaunt=True, with_breit=True)
e_ghf_breit = gmf.kernel()


# Energies from j-spinor and ghf-based calculations are expected to be the same.
# Coulomb: -76.08200768
# Gaunt:   -76.0665648
# Breit:   -76.06711578
print("Energy from spinor X2CAMF(Coulomb):    %16.10g" % e_spinor)
print("Energy from spinor X2CAMF(Gaunt):      %16.10g" % e_gaunt)
print("Energy from spinor X2CAMF(Breit):      %16.10g" % e_breit)
print("Energy from ghf-based X2CAMF(Coulomb): %16.10g" % e_ghf)
print("Energy from ghf-based X2CAMF(Gaunt):   %16.10g" % e_ghf_gaunt)
print("Energy from ghf-based X2CAMF(Breit):   %16.10g" % e_ghf_breit)