'''
Variational treatment of spin-orbit integrals using spinor Hartree-Fock calculations.
'''
import pyscf
from pyscf import scf, gto, x2c
from socutils.scf import spinor_hf
from socutils.somf import amf

mol = gto.M(verbose=0,
            atom=[["O", (0.,          0., -0.12390941)],
                  [  1, (0., -1.42993701,  0.98326612)],
                  [  1, (0.,  1.42993701,  0.98326612)]],
            basis='unc-ccpvdz',
            unit='Bohr')
mf = spinor_hf.JHF(mol)
mf.with_x2c = amf.SpinorX2CAMFHelper(mol,with_gaunt=False,with_breit=False)
e_spinor = mf.scf()
mf.with_x2c = amf.SpinorX2CAMFHelper(mol,with_gaunt=True,with_breit=False)
e_gaunt = mf.scf()
mf.with_x2c = amf.SpinorX2CAMFHelper(mol,with_gaunt=True,with_breit=True)
e_breit = mf.scf()

# Energies from j-spinor and ghf-based calculations are expected to be the same.
# Coulomb: -76.08200768
# Gaunt:   -76.0665648
# Breit:   -76.06711578
print("Energy from spinor X2CAMF(Coulomb):    %16.10g" % e_spinor)
print("Energy from spinor X2CAMF(Gaunt):      %16.10g" % e_gaunt)
print("Energy from spinor X2CAMF(Breit):      %16.10g" % e_breit)
