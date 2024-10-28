'''
Variational treatment of spin-orbit integrals using spinor Hartree-Fock calculations.
GHF should give same results as spinor HF calculations.
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
mf = spinor_hf.density_fit(spinor_hf.JHF(mol))
mf = spinor_hf.SpinorSCF(mol)
mf = mf.density_fit2()
mf.with_x2c = amf.SpinorX2CAMFHelper(mol,with_gaunt=True,with_breit=True)
e_jhf = mf.scf()

# Construct GHF object with x2camf helper.
mf = scf.GHF(mol).x2c1e().density_fit()
mf.with_x2c = amf.SpinOrbitalX2CAMFHelper(mol,with_gaunt=True,with_breit=True)
e_ghf = mf.scf()

# Energies from j-spinor and ghf-based calculations are expected to be the same.
# Ref:   -76.06711578
print("Energy from spinor HF:      %16.10g" % e_jhf)
print("Energy from GHF:            %16.10g" % e_ghf)
