import pyscf
from pyscf import gto,x2c
import x2camf
mol = gto.M(verbose=3,
                atom=[["H", (0., -1.42993701,    0.98326612)], 
                      ["O", (0.0,   -0.0,    -0.12390941)],
                      ["H", (0.0,    1.42993701,     0.98326612)]],
                unit="bohr",
                basis='unc-ccpvdz')
mf = x2camf.X2CAMF_RHF(mol)
mf.scf()
e1 = mf.e_tot
mf2 = x2camf.X2CAMF_RHF(mol,False,False,False,"sph_atm")
mf2.scf()
e2 = mf2.e_tot
print("e1 - e2 = ")
print(e1-e2)
# mf2 = x2camf.X2CAMF(mol)
# mf2.kernel()
# mf2 = x2camf.X2CAMF_RHF()
    # gmf = x2camf_ghf(scf.GHF(mol))
    # e_ghf = gmf.kernel()
    # print("Energy from spinor X2CAMF:    %16.8g" % e_spinor)
    # print("Energy from ghf-based X2CAMF: %16.8g" % e_ghf)
