'''
Calculate dipole moments within SFX2C-1e and X2CAMF scheme
'''
import pyscf
from pyscf import scf, gto, x2c
import x2camf_hf, somf_pt

mol = gto.M(verbose=4,
                atom=[["H", (0., 0., 0.0)],
                      ["O", (0., 0., 1.0)]],
                basis='unc-ccpvdz',
                spin = 1,
                unit = 'Bohr')

sf = scf.UHF(mol).sfx2c1e()
x2cdc = x2camf_hf.X2CAMF_UHF(mol, with_gaunt=False, with_breit=False)
x2cdcg = x2camf_hf.X2CAMF_UHF(mol, with_gaunt=True, with_breit=False)
x2cdcb = x2camf_hf.X2CAMF_UHF(mol, with_gaunt=True, with_breit=True)

e_sfx2c1e = sf.scf()
dipole_sf = somf_pt.sfx2c1e_dipole(sf)
e_x2cdc = x2cdc.scf()
dipole_dc = somf_pt.x2camf_dipole(x2cdc)
e_x2cdcg = x2cdcg.scf()
dipole_dcg = somf_pt.x2camf_dipole(x2cdcg)
e_x2cdcb = x2cdcb.scf()
dipole_dcb = somf_pt.x2camf_dipole(x2cdcb)

print("Energy from sfx2c-1e           :    %16.10g" % e_sfx2c1e)
print("Energy from spinor X2CAMF(DC)  :    %16.10g" % e_x2cdc)
print("Energy from spinor X2CAMF(DCG) :    %16.10g" % e_x2cdcg)
print("Energy from spinor X2CAMF(DCB) :    %16.10g" % e_x2cdcb)

print("Dipole z-component from sfx2c-1e           :    %16.10g" % dipole_sf[2])
print("Dipole z-component from spinor X2CAMF(DC)  :    %16.10g" % dipole_dc[2])
print("Dipole z-component from spinor X2CAMF(DCG) :    %16.10g" % dipole_dcg[2])
print("Dipole z-component from spinor X2CAMF(DCB) :    %16.10g" % dipole_dcb[2])


# Energy from sfx2c-1e           :        -74.83550604
# Energy from spinor X2CAMF(DC)  :        -74.83364783
# Energy from spinor X2CAMF(DCG) :        -74.81983089
# Energy from spinor X2CAMF(DCB) :        -74.81853161
# Dipole z-component from sfx2c-1e           :       -0.5818621874
# Dipole z-component from spinor X2CAMF(DC)  :       -0.5821840156
# Dipole z-component from spinor X2CAMF(DCG) :       -0.5803031343
# Dipole z-component from spinor X2CAMF(DCB) :       -0.5818832499
