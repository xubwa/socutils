'''
Calculate dipole moments within SFX2C-1e and X2CAMF scheme
'''
import pyscf
from pyscf import scf, gto
from socutils.scf import spinor_hf
from socutils.somf import amf, somf_pt

mol = gto.M(verbose=4,
                atom=[["H", (0., 0., 0.0)],
                      ["F", (0., 0., 1.0)]],
                basis='unc-ccpvdz',
                unit = 'Bohr')

sf = scf.RHF(mol).sfx2c1e()
x2cdc = spinor_hf.JHF(mol)
x2cdc.with_x2c = amf.SpinorX2CAMFHelper(mol, with_gaunt=False, with_breit=False)
x2cdcg = spinor_hf.JHF(mol)
x2cdcg.with_x2c = amf.SpinorX2CAMFHelper(mol, with_gaunt=True, with_breit=False)
x2cdcb = spinor_hf.JHF(mol)
x2cdcb.with_x2c = amf.SpinorX2CAMFHelper(mol, with_gaunt=True, with_breit=True)

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

# Energy from sfx2c-1e           :        -99.58238117
# Energy from spinor X2CAMF(DC)  :         -99.5823842
# Energy from spinor X2CAMF(DCG) :        -99.55851423
# Energy from spinor X2CAMF(DCB) :        -99.55957273
# Dipole z-component from sfx2c-1e           :       -0.518994
# Dipole z-component from spinor X2CAMF(DC)  :       -0.518995
# Dipole z-component from spinor X2CAMF(DCG) :       -0.518624
# Dipole z-component from spinor X2CAMF(DCB) :       -0.518596
