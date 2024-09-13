'''
Variational treatment of spin-orbit integrals using spinor Kohn-Sham calculations.
'''
from pyscf import gto
from socutils.dft import dft as socdft
from socutils.somf import amf

mol = gto.M(verbose=4,
            atom=[["O", (0.,          0., -0.12390941)],
                  [  1, (0., -1.42993701,  0.98326612)],
                  [  1, (0.,  1.42993701,  0.98326612)]],
            basis='unc-ccpvdz',
            unit='Bohr')
mf = socdft.SpinorDFT(mol, xc='b3lyp')
mf.with_x2c = amf.SpinorX2CAMFHelper(mol,with_gaunt=False,with_breit=False)
mf.kernel()


# X2CAMF(DC)-B3LYP:  -76.4827765800062
# X2CAMF(DCB)-B3LYP: -76.4678453729391

# With the old B3LYP_WITH_VWN5 = True
# X2CAMF(DC)-B3LYP:  -76.4456318070864
# X2CAMF(DCB)-B3LYP: -76.4307008465042