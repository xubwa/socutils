This repository adds utility for PySCF to include SOMF corrections within spinor style and GHF style calculations.
To do a spinor style calculation
```
import x2camf
from pyscf import gto
mol = gto.M(verbose=4,
                atom=[["O", (0., 0., 0.)], 
                      [1, (0., -0.757, 0.587)],
                      [1, (0., 0.757, 0.587)]],
                basis='ccpvdz')
    mf = x2camf_hf.X2CAMF_RHF(mol)
    e_spinor = mf.kernel()
    # gaunt term and breit term can be turned on by with_gaunt and with_breit
    mf2 = x2camf_hf.X2CAMF_RHF(mol, with_gaunt=True, with_breit=True)
    e_breit = mf2.kernel()
```
Note, spinor style calculation requires the "zquatev" module, one can install it through
```
pip instal git+https://github.com/sunqm/zquatev
```
To accelerates the calculation of SOC terms, "libx2camf" is required, and one can install it through
```
pip install git+https://github.com/warlocat/x2camf
```
To do a GHF style calculation
```
import x2camf_hf
from pyscf import gto, scf
mol = gto.M(verbose=4,
                atom=[["O", (0., 0., 0.)], 
                      [1, (0., -0.757, 0.587)],
                      [1, (0., 0.757, 0.587)]],
                basis='ccpvdz')
    gmf = x2camf_hf.x2camf_ghf(scf.GHF(mol))
    # gaunt term and breit term can be turned on and off by keyword with_gaunt and with_breit
    gmf2 = x2camf_hf.x2camf_ghf(scf.GHF(mol), with_gaunt=True, with_breit=True)
    e_ghf = gmf.kernel()
    e_breit = gmf2.kernel()
```
