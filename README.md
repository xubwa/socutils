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
    mf = x2camf.X2CAMF_RHF(mol)
    e_spinor = mf.kernel()
```
Note, spinor style calculation requires the "zquatev" module, one can install it through
```
pip instal git+https://github.com/sunqm/zquatev
```

To do a GHF style calculation
```
import x2camf
from pyscf import gto, scf
mol = gto.M(verbose=4,
                atom=[["O", (0., 0., 0.)], 
                      [1, (0., -0.757, 0.587)],
                      [1, (0., 0.757, 0.587)]],
                basis='ccpvdz')
    gmf = x2camf.x2camf_ghf(scf.GHF(mol))
    e_ghf = gmf.kernel()
```
