'''
Calculate the geometrical gradient of the hcore matrix elements
This is for X2C-1e part. The X2CAMF contribution is zero.
'''
from pyscf import gto
from pyscf.lib.parameters import BOHR
# numerical gradient
dx = 1e-3
molm = gto.Mole().build(atom=f'C {-dx} 0 0; O 1.2 0 0', basis='uncccpvdz')
molp = gto.Mole().build(atom=f'C {dx} 0 0; O 1.2 0 0', basis='uncccpvdz')
molm2 = gto.Mole().build(atom=f'C {-2.0*dx} 0 0; O 1.2 0 0', basis='uncccpvdz')
molp2 = gto.Mole().build(atom=f'C {2.0*dx} 0 0; O 1.2 0 0', basis='uncccpvdz')
molm3 = gto.Mole().build(atom=f'C {-3.0*dx} 0 0; O 1.2 0 0', basis='uncccpvdz')
molp3 = gto.Mole().build(atom=f'C {3.0*dx} 0 0; O 1.2 0 0', basis='uncccpvdz')
molm4 = gto.Mole().build(atom=f'C {-4.0*dx} 0 0; O 1.2 0 0', basis='uncccpvdz')
molp4 = gto.Mole().build(atom=f'C {4.0*dx} 0 0; O 1.2 0 0', basis='uncccpvdz')

from pyscf.x2c import x2c
mfm = x2c.X2C(molm)
mfp =  x2c.X2C(molp)
mfm2 = x2c.X2C(molm2)
mfp2 = x2c.X2C(molp2)
mfm3 = x2c.X2C(molm3)
mfp3 = x2c.X2C(molp3)
mfm4 = x2c.X2C(molm4)
mfp4 = x2c.X2C(molp4)

hm = mfm.get_hcore()
hp = mfp.get_hcore()
hm2 = mfm2.get_hcore()
hp2 = mfp2.get_hcore()
hm3 = mfm3.get_hcore()
hp3 = mfp3.get_hcore()
hm4 = mfm4.get_hcore()
hp4 = mfp4.get_hcore()

hgrad2 = (1.0/2.0*hp-1.0/2.0*hm)/(dx/BOHR)
hgrad4 = (-1.0/12.0*hp2+2.0/3.0*hp-2.0/3.0*hm+1.0/12.0*hm2)/(dx/BOHR)
hgrad6 = (1.0/60.0*hp3-3.0/20.0*hp2+3.0/4.0*hp-3.0/4.0*hm+3.0/20.0*hm2-1.0/60.0*hm3)/(dx/BOHR)
hgrad8 = (-1.0/280.0*hp4+4.0/105.0*hp3-1.0/5.0*hp2+4.0/5.0*hp-4.0/5.0*hm+1.0/5.0*hm2-4.0/105.0*hm3+1.0/280.0*hm4)/(dx/BOHR)

print('Numerical gradient hfw[0,0]')
print(hgrad2[0,0])
print(hgrad4[0,0])
print(hgrad6[0,0])
print(hgrad8[0,0])

mol = gto.Mole().build(atom='C 0 0 0; O 1.2 0 0', basis='uncccpvdz')
mf = x2c.X2C(mol)
from socutils.somf.x2c_grad import hcore_grad_generator_spinor
hcore_grad = hcore_grad_generator_spinor(mf)
print('Analytical gradient hfw[0,0]')
print(hcore_grad(0)[0,0,0])