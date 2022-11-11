import numpy, scipy, x2c_grad
import time
from functools import reduce
from pyscf import gto, scf, lib
from pyscf.gto import moleintor
from pyscf.lib.parameters import LIGHT_SPEED
from pyscf.x2c import x2c
from pyscf.x2c import sfx2c1e
from spinor2sph import spinor2sph_soc
x2camf  = None
try:
    import x2camf
except ImportError:
    pass
'''
Analytic energy gradients for (sf)x2c-1e and x2camf method

Ref.
JCP 135, 084114 (2011); DOI:10.1063/1.3624397
'''



'''
Perturbative treatment of spin-orbit coupling within X2CAMF scheme.
The SOC contributions are evaluated in a consistent way to include only first-order terms.

Ref.
Mol. Phys. 118, e1768313 (2020); DOI:10.1080/00268976.2020.1768313
'''
def get_psoc_x2camf(mol, gaunt=True, gauge=True):
    xmol, contr_coeff = sfx2c1e.SpinFreeX2C(mol).get_xmol()
        
    c = LIGHT_SPEED
    t = xmol.intor_symmetric('int1e_spsp_spinor') * .5
    v = xmol.intor_symmetric('int1e_nuc_spinor')
    s = xmol.intor_symmetric('int1e_ovlp_spinor')
    w = xmol.intor_symmetric('int1e_spnucsp_spinor')
    wsf = xmol.intor_symmetric('int1e_pnucp_spinor')
    wso = w - wsf

    n2c = s.shape[0]
    n4c = n2c * 2

    h4c1 = numpy.zeros((n4c, n4c), dtype=v.dtype)
    h4c1[n2c:, n2c:] = wso * (.25 / c**2)
    # Get x2camf 4c integrals and add to h4c1
    x2cobj = x2c.X2C(mol)
    spinor = x2camf.amfi(x2cobj, spin_free=True, two_c=True, with_gaunt=gaunt, with_gauge=gauge, pt=True)
    h4c1 += spinor
    
    hfw1 = x2c_grad.x2c1e_hfw1(t,v,wsf,s,h4c1)
    hfw1_sph = spinor2sph_soc(xmol, hfw1)[1:]
    # convert back to contracted basis
    result = numpy.zeros((3, mol.nao_nr(), mol.nao_nr()))
    for ic in range(3):
        result[ic] = reduce(numpy.dot, (contr_coeff.T, hfw1_sph[ic], contr_coeff))
    
    return result


'''
First-order (electric) properties that satisfy
        eri^x = 0    &&    S^x = 0
e.g., electric dipole moments and electric field gradients.
'''
def first_elec_prop_scf(dm, hfw1):
    # dE/dx = \sum_mn D_{mn} h1_{mn}
    size = dm.shape[0]
    prop = 0.0
    for ii in range(size):
        for jj in range(size):
            prop = prop + dm[ii,jj]*hfw1[ii,jj]
    return prop

def sfx2c1e_prop_elec(scalarscf, h4c1):
    xmol, contr_coeff = sfx2c1e.SpinFreeX2C(scalarscf.mol).get_xmol()
        
    s = xmol.intor_symmetric('int1e_ovlp')
    t = xmol.intor_symmetric('int1e_kin')
    v = xmol.intor_symmetric('int1e_nuc')
    w = xmol.intor_symmetric('int1e_pnucp')
    hfw1 = x2c_grad.x2c1e_hfw1(t,v,w,s,h4c1)
    hfw1 = reduce(numpy.dot, (contr_coeff.T, hfw1, contr_coeff))
    dm = scalarscf.make_rdm1()
    if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
        # UHF denisty matrices
        dm = dm[0] + dm[1]
    return first_elec_prop_scf(dm, hfw1)

def x2c1e_prop_elec(spinorscf, h4c1):
    xmol = spinorscf.mol        
    c = LIGHT_SPEED
    t = xmol.intor_symmetric('int1e_spsp_spinor') * .5
    v = xmol.intor_symmetric('int1e_nuc_spinor')
    s = xmol.intor_symmetric('int1e_ovlp_spinor')
    w = xmol.intor_symmetric('int1e_spnucsp_spinor')
    hfw1 = x2c_grad.x2c1e_hfw1(t,v,w,s,h4c1)

    return first_elec_prop_scf(spinorscf.make_rdm1(), hfw1)


    
def sfx2c1e_dipole(scalarscf):
    c = LIGHT_SPEED
    xmol, contr_coeff = sfx2c1e.SpinFreeX2C(scalarscf.mol).get_xmol()
    charges = xmol.atom_charges()
    mass = xmol.atom_mass_list()
    coords  = xmol.atom_coords()
    mass_center = numpy.einsum('i,ix->x', mass, coords)/sum(mass)
    coords_COM = coords
    for ii in range(coords_COM.shape[0]):
        coords_COM[ii] -= mass_center
    dip_nuc = numpy.einsum('i,ix->x', charges, coords_COM)
    with xmol.with_common_orig(mass_center):
        intDLL = xmol.intor_symmetric('int1e_r')
        intDSS = xmol.intor_symmetric('int1e_sprsp')
    size2c = intDLL.shape[1]
    intDSS = intDSS.reshape(3,4,size2c,size2c)[:,0] # take the scalar part
    h4c1 = numpy.zeros((3, size2c*2, size2c*2), dtype=intDLL.dtype)
    dip_elec = numpy.zeros((3))
    for xx in range(3):
        h4c1[xx,:size2c,:size2c] = -1.0*intDLL[xx]
        h4c1[xx,size2c:,size2c:] = -1.0*intDSS[xx]/4.0/c/c
        dip_elec[xx] = sfx2c1e_prop_elec(scalarscf, h4c1[xx])

    return dip_elec + dip_nuc

def x2c1e_dipole(spinorscf):
    c = LIGHT_SPEED
    xmol = spinorscf.mol
    charges = xmol.atom_charges()
    mass = xmol.atom_mass_list()
    coords  = xmol.atom_coords()
    mass_center = numpy.einsum('i,ix->x', mass, coords)/sum(mass)
    coords_COM = coords
    for ii in range(coords_COM.shape[0]):
        coords_COM[ii] -= mass_center
    dip_nuc = numpy.einsum('i,ix->x', charges, coords_COM)
    with xmol.with_common_orig(mass_center):
        intDLL = xmol.intor_symmetric('int1e_r_spinor')
        intDSS = xmol.intor_symmetric('int1e_sprsp_spinor')
    size2c = intDLL.shape[1]
    h4c1 = numpy.zeros((3, size2c*2, size2c*2), dtype=intDLL.dtype)
    dip_elec = numpy.zeros((3))
    for xx in range(3):
        h4c1[xx,:size2c,:size2c] = -1.0*intDLL[xx]
        h4c1[xx,size2c:,size2c:] = -1.0*intDSS[xx]/4.0/c/c
        dip_elec[xx] = x2c1e_prop_elec(spinorscf, h4c1[xx]).real

    return dip_elec + dip_nuc

# In x2camf-spinor calculations, h^{X2CAMF} = h^{X2C-1e} + h^{2c,AMF}
# h^{2c,AMF} integrals do not respond to external perturbations in the property calculations.
# This is rigorous for geometric gradient but an approximation in electric or magnetic properties.
x2camf_prop_elec = x2c1e_prop_elec
x2camf_dipole = x2c1e_dipole