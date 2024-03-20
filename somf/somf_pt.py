import numpy, scipy
import time
from functools import reduce
from pyscf import gto, scf, lib
from pyscf.gto import moleintor
from pyscf.lib.parameters import LIGHT_SPEED
from pyscf.x2c import x2c
from pyscf.x2c import sfx2c1e
from socutils.somf import x2c_grad
from socutils.tools.spinor2sph import spinor2sph_soc

x2camf  = None
try:
    import x2camf
except ImportError:
    pass

def get_psoc_x2camf(mol, gaunt=True, gauge=True, atm_pt=True):
    '''
    Perturbative treatment of spin-orbit coupling within X2CAMF scheme.
    The SOC contributions are evaluated in a consistent way to include only first-order terms.

    Ref.
    Mol. Phys. 118, e1768313 (2020); DOI:10.1080/00268976.2020.1768313
    Attention: The formula in the reference paper correspond to the case of atm_pt = False.

    Args:
        mol: molecule object
        gaunt: include Gaunt integrals
        gauge: include gauge integrals
        atm_pt: use atomic perturbation
        atm_pt = False:
            The molecular one-electron 4c SO integral (Wso) and AMF two-electron 4c SO integrals are
            treated as the perturbation together.
        atm_pt = True:
            The molecular one-electron 4c SO integral (Wso) is treated as the perturbation solely and
            then augmented with the AMF pt integrals.
        These two choices are supposed to give very similar results.

    Returns:
        Perturbative SOC integrals in pauli representation
    '''
    xmol, contr_coeff = sfx2c1e.SpinFreeX2C(mol).get_xmol()

    c = LIGHT_SPEED
    t = xmol.intor_symmetric('int1e_spsp_spinor') * .5
    v = xmol.intor_symmetric('int1e_nuc_spinor')
    s = xmol.intor_symmetric('int1e_ovlp_spinor')
    wsf = xmol.intor_symmetric('int1e_pnucp_spinor')
    wso = xmol.intor_symmetric('int1e_spnucsp_spinor') - wsf

    n2c = s.shape[0]
    n4c = n2c * 2

    h4c1 = numpy.zeros((n4c, n4c), dtype=v.dtype)
    h4c1[n2c:, n2c:] = wso * (.25 / c**2)

    if(atm_pt):
        hfw1 = x2c_grad.x2c1e_hfw1(t,v,wsf,s,h4c1)
        hfw1 += x2camf.amfi(x2c.X2C(mol), printLevel = mol.verbose,
                            with_gaunt=gaunt, with_gauge=gauge, pt=True, int4c=False)
    else:
        h4c1 += x2camf.amfi(x2c.X2C(mol), printLevel = mol.verbose,
                            with_gaunt=gaunt, with_gauge=gauge, pt=True, int4c=True)
        hfw1 = x2c_grad.x2c1e_hfw1(t,v,wsf,s,h4c1)
    
    hfw1_sph = spinor2sph_soc(xmol, hfw1)
    # convert back to contracted basis
    result = numpy.zeros((4, mol.nao_nr(), mol.nao_nr()))
    for ic in range(4):
        result[ic] = reduce(numpy.dot, (contr_coeff.T, hfw1_sph[ic], contr_coeff))
    return result

'''
First-order properties
'''
def first_prop_scf(dm, hfw1, eri1 = None, S1 = None):
    # dE/dx = \sum_{mn} D_{mn} h1_{mn} + 0.5\sum_{mnsr} Gamma_{mn,sr} <mn|sr>1 + \sum_{mn} I_{mn} S1_{mn}
    size = dm.shape[0]
    prop = 0.0
    if(eri1 is not None or S1 is not None):
        NotImplementedError("Analytical gradients with none-zero eri1 and S1 are not implemented yet.")
    else:
        for ii in range(size):
            for jj in range(size):
                prop = prop + dm[ii,jj]*hfw1[ii,jj]
    return prop

def x2c1e_elec_prop(method, t, v, w, s, integralKernel, mol = None, contr_coeff = None):
    if(mol is None):
        mol = method.mol
    if(contr_coeff is None):
        contr_coeff = numpy.eye(t.shape[0])

    dm = method.make_rdm1()
    if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
        # UHF denisty matrices
        dm = dm[0] + dm[1]

    a, e, x, st, r, l, h4c, m4c = x2c_grad.x2c1e_hfw0(t, v, w, s)

    h4c1, components = integralKernel(mol)
    if components == 1:
        hfw1 = x2c_grad.get_hfw1(a, x, st, m4c, h4c, e, r, l, h4c1)
        hfw1 = reduce(numpy.dot, (contr_coeff.T, hfw1, contr_coeff))
        return first_prop_scf(dm, hfw1)
    else:
        prop = numpy.zeros((components))
        for xx in range(components):
            hfw1 = x2c_grad.get_hfw1(a, x, st, m4c, h4c, e, r, l, h4c1[xx])
            hfw1 = reduce(numpy.dot, (contr_coeff.T, hfw1, contr_coeff))
            prop[xx] = first_prop_scf(dm, hfw1)
        return prop

def sfx2c1e_dipole(method):
    xmol, contr_coeff = sfx2c1e.SpinFreeX2C(method.mol).get_xmol()
    s = xmol.intor_symmetric('int1e_ovlp')
    t = xmol.intor_symmetric('int1e_kin')
    v = xmol.intor_symmetric('int1e_nuc')
    w = xmol.intor_symmetric('int1e_pnucp')
    elec_dipole = x2c1e_elec_prop(method, t, v, w, s, elec_dipole_integral_scalar, xmol, contr_coeff)
    nuc_dipole, mc = nuclear_dipole(xmol)
    return nuc_dipole + elec_dipole

def x2c1e_dipole(method):
    xmol = method.mol
    t = xmol.intor_symmetric('int1e_spsp_spinor') * .5
    v = xmol.intor_symmetric('int1e_nuc_spinor')
    s = xmol.intor_symmetric('int1e_ovlp_spinor')
    w = xmol.intor_symmetric('int1e_spnucsp_spinor')
    elec_dipole = x2c1e_elec_prop(method, t, v, w, s, elec_dipole_integral_spinor, xmol)
    nuc_dipole, mc = nuclear_dipole(xmol)
    return nuc_dipole + elec_dipole

def nuclear_dipole(mol):
    charges = mol.atom_charges()
    mass = mol.atom_mass_list()
    coords = mol.atom_coords()
    mass_center = numpy.einsum('i,ix->x', mass, coords)/sum(mass)
    coords_COM = coords
    for ii in range(coords_COM.shape[0]):
        coords_COM[ii] -= mass_center
    return numpy.einsum('i,ix->x', charges, coords_COM), mass_center

def elec_dipole_integral_scalar(mol):
    c = LIGHT_SPEED
    nuc_dipole, mass_center = nuclear_dipole(mol)
    with mol.with_common_orig(mass_center):
        intDLL = mol.intor_symmetric('int1e_r')
        size = intDLL.shape[1]
        intDSS_spinor = mol.intor_symmetric('int1e_sprsp_spinor')
        intDSS = numpy.zeros((3,size,size))
        for xx in range(3):
            intDSS[xx] = spinor2sph_soc(mol, intDSS_spinor[xx])[0]
    size2c = intDLL.shape[1]
    h4c1 = numpy.zeros((3, size2c*2, size2c*2), dtype=intDLL.dtype)
    for xx in range(3):
        h4c1[xx,:size2c,:size2c] = -1.0*intDLL[xx]
        h4c1[xx,size2c:,size2c:] = -1.0*intDSS[xx]/4.0/c/c
    return h4c1, 3

def elec_dipole_integral_spinor(mol):
    c = LIGHT_SPEED
    nuc_dipole, mass_center = nuclear_dipole(mol)
    with mol.with_common_orig(mass_center):
        intDLL = mol.intor_symmetric('int1e_r_spinor')
        intDSS = mol.intor_symmetric('int1e_sprsp_spinor')
    size2c = intDLL.shape[1]
    h4c1 = numpy.zeros((3, size2c*2, size2c*2), dtype=intDLL.dtype)
    for xx in range(3):
        h4c1[xx,:size2c,:size2c] = -1.0*intDLL[xx]
        h4c1[xx,size2c:,size2c:] = -1.0*intDSS[xx]/4.0/c/c
    return h4c1, 3

# In x2camf-spinor calculations, h^{X2CAMF} = h^{X2C-1e} + h^{2c,AMF}
# h^{2c,AMF} integrals do not respond to external perturbations in the property calculations.
# This is rigorous for geometric gradient but a convention in electric or magnetic properties.
x2camf_dipole = x2c1e_dipole
