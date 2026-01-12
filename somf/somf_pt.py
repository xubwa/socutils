import numpy
import scipy
import time
from functools import reduce
from pyscf import gto, scf, lib, dft
from pyscf.gto import moleintor
from pyscf.lib.parameters import LIGHT_SPEED
from pyscf.x2c import x2c
from pyscf.x2c import sfx2c1e
from socutils.somf import x2c_grad
from socutils.somf import x2cmp
from socutils.tools import addons
from socutils.tools.spinor2sph import spinor2sph_soc, spinor2spinor_sd
from socutils.scf.spinor_hf import sph2spinor, SpinorSCF
from socutils.dft.dft import SpinorDFT

x2camf = None
try:
    import x2camf
except ImportError:
    pass


def get_psoc_somf(mol, dm0, gaunt=True, gauge=True, form='scalar'):
    xmol, contr_coeff_nr = sfx2c1e.SpinFreeX2C(mol).get_xmol()
    x2cobj = x2cmp.SpinorX2CMPHelper(mol, x2cmp='sf1e')
    np, nc = contr_coeff_nr.shape
    contr_coeff = numpy.zeros((np * 2, nc * 2))
    contr_coeff[0::2, 0::2] = contr_coeff_nr
    contr_coeff[1::2, 1::2] = contr_coeff_nr

    c = LIGHT_SPEED
    wsf = xmol.intor('int1e_pnucp_spinor')
    wso = xmol.intor('int1e_spnucsp_spinor') - wsf

    n2c = wsf.shape[0]
    n4c = n2c * 2

    h4c1 = numpy.zeros((n4c, n4c), dtype=wso.dtype)
    h4c1[n2c:, n2c:] = wso * (.25 / c**2)

    mf_4c = scf.DHF(mol)
    mf_4c.with_gaunt = gaunt
    mf_4c.with_breit = gauge

    # start to handle two electron contributions

    # transform dm to uncontract form
    dm = reduce(numpy.dot, (contr_coeff, dm0, contr_coeff.T))
    nprim = dm.shape[0]
    dm_g = numpy.zeros((nprim, nprim), dtype=dm0.dtype)
    dm_2c = addons.ghf2jhf_dm(mol, dm_g)

    x, r = x2cobj.get_xr(xmol)
    dm4c = addons.hf2dhf_dm(dm_2c, x, r)

    vj_4c, vk_4c = mf_4c.get_jk(dm=dm4c)
    veff_4c = vj_4c - vk_4c

    h4c1 = spinor2spinor_sd(mol, veff_4c)

    hfw1 = x2c_grad.x2c1e_hfw1(xmol, h4c1, w=wsf)
    hfw1 = reduce(numpy.dot, (contr_coeff.T, hfw1, contr_coeff))
    
    if form == "spinor":
        return hfw1
    elif form == "sph":
        ca, cb = mol.sph2spinor_coeff()
        nao = mol.nao_nr()
        hso = numpy.zeros_like(hfw1, dtype=complex)
        hso[:nao, :nao] = reduce(numpy.dot, (ca, hfw1, ca.conj().T))
        hso[nao:, nao:] = reduce(numpy.dot, (cb, hfw1, cb.conj().T))
        hso[:nao, nao:] = reduce(numpy.dot, (ca, hfw1, cb.conj().T))
        hso[nao:, :nao] = reduce(numpy.dot, (cb, hfw1, ca.conj().T))
        hfw1 = hso
        return hfw1
    elif form == "scalar":
        hfw1 = spinor2sph_soc(mol, hfw1)
        return hfw1
    else:
        raise ValueError("Unknown form")


def get_psoc_x2camf(mol, gaunt=True, gauge=True, atm_pt=True, form="scalar"):
    r'''
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
        form: what to actually return
            "scalar": return the scalar integrals
            "spinor": return the spinor integrals
            "sph": return the 2c-spherical integrals

    Returns:
        Perturbative SOC integrals in pauli representation
    '''
    xmol, contr_coeff_nr = sfx2c1e.SpinFreeX2C(mol).get_xmol()
    np, nc = contr_coeff_nr.shape
    contr_coeff = numpy.zeros((np * 2, nc * 2))
    contr_coeff[0::2, 0::2] = contr_coeff_nr
    contr_coeff[1::2, 1::2] = contr_coeff_nr

    c = LIGHT_SPEED
    wsf = xmol.intor('int1e_pnucp_spinor')
    wso = xmol.intor('int1e_spnucsp_spinor') - wsf

    n2c = wsf.shape[0]
    n4c = n2c * 2

    h4c1 = numpy.zeros((n4c, n4c), dtype=wso.dtype)
    h4c1[n2c:, n2c:] = wso * (.25 / c**2)

    if (atm_pt):
        hfw1 = x2c_grad.x2c1e_hfw1(xmol, h4c1, w=wsf)
        hfw1 += x2camf.amfi(x2c.X2C(mol), printLevel=mol.verbose,
                            with_gaunt=gaunt, with_gauge=gauge, pt=True, int4c=False)
    else:
        h4c1 += x2camf.amfi(x2c.X2C(mol), printLevel=mol.verbose,
                            with_gaunt=gaunt, with_gauge=gauge, pt=True, int4c=True)
        hfw1 = x2c_grad.x2c1e_hfw1(xmol, h4c1, w=wsf)

    hfw1 = reduce(numpy.dot, (contr_coeff.T, hfw1, contr_coeff))
    if form == "spinor":
        return hfw1
    elif form == "sph":
        ca, cb = mol.sph2spinor_coeff()
        nao = mol.nao_nr()
        hso = numpy.zeros_like(hfw1, dtype=complex)
        hso[:nao, :nao] = reduce(numpy.dot, (ca, hfw1, ca.conj().T))
        hso[nao:, nao:] = reduce(numpy.dot, (cb, hfw1, cb.conj().T))
        hso[:nao, nao:] = reduce(numpy.dot, (ca, hfw1, cb.conj().T))
        hso[nao:, :nao] = reduce(numpy.dot, (cb, hfw1, ca.conj().T))
        hfw1 = hso
        return hfw1
    elif form == "scalar":
        hfw1 = spinor2sph_soc(mol, hfw1)
        return hfw1
    else:
        raise ValueError("Unknown form")


def get_psoc_x2c1e(mol, form="scalar"):
    xmol, contr_coeff_nr = sfx2c1e.SpinFreeX2C(mol).get_xmol()
    np, nc = contr_coeff_nr.shape
    contr_coeff = numpy.zeros((np * 2, nc * 2))
    contr_coeff[0::2, 0::2] = contr_coeff_nr
    contr_coeff[1::2, 1::2] = contr_coeff_nr

    c = LIGHT_SPEED
    wsf = xmol.intor('int1e_pnucp_spinor')
    wso = xmol.intor('int1e_spnucsp_spinor') - wsf

    n2c = wsf.shape[0]
    n4c = n2c * 2

    h4c1 = numpy.zeros((n4c, n4c), dtype=wso.dtype)
    h4c1[n2c:, n2c:] = wso * (.25 / c**2)
    hfw1 = x2c_grad.x2c1e_hfw1(xmol, h4c1, w=wsf)

    hfw1 = reduce(numpy.dot, (contr_coeff.T, hfw1, contr_coeff))
    if form == "spinor":
        return hfw1
    elif form == "sph":
        ca, cb = mol.sph2spinor_coeff()
        nao = mol.nao_nr()
        hso = numpy.zeros_like(hfw1, dtype=complex)
        hso[:nao, :nao] = reduce(numpy.dot, (ca, hfw1, ca.conj().T))
        hso[nao:, nao:] = reduce(numpy.dot, (cb, hfw1, cb.conj().T))
        hso[:nao, nao:] = reduce(numpy.dot, (ca, hfw1, cb.conj().T))
        hso[nao:, :nao] = reduce(numpy.dot, (cb, hfw1, ca.conj().T))
        hfw1 = hso
        return hfw1
    elif form == "scalar":
        hfw1 = spinor2sph_soc(mol, hfw1)
        return hfw1
    else:
        raise ValueError("Unknown form")


def get_soc_bp1e(mol, form="scalar"):
    nao = mol.nao_nr()
    hfw_scalar = mol.intor("int1e_pnucxp").reshape(
        3, nao, nao) / 2.0 / LIGHT_SPEED**2 / 2.0  # another 2.0 for spin operator
    if form == "scalar":
        return hfw_scalar
    hfw_sph = scalar2sph(mol, hfw_scalar)
    if form == "sph":
        return hfw_sph
    elif form == "spinor":
        return sph2spinor(mol, hfw_sph)
    else:
        raise ValueError("Unknown form")


def scalar2sph(mol, ints_scalar):
    # Construct 2c matrix using Pauli matrices
    nao = mol.nao_nr()
    assert ints_scalar.shape[0] == 3
    ints_sph = numpy.zeros((nao*2, nao*2), dtype=numpy.complex128)
    ints_sph[:nao, :nao] = 1.0j*ints_scalar[2]
    ints_sph[nao:, nao:] = -1.0j*ints_scalar[2]
    ints_sph[:nao, nao:] = 1.0j*ints_scalar[0] + ints_scalar[1]
    ints_sph[nao:, :nao] = 1.0j*ints_scalar[0] - ints_scalar[1]
    return ints_sph


def _get_soc_mf_rhf(mol, dm, soo=True):
    from pyscf.scf import jk
    fac = 0.5*0.5 / LIGHT_SPEED**2  # another 0.5 for RHF density
    fso_0, fso_1, fso_2 = jk.get_jk(
        mol, [dm, dm, dm],
        scripts=['ijkl,lk->ij', 'ijkl,jk->il', 'ijkl,li->kj'],
        intor='cint2e_p1vxp1_sph')
    if soo:
        return fac * (fso_0 - 1.5*fso_1 - 1.5*fso_2)
    else:
        output = numpy.zeros_like(fso_0)
        output[0] = fac*(0.5*fso_0[0] - 0.5*fso_1[0] - 0.5*fso_2[0])
        output[1] = fac*(0.5*fso_0[1] - 0.5*fso_1[1] - 0.5*fso_2[1])
        output[2] = fac*(fso_0[2] - 0.5*fso_1[2] - 0.5*fso_2[2])
        return output


def _get_soc_mf_bp(mol, dmaa, dmbb, dmab, soo=True):
    s_vec = [[[0, 0, 0.5], [0.5, -0.5j, 0]], [[0.5, 0.5j, 0], [0, 0, -0.5]]]
    nao = mol.nao_nr()
    dm = numpy.zeros((2*nao, 2*nao), dtype=numpy.complex128)
    dm[:nao, :nao] = dmaa
    dm[nao:, nao:] = dmbb
    dm[:nao, nao:] = dmab
    dm[nao:, :nao] = dmab.conj().T
    rp_ints = mol.intor("int2e_p1vxp1", comp=3).reshape(3, nao, nao, nao, nao)
    soc2_ints = numpy.zeros(
        (2*nao, 2*nao, 2*nao, 2*nao), dtype=numpy.complex128)
    for a1 in range(2):
        for a2 in range(2):
            for b1 in range(2):
                for b2 in range(2):
                    tmp = numpy.zeros((nao, nao, nao, nao),
                                      dtype=numpy.complex128)
                    for k in range(3):
                        tmp += s_vec[a1][b1][k] * rp_ints[k] if a2 == b2 else 0
                        if soo:
                            tmp += 2.0 * s_vec[a2][b2][k] * \
                                rp_ints[k] if a1 == b1 else 0
                    soc2_ints[a1*nao:(a1+1)*nao, b1*nao:(b1+1)*nao, a2*nao:(a2+1)
                              * nao, b2*nao:(b2+1)*nao] = 1j * tmp / LIGHT_SPEED**2
    soc2_ints = 0.5*(soc2_ints + soc2_ints.transpose(2, 3, 0, 1))
    return numpy.einsum("msnr,sm->nr", soc2_ints, dm, optimize=True) - numpy.einsum("mrns,sm->nr", soc2_ints, dm, optimize=True)


def get_soc_mf_bp(mf, mol=None, dm=None, soo=True):
    r'''
    Mean-field approximated SOC integrals for Breit-Pauli Hamiltonian,
    including the one-electron and two-electron parts.

    Args:
        mf: scf object
        soo: include spin-other-orbit integrals

    Returns:
        Mean-field approximated SOC integrals. 
        The format depends on the input mf object:
            JHF: integrals in spinor basis
            GHF: integrals in two-component spherical basis
            RHF/UHF: integrals in scalar basis
            ROHF: not supported
    '''
    if mol is None:
        mol = mf.mol
    if dm is None:
        dm = mf.make_rdm1()
    nao = mol.nao_nr()

    if isinstance(mf, scf.uhf.UHF) or isinstance(mf, dft.uks.UKS):
        raise NotImplementedError(
            "Unrestricted density matrix is not supported.")
    elif isinstance(mf, scf.hf.RHF) or isinstance(mf, dft.rks.RKS):
        return get_soc_bp1e(mol, form="scalar") + _get_soc_mf_rhf(mol, dm, soo)
    elif isinstance(mf, scf.ghf.GHF) or isinstance(mf, dft.gks.GKS):
        return _get_soc_mf_bp(mol, dm[:nao, :nao], dm[nao:, nao:], dm[:nao, nao:], soo)\
            + get_soc_bp1e(mol, form="sph")
    elif isinstance(mf, SpinorSCF) or isinstance(mf, SpinorDFT):
        raise NotImplementedError("Spinor mean-field is not supported.")
    else:
        raise ValueError("Unknown density matrix shape.")


'''
First-order properties
'''


def first_prop_scf(dm, hfw1, eri1=None, S1=None):
    # dE/dx = \sum_{mn} D_{mn} h1_{mn} + 0.5\sum_{mnsr} Gamma_{mn,sr} <mn|sr>1 + \sum_{mn} I_{mn} S1_{mn}
    size = dm.shape[0]
    prop = 0.0
    if (eri1 is not None or S1 is not None):
        NotImplementedError(
            "Analytical gradients with none-zero eri1 and S1 are not implemented yet.")
    else:
        for ii in range(size):
            for jj in range(size):
                prop = prop + dm[ii, jj]*hfw1[ii, jj]
    return prop


def x2c1e_elec_prop(method, t, v, w, s, integral_kernel, mol=None, contr_coeff=None):
    if (mol is None):
        mol = method.mol
    if (contr_coeff is None):
        contr_coeff = numpy.eye(t.shape[0])

    dm = method.make_rdm1()
    if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
        # UHF denisty matrices
        dm = dm[0] + dm[1]

    a, e, x, st, r, l, h4c, m4c = x2c_grad.x2c1e_hfw0(t, v, w, s)

    h4c1, components = integral_kernel(mol)
    if components == 1:
        hfw1 = x2c_grad.get_hfw1(a, x, st, m4c, h4c, e, r, l, h4c1)
        hfw1 = method.with_x2c.get_hfw1(h4c1)
        hfw1 = reduce(numpy.dot, (contr_coeff.T, hfw1, contr_coeff))
        return first_prop_scf(dm, hfw1)
    else:
        prop = numpy.zeros((components))
        for xx in range(components):
            # hfw1 = x2c_grad.get_hfw1(a, x, st, m4c, h4c, e, r, l, h4c1[xx])
            hfw1 = method.with_x2c.get_hfw1(h4c1[xx])
            hfw1 = reduce(numpy.dot, (contr_coeff.T, hfw1, contr_coeff))
            iprop = first_prop_scf(dm, hfw1)
            prop[xx] = iprop.real
            print(iprop.imag)
        return prop


def sfx2c1e_dipole(method):
    xmol, contr_coeff = sfx2c1e.SpinFreeX2C(method.mol).get_xmol()
    s = xmol.intor_symmetric('int1e_ovlp')
    t = xmol.intor_symmetric('int1e_kin')
    v = xmol.intor_symmetric('int1e_nuc')
    w = xmol.intor_symmetric('int1e_pnucp')
    elec_dipole = x2c1e_elec_prop(
        method, t, v, w, s, elec_dipole_integral_scalar, xmol, contr_coeff)
    nuc_dipole, mc = nuclear_dipole(xmol)
    print(nuc_dipole, elec_dipole)
    return nuc_dipole + elec_dipole


def x2c1e_dipole(method):
    xmol = method.mol
    t = xmol.intor_symmetric('int1e_spsp_spinor') * .5
    v = xmol.intor_symmetric('int1e_nuc_spinor')
    s = xmol.intor_symmetric('int1e_ovlp_spinor')
    w = xmol.intor_symmetric('int1e_spnucsp_spinor')
    elec_dipole = x2c1e_elec_prop(
        method, t, v, w, s, elec_dipole_integral_spinor, xmol)
    nuc_dipole, mc = nuclear_dipole(xmol)
    print(nuc_dipole, elec_dipole)
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
        intDSS = numpy.zeros((3, size, size))
        for xx in range(3):
            intDSS[xx] = spinor2sph_soc(mol, intDSS_spinor[xx])[0]
    size2c = intDLL.shape[1]
    h4c1 = numpy.zeros((3, size2c*2, size2c*2), dtype=intDLL.dtype)
    for xx in range(3):
        h4c1[xx, :size2c, :size2c] = -1.0*intDLL[xx]
        h4c1[xx, size2c:, size2c:] = -1.0*intDSS[xx]*(0.5/c)**2
    return h4c1, 3


def elec_dipole_integral_spinor(mol):
    c = LIGHT_SPEED
    nuc_dipole, mass_center = nuclear_dipole(mol)
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    with mol.with_common_orig(mass_center):
        intDLL = mol.intor_symmetric('int1e_r_spinor')
        intDSS = mol.intor_symmetric('int1e_sprsp_spinor')
    size2c = intDLL.shape[1]
    h4c1 = numpy.zeros((3, size2c*2, size2c*2), dtype=intDLL.dtype)
    for xx in range(3):
        h4c1[xx, :size2c, :size2c] = -1.0*intDLL[xx]
        h4c1[xx, size2c:, size2c:] = -1.0*intDSS[xx]*(0.5/c)**2
    return h4c1, 3


# In x2camf-spinor calculations, h^{X2CAMF} = h^{X2C-1e} + h^{2c,AMF}
# h^{2c,AMF} integrals do not respond to external perturbations in the property calculations.
# This is rigorous for geometric gradient but a convention in electric or magnetic properties.
x2camf_dipole = x2c1e_dipole
