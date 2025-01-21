'''
Dirac Hartree-Fock hyperfine coupling tensor (In testing)

Refs: JCP 134, 044111 (2011); DOI:10.1063/1.3526263
'''

from functools import reduce
import warnings
import numpy
from pyscf import lib
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor

warnings.warn('Module HFC is under testing')
au2MHz = nist.HARTREE2J / nist.PLANCK * 1e-6

def make_h01(mol, atm_id):
    mol.set_rinv_origin(mol.atom_coord(atm_id))
    t1 = mol.intor('int1e_sa01sp_spinor', 3)
    n2c = t1.shape[2]
    n4c = n2c * 2
    h1 = numpy.zeros((3, n4c, n4c), complex)
    for i in range(3):
        h1[i,:n2c,n2c:] += .5 * t1[i]
        h1[i,n2c:,:n2c] += .5 * t1[i].conj().T
    return h1

def int_hfc_4c(mol, atm_id, utm=False, h4c=None, m4c_inv=None):
    symb = mol.atom_symbol(atm_id)
    nuc_mag = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
    nuc_gyro = get_nuc_g_factor(symb) * nuc_mag
    e_gyro = .5 * nist.G_ELECTRON
    fac = nist.ALPHA**2 * nuc_gyro * e_gyro
    # fac = 0.5 # to compare with CFOUR
    int_4c = make_h01(mol, atm_id) * fac

    n2c = mol.nao_2c()
    for xx in range(3):
        if utm:
            if h4c is None or m4c_inv is None:
                raise ValueError('h4c and m4c_inv should be provided with utm=True')
            utm_tau = numpy.zeros_like(int_4c[xx])
            utm_tau[:n2c, n2c:] = -int_4c[xx][:n2c, n2c:]/2.0/nist.LIGHT_SPEED**2
            utm_tau[n2c:, :n2c] = int_4c[xx][n2c:, :n2c]/2.0/nist.LIGHT_SPEED**2
            int_4c[xx] += reduce(numpy.dot, (h4c, m4c_inv, utm_tau)) - reduce(numpy.dot, (utm_tau, m4c_inv, h4c))
    
    return int_4c


def kernel(method, hfc_nuc=None, utm=False, h4c_scheme='fock', dm=None):
    log = lib.logger.Logger(method.stdout, method.verbose)
    log.info('\n******** HFC for 4-component SCF methods (In testing) ********')
    if hfc_nuc is None:
        hfc_nuc = range(mol.natm)

    if dm is None:
        dm = method.make_rdm1()

    log.info('\nMagnetic Hyperfine Constants Results')
    n2c = mol.nao_2c()
    c = nist.LIGHT_SPEED
    if utm:
        t = mol.intor('int1e_kin_spinor')
        s = mol.intor('int1e_ovlp_spinor')
        v = mol.intor('int1e_nuc_spinor')
        w = mol.intor('int1e_spnucsp_spinor')
        h4c = numpy.zeros((n2c*2, n2c*2), dtype=v.dtype)
        m4c = numpy.zeros((n2c*2, n2c*2), dtype=v.dtype)
        if h4c_scheme == '1e':
            h4c[:n2c, :n2c] = v
            h4c[:n2c, n2c:] = t
            h4c[n2c:, :n2c] = t
            h4c[n2c:, n2c:] = w * (.25 / c**2) - t
            m4c[:n2c, :n2c] = s
            m4c[n2c:, n2c:] = t * (.5 / c**2)
        elif h4c_scheme == 'fock':
            h4c = method.get_fock()
            m4c = method.get_ovlp()
        else:
            raise ValueError
        m4c_inv = numpy.linalg.inv(m4c)
    else:
        h4c = m4c_inv = None
    
    hfc = []
    for i, atm_id in enumerate(hfc_nuc):
        int_4c = int_hfc_4c(mol, atm_id, utm=utm, h4c=h4c, m4c_inv=m4c_inv)
        hfc_atm = numpy.zeros(3, dtype=dm.dtype)
        for xx in range(3):
            hfc_atm[xx] = numpy.einsum('ij,ji->', int_4c[xx], dm)
        hfc.append(hfc_atm)
        print('\nAtom %d' % atm_id)
        print('HFC (a.u.):')
        print(hfc_atm.real)
        # print('HFC (MHz):')
        # print(hfc_atm.real * au2MHz)

    return numpy.asarray(hfc).real

# def kernel(hfcobj, verbose=None):
#     log = lib.logger.new_logger(hfcobj, verbose)
#     mf = hfcobj._scf
#     mol = mf.mol
# # Add finite field to remove degeneracy
#     nuc_spin = numpy.ones(3) * 1e-6
#     sc = numpy.dot(mf.get_ovlp(), mf.mo_coeff)
#     h0 = reduce(numpy.dot, (sc*mf.mo_energy, sc.conj().T))
#     c = lib.param.LIGHT_SPEED
#     n4c = h0.shape[0]
#     n2c = n4c // 2
#     Sigma = numpy.zeros((3,n4c,n4c), dtype=h0.dtype)
#     Sigma[:,:n2c,:n2c] = mol.intor('int1e_sigma_spinor', comp=3)
#     Sigma[:,n2c:,n2c:] = .25/c**2 * mol.intor('int1e_spsigmasp_spinor', comp=3)

#     hfc = []
#     for atm_id in range(mol.natm):
#         symb = mol.atom_symbol(atm_id)
#         nuc_mag = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
#         nuc_gyro = get_nuc_g_factor(symb) * nuc_mag
#         e_gyro = .5 * nist.G_ELECTRON
#         au2MHz = nist.HARTREE2J / nist.PLANCK * 1e-6
#         fac = nist.ALPHA**2 * nuc_gyro * e_gyro * au2MHz
#         #logger.debug('factor (MHz) %s', fac)

#         h01 = make_h01(mol, 0)
#         mo_occ = mf.mo_occ
#         mo_coeff = mf.mo_coeff
#         if 0:
#             h01b = h0 + numpy.einsum('xij,x->ij', h01, nuc_spin)
#             h01b = reduce(numpy.dot, (mf.mo_coeff.conj().T, h01b, mf.mo_coeff))
#             mo_energy, v = numpy.linalg.eigh(h01b)
#             mo_coeff = numpy.dot(mf.mo_coeff, v)
#             mo_occ = mf.get_occ(mo_energy, mo_coeff)

#         occidx = mo_occ > 0
#         orbo = mo_coeff[:,occidx]
#         dm0 = numpy.dot(orbo, orbo.T.conj())
#         e01 = numpy.einsum('xij,ji->x', h01, dm0) * fac

#         effspin = numpy.einsum('xij,ji->x', Sigma, dm0) * .5
#         log.debug('atom %d Eff-spin %s', atm_id, effspin.real)

#         e01 = (e01 / effspin).real
#         hfc.append(e01)
#     return numpy.asarray(hfc)

HFC = kernel
from pyscf.scf.dhf import DHF
DHF.HFC = lib.class_as_method(HFC)

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.M(
        atom = """
O        -0.00000000     0.00000000     0.68645937
H         0.00000000     0.00000000    -1.20326715
""",
        basis = 'ccpvdz', spin=1, unit="Bohr")
    mf = scf.DHF(mol).run()
    mf.HFC()
