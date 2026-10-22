'''
Two-component (j-spinor) Hamiltonians from socutils X2C SCF objects.

The one-electron operator is the SCF object's hcore (e.g. X2CAMF or X2CMP
including spin-orbit terms, or the spin-free 'sf1e' flavor); the two-electron
operator is the bare Coulomb interaction transformed with the j-spinor MO
coefficients, as in socutils.mcscf.zcasci, either exactly or from the
Cholesky vectors of a Cholesky-based CASSCF.
'''

import re

import numpy
from pyscf.ao2mo import nrr_outcore

from socutils.zmr.integrals import ERIs
from socutils.zmr.interface.ao2mo import IncoreAO2MO
from socutils.zmr.interface.cholesky import CholeskyAO2MO
from socutils.zmr.spaces import OrbitalSpaces


def _check_mol(mol, mo_coeff):
    if getattr(mol, 'so_contr', None) is not None:
        raise NotImplementedError('SO-contracted basis sets are not supported')
    if mo_coeff.shape[0] != mol.nao_2c():
        raise ValueError('mo_coeff is not in the j-spinor AO basis')


def spinor_eri(mol, mo_coeffs):
    '''(pq|rs) for four blocks of j-spinor MO coefficients, chemists'
    notation, p and q on electron 1.'''
    shape = tuple(c.shape[1] for c in mo_coeffs)
    eri = nrr_outcore.general_iofree(mol, mo_coeffs, intor='int2e_sph', motype='j-spinor')
    return numpy.asarray(eri).reshape(shape)


def from_mo_coeff(mol, hcore_ao, mo_coeff, nfrozen, ncore, nact):
    '''Spinor Hamiltonian.  All counts are numbers of spinors: the first
    nfrozen are frozen, then ncore correlated core, nact active, rest
    virtual.'''
    _check_mol(mol, mo_coeff)
    nmo = mo_coeff.shape[1]
    nvirt = nmo - nfrozen - ncore - nact
    if min(nfrozen, ncore, nact, nvirt) < 0:
        raise ValueError('inconsistent orbital partition')
    h = mo_coeff.conj().T @ hcore_ao @ mo_coeff
    eri = spinor_eri(mol, (mo_coeff,) * 4)
    f = slice(0, nfrozen)
    corr = slice(nfrozen, nmo)
    ff = eri[f, f, f, f]
    ecore = (mol.energy_nuc() + numpy.trace(h[f, f])
             + 0.5 * (numpy.einsum('iijj->', ff) - numpy.einsum('ijji->', ff)))
    h1 = h[corr, corr] + (numpy.einsum('pqff->pq', eri[corr, corr, f, f])
                          - numpy.einsum('pffq->pq', eri[corr, f, f, corr]))
    return ERIs(ecore.real, h1, eri[corr, corr, corr, corr],
                       OrbitalSpaces(ncore, nact, nvirt))


def mo_eris(mf, mol, mo_coeff, nfrozen, ncore, nact, cderi=None):
    '''Like from_mo_coeff, but integrals are transformed block by block on
    demand (zmr.integrals.MOERIs).

    cderi=None: exact integrals; the mean-field potential is mf.get_jk (bare
    Coulomb, as in socutils zcasci).  With cderi, the Cholesky vectors of the
    real spherical AO integrals (packed or square, zmr.interface.cholesky),
    integral blocks and mean-field potential are built from these vectors,
    i.e. with the two-electron operator of a Cholesky-based socutils CASSCF.'''
    from socutils.zmr.integrals import MOERIs
    from socutils.zmr.interface.pyscf_nr import _fold_frozen

    _check_mol(mol, mo_coeff)
    nvirt = mo_coeff.shape[1] - nfrozen - ncore - nact
    if min(nfrozen, ncore, nact, nvirt) < 0:
        raise ValueError('inconsistent orbital partition')

    ca, cb = mol.sph2spinor_coeff()

    def components(c):
        return ca @ c, cb @ c

    if cderi is None:
        def veff_fn(dm):
            vj, vk = mf.get_jk(mol, dm, hermi=0)
            return vj - vk

        def outcore(*cs):
            return nrr_outcore.general_iofree(mol, cs, intor='int2e_sph', motype='j-spinor',
                                              verbose=0)

        eri_fn = IncoreAO2MO(mol, components, outcore)
    else:
        eri_fn = CholeskyAO2MO(cderi, mol.nao_nr(), components)
        veff_fn = eri_fn.veff

    return _fold_frozen(MOERIs, mol, mf.get_hcore(), mo_coeff, nfrozen,
                        OrbitalSpaces(ncore, nact, nvirt), eri_fn, veff_fn)


def from_scf(mf, ncore, nact, nfrozen=0, mo_coeff=None, mol=None, lazy=True, cderi=None):
    '''Spinor Hamiltonian from a socutils spinor SCF object.  mol may differ
    from mf.mol in charge/spin only (e.g. orbitals of an ion).  lazy=True
    returns an MOERIs, otherwise all integrals are stored.  cderi: Cholesky
    vectors that replace the exact two-electron integrals (see mo_eris).'''
    mol = mf.mol if mol is None else mol
    mo_coeff = mf.mo_coeff if mo_coeff is None else mo_coeff
    if lazy:
        return mo_eris(mf, mol, mo_coeff, nfrozen, ncore, nact, cderi=cderi)
    if cderi is not None:
        return mo_eris(mf, mol, mo_coeff, nfrozen, ncore, nact, cderi=cderi).to_dense()
    return from_mo_coeff(mol, mf.get_hcore(), mo_coeff, nfrozen, ncore, nact)


_MJ = re.compile(r',\s*(-?\d+)/2\s*$')


def mj_of_spinor_aos(mol):
    '''m_j of every j-spinor AO, parsed from mol.spinor_labels().'''
    mj = []
    for label in mol.spinor_labels():
        m = _MJ.search(label)
        if m is None:
            raise ValueError('cannot parse m_j from %r' % label)
        mj.append(int(m.group(1)) / 2)
    return numpy.array(mj)


def jz_mo(mol, mo_coeff):
    '''Jz in the MO basis for a linear molecule on the z axis:
    Jz|mu> = m_j(mu)|mu> for every j-spinor AO centred on the axis.'''
    if abs(mol.atom_coords()[:, :2]).max() > 1e-10:
        raise ValueError('all atoms must lie on the z axis')
    s = mol.intor('int1e_ovlp_spinor')
    return mo_coeff.conj().T @ (s * mj_of_spinor_aos(mol)[None, :]) @ mo_coeff
