'''
Non-relativistic (spin-free) Hamiltonians from PySCF.

For the spinor code the spatial orbitals are expanded into spin-orbitals
p = 2*P + sigma (conventions.md section 1): from_mo_coeff, from_casci,
mo_eris, ghf_eris.

For the spin-restricted code (zmr.nr) the integrals stay real and spatial:
nr_eris, nr_from_casci, nr_civecs.
'''

import numpy
from pyscf import ao2mo

from socutils.zmr.integrals import ERIs
from socutils.zmr.spaces import OrbitalSpaces


def spin_orbital_integrals(h_spatial, eri_spatial):
    '''Expand spatial h[P,Q] and (PQ|RS) into spin-orbitals p = 2P + sigma.'''
    n = h_spatial.shape[0]
    delta = numpy.eye(2)
    h = numpy.einsum('PQ,st->PsQt', h_spatial, delta).reshape(2 * n, 2 * n)
    eri = numpy.einsum('PQRS,st,uv->PsQtRuSv', eri_spatial.reshape((n,) * 4),
                       delta, delta).reshape((2 * n,) * 4)
    return h, eri


def from_mo_coeff(mol, hcore_ao, mo_coeff, nfrozen, ncore, nact):
    '''Spin-orbital Hamiltonian for spatial orbitals mo_coeff.

    The first nfrozen spatial orbitals are frozen, the next ncore are the
    correlated core, then nact active, and the rest virtual.  ncore and nact
    count spatial orbitals; the returned OrbitalSpaces counts spin-orbitals.
    '''
    nmo_sp = mo_coeff.shape[1]
    nvirt = nmo_sp - nfrozen - ncore - nact
    if min(nfrozen, ncore, nact, nvirt) < 0:
        raise ValueError('inconsistent orbital partition')
    h_sp = mo_coeff.T @ hcore_ao @ mo_coeff
    eri_sp = ao2mo.restore(1, ao2mo.kernel(mol, mo_coeff), nmo_sp)
    h, eri = spin_orbital_integrals(h_sp, eri_sp)

    nf = 2 * nfrozen
    f = slice(0, nf)
    corr = slice(nf, 2 * nmo_sp)
    ff = eri[f, f, f, f]
    ecore = (mol.energy_nuc() + numpy.trace(h[f, f])
             + 0.5 * (numpy.einsum('iijj->', ff) - numpy.einsum('ijji->', ff)))
    h1 = h[corr, corr] + (numpy.einsum('pqff->pq', eri[corr, corr, f, f])
                          - numpy.einsum('pffq->pq', eri[corr, f, f, corr]))
    spaces = OrbitalSpaces(2 * ncore, 2 * nact, 2 * nvirt)
    return ERIs(ecore.real, h1, eri[corr, corr, corr, corr], spaces)


def from_casci(mc, nfrozen=0, mo_coeff=None, lazy=True, cderi=None):
    '''Spin-orbital Hamiltonian for a PySCF (RHF-based) CASCI/CASSCF object.
    lazy=True returns an MOERIs (integral blocks on demand), otherwise
    all integrals are stored.  cderi: Cholesky vectors that replace the exact
    two-electron integrals (zmr.interface.cholesky).  Returns (Hamiltonian,
    number of active electrons).'''
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if nfrozen > mc.ncore:
        raise ValueError('nfrozen larger than the CASCI core')
    args = (mc.mol, mc.get_hcore(), mo_coeff, nfrozen, mc.ncore - nfrozen, mc.ncas)
    if lazy or cderi is not None:
        eris = mo_eris(*args, cderi=cderi)
        if not lazy:
            eris = eris.to_dense()
    else:
        eris = from_mo_coeff(*args)
    nelecas = mc.nelecas
    nelec = nelecas if isinstance(nelecas, (int, numpy.integer)) else sum(nelecas)
    return eris, int(nelec)


def _spin_orbital_coeff(mo_coeff):
    '''GHF-form coefficients [alpha rows; beta rows] of the spin-orbitals
    p = 2P + sigma.'''
    nao, n = mo_coeff.shape
    c = numpy.zeros((2 * nao, 2 * n))
    c[:nao, 0::2] = mo_coeff
    c[nao:, 1::2] = mo_coeff
    return c


def mo_eris(mol, hcore_ao, mo_coeff, nfrozen, ncore, nact, cderi=None):
    '''Like from_mo_coeff, but integrals are transformed block by block on
    demand (zmr.integrals.MOERIs).  Counts are spatial orbitals.'''
    nmo_sp = mo_coeff.shape[1]
    nvirt = nmo_sp - nfrozen - ncore - nact
    if min(nfrozen, ncore, nact, nvirt) < 0:
        raise ValueError('inconsistent orbital partition')
    return ghf_eris(mol, hcore_ao, _spin_orbital_coeff(mo_coeff),
                    2 * nfrozen, 2 * ncore, 2 * nact, cderi=cderi)


def ghf_eris(mol, hcore_ao, c_ghf, nfrozen, ncore, nact, cderi=None):
    '''Integrals for arbitrary (complex) GHF-form spin-orbitals
    c_ghf = [alpha rows; beta rows] of shape (2 nao, nso); counts are
    spin-orbitals.  Used, e.g., with spin-orbitals mixed by complex unitary
    rotations to exercise the spinor code paths with a spin-free
    Hamiltonian.  cderi: Cholesky vectors that replace the exact
    two-electron integrals (zmr.interface.cholesky).'''
    from pyscf.ao2mo import nrr_outcore
    from pyscf.scf import ghf, hf
    from socutils.zmr.integrals import MOERIs
    from socutils.zmr.interface.ao2mo import IncoreAO2MO
    from socutils.zmr.interface.cholesky import CholeskyAO2MO

    nso = c_ghf.shape[1]
    nvirt = nso - nfrozen - ncore - nact
    if min(nfrozen, ncore, nact, nvirt) < 0:
        raise ValueError('inconsistent orbital partition')
    h_ao = numpy.kron(numpy.eye(2), hcore_ao)
    nao = mol.nao_nr()

    def components(c):
        return c[:nao], c[nao:]

    if cderi is None:
        def jkbuild(mol, dms, hermi, with_j, with_k, omega=None):
            # ghf.get_jk passes these positionally, which does not match the
            # signature of hf.get_jk in this PySCF version
            return hf.get_jk(mol, dms, hermi, with_j=with_j, with_k=with_k, omega=omega)

        def veff_fn(dm):
            vj, vk = ghf.get_jk(mol, dm, hermi=0, jkbuild=jkbuild)
            return vj - vk

        def outcore(*cs):
            return nrr_outcore.general_iofree(mol, cs, intor='int2e_sph', motype='ghf',
                                              verbose=0)

        eri_fn = IncoreAO2MO(mol, components, outcore)
    else:
        eri_fn = CholeskyAO2MO(cderi, nao, components)
        veff_fn = eri_fn.veff

    return _fold_frozen(MOERIs, mol, h_ao, c_ghf, nfrozen,
                        OrbitalSpaces(ncore, nact, nvirt), eri_fn, veff_fn)


def _fold_frozen(cls, mol, h_ao, c, nf, spaces, eri_fn, veff_fn):
    cf = c[:, :nf]
    dm = cf @ cf.conj().T
    ecore = mol.energy_nuc()
    if nf:
        vf = veff_fn(dm)
        ecore += (numpy.einsum('ij,ji->', dm, h_ao) + 0.5 * numpy.einsum('ij,ji->', dm, vf)).real
        h_ao = h_ao + vf
    return cls(ecore, h_ao, c[:, nf:], spaces, eri_fn, veff_fn)


# ----------------------------------------------------------------------
# spin-restricted code (zmr.nr): real integrals over spatial orbitals

def _real_ao2mo(mol, max_memory=None):
    '''eri_fn(c1, c2, c3, c4) -> real (pq|rs): from AO integrals kept in
    memory with their eightfold symmetry if they fit, else integral-direct.'''
    max_memory = mol.max_memory if max_memory is None else max_memory
    nao = mol.nao_nr()
    npair = nao * (nao + 1) // 2
    state = {}

    def eri_fn(c1, c2, c3, c4):
        cs = [numpy.asarray(c, order='F') for c in (c1, c2, c3, c4)]
        shape = tuple(c.shape[1] for c in cs)
        if npair * (npair + 1) // 2 * 8 / 1e6 < 0.4 * max_memory:
            if 'eri' not in state:
                state['eri'] = mol.intor('int2e', aosym='s8')
            out = ao2mo.incore.general(state['eri'], cs, compact=False)
        else:
            out = ao2mo.outcore.general_iofree(mol, cs, compact=False, verbose=0)
        return out.reshape(shape)

    return eri_fn


def nr_eris(mol, hcore_ao, mo_coeff, nfrozen, ncore, nact, cderi=None):
    '''Integrals for the spin-restricted code: zmr.nr.integrals.MOERIs for
    real spatial orbitals mo_coeff (frozen, core, active, virtual; counts are
    spatial orbitals).  Blocks are transformed on demand, from the exact AO
    integrals or, with cderi, from Cholesky / density-fitting vectors
    (zmr.interface.cholesky).'''
    from pyscf.scf import hf
    from socutils.zmr.interface.cholesky import CholeskyAO2MO
    from socutils.zmr.nr.integrals import MOERIs

    mo_coeff = numpy.asarray(mo_coeff)
    if numpy.iscomplexobj(mo_coeff):
        raise ValueError('the spin-restricted code needs real orbitals')
    nvirt = mo_coeff.shape[1] - nfrozen - ncore - nact
    if min(nfrozen, ncore, nact, nvirt) < 0:
        raise ValueError('inconsistent orbital partition')

    if cderi is None:
        def veff_fn(dm):
            vj, vk = hf.get_jk(mol, dm, hermi=1)
            return vj - 0.5 * vk

        eri_fn = _real_ao2mo(mol)
    else:
        eri_fn = CholeskyAO2MO(cderi, mol.nao_nr(), exchange=0.5)
        veff_fn = eri_fn.veff

    cf = mo_coeff[:, :nfrozen]
    ecore = mol.energy_nuc()
    h_ao = hcore_ao
    if nfrozen:
        dm = 2 * cf @ cf.T
        vf = veff_fn(dm)
        ecore += numpy.einsum('ij,ji->', dm, h_ao) + 0.5 * numpy.einsum('ij,ji->', dm, vf)
        h_ao = h_ao + vf
    return MOERIs(ecore, h_ao, mo_coeff[:, nfrozen:], OrbitalSpaces(ncore, nact, nvirt),
                  eri_fn, veff_fn)


def nr_from_casci(mc, nfrozen=0, mo_coeff=None, cderi=None):
    '''(eris, (n_alpha, n_beta)) of the spin-restricted code for a PySCF
    (RHF/ROHF-based) CASCI/CASSCF object.'''
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if nfrozen > mc.ncore:
        raise ValueError('nfrozen larger than the CASCI core')
    eris = nr_eris(mc.mol, mc.get_hcore(), mo_coeff, nfrozen, mc.ncore - nfrozen, mc.ncas,
                   cderi=cderi)
    nelecas = mc.nelecas
    if isinstance(nelecas, (int, numpy.integer)):
        spin = mc.mol.spin
        nelecas = ((nelecas + spin) // 2, (nelecas - spin) // 2)
    return eris, (int(nelecas[0]), int(nelecas[1]))


def nr_civecs(mc, roots=None):
    '''CI vectors of a solved PySCF CASCI/CASSCF as columns, in the
    determinant order of zmr.nr.active (that of pyscf.fci).'''
    ci = mc.ci
    if isinstance(ci, numpy.ndarray) and ci.ndim == 2:
        ci = [ci]
    if roots is not None:
        ci = [ci[r] for r in roots]
    return numpy.stack([numpy.asarray(c).ravel() for c in ci], axis=1)
