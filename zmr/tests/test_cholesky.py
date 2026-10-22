'''Integrals and mean-field potentials from Cholesky vectors (zmr.interface.cholesky)
against the exact AO integrals.  The vectors are an exact factorization of the
AO integral matrix (all eigenvectors), so both routes must agree to round-off;
truncated vectors are checked against the integrals they represent.'''
import numpy
import pytest
from pyscf import ao2mo, gto, mcscf, scf
from pyscf.ao2mo import nrr_outcore
from pyscf.scf import ghf, hf

from socutils.zmr import fock
from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.interface import pyscf_nr
from socutils.zmr.interface.cholesky import CholeskyAO2MO
from socutils.zmr.nr import PCNEVPT2 as NRPCNEVPT2
from socutils.zmr.pc.nevpt2 import PCNEVPT2
from socutils.zmr.uc.nevpt2 import UCNEVPT2


def factorize(mol, tol=1e-13):
    '''Vectors L with (mn|ls) = sum_L L_mn L_ls up to the eigenvalues below
    tol: (naux, npair) in packed form.'''
    eri = mol.intor('int2e_sph', aosym='s4')
    w, v = numpy.linalg.eigh(eri)
    keep = w > tol
    return (v[:, keep] * numpy.sqrt(w[keep])).T


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='6-31g', verbose=0)
    return mol, factorize(mol)


def _rand(shape, seed, real=False):
    rng = numpy.random.default_rng(seed)
    x = rng.normal(size=shape)
    return x if real else x + 1j * rng.normal(size=shape)


def _spinor(mol):
    ca, cb = mol.sph2spinor_coeff()

    def exact(*cs):
        return nrr_outcore.general_iofree(mol, cs, intor='int2e_sph', motype='j-spinor',
                                          verbose=0)
    return mol.nao_2c(), (lambda c: (ca @ c, cb @ c)), exact


def _ghf(mol):
    nao = mol.nao_nr()

    def exact(*cs):
        return nrr_outcore.general_iofree(mol, cs, intor='int2e_sph', motype='ghf', verbose=0)
    return 2 * nao, (lambda c: (c[:nao], c[nao:])), exact


@pytest.mark.parametrize('setup', [_spinor, _ghf])
@pytest.mark.parametrize('sizes', [(2, 3, 4, 5), (5, 2, 3, 3), (3, 3, 1, 6)])
def test_two_component_block(water, setup, sizes):
    '''Both orders of the half transformation (n1 >= n2 and n1 < n2).'''
    mol, cderi = water
    n, components, exact = setup(mol)
    cs = [_rand((n, k), 10 + i) for i, k in enumerate(sizes)]
    eri = CholeskyAO2MO(cderi, mol.nao_nr(), components)(*cs)
    ref = numpy.asarray(exact(*cs)).reshape(sizes)
    assert eri.shape == ref.shape
    assert abs(eri - ref).max() < 1e-11


def test_vector_layouts_and_blocking(water):
    '''Packed, flattened and square vectors; many small blocks of vectors.'''
    mol, cderi = water
    nao = mol.nao_nr()
    n, components, exact = _spinor(mol)
    cs = [_rand((n, k), 20 + i) for i, k in enumerate((4, 2, 3, 5))]
    ref = numpy.asarray(exact(*cs)).reshape(4, 2, 3, 5)
    square = numpy.zeros((cderi.shape[0], nao, nao))
    idx = numpy.tril_indices(nao)
    square[:, idx[0], idx[1]] = cderi
    square[:, idx[1], idx[0]] = cderi
    for vectors in (cderi, square, square.reshape(-1, nao * nao)):
        for max_memory in (200, 1e-3):
            chol = CholeskyAO2MO(vectors, nao, components, max_memory=max_memory)
            assert abs(chol(*cs) - ref).max() < 1e-11
    with pytest.raises(ValueError):
        CholeskyAO2MO(cderi[:, :-1], nao, components)


def test_same_bra_and_ket(water):
    mol, cderi = water
    n, components, exact = _spinor(mol)
    c1, c2 = _rand((n, 5), 30), _rand((n, 3), 31)
    eri = CholeskyAO2MO(cderi, mol.nao_nr(), components)(c1, c2, c1.copy(), c2)
    ref = numpy.asarray(exact(c1, c2, c1, c2)).reshape(5, 3, 5, 3)
    assert abs(eri - ref).max() < 1e-11
    assert abs(eri - eri.transpose(2, 3, 0, 1)).max() < 1e-12


def test_empty_block(water):
    mol, cderi = water
    n, components, _ = _spinor(mol)
    cs = [_rand((n, k), 40 + i) for i, k in enumerate((2, 0, 3, 1))]
    eri = CholeskyAO2MO(cderi, mol.nao_nr(), components)(*cs)
    assert eri.shape == (2, 0, 3, 1)


def test_real_orbitals_stay_real(water):
    mol, cderi = water
    nao = mol.nao_nr()
    cs = [_rand((nao, k), 50 + i, real=True) for i, k in enumerate((3, 5, 2, 4))]
    eri = CholeskyAO2MO(cderi, nao)(*cs)
    ref = ao2mo.incore.general(mol.intor('int2e', aosym='s8'), cs, compact=False)
    assert eri.dtype == numpy.float64
    assert abs(eri - ref.reshape(3, 5, 2, 4)).max() < 1e-11


def test_veff_spin_restricted(water):
    mol, cderi = water
    nao = mol.nao_nr()
    dm = _rand((nao, nao), 60, real=True)
    dm = dm + dm.T
    vj, vk = hf.get_jk(mol, dm, hermi=1)
    v = CholeskyAO2MO(cderi, nao, exchange=0.5).veff(dm)
    assert v.dtype == numpy.float64
    assert abs(v - (vj - 0.5 * vk)).max() < 1e-10


@pytest.mark.parametrize('hermitian', [True, False])
def test_veff_two_component(water, hermitian):
    '''J - K for a GHF density with alpha-beta blocks, also non-Hermitian.'''
    mol, cderi = water
    nao = mol.nao_nr()
    dm = _rand((2 * nao, 2 * nao), 70)
    if hermitian:
        dm = dm + dm.conj().T

    def jkbuild(mol, dms, hermi, with_j, with_k, omega=None):
        return hf.get_jk(mol, dms, hermi, with_j=with_j, with_k=with_k, omega=omega)

    vj, vk = ghf.get_jk(mol, dm, hermi=0, jkbuild=jkbuild)
    v = CholeskyAO2MO(cderi, nao, lambda c: (c[:nao], c[nao:])).veff(dm)
    assert abs(v - (vj - vk)).max() < 1e-10


def test_veff_low_rank_density(water):
    '''A density of a few occupied spinors, the case veff is built for.'''
    mol, cderi = water
    n, components, exact = _spinor(mol)
    c = numpy.linalg.qr(_rand((n, n), 80))[0][:, :4]
    dm = c @ c.conj().T
    v = CholeskyAO2MO(cderi, mol.nao_nr(), components).veff(dm)
    unit = numpy.eye(n, dtype=complex)
    coul = numpy.asarray(exact(unit, unit, c, c)).reshape(n, n, 4, 4)
    exch = numpy.asarray(exact(unit, c, c, unit)).reshape(n, 4, 4, n)
    ref = numpy.einsum('mnii->mn', coul) - numpy.einsum('miin->mn', exch)
    assert abs(v - ref).max() < 1e-10
    assert abs(CholeskyAO2MO(cderi, mol.nao_nr(), components).veff(0 * dm)).max() == 0


@pytest.fixture(scope='module')
def water_casscf(water):
    mol, cderi = water
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASSCF(mf, 4, 4)
    mc.conv_tol = 1e-11
    mc.kernel()
    return mc, cderi


@pytest.mark.parametrize('nfrozen', [0, 1])
def test_spin_orbital_nevpt2(water_casscf, nfrozen):
    '''Frozen-core folding, generalized Fock matrix and all classes.'''
    mc, cderi = water_casscf
    ref, nelec = pyscf_nr.from_casci(mc, nfrozen=nfrozen)
    eris, _ = pyscf_nr.from_casci(mc, nfrozen=nfrozen, cderi=cderi)
    assert abs(eris.ecore - ref.ecore) < 1e-10
    assert abs(eris.h1 - ref.h1).max() < 1e-10
    # the same model states for both (the second level is a degenerate triplet)
    civecs = ActiveSpace.from_eris(ref).solve(nelec, 4)[1]
    e_ref = PCNEVPT2(ref, nelec, civecs=civecs)
    e_ref.kernel()
    e = PCNEVPT2(eris, nelec, civecs=civecs)
    e.kernel()
    assert abs(e.heff - e_ref.heff).max() < 1e-10
    for name, a in e.h2_by_class.items():
        assert abs(a - e_ref.h2_by_class[name]).max() < 1e-10


def test_dense_hamiltonian_from_vectors(water_casscf):
    mc, cderi = water_casscf
    dense, nelec = pyscf_nr.from_casci(mc, nfrozen=1, lazy=False, cderi=cderi)
    ref, _ = pyscf_nr.from_casci(mc, nfrozen=1, lazy=False)
    assert abs(dense.eri_block('pppp') - ref.eri_block('pppp')).max() < 1e-10
    assert abs(dense.core_energy() - ref.core_energy()) < 1e-10


@pytest.mark.parametrize('nfrozen', [0, 1])
def test_spin_restricted_nevpt2(water_casscf, nfrozen):
    mc, cderi = water_casscf
    civecs = pyscf_nr.nr_civecs(mc)
    ref, nelec = pyscf_nr.nr_from_casci(mc, nfrozen=nfrozen)
    eris, _ = pyscf_nr.nr_from_casci(mc, nfrozen=nfrozen, cderi=cderi)
    assert eris.dtype == numpy.float64
    e_ref = NRPCNEVPT2(ref, nelec, civecs=civecs)
    e_ref.kernel()
    e = NRPCNEVPT2(eris, nelec, civecs=civecs)
    e.kernel()
    assert abs(e.heff - e_ref.heff).max() < 1e-10
    assert abs(e.hmodel[0, 0] - mc.e_tot) < 1e-9


def test_truncated_vectors_are_a_consistent_hamiltonian(water_casscf):
    '''Vectors that reproduce the integrals to 1e-4: every quantity is that
    of the approximate integrals, and the energies differ by that order.'''
    mc, _ = water_casscf
    mol = mc.mol
    nao = mol.nao_nr()
    cderi = factorize(mol, tol=1e-4)
    full = factorize(mol)
    assert cderi.shape[0] < full.shape[0]
    approx = cderi.T @ cderi
    assert 1e-6 < abs(approx - mol.intor('int2e_sph', aosym='s4')).max() < 1e-3
    c = pyscf_nr._spin_orbital_coeff(mc.mo_coeff)
    eris, nelec = pyscf_nr.from_casci(mc, nfrozen=1, cderi=cderi)
    coeff = c[:, 2:]
    ca, cb = coeff[:nao], coeff[nao:]
    b = (numpy.einsum('mp,lmn,nq->lpq', ca.conj(), _square(cderi, nao), ca)
         + numpy.einsum('mp,lmn,nq->lpq', cb.conj(), _square(cderi, nao), cb))
    ref = numpy.einsum('lpq,lrs->pqrs', b, b)
    assert abs(eris.eri_block('pppp') - ref).max() < 1e-11
    exact, _ = pyscf_nr.from_casci(mc, nfrozen=1)
    e = ActiveSpace.from_eris(eris).solve(nelec, 1)[0][0]
    e_exact = ActiveSpace.from_eris(exact).solve(nelec, 1)[0][0]
    assert 1e-8 < abs(e - e_exact) < 1e-3


def _square(cderi, nao):
    out = numpy.zeros((cderi.shape[0], nao, nao))
    idx = numpy.tril_indices(nao)
    out[:, idx[0], idx[1]] = cderi
    out[:, idx[1], idx[0]] = cderi
    return out


# ----------------------------------------------------------------------
# virtual space truncation

def _model(water_casscf, lazy, nstate=2):
    mc, cderi = water_casscf
    eris, nelec = pyscf_nr.from_casci(mc, nfrozen=1, lazy=lazy)
    act = ActiveSpace.from_eris(eris)
    _, civecs = act.solve(nelec, nstate)
    return eris, nelec, civecs, act.averaged_rdm1(civecs, nelec)


@pytest.mark.parametrize('lazy', [True, False])
def test_truncate_virtuals_keeps_everything_below_the_cut(water_casscf, lazy):
    '''No orbital above the cut: the virtual space is only rotated, which
    leaves NEVPT2 unchanged.'''
    eris, nelec, civecs, gamma = _model(water_casscf, lazy)
    new, kept, dropped = fock.truncate_virtuals(eris, gamma, 1e6)
    assert len(dropped) == 0 and len(kept) == eris.spaces.nvirt
    assert new.spaces.nvirt == eris.spaces.nvirt
    ref = UCNEVPT2(eris, nelec, civecs=civecs)
    ref.kernel()
    pt = UCNEVPT2(new, nelec, civecs=civecs)
    pt.kernel()
    assert abs(pt.heff - ref.heff).max() < 1e-10


def test_truncate_virtuals_dense_and_lazy_agree(water_casscf):
    eris, nelec, civecs, gamma = _model(water_casscf, True)
    dense = _model(water_casscf, False)[0]
    # virtual energies: 0.68, 0.79, 1.16, 1.18, 1.21, 1.66 (two spin-orbitals
    # each).  The cut is the largest gap within 20 % of emax: a threshold of
    # 1.2 would separate 1.18 from 1.21, the cut is made at 1.21 | 1.66.
    emax = 1.2
    new, kept, dropped = fock.truncate_virtuals(eris, gamma, emax)
    new_dense, kept_dense, _ = fock.truncate_virtuals(dense, gamma, emax)
    assert len(kept) == 10 and len(dropped) == 2
    assert new.spaces.nvirt == len(kept) == new_dense.spaces.nvirt
    assert abs(kept - kept_dense).max() < 1e-10
    assert emax < kept.max() < 1.2 * emax < dropped.min()
    assert len(fock.truncate_virtuals(eris, gamma, emax, window=0)[1]) == 8
    pt = PCNEVPT2(new, nelec, civecs=civecs)
    pt.kernel()
    pt_dense = PCNEVPT2(new_dense, nelec, civecs=civecs)
    pt_dense.kernel()
    assert abs(pt.heff - pt_dense.heff).max() < 1e-10
    # removing virtual orbitals removes part of the (negative) correlation energy
    full = PCNEVPT2(eris, nelec, civecs=civecs)
    full.kernel()
    assert pt.h2[0, 0].real > full.h2[0, 0].real
    assert abs(pt.hmodel - full.hmodel).max() < 1e-10


def test_truncate_virtuals_spin_restricted(water_casscf):
    mc, _ = water_casscf
    eris, nelec = pyscf_nr.nr_from_casci(mc, nfrozen=1)
    civecs = pyscf_nr.nr_civecs(mc)
    gamma = mc.make_rdm1s()
    mo = mc.mo_coeff[:, mc.ncore:mc.ncore + mc.ncas]
    s = mc.mol.intor('int1e_ovlp')
    dm_act = mo.T @ s @ (gamma[0] + gamma[1]) @ s @ mo
    new, kept, dropped = fock.truncate_virtuals(eris, dm_act, 1.2)
    assert new.dtype == numpy.float64 and new.spaces.nvirt == len(kept) == 5
    assert len(dropped) == 1
    so, _, _, gamma_so = _model(water_casscf, True, nstate=1)
    kept_so = fock.truncate_virtuals(so, gamma_so, 1.2)[1]
    # PySCF's CASSCF density against the exact CASCI one: converged to 1e-6
    assert abs(numpy.repeat(kept, 2) - kept_so).max() < 1e-6
    pt = NRPCNEVPT2(new, nelec, civecs=civecs)
    pt.kernel()
    full = NRPCNEVPT2(eris, nelec, civecs=civecs)
    full.kernel()
    assert pt.h2[0, 0] > full.h2[0, 0]


def test_with_virtuals_checks_its_argument(water_casscf):
    eris = _model(water_casscf, True)[0]
    nv = eris.spaces.nvirt
    with pytest.raises(ValueError):
        eris.with_virtuals(numpy.eye(nv + 1)[:, :2])
    with pytest.raises(ValueError):
        eris.with_virtuals(2 * numpy.eye(nv)[:, :2])


def test_density_fitting_object_as_source(water):
    '''A PySCF with_df object is read block by block (as socutils _CDERIS does)
    and gives what its vectors give as an array.'''
    from pyscf import df
    mol, _ = water
    nao = mol.nao_nr()
    with_df = df.DF(mol, auxbasis='weigend')
    with_df.build()
    vectors = numpy.vstack([numpy.asarray(b) for b in with_df.loop()])
    assert vectors.shape == (with_df.get_naoaux(), nao * (nao + 1) // 2)
    n, components, _ = _spinor(mol)
    cs = [_rand((n, k), 90 + i) for i, k in enumerate((5, 2, 3, 6))]
    ref = CholeskyAO2MO(vectors, nao, components)
    for max_memory in (200, 1e-3):
        new = CholeskyAO2MO(with_df, nao, components, max_memory=max_memory)
        assert new.naux == ref.naux
        assert abs(new(*cs) - ref(*cs)).max() < 1e-12
    dm = _rand((n, n), 95)
    dm = dm + dm.conj().T
    assert abs(CholeskyAO2MO(with_df, nao, components).veff(dm) - ref.veff(dm)).max() < 1e-11


def test_three_index_integrals_are_kept_four_index_blocks_are_not(water_casscf):
    '''With Cholesky vectors NEVPT2 keeps cd_pa (all x active) and cd_vc
    (virtual x core), as socutils' _CDERIS, forms the four-index blocks class
    by class and the doubly external ones only one core orbital at a time.'''
    mc, cderi = water_casscf
    ref, nelec = pyscf_nr.from_casci(mc, nfrozen=1)
    eris, _ = pyscf_nr.from_casci(mc, nfrozen=1, cderi=cderi)
    assert eris.factorized and not ref.factorized
    civecs = ActiveSpace.from_eris(ref).solve(nelec, 2)[1]
    for cls in (PCNEVPT2, UCNEVPT2):
        pt = cls(eris, nelec, civecs=civecs)
        pt.verbose = 0
        pt.kernel()
        used = pt.quasi_canonical.eris
        sp = used.spaces
        naux = cderi.shape[0]
        assert used.cd_pa.shape == (naux, sp.nmo, sp.nact)
        assert used.cd_pc.shape == (naux, sp.nmo, sp.ncore)
        assert used.cd_vc.shape == (naux, sp.nvirt, sp.ncore)
        assert numpy.shares_memory(used.cd_vc, used.cd_pc)
        assert not used._super and not used._cache
        # the blocks of ccvv, cavv and aavv were never formed
        assert not {'vcvc', 'vcva', 'vavc', 'vava'} & set(used.computed_blocks)
        exact = cls(ref, nelec, civecs=civecs)
        exact.verbose = 0
        exact.kernel()
        assert abs(pt.heff - exact.heff).max() < 1e-10
        for name, a in pt.h2_by_class.items():
            assert abs(a - exact.h2_by_class[name]).max() < 1e-10
    # a block is the product of its two factors; the other pairs are slices
    # of cd_pa or conjugate transposes
    block = used.eri_block('vcaa')
    assert abs(block - numpy.einsum('lai,ltu->aitu', used.cd_vc, used.cd_block('aa'))).max() < 1e-12
    assert numpy.shares_memory(used.cd_block('va'), used.cd_pa)
    assert abs(used.cd_block('ac') - used.cd_block('ca').transpose(0, 2, 1).conj()).max() < 1e-14
    assert abs(used.cd_block('cv') - used.cd_vc.transpose(0, 2, 1).conj()).max() < 1e-14


def test_core_valence_separation_with_three_index_integrals(water_casscf):
    mc, cderi = water_casscf
    ref, nelec = pyscf_nr.from_casci(mc, nfrozen=0)
    eris, _ = pyscf_nr.from_casci(mc, nfrozen=0, cderi=cderi)
    civecs = ActiveSpace.from_eris(ref).solve(nelec, 1)[1]
    out = []
    for h in (ref, eris):
        pt = PCNEVPT2(h, nelec, civecs=civecs)
        pt.verbose = 0
        pt.cvs = True
        pt.kernel()
        out.append(pt.heff)
    assert abs(out[0] - out[1]).max() < 1e-10


@pytest.mark.parametrize('mode', [-1, 0, 1])
def test_external_sum_kernels(water_casscf, mode):
    '''W at a set of energy arguments from the three-index integrals (one
    task per core orbital or pair, or threads inside a task) against the
    same sums over the listed blocks.'''
    from socutils.zmr.lib import kernels
    from socutils.zmr.uc import classes, engine
    mc, cderi = water_casscf
    eris, nelec = pyscf_nr.from_casci(mc, nfrozen=0, cderi=cderi)
    rng = numpy.random.default_rng(5)
    sp = eris.spaces
    eps_c = numpy.sort(rng.normal(size=sp.ncore)) - 3
    eps_v = numpy.sort(rng.normal(size=sp.nvirt)) + 3
    fcore = eris.generalized_fock(numpy.zeros((sp.nact, sp.nact)))
    ctx = classes.Context(eris, eps_c, eps_v, fcore)
    d = numpy.array([-0.3, 0.0, 0.7, 2.5])
    bvc = eris.cd_block('vc')
    got = {'ccvv': kernels.w_ccvv(bvc, eps_v, eps_c, d, mode),
           'cavv': kernels.w_cavv(bvc, eris.cd_block('va'), eps_v, eps_c, d, mode),
           'aavv': kernels.w_aavv(eris.cd_block('va'), eps_v, d)}
    if got['ccvv'] is None:
        pytest.skip('libzmr not built')
    for name, w in got.items():
        pclass = classes.CLASS_BY_NAME[name]
        ext = engine.class_blocks(pclass, ctx)
        assert isinstance(ext, classes.ExternalBlocks) and ext._list is None
        blocks = list(pclass.blocks(ctx))
        delta = numpy.concatenate([x for x, _ in blocks])
        v = numpy.concatenate([x for _, x in blocks], axis=0)
        assert ext.nlabel == len(delta)
        assert abs(ext.delta_min - delta.min()) < 1e-12
        ref = numpy.array([v.conj().T @ (v / (x + delta)[:, None]) for x in d])
        assert abs(w.reshape(ref.shape) - ref).max() < 1e-11 * abs(ref).max(), name
        assert abs(ext.w_nodes(d).reshape(ref.shape) - ref).max() < 1e-11 * abs(ref).max()


def test_union_contraction_does_not_list_the_doubly_external_blocks(water_casscf):
    from socutils.zmr.uc import classes, engine
    mc, cderi = water_casscf
    ref, nelec = pyscf_nr.from_casci(mc, nfrozen=1)
    eris, _ = pyscf_nr.from_casci(mc, nfrozen=1, cderi=cderi)
    civecs = ActiveSpace.from_eris(ref).solve(nelec, 3)[1]
    made = []
    original = classes.ExternalBlocks.materialize

    def spy(self):
        made.append(self)
        return original(self)

    classes.ExternalBlocks.materialize = spy
    try:
        out = []
        for h in (eris, ref):
            for e0 in ('state', 'average'):
                pt = PCNEVPT2(h, nelec, civecs=civecs, e0=e0)
                pt.verbose = 0
                pt.kernel()
                out.append(pt.heff)
    finally:
        classes.ExternalBlocks.materialize = original
    from socutils.zmr import lib as zlib
    if zlib.zmr() is not None:
        assert not made
    assert abs(out[0] - out[2]).max() < 1e-10
    assert abs(out[1] - out[3]).max() < 1e-10


@pytest.mark.parametrize('nfrozen', [0, 1])
def test_fock_matrix_and_core_energy_from_three_index_integrals(water_casscf, nfrozen):
    '''Exchange from cd_pc and cd_pa, Coulomb from their charge; rotated and
    truncated orbitals take rotated three-index integrals.'''
    mc, cderi = water_casscf
    ref, nelec = pyscf_nr.from_casci(mc, nfrozen=nfrozen)
    eris, _ = pyscf_nr.from_casci(mc, nfrozen=nfrozen, cderi=cderi)
    sp = ref.spaces
    gamma = _rand((sp.nact, sp.nact), 7)
    gamma = gamma + gamma.conj().T
    assert abs(eris.generalized_fock(gamma) - ref.generalized_fock(gamma)).max() < 1e-10
    assert abs(eris.core_energy() - ref.core_energy()) < 1e-10
    u = numpy.eye(sp.nmo, dtype=complex)
    for label in 'cav':
        sl = sp.slice(label)
        u[sl, sl] = numpy.linalg.qr(_rand((sp.size(label),) * 2, 8))[0]
    rot, rot_ref = eris.rotate(u), ref.rotate(u)
    # the active orbitals were rotated: nothing is shared
    assert rot._occ is None
    # rotations among core or virtual orbitals share the occupied-orbital pass
    u2 = u.copy()
    a = sp.slice('a')
    u2[a, a] = numpy.eye(sp.nact)
    kept = eris.rotate(u2)
    assert kept._occ is eris._occ and kept._cd_pc is None
    assert abs(kept.generalized_fock(gamma) - ref.rotate(u2).generalized_fock(gamma)).max() < 1e-10
    assert abs(kept.eri_block('vcva') - ref.rotate(u2).eri_block('vcva')).max() < 1e-10
    g2 = u[sp.slice('a'), sp.slice('a')].T @ gamma @ u[sp.slice('a'), sp.slice('a')].conj()
    assert abs(rot.generalized_fock(g2) - rot_ref.generalized_fock(g2)).max() < 1e-10
    cut = eris.with_virtuals(numpy.eye(sp.nvirt)[:, :sp.nvirt - 2])
    assert cut.cd_pa.shape[1] == sp.nmo - 2
    assert abs(cut.eri_block('vcva') - ref.eri_block('vcva')[:-2, :, :-2]).max() < 1e-10


def test_spin_restricted_fock_from_three_index_integrals(water_casscf):
    mc, cderi = water_casscf
    ref, nelec = pyscf_nr.nr_from_casci(mc, nfrozen=1)
    eris, _ = pyscf_nr.nr_from_casci(mc, nfrozen=1, cderi=cderi)
    sp = ref.spaces
    dm = _rand((sp.nact, sp.nact), 9, real=True)
    dm = dm + dm.T
    f = eris.generalized_fock(dm)
    assert f.dtype == numpy.float64
    assert abs(f - ref.generalized_fock(dm)).max() < 1e-10
    assert abs(eris.core_energy() - ref.core_energy()) < 1e-10
