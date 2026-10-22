'''Density-matrix route of the spin-free PC-NEVPT2 (zmr.nr.rdm):

* operator bases: the functions of zmr.nr.classes are the stated combinations
  of the basis vectors;
* metric and Koopmans matrices against the brute-force ones (H_act applied to
  every function, zmr.reference.nr_koopmans), for eigenstates and for
  arbitrary vectors, closed and open shells;
* energies against the CI-vector route: one to three states, states of
  different spin in one model space, both contractions, both E0, dense and
  matrix-free sectors, a molecule with exact and with Cholesky integrals;
* nothing of the size of a 4-RDM is formed.'''
import numpy
import pytest
from pyscf import gto, mcscf, scf

from socutils.zmr import lib
from socutils.zmr.active import rdm as active_rdm
from socutils.zmr.interface import pyscf_nr
from socutils.zmr.nr import PCNEVPT2, classes, rdm
from socutils.zmr.nr.active import ActiveSpace
from socutils.zmr.reference import nr_koopmans
from socutils.zmr.tests._helpers import nr_model_eris
from socutils.zmr.tests.test_cholesky import factorize

# some of the model Hamiltonians have intruder states for the higher model
# states; both routes are compared there as well
pytestmark = pytest.mark.filterwarnings('ignore:WARNING. class')

# (nact, (n_alpha, n_beta), seed)
SPACES = [(3, (2, 2), 1), (4, (2, 1), 2), (3, (3, 0), 3), (4, (1, 3), 4), (3, (3, 3), 5),
          (3, (0, 0), 6), (4, (4, 1), 7), (4, (2, 2), 8)]


def _space(nact, nelec, seed, nstate, eigen):
    eris = nr_model_eris(2, nact, 3, seed=seed)
    act = ActiveSpace.from_eris(eris)
    dim = act.dim(nelec)
    nstate = min(nstate, dim)
    if eigen:
        c = act.sector(nelec).vecs[:, :nstate]
    else:
        rng = numpy.random.default_rng(seed)
        c = numpy.linalg.qr(rng.normal(size=(dim, nstate)))[0]
    return eris, act, numpy.ascontiguousarray(c)


@pytest.mark.parametrize('nact, nelec, seed', SPACES)
def test_functions_are_combinations_of_the_basis(nact, nelec, seed):
    '''A_(P,p) |Psi> = sum_w W_w |Psi> C_p[w, P] for every pattern.'''
    _, act, c = _space(nact, nelec, seed, 2, False)
    ms = rdm.ModelStates(act, c, nelec)
    for pclass in classes.ALL_CLASSES:
        nparams = pclass.nparams(nact)
        patterns = pclass.patterns(nact)
        for pattern, (target, phi) in zip(patterns, pclass.functions(act, c.T, nelec)):
            if act.dim(target) == 0:
                continue
            basis = rdm.BASES[pclass.name](ms, pattern.dn)
            coef = rdm.coefficients(pattern, nparams, basis.index())
            mine = numpy.einsum('jwd,wp->jpd', basis.w, coef.toarray())
            assert abs(mine - phi).max() < 1e-13, (type(pclass).__name__, pattern.dn)


@pytest.mark.parametrize('eigen', [True, False])
@pytest.mark.parametrize('nact, nelec, seed', SPACES)
def test_matrices_against_brute_force(nact, nelec, seed, eigen):
    '''S and K of every class; model vectors that are not eigenstates check
    that no invariance of the model space is used.'''
    _, act, c = _space(nact, nelec, seed, 2, eigen)
    for kw in (dict(check=True, flip_tol=0.0), dict(), dict(diagonal=True)):
        red = rdm.Reduced(act, c, nelec, **kw)
        for pclass in classes.ALL_CLASSES:
            ref = nr_koopmans.class_matrices(pclass, act, c, nelec)
            new = red.class_matrices(pclass)
            assert (ref is None) == (new is None)
            if ref is None:
                continue
            scale = max(abs(ref[1]).max(), 1.0)
            assert abs(new[0] - ref[0]).max() < 1e-12
            if kw.get('diagonal'):
                for j in range(c.shape[1]):
                    assert abs(new[1][j, :, j] - ref[1][j, :, j]).max() < 1e-12 * scale
            else:
                assert abs(new[1] - ref[1]).max() < 1e-12 * scale, type(pclass).__name__
        # K = W^T R + G M comes out symmetric only if the commutators are right
        if kw.get('check'):
            assert red.asymmetry < 1e-12
        assert red.max_basis <= nact + nact**3


@pytest.mark.parametrize('nact, nelec, seed', SPACES[:5])
def test_intermediate_states_of_e(nact, nelec, seed):
    '''H E_uv |J> from the one-operator intermediate states.'''
    _, act, c = _space(nact, nelec, seed, 2, False)
    ms = rdm.ModelStates(act, c, nelec)
    assert abs(ms.h_e() - ms.h_e_direct()).max() < 1e-12


def _run(eris, nelec, c, **kw):
    attrs = {k: kw.pop(k) for k in list(kw) if k.startswith('rdm_')}
    pt = PCNEVPT2(eris, nelec, civecs=c, **kw)
    pt.verbose = 0
    for k, v in attrs.items():
        setattr(pt, k, v)
    pt.kernel()
    return pt


def _same(a, b, tol=1e-10):
    for name in classes.CLASS_NAMES:
        assert abs(a.h2_by_class[name] - b.h2_by_class[name]).max() < tol, name
    assert abs(a.heff - b.heff).max() < tol
    assert a.heff.dtype == b.heff.dtype == numpy.float64


# (nact, nelec, seed, nstate): closed and open shells, n_alpha != n_beta
MODELS = [(3, (2, 2), 1, 3), (4, (2, 1), 2, 2), (4, (3, 1), 3, 3), (4, (1, 3), 4, 1),
          (5, (3, 2), 5, 2), (4, (2, 2), 6, 3)]


@pytest.mark.parametrize('contraction', ['union', 'state'])
@pytest.mark.parametrize('e0', ['state', 'average'])
@pytest.mark.parametrize('nact, nelec, seed, nstate', MODELS)
def test_energies_against_ci_vector_route(nact, nelec, seed, nstate, contraction, e0):
    '''The lowest states of the (n_alpha, n_beta) sector: for Ms = 0 they are
    singlets and triplets mixed in one model space.'''
    eris, act, c = _space(nact, nelec, seed, nstate, True)
    kw = dict(contraction=contraction, e0=e0)
    ref = _run(eris, nelec, c, amplitudes='spectral', **kw)
    new = _run(eris, nelec, c, amplitudes='rdm', **kw)
    _same(new, ref)
    assert set(sum(new.routes.values(), [])) == {'rdm'}
    assert set(sum(ref.routes.values(), [])) == {'spectral'}
    assert new.rdm_info['max_basis'] == nact + nact**3 and ref.rdm_info is None
    # the vectors are not kept after the calculation
    assert new._reduced is None


@pytest.mark.parametrize('contraction', ['union', 'state'])
def test_matrix_free_sectors(monkeypatch, contraction):
    eris, act, c = _space(5, (3, 2), 5, 2, True)
    ref = _run(eris, (3, 2), c, amplitudes='spectral', contraction=contraction)
    monkeypatch.setattr(ActiveSpace, 'dense_limit', 5)
    monkeypatch.setattr(ActiveSpace, 'make_hmat', None)       # no dense sector Hamiltonian
    new = _run(eris, (3, 2), c, amplitudes='rdm', contraction=contraction)
    _same(new, ref)


def test_spin_flip_symmetry(model=MODELS[5]):
    '''Ms = 0 singlets and a triplet: the beta sectors are taken from the
    alpha ones with the parities of the states.'''
    nact, nelec, seed, _ = model
    eris, act, c = _space(nact, nelec, seed, 4, True)
    red = rdm.Reduced(act, c, nelec)
    parities = red.flip_parities()
    assert parities is not None and set(parities) == {1.0, -1.0}
    with_flip = _run(eris, nelec, c, amplitudes='rdm')
    without = _run(eris, nelec, c, amplitudes='rdm', rdm_flip_tol=0.0, rdm_check=True)
    assert with_flip.rdm_info['spin_flip'] and not without.rdm_info['spin_flip']
    assert with_flip.rdm_info['flip_error'] < 1e-12
    assert without.rdm_info['asymmetry'] < 1e-12
    _same(with_flip, without, 1e-11)
    # a vector without the symmetry, n_alpha != n_beta
    rng = numpy.random.default_rng(0)
    mixed = numpy.linalg.qr(c[:, :2] + 0.1 * rng.normal(size=(c.shape[0], 2)))[0]
    assert rdm.Reduced(act, mixed, nelec).flip_parities() is None
    eris2, act2, c2 = _space(4, (3, 1), 3, 2, True)
    assert rdm.Reduced(act2, c2, (3, 1)).flip_parities() is None


def test_no_rank_four_quantity(monkeypatch):
    '''No RDM is built, H_act is applied only to the model states and their
    one-operator intermediate states, and the largest reduced matrix has
    (n + n^3)^2 elements per pair of states.'''
    nact, nelec, nstate = 4, (2, 2), 2
    eris, act, c = _space(nact, nelec, 6, nstate, True)

    def forbidden(*args, **kwargs):
        raise AssertionError('an RDM was requested')
    monkeypatch.setattr(active_rdm, 'make_rdm', forbidden)
    monkeypatch.setattr(active_rdm, 'annihilation_amplitudes', forbidden)
    monkeypatch.setattr(nr_koopmans, 'class_matrices', forbidden)
    shapes, applied = [], []
    reduced_matrices = rdm.reduced_matrices

    def recording(basis, *args, **kwargs):
        out = reduced_matrices(basis, *args, **kwargs)
        shapes.extend(x.shape for x in out[:2])
        return out
    apply_h = rdm.ModelStates.apply_h

    def counting(self, sector, vecs):
        applied.append(vecs.size // vecs.shape[-1])
        return apply_h(self, sector, vecs)
    monkeypatch.setattr(rdm, 'reduced_matrices', recording)
    monkeypatch.setattr(rdm.ModelStates, 'apply_h', counting)
    monkeypatch.setattr(ActiveSpace, 'dense_limit', 5)
    pt = _run(eris, nelec, c, amplitudes='rdm', rdm_flip_tol=0.0)
    nw = nact + nact**3
    assert max(s[1] for s in shapes) == nw
    assert all(s == (nstate, s[1], nstate, s[1]) and s[1] <= nw for s in shapes)
    assert max(numpy.prod(s) for s in shapes) < nstate**2 * nact**8
    # H |J> and H y |J> for y = t^+_s, t_s: (1 + 4 n) vectors per state; the
    # driver itself applies H_act to the model states once more
    assert sum(applied) <= nstate * (2 + 4 * nact)
    assert abs(pt.heff - _run(eris, nelec, c, amplitudes='spectral').heff).max() < 1e-10


def test_automatic_route():
    '''Few states: reduced matrices; many states (more functions than
    determinants): CI vectors.  Complex model states: CI vectors.'''
    nact, nelec = 5, (3, 2)
    eris, act, c = _space(nact, nelec, 5, 12, True)
    for nstate, expected in ((1, {'rdm'}), (12, {'rdm', 'spectral'})):
        auto = _run(eris, nelec, c[:, :nstate], e0='average')
        assert auto.amplitudes == 'auto'
        assert set(sum(auto.routes.values(), [])) == expected
        for name, used in auto.routes.items():
            for pclass, route in zip([p for p in classes.ALL_CLASSES if p.name == name], used):
                dim = sum(act.dim(act.target(nelec, p.dn)) for p in pclass.patterns(nact))
                assert (route == 'rdm') == (nstate * sum(pclass.nparams(nact)) < dim)
        _same(auto, _run(eris, nelec, c[:, :nstate], e0='average', amplitudes='spectral'), 1e-9)
    eris, act, c = _space(4, (2, 2), 6, 1, True)
    phase = numpy.exp(0.3j)
    z = _run(eris, (2, 2), c[:, :1] * phase)
    assert set(sum(z.routes.values(), [])) == {'spectral'}
    with pytest.raises(NotImplementedError):
        PCNEVPT2(eris, (2, 2), civecs=c[:, :1] * phase, amplitudes='rdm')
    with pytest.raises(ValueError):
        PCNEVPT2(eris, (2, 2), civecs=c[:, :1], amplitudes='dm')


def test_without_native_library(monkeypatch):
    eris, act, c = _space(3, (2, 1), 2, 2, True)
    ref = _run(eris, (2, 1), c, amplitudes='rdm')
    monkeypatch.setenv('ZMR_NO_NATIVE', '1')
    lib.reset()
    try:
        assert lib.zmr() is None
        new = _run(eris, (2, 1), c, amplitudes='rdm')
    finally:
        monkeypatch.delenv('ZMR_NO_NATIVE')
        lib.reset()
    _same(new, ref)


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASSCF(mf, 4, 4)
    mc.conv_tol = 1e-11
    mc.kernel()
    return mc, factorize(mol)


@pytest.mark.parametrize('integrals', ['exact', 'cholesky'])
@pytest.mark.parametrize('nfrozen', [0, 1])
def test_water(water, integrals, nfrozen):
    mc, cderi = water
    civecs = pyscf_nr.nr_civecs(mc)
    eris, nelec = pyscf_nr.nr_from_casci(mc, nfrozen=nfrozen,
                                         cderi=cderi if integrals == 'cholesky' else None)
    ref = _run(eris, nelec, civecs, amplitudes='spectral')
    new = _run(eris, nelec, civecs, amplitudes='rdm')
    _same(new, ref)
    assert all(abs(new.h2_by_class[name]).max() > 1e-5 for name in classes.CLASS_NAMES)
    assert abs(new.hmodel[0, 0] - mc.e_tot) < 1e-9


def test_water_states_of_different_spin(water):
    '''Three CASCI states of the Ms = 0 sector (singlets and a triplet) in
    the CASSCF orbitals, both contractions.'''
    mc, cderi = water
    eris, nelec = pyscf_nr.nr_from_casci(mc, cderi=cderi)
    c = ActiveSpace.from_eris(eris).sector(nelec).vecs[:, :3]
    for contraction in ('union', 'state'):
        ref = _run(eris, nelec, c, amplitudes='spectral', contraction=contraction)
        new = _run(eris, nelec, c, amplitudes='rdm', contraction=contraction)
        _same(new, ref)
        assert new.rdm_info['spin_flip']
