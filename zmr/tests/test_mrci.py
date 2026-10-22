'''MRCI in the product basis against the brute-force determinant references.'''
import numpy
import pytest

from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.mrci import MRCI
from socutils.zmr.reference import icmrci_bruteforce, mrci_bruteforce
from socutils.zmr.tests._helpers import model_eris

CASES = [((2, 4, 3), 2, 1), ((3, 4, 2), 3, 2), ((2, 3, 3), 1, 1), ((1, 4, 2), 2, 2)]


def _setup(sizes, nelec, nstate, seed=7):
    eris = model_eris(*sizes, seed=seed)
    c = ActiveSpace.from_eris(eris).solve(nelec, nstate)[1]
    return eris, c


@pytest.mark.parametrize('sizes, nelec, nstate', CASES)
def test_uncontracted_is_the_determinant_mrcisd(sizes, nelec, nstate):
    eris, c = _setup(sizes, nelec, nstate)
    ref, _, strs = mrci_bruteforce.kernel(eris, nelec, nroots=4)
    ci = MRCI(eris, nelec, civecs=c, contraction='none')
    e = ci.kernel(4)
    assert ci.basis.dim == len(strs)
    assert ci.hermiticity < 1e-10
    assert abs(e - ref).max() < 1e-10


@pytest.mark.parametrize('sizes, nelec, nstate', CASES)
@pytest.mark.parametrize('internal', ['cas', 'states'])
def test_fully_internally_contracted(sizes, nelec, nstate, internal):
    eris, c = _setup(sizes, nelec, nstate)
    ref = icmrci_bruteforce.kernel(eris, nelec, c, nroots=nstate, internal=internal)
    ci = MRCI(eris, nelec, civecs=c, contraction='fic', internal=internal)
    e = ci.kernel(nstate)
    assert ci.basis.dim == ref.basis.shape[1]
    assert ci.hermiticity < 1e-10
    assert abs(e - ref.e).max() < 1e-10


@pytest.mark.parametrize('sizes, nelec, nstate', CASES[:2])
def test_werner_knowles_type_lies_between(sizes, nelec, nstate):
    eris, c = _setup(sizes, nelec, nstate)
    e = {k: MRCI(eris, nelec, civecs=c, contraction=k).kernel(nstate) for k in ('none', 'wk', 'fic')}
    assert numpy.all(e['none'] <= e['wk'] + 1e-10)
    assert numpy.all(e['wk'] <= e['fic'] + 1e-10)


def test_sigma_keeps_the_antisymmetry():
    eris, c = _setup((3, 4, 3), 2, 1)
    ci = MRCI(eris, 2, civecs=c).build()
    rng = numpy.random.default_rng(0)
    x = rng.normal(size=(2, ci.basis.dim)) + 1j * rng.normal(size=(2, ci.basis.dim))
    s = ci.sigma(ci.basis.unpack(x))
    for name, block in ci.basis.blocks.items():
        assert abs(block.unpack(block.pack(s[name])) - s[name]).max() < 1e-10, name


@pytest.mark.parametrize('contraction', ['none', 'fic'])
def test_davidson_equals_dense(contraction):
    eris, c = _setup((3, 4, 3), 2, 2)
    dense = MRCI(eris, 2, civecs=c, contraction=contraction)
    dense.dense_limit = 10**6
    e_dense = dense.kernel()
    it = MRCI(eris, 2, civecs=c, contraction=contraction)
    it.dense_limit = 0
    e = it.kernel()
    assert it.converged
    assert abs(e - e_dense).max() < 1e-8
    # the zeroth-order energies used as preconditioner: exact in the internal
    # block (eigenvectors of H_act), the Dyall energies elsewhere
    h = dense.hamiltonian()
    ncas = dense.basis.blocks['cas'].size
    assert abs(h.diagonal().real - dense.hdiag)[:ncas].max() < 1e-10
    assert abs(h.diagonal().real - dense.hdiag).max() < 3.0
    assert abs(dense.weights - it.weights).max() < 1e-6


def test_orbital_rotations_do_not_change_the_energies():
    '''Unitary rotations among the core and among the virtual orbitals.'''
    from scipy.stats import unitary_group
    eris, c = _setup((3, 4, 3), 2, 1)
    u = numpy.eye(eris.nmo, dtype=complex)
    sp = eris.spaces
    for label, seed in (('c', 1), ('v', 2)):
        s = sp.slice(label)
        u[s, s] = unitary_group.rvs(s.stop - s.start, random_state=seed)
    for contraction in ('fic', 'wk'):
        e0 = MRCI(eris, 2, civecs=c, contraction=contraction).kernel()
        e1 = MRCI(eris.rotate(u), 2, civecs=c, contraction=contraction).kernel()
        assert abs(e0 - e1).max() < 1e-9


def test_davidson_correction_and_weights():
    eris, c = _setup((3, 4, 3), 2, 1)
    ci = MRCI(eris, 2, civecs=c)
    e = ci.kernel()
    assert 0.5 < ci.weights[0] < 1.0
    assert ci.e_davidson[0] < e[0] and ci.e_davidson_renormalized[0] < ci.e_davidson[0]
    assert ci.e_rotated_ref[0] >= ci.e_ref[0] - 1e-10


@pytest.mark.parametrize('functional', ['acpf', 'aqcc'])
def test_acpf_aqcc(functional):
    '''Stationary points of the functional: checked against a direct
    evaluation with the dense Hamiltonian, and ordered E(ACPF) <= E(AQCC) <=
    E(CI) for the ground state.'''
    eris, c = _setup((3, 4, 3), 2, 1)
    e_ci = MRCI(eris, 2, civecs=c).kernel()
    ci = MRCI(eris, 2, civecs=c, functional=functional)
    e = ci.kernel()
    assert e[0] < e_ci[0]
    h = ci.hamiltonian()
    x = ci.ci[0]
    m = ci._metric()
    val = (x.conj() @ (h @ x) - ci.e_ref[0] * numpy.vdot(x, x)).real / (m * abs(x)**2).sum()
    assert abs(val - (e[0] - ci.e_ref[0])) < 1e-9
    # stationarity: the gradient of the functional vanishes
    grad = h @ x - ci.e_ref[0] * x - (e[0] - ci.e_ref[0]) * m * x
    assert abs(grad).max() < 1e-7
    if functional == 'aqcc':
        e_acpf = MRCI(eris, 2, civecs=c, functional='acpf').kernel()
        assert e_acpf[0] <= e[0] + 1e-10


def test_vvvv_terms_from_three_index_integrals():
    '''Water with Cholesky vectors: the (vv|vv) terms through the
    three-index integrals, the dense (vv|vv) block, and the exact integrals
    give the same energies.'''
    from pyscf import gto, mcscf, scf
    from socutils.zmr.interface import pyscf_nr
    from socutils.zmr.mrci.sigma import Sigma
    from socutils.zmr.tests.test_cholesky import factorize
    mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASSCF(mf, 3, 4)
    mc.kernel()
    exact, nelec = pyscf_nr.from_casci(mc, nfrozen=1)
    fact, _ = pyscf_nr.from_casci(mc, nfrozen=1, cderi=factorize(mol))
    # the same model states for all (the second level is a degenerate triplet)
    civecs = ActiveSpace.from_eris(exact).solve(nelec, 2)[1]
    e_ref = MRCI(exact, nelec, civecs=civecs).kernel()
    out = {}
    # the dense block of the integral source, the three-index integrals
    # directly, and the dense matrix built from the three-index integrals
    for direct, limit, kind in ((False, 2e9, None), (True, 2e9, '_VVVV'), (None, 0, '_VVVVDense')):
        ci = MRCI(fact, nelec, civecs=civecs)
        ci.dense_limit = 0
        Sigma.direct_vvvv, Sigma.vvvv_memory = direct, limit
        try:
            out[kind] = ci.kernel()
            used = [type(x).__name__ for pair in ci.sigma._compiled.values()
                    for _, cons in pair[1] for x in cons if type(x).__name__.startswith('_VVVV')]
        finally:
            Sigma.direct_vvvv, Sigma.vvvv_memory = None, 2e9
        assert used == ([kind] * 3 if kind else [])
        assert abs(out[kind] - e_ref).max() < 1e-8
    assert abs(out['_VVVV'] - out[None]).max() < 1e-9
    assert abs(out['_VVVVDense'] - out[None]).max() < 1e-9


def test_two_stage_contractions():
    '''Terms whose dressed integral is not formed (Gamma applied to the
    coefficients first) give the same sigma.'''
    from socutils.zmr.mrci.sigma import Sigma
    eris, c = _setup((3, 4, 3), 2, 2)
    rng = numpy.random.default_rng(1)
    out = []
    for limit in (2e7, 0):
        ci = MRCI(eris, 2, civecs=c)
        Sigma.w_limit = limit
        Sigma.w_max = 2e8 if limit else 0
        try:
            ci.build()
            if not out:
                x = rng.normal(size=(2, ci.basis.dim)) + 1j * rng.normal(size=(2, ci.basis.dim))
            out.append(ci.sigma.matvec(x))
            kinds = {type(con).__name__ for pair in ci.sigma._compiled.values() for con in pair[0]}
        finally:
            Sigma.w_limit, Sigma.w_max = 2e7, 2e8
        assert bool(kinds & {'_TwoStage', '_IntegralFirst'}) == (limit == 0)
    assert abs(out[0] - out[1]).max() < 1e-10


def test_replaced_active_integrals():
    '''active_eri replaces (tu|vw) in the all-active part of H: the same
    energies as a Hamiltonian whose active block is changed.'''
    from socutils.zmr.integrals import ERIs
    eris, _ = _setup((2, 4, 3), 2, 1)
    sp = eris.spaces
    rng = numpy.random.default_rng(3)
    d = 0.05 * (rng.normal(size=(4,) * 4) + 1j * rng.normal(size=(4,) * 4))
    d = d + d.transpose(2, 3, 0, 1)
    d = d + d.conj().transpose(1, 0, 3, 2)
    eri = numpy.array(eris.eri_block('pppp'))
    a = sp.slice('a')
    new = eri[a, a, a, a] + d
    eri2 = eri.copy()
    eri2[a, a, a, a] = new
    changed = ERIs(eris.ecore, eris.h1, eri2, sp)
    c = ActiveSpace.from_eris(changed).solve(2, 2)[1]
    ref = MRCI(changed, 2, civecs=c).kernel()
    e = MRCI(eris, 2, civecs=c, active_eri=new).kernel()
    assert abs(e - ref).max() < 1e-10
    assert abs(e - MRCI(eris, 2, civecs=c).kernel()).max() > 1e-6


def test_sliced_intermediates_and_streamed_gamma():
    '''Limits set to zero: matrices Gamma are not stored but streamed into
    the dressed integrals, two-step contractions are sliced along a
    spectator index, the coefficients times a block overlap along the bra
    functions, and the strings act on one function at a time.  The same
    sigma.'''
    from socutils.zmr.mrci import sigma as sg
    eris, c = _setup((3, 4, 4), 2, 2)
    rng = numpy.random.default_rng(2)
    ref = MRCI(eris, 2, civecs=c).build()
    x = rng.normal(size=(2, ref.basis.dim)) + 1j * rng.normal(size=(2, ref.basis.dim))
    s0 = ref.sigma.matvec(x)
    names = 'gamma_limit', 'x_limit', 'w_limit', 'w_max', 'string_limit'
    saved = [getattr(sg.Sigma, n) for n in names]
    for settings in ((1.0, 2e8, 2e7, 2e8, 1e8), (5e7, 1.0, 0, 0, 1e8), (1.0, 1.0, 0, 0, 1.0),
                     (5e7, 2e8, 2e7, 2e8, 1.0)):
        for n, v in zip(names, settings):
            setattr(sg.Sigma, n, v)
        try:
            ci = MRCI(eris, 2, civecs=c).build()
            s1 = ci.sigma.matvec(x)
        finally:
            for n, v in zip(names, saved):
                setattr(sg.Sigma, n, v)
        assert abs(s1 - s0).max() < 1e-10, settings


def test_printed_output(capsys):
    '''Settings, block table, setup steps, one line per Davidson iteration,
    results and timings; nothing with verbose = 0.'''
    eris, c = _setup((3, 4, 3), 2, 2)
    ci = MRCI(eris, 2, civecs=c)
    ci.dense_limit = 0
    e = ci.kernel()
    text = capsys.readouterr().out
    for piece in ('zmr  MRCI', 'fully internally contracted', 'Blocks of the MRCI space',
                  'Contractions for block ccvv', 'Davidson iterations', 'converged after',
                  'MRCI Results', 'Weight of the internal block', 'MRCI+Q (Davidson)',
                  'TIMINGS MRCI'):
        assert piece in text, piece
    assert '%.10f' % e[0] in text
    assert sum(abs(v - 1) < 1e-10 for v in map(lambda w: sum(w.values()), ci.block_weights)) == 2
    quiet = MRCI(eris, 2, civecs=c)
    quiet.verbose = 0
    quiet.kernel()
    assert capsys.readouterr().out == ''


def test_pair_packing():
    '''Matrices packed over antisymmetric index pairs (forced for all sizes)
    and unpacked ones give the same sigma.'''
    from socutils.zmr.mrci import sigma as sg
    eris, c = _setup((3, 4, 4), 2, 2)
    rng = numpy.random.default_rng(4)
    out = []
    for limit in (1e30, 0):
        saved = sg.Sigma.pack_limit, sg.Sigma.w_limit, sg.Sigma.w_max
        sg.Sigma.pack_limit = limit
        try:
            for w_limit in (2e7, 0):
                sg.Sigma.w_limit, sg.Sigma.w_max = w_limit, (2e8 if w_limit else 0)
                ci = MRCI(eris, 2, civecs=c)
                ci.verbose = 0
                ci.build()
                if not out:
                    x = rng.normal(size=(2, ci.basis.dim)) + 1j * rng.normal(size=(2, ci.basis.dim))
                out.append(ci.sigma.matvec(x))
        finally:
            sg.Sigma.pack_limit, sg.Sigma.w_limit, sg.Sigma.w_max = saved
    for s1 in out[1:]:
        assert abs(s1 - out[0]).max() < 1e-10


def test_two_step_contraction_by_active_index():
    '''The loop over the active indices (large intermediates) and the single
    intermediate give the same sigma.'''
    from socutils.zmr.mrci import sigma as sg
    eris, c = _setup((3, 4, 4), 2, 2)
    rng = numpy.random.default_rng(5)
    out = []
    saved = sg.Sigma.loop_limit, sg.Sigma.w_limit, sg.Sigma.w_max, sg.Sigma.pack_limit
    try:
        for loop, pack in ((2e7, 5e6), (0, 5e6), (0, 0)):
            sg.Sigma.loop_limit, sg.Sigma.w_limit, sg.Sigma.w_max, sg.Sigma.pack_limit = loop, 0, 0, pack
            ci = MRCI(eris, 2, civecs=c)
            ci.verbose = 0
            ci.build()
            if not out:
                x = rng.normal(size=(2, ci.basis.dim)) + 1j * rng.normal(size=(2, ci.basis.dim))
            out.append(ci.sigma.matvec(x))
            used = any(getattr(con, 'g3', None) is not None
                       for pair in ci.sigma._compiled.values() for con in pair[0])
            assert used == (loop == 0)
    finally:
        sg.Sigma.loop_limit, sg.Sigma.w_limit, sg.Sigma.w_max, sg.Sigma.pack_limit = saved
    for s1 in out[1:]:
        assert abs(s1 - out[0]).max() < 1e-10


def test_block_davidson():
    '''Own block Davidson against dense diagonalization: several roots, a
    near-degenerate pair, restarts with a small subspace.'''
    from socutils.zmr.mrci import davidson
    rng = numpy.random.default_rng(7)
    n = 400
    d = numpy.sort(rng.uniform(0, 30, n))
    d[1] = d[0] + 1e-6
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    h = numpy.diag(d) + 0.05 * (a + a.conj().T)
    ref = numpy.linalg.eigvalsh(h)[:4]
    calls = []

    def aop(x):
        calls.append(len(x))
        return x @ h.T

    def precond(r, e):
        den = h.diagonal().real - e
        den[abs(den) < 1e-6] = 1e-6
        return r / den

    for max_space in (12, 3):
        x0 = numpy.eye(n)[:4]
        conv, e, x = davidson.davidson(aop, list(x0), precond, 4, tol=1e-11, max_space=max_space)
        assert conv
        assert abs(e - ref).max() < 1e-9
        assert abs(x.conj() @ x.T - numpy.eye(4)).max() < 1e-8
        assert abs(x @ h.T - e[:, None] * x).max() < 1e-4


def test_davidson_converges_on_the_residual():
    '''Converged when every residual norm is below tol; the energies are
    then right to second order in it.'''
    from socutils.zmr.mrci import davidson
    rng = numpy.random.default_rng(5)
    n = 300
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    h = numpy.diag(numpy.arange(n) * 0.1) + 0.02 * (a + a.conj().T)
    diag = h.diagonal().real
    counts = {}
    for tol in (5e-5, 1e-9):
        norms = []
        conv, e, x = davidson.davidson(
            lambda v: v @ h.T, list(numpy.eye(n)[:3]),
            lambda r, e0: r / numpy.where(abs(diag - e0) < 1e-6, 1e-6, diag - e0), 3, tol=tol,
            callback=lambda icyc, space, e, de, res, sec: norms.append(res.max()))
        assert conv and norms[-1] < tol and all(r >= tol for r in norms[:-1])
        assert abs(numpy.linalg.norm(x @ h.T - e[:, None] * x, axis=1)).max() < tol
        assert abs(e - numpy.linalg.eigvalsh(h)[:3]).max() < 10 * tol**2 + 1e-12
        counts[tol] = len(norms)
    assert counts[5e-5] < counts[1e-9]


def test_davidson_follows_the_initial_vectors():
    '''follow=True: the roots are the Ritz vectors with the largest weight in
    the span of the initial vectors.  Two symmetry blocks coupled by noise
    only; the initial vectors approximate the two lowest states of the upper
    block (as the model states do in MRCI).  The plain iteration falls to the
    lowest eigenvalues of the whole matrix, the following one stays.'''
    from socutils.zmr.mrci import davidson
    rng = numpy.random.default_rng(8)
    n = 120

    def block(shift):
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        return numpy.diag(shift + 0.1 * numpy.arange(n)) + 0.02 * (a + a.conj().T)

    h = numpy.zeros((2 * n, 2 * n), dtype=complex)
    h[:n, :n], h[n:, n:] = block(0.0), block(0.35)
    noise = 1e-9 * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    h[:n, n:], h[n:, :n] = noise, noise.conj().T
    diag = h.diagonal().real
    target, vec = numpy.linalg.eigh(h[n:, n:])
    target = target[:2]
    x0 = numpy.zeros((2, 2 * n), dtype=complex)
    x0[:, n:] = vec[:, :2].T + 0.02 * rng.normal(size=(2, n))      # weight 0.95 in the targets

    # denominators bounded from below, as MRCI does for followed roots: a small
    # one would amplify the noise on the functions of the other block
    def precond(r, e0):
        return r / numpy.where(abs(diag - e0) < 0.02, 0.02, diag - e0)

    for nroots, max_space in ((2, 6), (2, 12), (1, 6)):
        conv, e, x = davidson.davidson(lambda v: v @ h.T, list(x0[:nroots]), precond, nroots,
                                       tol=1e-6, max_space=max_space, follow=True)
        assert conv and abs(e - target[:nroots]).max() < 1e-9, (nroots, max_space)
        assert (abs(x[:, :n])**2).sum() < 1e-8
    # without following: the lowest states of the other block
    conv, e, x = davidson.davidson(lambda v: v @ h.T, list(x0), precond, 2, tol=1e-6,
                                   max_space=6, max_cycle=300)
    assert e[0] < target[0] - 0.05


def test_davidson_vectors_in_scratch_files(tmp_path, monkeypatch):
    '''incore = 0: the trial vectors and their images live in mapped scratch
    files; the same iteration, and the files are removed afterwards.'''
    from socutils.zmr.mrci import davidson
    monkeypatch.setenv('ZMR_TMPDIR', str(tmp_path))
    rng = numpy.random.default_rng(9)
    n = 200
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    h = numpy.diag(numpy.arange(n) * 0.1) + 0.02 * (a + a.conj().T)
    diag = h.diagonal().real

    def precond(r, e0):
        return r / numpy.where(abs(diag - e0) < 1e-6, 1e-6, diag - e0)

    out = {}
    for incore in (None, 0.0):
        counts = []
        out[incore] = davidson.davidson(lambda v: v @ h.T, list(numpy.eye(n)[:3]), precond, 3,
                                        tol=1e-8, max_space=4, incore=incore,
                                        callback=lambda *args: counts.append(args[0])), len(counts)
        assert davidson.last['on_disk'] == (incore is not None)
    (conv0, e0, x0), n0 = out[None]
    (conv1, e1, x1), n1 = out[0.0]
    assert conv0 and conv1 and n0 == n1
    assert abs(e0 - e1).max() < 1e-12 and abs(x0 - x1).max() < 1e-10
    assert not list(tmp_path.iterdir())
