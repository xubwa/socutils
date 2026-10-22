'''Active space of spin-restricted orbitals: order and phases of pyscf.fci,
agreement with the spinor code in the corresponding spin-orbital order.'''
import numpy
import pytest
from pyscf import fci

from socutils.zmr import lib
from socutils.zmr.active.sectors import ActiveSpace as SpinorActiveSpace
from socutils.zmr.nr.active import (ALPHA, BETA, ActiveSpace, sector_index, so,
                           spin_orbital_hamiltonian)

NORB = 5


def _hamiltonian(seed=0):
    rng = numpy.random.default_rng(seed)
    h = rng.normal(size=(NORB, NORB))
    h = h + h.T
    b = rng.normal(size=(6, NORB, NORB))
    b = b + b.transpose(0, 2, 1)
    return h, 0.1 * numpy.einsum('lpq,lrs->pqrs', b, b)


@pytest.fixture(params=[False, True], ids=['native', 'fallback'])
def act(request, monkeypatch):
    if request.param:
        monkeypatch.setenv('ZMR_NO_NATIVE', '1')
    lib.reset()
    yield ActiveSpace(*_hamiltonian(), 0.3)
    monkeypatch.delenv('ZMR_NO_NATIVE', raising=False)
    lib.reset()


@pytest.mark.parametrize('nelec', [(2, 2), (3, 1), (1, 3), (0, 2), (5, 5), (0, 0), (3, 2)])
def test_sector_hamiltonian(act, nelec):
    h, eri = _hamiltonian()
    hm = act.make_hmat(nelec)
    dim = act.dim(nelec)
    assert hm.shape == (dim, dim) and hm.dtype == numpy.float64
    # the spinor code with the spin-orbitals [beta block, alpha block]
    spinor = SpinorActiveSpace(*spin_orbital_hamiltonian(h, eri))
    idx = sector_index(NORB, nelec)
    ref = spinor.make_hmat_numpy(sum(nelec))[numpy.ix_(idx, idx)]
    assert abs(ref.imag).max(initial=0) == 0
    assert abs(hm - ref.real).max() < 1e-13
    if dim > 1:
        # pyscf.fci: same determinant order and phases
        addr, h0 = fci.direct_spin1.pspace(h, eri, NORB, nelec, np=dim + 1)
        assert (addr == numpy.arange(dim)).all()
        assert abs(hm - h0).max() < 1e-13


def test_dimensions_and_targets(act):
    assert act.dim((2, 3)) == 10 * 10
    assert act.dim((-1, 2)) == 0 and act.dim((2, 6)) == 0
    assert act.target((2, 3), (1, -1)) == (3, 2)
    assert act.spin_change((True, False), [[so(1, ALPHA, NORB), so(2, BETA, NORB)]]) == (1, -1)
    with pytest.raises(ValueError):
        act.spin_change((True,), [[0], [NORB]])


def test_pyscf_ci_vectors_are_eigenvectors(act):
    h, eri = _hamiltonian()
    nelec = (3, 2)
    e, c = fci.direct_spin1.kernel(h, eri, NORB, nelec, nroots=3)
    for ek, ck in zip(e, c):
        v = ck.ravel()
        assert abs(act.hmat(nelec) @ v - ek * v).max() < 1e-8
    e0, c0 = act.solve(nelec, 3)
    assert abs(e0 - 0.3 - e).max() < 1e-9
    # spin-summed 1-RDM (pyscf stores <q^+ p>)
    d = act.trans_rdm1(c[0].ravel(), c[0].ravel(), nelec)
    assert abs(d - fci.direct_spin1.make_rdm1(c[0], NORB, nelec).T).max() < 1e-12
    assert abs(numpy.trace(act.averaged_rdm1(c0, nelec)) - 5) < 1e-12


def test_operator_strings(act):
    '''Strings of spin-orbital operators vs the spinor kernel on the
    embedded vectors.'''
    from socutils.zmr.lib import kernels
    nelec = (3, 2)
    rng = numpy.random.default_rng(1)
    vecs = rng.normal(size=(2, 3, act.dim(nelec)))
    t, u, v = numpy.indices((NORB,) * 3).reshape(3, -1)
    blocks = [((True,), so(numpy.arange(NORB), ALPHA, NORB)[:, None]),
              ((True, True, False), numpy.stack([so(t, ALPHA, NORB), so(u, BETA, NORB),
                                                 so(v, BETA, NORB)], axis=1)),
              ((True, True, False), numpy.stack([so(t, ALPHA, NORB), so(u, ALPHA, NORB),
                                                 so(v, ALPHA, NORB)], axis=1))]
    out = act.apply_strings(vecs, nelec, blocks)
    target = (4, 2)
    assert out.shape == (2, 3, NORB + 2 * NORB**3, act.dim(target))
    src, dst = sector_index(NORB, nelec), sector_index(NORB, target)
    full = numpy.zeros((6, 252), dtype=complex)
    full[:, src] = vecs.reshape(6, -1)
    x0 = 0
    for daggers, ops in blocks:
        ref = kernels.apply_strings(2 * NORB, 5, daggers, ops, full)
        assert abs(numpy.delete(ref, dst, axis=2)).max() == 0     # stays in the sector
        assert abs(out.reshape(6, -1, len(dst))[:, x0:x0 + len(ops)] - ref[:, :, dst]).max() == 0
        x0 += len(ops)
    # complex vectors, an empty target sector, blocks of different sectors
    z = act.apply_strings(vecs[0] + 1j * vecs[1], nelec, blocks)
    assert abs(z - (out[0] + 1j * out[1])).max() == 0
    assert act.apply_strings(vecs, nelec, [((False,) * 3, [[0, 1, 2]])]).shape == (2, 3, 1, 0)
    with pytest.raises(ValueError):
        act.apply_strings(vecs, nelec, [((True,), [[0]]), ((True,), [[NORB]])])


# ----------------------------------------------------------------------
# matrix-free H_act (kernels.sigma_ab) against the dense sector Hamiltonian

def _sigma_hamiltonian(norb, kind, seed=3):
    '''eight: (pq|rs) with all eight permutational symmetries; four: without
    (pq|rs) = (rs|pq); none: no symmetry at all (h1 not symmetric either).'''
    rng = numpy.random.default_rng(seed)
    h = rng.normal(size=(norb, norb))
    eri = rng.normal(size=(norb,) * 4)
    if kind != 'none':
        h = h + h.T
        eri = eri + eri.transpose(1, 0, 2, 3)
        eri = eri + eri.transpose(0, 1, 3, 2)
    if kind == 'eight':
        eri = eri + eri.transpose(2, 3, 0, 1)
    return h, eri


SIGMA_SECTORS = [(6, 3, 3), (6, 4, 2), (6, 1, 4), (5, 0, 3), (5, 2, 0), (5, 5, 2), (5, 3, 5),
                 (4, 0, 0), (4, 4, 4), (4, 0, 4), (1, 1, 0), (7, 4, 3)]


@pytest.mark.parametrize('kind', ['eight', 'four', 'none'])
@pytest.mark.parametrize('norb, na, nb', SIGMA_SECTORS)
def test_sigma_ab(norb, na, nb, kind):
    from socutils.zmr.lib import kernels
    lib.reset()
    if lib.zmr() is None:
        pytest.skip('libzmr is not built (run make)')
    h, eri = _sigma_hamiltonian(norb, kind)
    hmat = kernels.build_hmat_ab(norb, (na, nb), h, eri)
    dim = hmat.shape[0]
    rng = numpy.random.default_rng(5)
    tol = 1e-11 * max(abs(hmat).max(), 1.0)
    for nvec in (1, 2, 5, 37, 150):
        q = rng.normal(size=(dim, nvec))
        ref = hmat @ q
        out = kernels.sigma_ab(norb, (na, nb), h, eri, q)
        assert out.shape == q.shape and abs(out - ref).max() < tol
        if kind == 'eight':
            # the determinant-driven kernel and pyscf.fci.direct_spin1
            assert abs(kernels.sigma_ab_det(norb, (na, nb), h, eri, q) - ref).max() < tol
        if nvec <= 5:
            assert abs(kernels.sigma_ab_pyscf(norb, (na, nb), h, eri, q) - ref).max() < tol
    # non-contiguous input: every second column of a Fortran-ordered array
    big = numpy.asfortranarray(rng.normal(size=(dim, 12)))
    q = big[:, 1::2]
    assert not q.flags.c_contiguous
    assert abs(kernels.sigma_ab(norb, (na, nb), h, eri, q) - hmat @ q).max() < tol
    # no vectors
    assert kernels.sigma_ab(norb, (na, nb), h, eri, numpy.zeros((dim, 0))).shape == (dim, 0)


@pytest.mark.parametrize('threads', [1, 3, 16])
def test_sigma_ab_threading(threads):
    '''Both threading modes (threads over batches of vectors, threads over
    alpha strings) on a sector large enough to start a team.'''
    from socutils.zmr.lib import kernels
    lib.reset()
    if lib.zmr() is None:
        pytest.skip('libzmr is not built (run make)')
    norb, nelec = 8, (4, 3)
    h, eri = _sigma_hamiltonian(norb, 'eight')
    hmat = kernels.build_hmat_ab(norb, nelec, h, eri)
    rng = numpy.random.default_rng(6)
    before = lib.num_threads()
    lib.set_num_threads(threads)
    try:
        for nvec in (1, 2, 9, 15, 16, 17, 40):
            q = rng.normal(size=(hmat.shape[0], nvec))
            ref = hmat @ q
            out = kernels.sigma_ab(norb, nelec, h, eri, q)
            assert abs(out - ref).max() < 1e-11 * abs(ref).max()
    finally:
        lib.set_num_threads(before)


def test_apply_h_matrix_free(act):
    '''ActiveSpace.apply_h above dense_limit (libzmr or pyscf.fci), real and
    complex vectors.'''
    nelec = (3, 2)
    hmat = act.make_hmat(nelec)
    rng = numpy.random.default_rng(7)
    q = rng.normal(size=(act.dim(nelec), 4)) + 1j * rng.normal(size=(act.dim(nelec), 4))
    act.dense_limit = 10
    assert not act.is_dense(nelec)
    assert abs(act.apply_h(nelec, q) - hmat @ q).max() < 1e-12
    assert abs(act.apply_h(nelec, q.real) - hmat @ q.real).max() < 1e-12
