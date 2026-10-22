'''CASPT2 (zmr.mrci.caspt2) against dense calculations in the determinant basis.'''
import numpy
import pytest
from socutils.fci import zfci

from socutils.zmr import fock
from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.mrci.caspt2 import CASPT2
from socutils.zmr.pc.nevpt2 import PCNEVPT2
from socutils.zmr.reference import dets, icmrci_bruteforce, mrci_bruteforce
from socutils.zmr.tests._helpers import model_eris


def _reference(eris, nelec, nstate, contracted):
    '''H0 = P F P + S F S with the full generalized Fock matrix of the
    averaged density; S = all external determinants, or the internally
    contracted space built from its definition.'''
    sp = eris.spaces
    act = ActiveSpace.from_eris(eris)
    _, civecs = act.solve(nelec, nstate)
    f = fock.generalized_fock(eris, act.averaged_rdm1(civecs, nelec))
    if contracted:
        strs, basis, count = icmrci_bruteforce.contracted_basis(sp, nelec, civecs,
                                                                internal='states')
        nint = count['cas']
    else:
        strs, labels = mrci_bruteforce.mrci_strings(sp, nelec)
        order = numpy.argsort(strs)
        strs = strs[order]
        cas = numpy.array([labels[k] == (0, 0) for k in order])
        basis, nint = None, None
    occs = [dets.str_to_occ(s) for s in strs]
    h = zfci.make_hmat_dets(eris.h1, eris.eri_block('pppp'), sp.nmo, occs)
    h = h + eris.ecore * numpy.eye(len(strs))
    h0 = zfci.make_hmat_dets(f, numpy.zeros((sp.nmo,) * 4, dtype=complex), sp.nmo, occs)
    if contracted:
        h, h0 = basis.conj().T @ h @ basis, basis.conj().T @ h0 @ basis
        p = numpy.arange(len(h)) < nint
        e, c = numpy.linalg.eigh(h[numpy.ix_(p, p)])
        c = c[:, :nstate]
    else:
        e, c = numpy.linalg.eigh(h[numpy.ix_(cas, cas)])
        c = c[:, :nstate]
        p = cas
    s = ~p
    w, u = numpy.linalg.eigh(c.conj().T @ h0[numpy.ix_(p, p)] @ c)
    c = c @ u
    v = h[numpy.ix_(s, p)] @ c
    a = numpy.zeros((nstate, nstate), dtype=complex)
    for j in range(nstate):
        x = numpy.linalg.solve(w[j] * numpy.eye(s.sum()) - h0[numpy.ix_(s, s)], v[:, j])
        a[:, j] = v.conj().T @ x
    heff = c.conj().T @ h[numpy.ix_(p, p)] @ c + 0.5 * (a + a.conj().T)
    return numpy.linalg.eigvalsh(heff)


def _run(eris, nelec, nstate, contraction, **kw):
    pt = CASPT2(eris, nelec, nstate=nstate, contraction=contraction, **kw)
    pt.verbose = 0
    e = pt.kernel()
    assert pt.converged
    return pt, e


@pytest.mark.parametrize('sizes,nelec,nstate', [((2, 4, 3), 2, 1), ((2, 4, 3), 2, 3),
                                                ((2, 4, 3), 2, 6), ((1, 4, 2), 3, 2),
                                                ((2, 3, 3), 1, 3)])
def test_uncontracted(sizes, nelec, nstate):
    eris = model_eris(*sizes, seed=4)
    _, e = _run(eris, nelec, nstate, 'none')
    assert abs(e - _reference(eris, nelec, nstate, False)).max() < 1e-7


@pytest.mark.parametrize('sizes,nelec,nstate', [((2, 4, 3), 2, 1), ((2, 4, 3), 2, 3),
                                                ((1, 4, 2), 3, 2)])
def test_internally_contracted(sizes, nelec, nstate):
    eris = model_eris(*sizes, seed=6)
    _, e = _run(eris, nelec, nstate, 'fic')
    assert abs(e - _reference(eris, nelec, nstate, True)).max() < 1e-7


def test_contracted_equals_uncontracted_for_the_complete_model_space():
    eris = model_eris(2, 4, 3, seed=8)
    assert abs(_run(eris, 2, 6, 'fic')[1] - _run(eris, 2, 6, 'none')[1]).max() < 1e-7


def test_invariant_under_rotations_of_the_model_states():
    eris = model_eris(2, 4, 3, seed=10)
    pt, e = _run(eris, 2, 3, 'fic')
    rng = numpy.random.default_rng(2)
    q, _ = numpy.linalg.qr(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))
    pt2 = CASPT2(eris, 2, civecs=pt.civecs @ q)
    pt2.verbose = 0
    assert abs(pt2.kernel() - e).max() < 1e-7


def test_fock_h0_of_nevpt2_is_caspt2_without_the_offdiagonal_fock_blocks():
    '''With the core-active, active-virtual and core-virtual blocks of the
    Fock matrix removed, CASPT2 is PC-NEVPT2 with h0 = "fock".'''
    eris = model_eris(2, 4, 3, seed=12)
    nelec, nstate = 2, 6

    class Diagonal(CASPT2):
        pass

    import socutils.zmr.mrci.caspt2 as module
    full = fock.generalized_fock

    def block_diagonal(er, gamma):
        f = full(er, gamma)
        out = numpy.zeros_like(f)
        for x in 'cav':
            out[er.spaces.slice(x), er.spaces.slice(x)] = f[er.spaces.slice(x), er.spaces.slice(x)]
        return out

    pt = Diagonal(eris, nelec, nstate=nstate, contraction='none')
    pt.verbose = 0
    saved = module.fock.generalized_fock
    try:
        # the quasi-canonical orbitals of build() need the full matrix
        pt.build()
        build = pt.build
        pt.build = lambda log=None: pt
        module.fock.generalized_fock = block_diagonal
        e = pt.kernel()
    finally:
        module.fock.generalized_fock = saved
    nev = PCNEVPT2(eris, nelec, nstate=nstate)
    nev.h0, nev.verbose = 'fock', 0
    assert abs(e - nev.kernel()).max() < 1e-7
