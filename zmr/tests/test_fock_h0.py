'''Second-order perturbation theory with the Fock operator as zeroth-order
Hamiltonian in the active space (h0 = 'fock' of the NEVPT2 drivers).'''
import numpy
import pytest
from socutils.fci import zfci

from socutils.zmr.integrals import ERIs
from socutils.zmr.pc.nevpt2 import PCNEVPT2
from socutils.zmr.reference import dets, mrci_bruteforce
from socutils.zmr.tests._helpers import model_eris
from socutils.zmr.uc.nevpt2 import UCNEVPT2


def _run(cls, eris, nelec, h0, **kw):
    pt = cls(eris, nelec, **kw)
    pt.h0 = h0
    pt.verbose = 0
    return pt, pt.kernel()


def _bruteforce(pt, nelec, nstate):
    '''Effective Hamiltonian in the determinant basis: H0 = the generalized
    Fock matrix without its blocks between core, active and virtual
    orbitals, P = the nstate lowest CASCI states rotated to diagonalize H0,
    Q = the determinants outside the active space.'''
    eris, f = pt.quasi_canonical.eris, pt.quasi_canonical.fock
    sp = eris.spaces
    strs, labels = mrci_bruteforce.mrci_strings(sp, nelec)
    occs = [dets.str_to_occ(s) for s in strs]
    h = zfci.make_hmat_dets(eris.h1, eris.eri_block('pppp'), sp.nmo, occs)
    h = h + eris.ecore * numpy.eye(len(strs))
    f0 = numpy.zeros_like(f)
    for x in 'cav':
        s = sp.slice(x)
        f0[s, s] = f[s, s]
    h0 = zfci.make_hmat_dets(f0, numpy.zeros((sp.nmo,) * 4, dtype=complex), sp.nmo, occs)
    cas = numpy.array([lab == (0, 0) for lab in labels])
    ext = ~cas
    e, c = numpy.linalg.eigh(h[numpy.ix_(cas, cas)])
    c = c[:, :nstate]
    w, u = numpy.linalg.eigh(c.conj().T @ h0[numpy.ix_(cas, cas)] @ c)
    c = c @ u
    v = h[numpy.ix_(ext, cas)] @ c
    a = numpy.zeros((nstate, nstate), dtype=complex)
    for j in range(nstate):
        x = numpy.linalg.solve(w[j] * numpy.eye(ext.sum()) - h0[numpy.ix_(ext, ext)], v[:, j])
        a[:, j] = v.conj().T @ x
    heff = c.conj().T @ h[numpy.ix_(cas, cas)] @ c + 0.5 * (a + a.conj().T)
    return numpy.linalg.eigvalsh(heff)


@pytest.mark.parametrize('sizes,nelec,nstate', [((2, 4, 3), 2, 6), ((2, 4, 3), 2, 2),
                                                ((1, 4, 2), 3, 4), ((2, 3, 3), 1, 3)])
def test_against_determinants(sizes, nelec, nstate):
    eris = model_eris(*sizes, seed=3)
    pt, e = _run(UCNEVPT2, eris, nelec, 'fock', nstate=nstate)
    ref = _bruteforce(pt, nelec, nstate)
    assert abs(e - ref).max() < 1e-9
    # the Dyall Hamiltonian gives something else
    assert abs(_run(UCNEVPT2, eris, nelec, 'dyall', nstate=nstate)[1] - ref).max() > 1e-6


def test_contracted_equals_uncontracted_for_the_complete_model_space():
    eris = model_eris(2, 4, 3, seed=5)
    nstate = 6                      # all states of 2 electrons in 4 spinors
    e_uc = _run(UCNEVPT2, eris, 2, 'fock', nstate=nstate)[1]
    e_pc = _run(PCNEVPT2, eris, 2, 'fock', nstate=nstate)[1]
    assert abs(e_uc - e_pc).max() < 1e-9


def test_invariant_under_rotations_of_the_model_states():
    eris = model_eris(2, 4, 3, seed=7)
    pt, e = _run(PCNEVPT2, eris, 2, 'fock', nstate=3)
    rng = numpy.random.default_rng(1)
    q, _ = numpy.linalg.qr(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))
    e_rot = _run(PCNEVPT2, eris, 2, 'fock', civecs=pt.civecs @ q)[1]
    assert abs(e - e_rot).max() < 1e-9


def test_equals_dyall_without_active_two_electron_integrals():
    eris = model_eris(2, 4, 3, seed=9)
    sp = eris.spaces
    a = sp.slice('a')
    eri = eris.eri_block('pppp').copy()
    eri[a, a, a, a] = 0.0
    eris = ERIs(eris.ecore, eris.h1, eri, sp)
    for cls in (UCNEVPT2, PCNEVPT2):
        e_f = _run(cls, eris, 2, 'fock', nstate=3)[1]
        e_d = _run(cls, eris, 2, 'dyall', nstate=3)[1]
        assert abs(e_f - e_d).max() < 1e-9


def test_single_state_energies_on_the_diagonal():
    eris = model_eris(2, 4, 3, seed=11)
    nelec, nstate = 2, 3
    pt = UCNEVPT2(eris, nelec, nstate=nstate)
    pt.h0, pt.fock_rotate, pt.verbose = 'fock', False, 0
    pt.kernel()
    eris, f = pt.quasi_canonical.eris, pt.quasi_canonical.fock
    sp = eris.spaces
    strs, labels = mrci_bruteforce.mrci_strings(sp, nelec)
    occs = [dets.str_to_occ(s) for s in strs]
    h = zfci.make_hmat_dets(eris.h1, eris.eri_block('pppp'), sp.nmo, occs)
    h = h + eris.ecore * numpy.eye(len(strs))
    f0 = numpy.zeros_like(f)
    for x in 'cav':
        f0[sp.slice(x), sp.slice(x)] = f[sp.slice(x), sp.slice(x)]
    h0 = zfci.make_hmat_dets(f0, numpy.zeros((sp.nmo,) * 4, dtype=complex), sp.nmo, occs)
    cas = numpy.array([lab == (0, 0) for lab in labels])
    ext = ~cas
    e, c = numpy.linalg.eigh(h[numpy.ix_(cas, cas)])
    ref = []
    for j in range(nstate):
        e0 = (c[:, j].conj() @ h0[numpy.ix_(cas, cas)] @ c[:, j]).real
        v = h[numpy.ix_(ext, cas)] @ c[:, j]
        x = numpy.linalg.solve(e0 * numpy.eye(ext.sum()) - h0[numpy.ix_(ext, ext)], v)
        ref.append(e[j] + (v.conj() @ x).real)
    assert abs(pt.heff.diagonal().real - numpy.array(ref)).max() < 1e-9
