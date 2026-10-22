'''RDMs by definition vs socutils.fci.zfci and pyscf.fci.fci_dhf_slow.
Pins the convention gamma_pq = <p^+ q>, Gamma_pqrs = <p^+ r^+ s q>
(conventions.md section 5) for all three implementations.'''
import numpy
from pyscf.fci import fci_dhf_slow
from socutils.fci import zfci

from socutils.zmr.reference import dets
from socutils.zmr.tests._helpers import random_hamiltonian

NORB, NELEC = 6, 3


def _states(nroots=2):
    h, eri = random_hamiltonian(NORB, seed=11)
    strs = dets.gen_strings(range(NORB), NELEC)
    hmat = dets.operator_matrix(dets.hamiltonian_operator(h, eri), strs).toarray()
    e, c = numpy.linalg.eigh(hmat)
    return h, eri, strs, e[:nroots], c[:, :nroots]


def test_rdm_energy():
    h, eri, strs, e, c = _states(1)
    v = dets.SparseVector(strs, c[:, 0])
    g1 = dets.trans_rdm1(v, v, NORB)
    g2 = dets.trans_rdm2(v, v, NORB)
    energy = numpy.einsum('pq,pq', h, g1) + 0.5 * numpy.einsum('pqrs,pqrs', eri, g2)
    assert abs(energy - e[0]) < 1e-11


def test_rdm_matches_zfci_and_fci_dhf_slow():
    _, _, strs, _, c = _states(1)
    v = dets.SparseVector(strs, c[:, 0])
    g1 = dets.trans_rdm1(v, v, NORB)
    g2 = dets.trans_rdm2(v, v, NORB)
    occs = [dets.str_to_occ(s) for s in strs]
    z1, z2 = zfci.make_rdm12_dets(c[:, 0], NORB, occs)
    d1, d2 = fci_dhf_slow.make_rdm12(c[:, 0], NORB, NELEC)
    for a, b in [(g1, z1), (g1, d1), (g2, z2), (g2, d2)]:
        assert abs(a - b).max() < 1e-13


def test_transition_rdm1_matches_zfci():
    _, _, strs, _, c = _states(2)
    bra = dets.SparseVector(strs, c[:, 0])
    ket = dets.SparseVector(strs, c[:, 1] * numpy.exp(0.3j))
    t = dets.trans_rdm1(bra, ket, NORB)
    occs = [dets.str_to_occ(s) for s in strs]
    ref = zfci.trans_rdm1_dets(bra.on(strs), ket.on(strs), NORB, occs)
    assert abs(t - ref).max() < 1e-13
    assert abs(t - dets.trans_rdm1(ket, bra, NORB).conj().T).max() < 1e-13
