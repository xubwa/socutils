import itertools
import numpy
import pytest
from pyscf.fci import cistring
from socutils.fci import zfci

from socutils.zmr.reference import dets
from socutils.zmr.tests._helpers import random_hamiltonian, random_vector


def test_gen_strings_matches_cistring_order():
    for norb, nelec in [(6, 3), (7, 2), (5, 0), (5, 5)]:
        ref = numpy.asarray(cistring.gen_strings4orblist(range(norb), nelec),
                            dtype=numpy.uint64)
        assert numpy.array_equal(dets.gen_strings(range(norb), nelec), ref)


def test_single_operator_phase_matches_cistring():
    strs = dets.gen_strings(range(8), 4)
    for p in range(8):
        for dagger, ref_sign in [(True, cistring.cre_sign), (False, cistring.des_sign)]:
            new, sign = dets.apply_opstring((dagger,), [[p]], strs)
            for s, n, g in zip(strs, new[0], sign[0]):
                assert g == ref_sign(p, int(s))
                if g:
                    assert int(n) == int(s) ^ (1 << p)


def test_determinant_from_vacuum():
    '''|q_1 ... q_n> = a^+_{q_n} ... a^+_{q_1} |vac> with a positive sign.'''
    occ = [1, 3, 4, 7]
    orbs = [occ[::-1]]
    new, sign = dets.apply_opstring((True,) * len(occ), orbs, [0])
    assert sign[0, 0] == 1
    assert int(new[0, 0]) == int(dets.occ_to_str(occ))
    # creating in the opposite order gives the permutation parity
    new, sign = dets.apply_opstring((True,) * len(occ), [occ], [0])
    assert sign[0, 0] == (-1) ** (len(occ) * (len(occ) - 1) // 2)


def _op(dagger, p):
    return [((dagger,), [[p]], [1.0])]


def test_anticommutation():
    norb = 6
    for nelec in range(norb + 1):
        strs = dets.gen_strings(range(norb), nelec)
        vec = dets.SparseVector(strs, random_vector(len(strs), seed=nelec))
        for p, q in itertools.product(range(norb), repeat=2):
            # {a_p, a_q^+} = delta_pq
            x = (dets.apply_operator(_op(False, p), dets.apply_operator(_op(True, q), vec))
                 + dets.apply_operator(_op(True, q), dets.apply_operator(_op(False, p), vec)))
            ref = vec if p == q else dets.SparseVector([], [])
            assert (x - ref).norm() < 1e-14
            # {a_p, a_q} = 0
            y = (dets.apply_operator(_op(False, p), dets.apply_operator(_op(False, q), vec))
                 + dets.apply_operator(_op(False, q), dets.apply_operator(_op(False, p), vec)))
            assert y.norm() < 1e-14


def test_hamiltonian_matches_zfci_full_space():
    norb, nelec = 6, 3
    h, eri = random_hamiltonian(norb, seed=1)
    strs = dets.gen_strings(range(norb), nelec)
    hmat = dets.operator_matrix(dets.hamiltonian_operator(h, eri), strs).toarray()
    assert abs(hmat - zfci.make_hmat(h, eri, norb, nelec)).max() < 1e-12
    assert abs(hmat - hmat.conj().T).max() < 1e-12


def test_hamiltonian_matches_zfci_on_determinant_list():
    '''Arbitrary subset of determinants in arbitrary order.'''
    norb, nelec = 7, 3
    h, eri = random_hamiltonian(norb, seed=2)
    rng = numpy.random.default_rng(3)
    strs = dets.gen_strings(range(norb), nelec)
    strs = rng.permutation(strs)[:20]
    occs = [dets.str_to_occ(s) for s in strs]
    hmat = dets.operator_matrix(dets.hamiltonian_operator(h, eri, ecore=0.7), strs).toarray()
    ref = zfci.make_hmat_dets(h, eri, norb, occs) + 0.7 * numpy.eye(len(strs))
    assert abs(hmat - ref).max() < 1e-12


def test_apply_operator_matches_operator_matrix():
    norb, nelec = 6, 3
    h, eri = random_hamiltonian(norb, seed=4)
    op = dets.hamiltonian_operator(h, eri, ecore=-1.3)
    strs = dets.gen_strings(range(norb), nelec)
    c = random_vector(len(strs), seed=5)
    hv = dets.apply_operator(op, dets.SparseVector(strs, c))
    ref = dets.operator_matrix(op, strs) @ c
    assert abs(hv.on(strs) - ref).max() < 1e-12
    assert set(hv.strs.tolist()) <= set(strs.tolist())


def test_electron_number_changing_operators():
    '''(a_p^+)^+ = a_p between the N and N+1 electron spaces, and a string
    with net charge maps between the right spaces.'''
    norb = 6
    s3 = dets.gen_strings(range(norb), 3)
    s4 = dets.gen_strings(range(norb), 4)
    s1 = dets.gen_strings(range(norb), 1)
    for p in range(norb):
        cre = dets.operator_matrix(_op(True, p), s4, s3).toarray()
        des = dets.operator_matrix(_op(False, p), s3, s4).toarray()
        assert abs(cre - des.conj().T).max() == 0
    # a_p a_q |3 electrons> lands in the 1-electron space
    c = random_vector(len(s3), seed=6)
    v = dets.SparseVector(s3, c)
    op = [((False, False), [[0, 2], [4, 1]], [1.0, 0.5j])]
    out = dets.apply_operator(op, v)
    assert set(out.strs.tolist()) <= set(s1.tolist())
    ref = dets.operator_matrix(op, s1, s3) @ c
    assert abs(out.on(s1) - ref).max() < 1e-14


def test_rejects_too_many_orbitals():
    with pytest.raises(ValueError):
        dets.apply_opstring((True,), [[64]], [0])
