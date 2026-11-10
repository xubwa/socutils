'''zmr.kramers: time reversal in an orbital basis, Kramers-paired eigenvectors, and the
NEVPT2 with one core pair of every Kramers-related couple against all pairs.'''
import os
import sys

import numpy
import pytest
from pyscf import gto

from socutils.zmr import kramers

sys.path.insert(0, os.path.dirname(__file__))


def paired_unitary(n, rng):
    '''A unitary that maps the interleaved pairs onto pairs: blocks [[a, -conj(b)],
    [b, conj(a)]] in a random unitary of quaternion form.'''
    tr = kramers.TimeReversal.interleaved(2 * n)
    th = tr.matrix()
    x = rng.normal(size=(2 * n, 2 * n)) + 1j * rng.normal(size=(2 * n, 2 * n))
    h = x + x.conj().T
    h = 0.5 * (h + th @ h.conj() @ th.conj().T)
    res = kramers.paired_eigh(h, tr)
    assert res is not None
    return res[1], tr, h


def test_time_reversal_basics():
    rng = numpy.random.default_rng(3)
    tr = kramers.TimeReversal.interleaved(6)
    v = rng.normal(size=6) + 1j * rng.normal(size=6)
    # T^2 = -1
    assert abs(tr.apply(tr.apply(v)) + v).max() < 1e-14
    # <v|Tv> = 0 for every vector
    assert abs(numpy.vdot(v, tr.apply(v))) < 1e-12
    with pytest.raises(ValueError):
        kramers.TimeReversal([1, 0], [1.0, 1.0])          # T^2 = +1
    assert list(tr.representatives()) == [0, 2, 4]
    sub = tr.subset([2, 3])
    assert list(sub.partner) == [1, 0]
    assert tr.subset([0, 2]) is None


def test_paired_eigh_and_rotation():
    rng = numpy.random.default_rng(4)
    u, tr, h = paired_unitary(5, rng)
    e = numpy.diag(u.conj().T @ h @ u).real
    assert abs(u.conj().T @ h @ u - numpy.diag(e)).max() < 1e-10
    assert abs(u.conj().T @ u - numpy.eye(10)).max() < 1e-12
    # consecutive columns are (v, Tv), equal eigenvalues
    for k in range(0, 10, 2):
        assert abs(tr.apply(u[:, k]) - u[:, k + 1]).max() < 1e-12
        assert abs(e[k] - e[k + 1]) < 1e-10
    new = tr.rotated(u)
    assert new is not None
    assert list(new.partner) == list(numpy.arange(10) ^ 1)
    assert abs(new.phase - kramers.TimeReversal.interleaved(10).phase).max() < 1e-10
    # a rotation that mixes the members of different pairs loses them
    w = numpy.linalg.qr(rng.normal(size=(10, 10)) + 1j * rng.normal(size=(10, 10)))[0]
    assert tr.rotated(w) is None
    # a matrix that does not commute with T has no paired eigenvectors
    x = rng.normal(size=(10, 10)) + 1j * rng.normal(size=(10, 10))
    assert kramers.paired_eigh(x + x.conj().T, tr) is None


@pytest.fixture(scope='module')
def hbr():
    from pyscf.x2c import x2c
    from socutils.scf import double_group_symm as dgs
    from socutils.symm import spinor_basis as sb
    from test_cholesky import factorize
    mol = sb.to_symm_frame(gto.M(atom='H 0 0 0; Br 0 0 1.4144', basis='6-31g',
                                 symmetry=True, verbose=0))
    mf = dgs.SymAdaptedRHF(mol)
    mf.with_x2c = x2c.SpinorX2CHelper(mol)
    mf.conv_tol = 1e-11
    mf.verbose = 0
    mf.kernel()
    return mol, mf, factorize(mol)


def test_nevpt2_kramers_pairs(hbr):
    '''PC-NEVPT2 of HBr (Kramers-restricted double-group SCF): the core pairs of ccvv
    reduced to one of every Kramers-related couple, and cavv from one core orbital of
    every pair, give the energies of all labels.'''
    from socutils.zmr.interface import socutils_x2c
    from socutils.zmr.pc.nevpt2 import PCNEVPT2
    from socutils.zmr.uc import classes
    mol, mf, cderi = hbr
    eris = socutils_x2c.from_scf(mf, ncore=mol.nelectron - 2, nact=4, cderi=cderi)
    assert eris.time_reversal is not None
    out = []
    for use in (True, False):
        e = socutils_x2c.from_scf(mf, ncore=mol.nelectron - 2, nact=4, cderi=cderi)
        if not use:
            e.time_reversal = None
        pt = PCNEVPT2(e, 2, nstate=1)
        pt.verbose = 0
        out.append((pt.kernel(), pt))
    (ea, pa), (eb, pb) = out
    qc = pa.quasi_canonical
    assert qc.eris.time_reversal is not None
    ctx = classes.Context(qc.eris, qc.eps_core, qc.eps_virt, None)
    pairs = classes.kramers_pairs(ctx)
    nc = qc.eris.spaces.ncore
    assert pairs is not None and len(pairs[0]) < nc * (nc - 1) // 2
    assert abs(pairs[2].sum() - nc * (nc - 1) // 2) < 1e-12
    assert pb.quasi_canonical.eris.time_reversal is None
    assert abs(numpy.asarray(ea) - numpy.asarray(eb)).max() < 1e-10
    assert classes.kramers_core_half(ctx) is not None
    for name in ('ccvv', 'cavv', 'ccav'):
        assert abs(pa.h2_by_class[name] - pb.h2_by_class[name]).max() < 1e-11, name


def test_nevpt2_memory_check(hbr):
    '''Too small max_memory stops NEVPT2 before the integrals, with the minimum needed.'''
    from socutils.zmr.interface import socutils_x2c
    from socutils.zmr.pc.nevpt2 import PCNEVPT2
    mol, mf, cderi = hbr
    eris = socutils_x2c.from_scf(mf, ncore=mol.nelectron - 2, nact=4, cderi=cderi)
    pt = PCNEVPT2(eris, 2, nstate=1)
    pt.verbose = 0
    pt.max_memory = 1
    with pytest.raises(MemoryError, match='needs max_memory >='):
        pt.kernel()
