'''Block-wise (on-demand) integrals vs stored four-index integrals.'''
import numpy
import pytest

from socutils.zmr import fock
from socutils.zmr.interface import pyscf_nr, socutils_x2c
from socutils.zmr.tests import systems
from socutils.zmr.tests.systems_x2c import o2_dianion_orbitals
from socutils.zmr.uc.nevpt2 import UCNEVPT2

BLOCKS = ['vcvc', 'vcac', 'vavc', 'acac', 'vava', 'vcaa', 'vaac', 'acaa', 'aaac', 'vaaa', 'aaaa']


def _pairs():
    mc = systems.h2o_casci(4, 4, basis='6-31g')
    for nfrozen in (0, 1):
        dense, _ = pyscf_nr.from_casci(mc, nfrozen=nfrozen, lazy=False)
        lazy, _ = pyscf_nr.from_casci(mc, nfrozen=nfrozen, lazy=True)
        yield 'nr%d' % nfrozen, dense, lazy, 4
    mf, mol = o2_dianion_orbitals('sto-3g', True)
    dense = socutils_x2c.from_scf(mf, 10, 4, nfrozen=4, mol=mol, lazy=False)
    lazy = socutils_x2c.from_scf(mf, 10, 4, nfrozen=4, mol=mol, lazy=True)
    yield 'x2c', dense, lazy, 2


@pytest.fixture(scope='module')
def pairs():
    return list(_pairs())


def test_lazy_matches_dense(pairs):
    rng = numpy.random.default_rng(0)
    for name, dense, lazy, _ in pairs:
        na = dense.spaces.nact
        g = rng.normal(size=(na, na)) + 1j * rng.normal(size=(na, na))
        g = 0.1 * (g + g.conj().T) + 0.4 * numpy.eye(na)
        assert abs(dense.ecore - lazy.ecore) < 1e-10, name
        assert abs(dense.h1 - lazy.h1).max() < 1e-10, name
        assert abs(dense.core_energy() - lazy.core_energy()) < 1e-10, name
        assert abs(fock.generalized_fock(dense, g) - fock.generalized_fock(lazy, g)).max() < 1e-10, name
        for b in BLOCKS:
            assert abs(dense.eri_block(b) - lazy.eri_block(b)).max() < 1e-12, (name, b)


def test_uc_nevpt2_never_builds_large_blocks(pairs):
    for name, dense, lazy, nelec in pairs:
        ref = UCNEVPT2(dense, nelec, nstate=2, e0='average')
        ref.kernel()
        # same model vectors: a truncated degenerate multiplet is otherwise
        # resolved differently by the two diagonalizations
        uc = UCNEVPT2(lazy, nelec, civecs=ref.civecs, e0='average')
        uc.kernel()
        assert abs(uc.heff - ref.heff).max() < 1e-10, name
        blocks = uc.quasi_canonical.eris.computed_blocks
        assert 'pppp' not in blocks and 'vvvv' not in blocks, name
        assert not any(b.count('v') > 2 for b in blocks), (name, blocks)


def test_rotation_must_respect_spaces(pairs):
    _, _, lazy, _ = pairs[0]
    n = lazy.nmo
    u = numpy.eye(n)
    c, v = lazy.spaces.slice('c'), lazy.spaces.slice('v')
    u[c.start, v.start] = u[v.start, c.start] = 1e-3
    with pytest.raises(ValueError):
        lazy.rotate(u)


def test_attribute_block_access(pairs):
    for name, dense, lazy, _ in pairs:
        for eris in (dense, lazy):
            assert eris.cvcv is eris.eri_block('cvcv') or numpy.array_equal(eris.cvcv, eris.eri_block('cvcv'))
            assert eris.ppaa.shape == (eris.nmo, eris.nmo, eris.spaces.nact, eris.spaces.nact)
        with pytest.raises(AttributeError):
            lazy.cvcx


def test_prefetched_superblock(pairs):
    for name, dense, lazy, _ in pairs:
        fresh = lazy.rotate(numpy.eye(lazy.nmo))
        fresh.prefetch(('av', 'ca', 'av', 'ca'))
        for b in BLOCKS:
            assert abs(fresh.eri_block(b) - dense.eri_block(b)).max() < 1e-12, (name, b)
        # everything was served by the single superblock transform
        assert fresh.computed_blocks == ['(av,ca,av,ca)'], fresh.computed_blocks
