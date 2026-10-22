'''In-core s4 AO -> MO transformation vs pyscf.ao2mo.nrr_outcore.'''
import numpy
import pytest
from pyscf import gto
from pyscf.ao2mo import nrr_outcore

from socutils.zmr.interface.ao2mo import IncoreAO2MO


@pytest.fixture(scope='module')
def mol():
    return gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='sto-3g', verbose=0)


def _rand(shape, seed):
    rng = numpy.random.default_rng(seed)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _spinor(mol):
    ca, cb = mol.sph2spinor_coeff()

    def outcore(*cs):
        return nrr_outcore.general_iofree(mol, cs, intor='int2e_sph', motype='j-spinor',
                                          verbose=0)
    return mol.nao_2c(), (lambda c: (ca @ c, cb @ c)), outcore


def _ghf(mol):
    nao = mol.nao_nr()

    def outcore(*cs):
        return nrr_outcore.general_iofree(mol, cs, intor='int2e_sph', motype='ghf', verbose=0)
    return 2 * nao, (lambda c: (c[:nao], c[nao:])), outcore


@pytest.mark.parametrize('setup', [_spinor, _ghf])
def test_general_block(mol, setup):
    n2c, components, outcore = setup(mol)
    cs = [_rand((n2c, n), 10 + k) for k, n in enumerate((2, 3, 4, 5))]
    eri = IncoreAO2MO(mol, components, outcore)(*cs)
    ref = numpy.asarray(outcore(*cs)).reshape(2, 3, 4, 5)
    assert eri.shape == ref.shape
    assert abs(eri - ref).max() < 1e-12


@pytest.mark.parametrize('setup', [_spinor, _ghf])
def test_same_bra_and_ket(mol, setup):
    '''(pq|pq) blocks reuse the folded density of the bra.'''
    n2c, components, outcore = setup(mol)
    c1, c2 = _rand((n2c, 3), 20), _rand((n2c, 4), 21)
    eri = IncoreAO2MO(mol, components, outcore)(c1, c2, c1.copy(), c2)
    ref = numpy.asarray(outcore(c1, c2, c1, c2)).reshape(3, 4, 3, 4)
    assert abs(eri - ref).max() < 1e-12
    assert abs(eri - eri.transpose(2, 3, 0, 1)).max() < 1e-12


def test_empty_block(mol):
    n2c, components, outcore = _spinor(mol)
    cs = [_rand((n2c, n), 30 + k) for k, n in enumerate((2, 0, 3, 1))]
    eri = IncoreAO2MO(mol, components, outcore)(*cs)
    assert eri.shape == (2, 0, 3, 1)


def test_fallback_when_integrals_do_not_fit(mol):
    n2c, components, outcore = _spinor(mol)
    calls = []

    def fallback(*cs):
        calls.append(len(cs))
        return numpy.asarray(outcore(*cs)).reshape([c.shape[1] for c in cs])

    cs = [_rand((n2c, 2), 40 + k) for k in range(4)]
    small = IncoreAO2MO(mol, components, fallback, max_memory=1e-6)
    assert not small.incore
    eri = small(*cs)
    assert calls == [4] and small._eri is None
    assert abs(eri - IncoreAO2MO(mol, components, fallback)(*cs)).max() < 1e-12
