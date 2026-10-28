'''Integrals read from an HDF5 file (DiskERIs) against the stored four-index
integrals: blocks, rotations, the npz conversion and NEVPT4(SD) with the
(vv|vv) term from rows of the file.'''
import numpy
import pytest

from socutils.zmr import fock
from socutils.zmr.integrals import DiskERIs, ERIs, add_vvvv_pairs, canonical_labels, write_h5
from socutils.zmr.interface import npz
from socutils.zmr.mrci import sigma as sg
from socutils.zmr.pt4 import NEVPT4SD
from socutils.zmr.tests._helpers import model_eris

LABELS = [p + q + r + s for p in 'cav' for q in 'cav' for r in 'cav' for s in 'cav']


def _write(eris, path, rows=None):
    def block(lab, i0, i1):
        return eris.eri_block(lab)[i0:i1]

    write_h5(path, eris.ecore, eris.h1, eris.spaces, block, rows=rows)
    return DiskERIs(str(path))


def _unitary(n, rng):
    q, _ = numpy.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    return q


def test_blocks_and_mean_field(tmp_path):
    eris = model_eris(2, 3, 4, seed=3)
    # rows=7: several slices per block
    disk = _write(eris, tmp_path / 'm.h5', rows=7)
    assert len(canonical_labels()) < len(LABELS)
    for lab in LABELS + ['pppp', 'ppcc', 'paap']:
        assert abs(disk.eri_block(lab) - eris.eri_block(lab)).max() < 1e-14, lab
    assert abs(disk.h1 - eris.h1).max() < 1e-14
    g = numpy.diag([0.9, 0.6, 0.5]).astype(complex)
    assert abs(disk.core_energy() - eris.core_energy()) < 1e-12
    assert abs(fock.generalized_fock(disk, g) - fock.generalized_fock(eris, g)).max() < 1e-12
    assert abs(disk.vvvv_rows(1, 3) - eris.eri_block('vvvv')[1:3]).max() < 1e-14


def test_rotation(tmp_path):
    eris = model_eris(2, 3, 4, seed=5)
    disk = _write(eris, tmp_path / 'r.h5')
    rng = numpy.random.default_rng(1)
    u = numpy.zeros((9, 9), dtype=complex)
    for x in 'cav':
        s = eris.spaces.slice(x)
        u[s, s] = _unitary(s.stop - s.start, rng)
    u2 = numpy.eye(9, dtype=complex)
    s = eris.spaces.slice('c')
    u2[s, s] = _unitary(2, rng)
    ref = eris.rotate(u).rotate(u2)
    rot = disk.rotate(u).rotate(u2)
    assert abs(rot.h1 - ref.h1).max() < 1e-12
    for lab in LABELS:
        assert abs(rot.eri_block(lab) - ref.eri_block(lab)).max() < 1e-12, lab
    with pytest.raises(NotImplementedError):
        rot.vvvv_rows(0, 1)


def test_npz_conversion(tmp_path):
    eris = model_eris(2, 3, 4, seed=7)
    sp = eris.spaces
    src = tmp_path / 'm.npz'
    numpy.savez(src, h1=eris.h1, eri=eris.eri_block('pppp'), ecore=eris.ecore, ncore=sp.ncore,
                nact=sp.nact, nvirt=sp.nvirt, nelec=sp.ncore + 2)
    mm = npz.npz_memmap(str(src), 'eri')
    assert abs(mm - eris.eri_block('pppp')).max() == 0
    npz.convert(str(src), str(tmp_path / 'm.h5'), rows=11)
    disk = DiskERIs(str(tmp_path / 'm.h5'))
    for lab in LABELS:
        assert abs(disk.eri_block(lab) - eris.eri_block(lab)).max() == 0, lab
    assert int(disk.meta['nelec'][()]) == sp.ncore + 2


def test_vvvv_pairs(tmp_path):
    eris = model_eris(2, 3, 5, seed=4)
    disk = _write(eris, tmp_path / 'p.h5')
    disk._f.close()
    add_vvvv_pairs(str(tmp_path / 'p.h5'), rows=7)
    disk = DiskERIs(str(tmp_path / 'p.h5'))
    w = eris.eri_block('vvvv')
    i, j = numpy.triu_indices(5, k=1)
    ref = w[i[:, None], i[None, :], j[:, None], j[None, :]] - w[i[:, None], j[None, :], j[:, None], i[None, :]]
    assert abs(disk.vvvv_pairs[()] - ref).max() < 1e-14


@pytest.mark.parametrize('pairs', [False, True])
@pytest.mark.parametrize('dense', [True, False])
def test_nevpt4_vvvv_rows(tmp_path, dense, pairs):
    '''NEVPT4(SD) with the (vv|vv) term from rows of the file, the matrix over
    pairs in memory (dense) or formed again in every product, against the
    stored integrals (quasi-canonical orbitals in the file).'''
    eris = model_eris(2, 3, 4, seed=11)
    ref = NEVPT4SD(eris, 2, nstate=2, contraction='union')
    ref.verbose = 0
    ref.kernel()
    disk = _write(ref.eris, tmp_path / 'q.h5')
    if pairs:
        disk._f.close()
        add_vvvv_pairs(str(tmp_path / 'q.h5'))
        disk = DiskERIs(str(tmp_path / 'q.h5'))
    calls = []
    rows = disk.vvvv_rows
    disk.vvvv_rows = lambda a0, a1: calls.append(a0) or rows(a0, a1)
    saved = sg.Sigma.direct_vvvv, sg.Sigma.vvvv_dense_memory
    try:
        sg.Sigma.direct_vvvv = True
        sg.Sigma.vvvv_dense_memory = 1e12 if dense else 0
        pt = NEVPT4SD(disk, 2, nstate=2, contraction='union')
        pt.verbose = 0
        pt.kernel()
    finally:
        sg.Sigma.direct_vvvv, sg.Sigma.vvvv_dense_memory = saved
    assert bool(calls) != pairs and 'vvvv' not in disk.computed_blocks
    for a, b in ((pt.e2, ref.e2), (pt.e3, ref.e3), (pt.e4, ref.e4)):
        assert abs(a - b).max() < 1e-10
