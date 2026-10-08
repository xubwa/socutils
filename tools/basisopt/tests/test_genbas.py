"""GENBAS <-> PySCF internal format (plain and j-adapted)."""
import os
import numpy as np
import pytest
from pyscf import gto, scf
from socutils.tools.basisopt import genbas, contract, engine
from socutils.tools import basis_parser


@pytest.fixture(scope='module')
def so_entry():
    prims, pattern = contract.pattern_from_pyscf('O', 'cc-pvdz')
    return contract.contract('O', prims, pattern, kind='sog', name='PVDZ-TEST')


def same(a, b, tol=1e-7):       # GENBAS files carry 8 decimals
    assert sorted(a.shells) == sorted(b.shells)
    for l in a.shells:
        assert np.allclose(a.shells[l][0], b.shells[l][0], rtol=1e-8, atol=1e-8)
        assert np.allclose(a.shells[l][1], b.shells[l][1], atol=tol)


def test_so_roundtrip(so_entry, tmp_path):
    assert so_entry.so_info() == {1: 1}
    kb = genbas.to_pyscf(so_entry, kappa=True)
    assert sorted({(sh[0], sh[1]) for sh in kb}) == [(0, 0), (1, -2), (1, 0), (1, 1), (2, 0)]
    back = genbas.from_pyscf('O:BACK', kb)
    same(so_entry, back, tol=1e-12)
    assert back.comment.split() == so_entry.comment.split()
    # through files: GENBAS -> PySCF file -> GENBAS
    g, p = str(tmp_path / 'GENBAS'), str(tmp_path / 'o.py')
    genbas.append(g, so_entry)
    genbas.write_pyscf(p, {'O': genbas.to_pyscf(genbas.read(g, 'O:PVDZ-TEST'), kappa=True)})
    again = genbas.from_pyscf('O:PVDZ-TEST', genbas.read_pyscf(p)['O'])
    same(genbas.read(g, 'O:PVDZ-TEST'), again, tol=1e-12)
    # the same kappa basis as the existing socutils parser gives
    ref = basis_parser.parse_genbas('O:PVDZ-TEST', g, so_basis=True)
    mine = genbas.to_pyscf(genbas.read(g, 'O:PVDZ-TEST'), kappa=True)
    assert len(ref) == len(mine)
    for a, b in zip(ref, mine):
        assert a[:2] == b[:2] and np.allclose(np.array(a[2:]), np.array(b[2:]))


def test_so_spinor_count(so_entry):
    mol = gto.M(atom='O', basis={'O': genbas.to_pyscf(so_entry, kappa=True)}, spin=2, verbose=0)
    # 3s -> 6 spinors; p: 1 x (2 + 4) contracted + 1 primitive x 6; d: 1 x 10
    assert mol.nao_2c() == 6 + 6 + 6 + 10


def test_segmented_roundtrip():
    """A segmented PySCF set (shells with different primitives) keeps its SCF energy."""
    b0 = gto.basis.load('cc-pvtz', 'O')
    entry = genbas.from_pyscf('O:PVTZ', b0)
    assert entry.so_info() == {}
    e = [scf.RHF(gto.M(atom='O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24', basis={'O': b, 'H': 'cc-pvdz'},
                       verbose=0)).kernel() for b in (b0, genbas.to_pyscf(entry))]
    assert abs(e[0] - e[1]) < 1e-9
    back = genbas.from_pyscf('O:PVTZ', genbas.to_pyscf(entry))
    same(entry, back, tol=1e-14)


def test_unequal_kappa_rejected():
    with pytest.raises(ValueError):
        genbas.from_pyscf('X:BAD', [[1, 1, [1.0, 1.0], [0.3, 0.5]], [1, 0, [0.3, 1.0]]])
