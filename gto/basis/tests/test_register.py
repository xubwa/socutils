"""Registration of j-adapted basis modules with PySCF and the warning for scalar calculations."""
import warnings
import pytest
from pyscf import gto, scf
from socutils.gto import basis as sbasis
from socutils.tools.basisopt import genbas, contract


@pytest.fixture(scope='module')
def name(tmp_path_factory):
    prims, pattern = contract.pattern_from_pyscf('O', 'cc-pvdz')
    entry = contract.contract('O', prims, pattern, kind='sog', name='PVDZ-TEST')
    d = tmp_path_factory.mktemp('basis')
    genbas.write_pyscf(str(d / 'o_test-sog.py'), {'O': genbas.to_pyscf(entry, kappa=True)})
    assert sbasis.register(str(d)) == ['o_test-sog']
    return 'o_test-sog', genbas.to_pyscf(entry, kappa=True)


def test_by_name(name):
    mol = gto.M(atom='O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24', basis={'O': name[0], 'H': 'cc-pvdz'}, verbose=0)
    ref = gto.M(atom='O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24', basis={'O': name[1], 'H': 'cc-pvdz'}, verbose=0)
    assert sbasis.has_kappa(mol)
    assert mol.nao_2c() == ref.nao_2c() and mol.nao_2c() < 2 * mol.nao_nr()


def test_scalar_warns_spinor_does_not(name):
    mol = gto.M(atom='O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24', basis={'O': name[0], 'H': 'cc-pvdz'}, verbose=0)
    for make in (scf.RHF, scf.GHF, lambda m: scf.RHF(m).sfx2c1e()):
        with pytest.warns(sbasis.ScalarWithSpinorBasisWarning):
            make(mol).build()
    with warnings.catch_warnings():
        warnings.simplefilter('error', sbasis.ScalarWithSpinorBasisWarning)
        scf.X2C(mol).build()                                  # spinor: no warning
        scf.DHF(mol).build()
        scf.RHF(gto.M(atom='O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24', basis='cc-pvdz', verbose=0)).build()
