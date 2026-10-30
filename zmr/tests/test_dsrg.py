'''DSRG-MRPT2, DSRG-MRPT3 and MR-LDSRG(2) (zmr.dsrg): three-index against four-index
integrals (the core loop of the ccvv energy, the pair-packed four-virtual ladder),
invariance to a unitary rotation of the active spinors, and reference values.

The OH radical (X2CAMF, cc-pVDZ, SA-4, s = 0.5) agrees with forte2 (RelDSRG_MRPT2,
RelDSRG_MRPT3, RelMRLDSRG2) to 1e-10 Eh in the energy of the ensemble and in the relaxed
levels; that comparison needs forte2 and is not repeated here.'''
import numpy
import pytest
from pyscf import gto, mcscf, scf

from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.dsrg import DSRGMRPT2, DSRGMRPT3, LDSRG2
from socutils.zmr.interface import pyscf_nr
from socutils.zmr.tests._helpers import complete_levels, model_eris
from socutils.zmr.tests.test_cholesky import factorize

METHODS = {'mrpt2': DSRGMRPT2, 'mrpt3': DSRGMRPT3, 'ldsrg2': LDSRG2}


def run(cls, eris, nelec, nstate, s=0.5):
    # complete multiplets only: a cut degenerate level makes the ensemble arbitrary
    act = ActiveSpace.from_eris(eris)
    e, c = act.solve(nelec, nstate + 6)
    c = c[:, :complete_levels(e, nstate)]
    pt = cls(eris, nelec, civecs=c, s=s)
    pt.verbose = False
    if cls is LDSRG2:
        pt.conv_tol, pt.conv_tol_amp = 1e-11, 1e-9
    pt.kernel()
    return pt


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-13, conv_tol_grad=1e-10)
    mc = mcscf.CASCI(mf, 4, 4)
    mc.verbose = 0
    mc.kernel()
    dense, nelec = pyscf_nr.from_casci(mc, nfrozen=1)
    chol, _ = pyscf_nr.from_casci(mc, nfrozen=1, cderi=factorize(mol))
    return dense, chol, nelec


# E_corr of the ensemble and the relaxed levels, H2O 6-31G CAS(4,4) on RHF orbitals,
# singlet ground state and the triplet above it (SA-4), s = 0.5
REFERENCE = {
    'mrpt2': (-0.1730884280, (-76.1165245027,) + (-75.8454216239,) * 3),
    'mrpt3': (-0.1484352871, (-76.1189940271,) + (-75.8112684562,) * 3),
    'ldsrg2': (-0.1619623390, (-76.1244384458,) + (-75.8275675574,) * 3),
}


@pytest.mark.parametrize('name', sorted(METHODS))
def test_three_index_integrals(water, name):
    dense, chol, nelec = water
    a = run(METHODS[name], dense, nelec, 2)
    b = run(METHODS[name], chol, nelec, 2)
    tol = 1e-7 if name == 'ldsrg2' else 1e-9
    assert abs(a.e_corr - b.e_corr) < tol
    assert abs(numpy.asarray(a.e_tot) - b.e_tot).max() < tol
    ref = REFERENCE[name]
    if ref is not None:
        assert abs(b.e_corr - ref[0]) < 1e-8
        assert abs(numpy.asarray(b.e_tot) - ref[1]).max() < 1e-8


@pytest.mark.parametrize('name', ['mrpt2', 'mrpt3'])
def test_active_rotation(name):
    '''Complex spinors: the levels do not depend on the active orbitals.'''
    eris = model_eris(2, 4, 3, seed=11)
    sp = eris.spaces
    rng = numpy.random.default_rng(5)
    x = rng.normal(size=(sp.nact, sp.nact)) + 1j * rng.normal(size=(sp.nact, sp.nact))
    u = numpy.eye(sp.nmo, dtype=complex)
    u[sp.slice('a'), sp.slice('a')] = numpy.linalg.qr(x)[0]
    a = run(METHODS[name], eris, 2, 3)
    b = run(METHODS[name], eris.rotate(u), 2, 3)
    assert abs(a.e_dsrg - b.e_dsrg) < 1e-10
    assert abs(numpy.asarray(a.e_tot) - b.e_tot).max() < 1e-10
