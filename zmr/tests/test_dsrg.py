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


def test_vvvv_on_disk(water):
    '''MR-LDSRG(2) with the (vv|vv) ladder in a temporary HDF5 file, read in row blocks,
    and rebuilt for every use, against memory.'''
    _, chol, nelec = water
    ref = run(LDSRG2, chol, nelec, 2)
    c = ActiveSpace.from_eris(chol).solve(nelec, ref.nstate)[1]
    for disk in (True, False):
        pt = LDSRG2(chol, nelec, civecs=c, s=0.5)
        pt.verbose = False
        pt.conv_tol, pt.conv_tol_amp = 1e-11, 1e-9
        if disk:
            pt.build()
            pt.min_memory = pt.max_memory              # nothing beyond the minimum
            pt._w = pt._store_vvvv()
            assert not isinstance(pt._w, numpy.ndarray)
        else:
            pt.vvvv = 'direct'
            pt.build()
        pt.solve_amplitudes()
        # converged to 1e-9 in the amplitudes
        assert abs(pt.e_corr - ref.e_corr) < 1e-8


def test_memory_check(water):
    '''Too small max_memory stops before the calculation with the minimum needed.'''
    _, chol, nelec = water
    for cls in METHODS.values():
        pt = cls(chol, nelec, nstate=1)
        pt.verbose = False
        pt.max_memory = 1
        with pytest.raises(MemoryError, match='needs max_memory >='):
            pt.kernel()


def test_vvvv_pair_rows():
    '''Native rows of the pair-packed (vv|vv) against numpy.'''
    from socutils.zmr.lib import kernels
    rng = numpy.random.default_rng(2)
    n, a0, nb = 7, 2, 3
    d = rng.normal(size=(nb, n, n - a0, n)) + 1j * rng.normal(size=(nb, n, n - a0, n))
    iu = numpy.triu_indices(n, 1)
    ref = numpy.concatenate([
        numpy.conj(d[a - a0].transpose(1, 0, 2)[a + 1 - a0:][:, iu[0], iu[1]]
                   - d[a - a0].transpose(1, 0, 2)[a + 1 - a0:][:, iu[1], iu[0]])
        for a in range(a0, a0 + nb)])
    assert abs(kernels.vvvv_pair_rows(d, a0) - ref).max() < 1e-14


SYMMETRY_MOLECULES = {
    'C2': 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',            # H2O, C2v
    'C2h': ('N 0 0.6123 0.0; N 0 -0.6123 0.0; H 0.9458 0.9214 0.0; '
            'H -0.9458 -0.9214 0.0'),                                         # trans-N2H2
    'Coov': 'H 0 0 0; Br 0 0 1.4144',                                         # HBr
}


@pytest.fixture(scope='module', params=sorted(SYMMETRY_MOLECULES))
def double_group(request):
    from pyscf.x2c import x2c
    from socutils.scf import double_group_symm as dgs
    from socutils.symm import spinor_basis as sb
    from socutils.zmr.interface import socutils_x2c
    mol = sb.to_symm_frame(gto.M(atom=SYMMETRY_MOLECULES[request.param], basis='6-31g',
                                 symmetry=True, verbose=0))
    mf = dgs.SymAdaptedRHF(mol)
    mf.with_x2c = x2c.SpinorX2CHelper(mol)
    mf.conv_tol = 1e-11
    mf.verbose = 0
    mf.kernel()
    dense = socutils_x2c.from_scf(mf, ncore=mol.nelectron - 2, nact=4, lazy=False)
    chol = socutils_x2c.from_scf(mf, ncore=mol.nelectron - 2, nact=4, cderi=factorize(mol))
    assert dense.doublegroupname == chol.doublegroupname == request.param
    return dense, chol


@pytest.mark.parametrize('integrals', ['dense', 'three-index'])
@pytest.mark.parametrize('name', sorted(METHODS))
def test_double_group_sectors(double_group, name, integrals):
    '''Contractions blocked by the double-group sectors of a socutils double-group SCF
    (C2*, C2h*, Coov*) against the same calculation without them; with three-index
    integrals also the blocked (vv|vv) rows and ladder.'''
    eris = double_group[0 if integrals == 'dense' else 1]
    out = []
    for use in (True, False):
        cls = type('X', (METHODS[name],), {'use_symmetry': use})
        out.append(run(cls, eris, 2, 2))
    a, b = out
    assert a.dims.sym is not None and b.dims.sym is None
    assert abs(a.e_corr - b.e_corr) < 1e-9
    assert abs(numpy.asarray(a.e_tot) - b.e_tot).max() < 1e-8


def test_vvvv_pair_rows_sym():
    '''Native pair rows with pair irreps against numpy (C2h*-like irreps 0..7).'''
    from socutils.zmr.lib import kernels
    rng = numpy.random.default_rng(3)
    n, a0, nb = 7, 2, 3
    d = rng.normal(size=(nb, n, n - a0, n)) + 1j * rng.normal(size=(nb, n, n - a0, n))
    pirr = rng.integers(0, 3, size=n * (n - 1) // 2)
    full = kernels.vvvv_pair_rows(d, a0)
    first = a0 * n - a0 * (a0 + 1) // 2
    rows = pirr[first:first + len(full)]
    ref = numpy.where(rows[:, None] == pirr[None, :], full, 0)
    assert abs(kernels.vvvv_pair_rows_sym(d, a0, pirr) - ref).max() < 1e-14


def test_vvvv_pair_rows_packed():
    '''Pair rows from D in packed blocks against the dense D (pair irreps of a group with
    additive irreps 0..2 for the orbitals).'''
    from socutils.zmr.lib import kernels
    rng = numpy.random.default_rng(4)
    n, a0, nb = 7, 2, 3
    d = rng.normal(size=(nb, n, n - a0, n)) + 1j * rng.normal(size=(nb, n, n - a0, n))
    pirr = rng.integers(0, 3, size=n * (n - 1) // 2)
    # every element in its own one-element block: rowoff + coloff is the dense index
    rowoff = numpy.arange(nb * n) * (n - a0) * n
    coloff = numpy.arange((n - a0) * n)
    dense = d.transpose(0, 1, 2, 3).reshape(-1)
    ref = kernels.vvvv_pair_rows_sym(d, a0, pirr)
    out = kernels.vvvv_pair_rows_packed(dense, rowoff, coloff, nb, n, a0, pirr)
    assert abs(out - ref).max() < 1e-14
