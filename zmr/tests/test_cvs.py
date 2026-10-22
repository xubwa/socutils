'''Core-valence separation for model states with holes in a deep active shell.'''
import numpy
import pytest
from pyscf import gto, mcscf, scf

from socutils.zmr import output
from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.interface import pyscf_nr
from socutils.zmr.pc.nevpt2 import PCNEVPT2
from socutils.zmr.uc.nevpt2 import UCNEVPT2

DN_POSITIVE = ('ccaa', 'ccav', 'caaa')


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASCI(mf, 2, 2)
    return mc, mf.mo_coeff


def _hamiltonian(water, active):
    '''Spin-orbital Hamiltonian with the spatial orbitals `active` as the
    active space with one hole pair; the other occupied ones are the core.'''
    mc, mo = water
    occupied = [i for i in range(5) if i not in active]
    order = occupied + list(active) + list(range(5, mo.shape[1]))
    eris = pyscf_nr.mo_eris(mc.mol, mc.get_hcore(), mo[:, order], 0, len(occupied), len(active))
    nelec = 2 * len(active) - 2
    # all states of the lowest level and above: the ground level of two
    # holes is degenerate, a single state of it is not a defined model space
    act = ActiveSpace.from_eris(eris)
    civecs = act.solve(nelec, act.dim(nelec))[1]
    return eris, nelec, civecs


@pytest.mark.parametrize('cls', [PCNEVPT2, UCNEVPT2])
def test_cvs_does_nothing_for_valence_active_space(water, cls):
    eris, nelec, civecs = _hamiltonian(water, [3, 4])
    ref = cls(eris, nelec, civecs=civecs)
    ref.kernel()
    pt = cls(eris, nelec, civecs=civecs)
    pt.cvs = True
    pt.kernel()
    assert len(pt.cvs_core) == eris.spaces.ncore
    assert abs(pt.heff - ref.heff).max() < 1e-12


def test_cvs_with_no_inner_core_drops_the_filling_classes(water):
    '''Holes in the deepest orbital: every core orbital lies above it.'''
    eris, nelec, civecs = _hamiltonian(water, [0])
    pt = PCNEVPT2(eris, nelec, civecs=civecs)
    pt.cvs = True
    pt.kernel()
    assert len(pt.cvs_core) == 0
    skip = PCNEVPT2(eris, nelec, civecs=civecs)
    skip.skip_classes = DN_POSITIVE
    skip.kernel()
    assert abs(pt.heff - skip.heff).max() < 1e-12
    for name in DN_POSITIVE:
        assert abs(pt.h2_by_class.get(name, numpy.zeros(1))).max() == 0
    assert pt.h2[0, 0].real < 0


def test_cvs_keeps_the_inner_core(water):
    '''Holes in the second orbital (O 2s): 1s stays, the outer ones go.'''
    eris, nelec, civecs = _hamiltonian(water, [1])
    pt = PCNEVPT2(eris, nelec, civecs=civecs)
    pt.cvs = True
    pt.kernel()
    assert len(pt.cvs_core) == 2
    full = PCNEVPT2(eris, nelec, civecs=civecs)
    full.kernel()
    skip = PCNEVPT2(eris, nelec, civecs=civecs)
    skip.skip_classes = DN_POSITIVE
    skip.kernel()
    for name in pt.h2_by_class:
        d = abs(pt.h2_by_class[name] - full.h2_by_class[name]).max()
        assert (d > 1e-8) == (name in DN_POSITIVE)
    # filling the 2s holes from 1s costs energy: regular, negative contributions
    filled = sum(pt.h2_by_class[name][0, 0].real for name in DN_POSITIVE)
    assert -0.1 < filled < 0
    assert abs(pt.h2[0, 0] - skip.h2[0, 0] - filled) < 1e-12


def test_intruder_states_are_reported(water, capsys):
    '''Holes in O 2s without the core-valence separation: filling them from
    the outer shells has negative denominators.  The class is named in the
    output and a warning is raised; with the separation nothing is reported.'''
    import warnings
    eris, nelec, civecs = _hamiltonian(water, [1])
    pt = PCNEVPT2(eris, nelec, civecs=civecs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        pt.kernel()
    text = capsys.readouterr().out
    bad = {name for name, d in pt.min_denominators.items() if d < output.INTRUDER_THRESHOLD}
    assert bad and bad <= set(DN_POSITIVE)
    assert any('intruder' in str(w.message) for w in caught)
    assert text.count('WARNING: class') >= 2 * len(bad)
    assert 'min. denominator' in text
    clean = PCNEVPT2(eris, nelec, civecs=civecs)
    clean.cvs = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        clean.kernel()
    assert not caught
    assert min(clean.min_denominators.values()) > output.INTRUDER_THRESHOLD
    assert 'WARNING' not in capsys.readouterr().out


@pytest.mark.parametrize('cls', [PCNEVPT2, UCNEVPT2])
def test_decay_channels_complete_the_classes(water, cls, capsys):
    '''Holes in O 2s: without a window the CVS part and the decay channels
    of ccaa, ccav, caaa add up to the full classes; the CVS energies do not
    depend on the decay pass; both sets of energies are printed.'''
    eris, nelec, civecs = _hamiltonian(water, [1])
    full = cls(eris, nelec, civecs=civecs)
    full.verbose = 0
    full.kernel()
    pt = cls(eris, nelec, civecs=civecs)
    pt.cvs = True
    pt.cvs_window = 0.0
    pt.kernel()
    assert set(pt.h2_decay_by_class) == set(DN_POSITIVE)
    for name in DN_POSITIVE:
        total = pt.h2_by_class[name] + pt.h2_decay_by_class[name]
        assert abs(total - full.h2_by_class[name]).max() < 1e-10, name
    assert abs(pt.heff_decay - full.heff).max() < 1e-10
    assert pt.decay_window[1] == 0
    text = capsys.readouterr().out
    assert 'E(CVS+Q)' in text and 'Decay channels ccav' in text
    plain = cls(eris, nelec, civecs=civecs)
    plain.verbose = 0
    plain.cvs = True
    plain.cvs_decay = False
    plain.kernel()
    assert abs(plain.e_tot - pt.e_tot).max() < 1e-12
    assert not plain.h2_decay_by_class


def test_window_leaves_out_the_terms_near_zero(water):
    eris, nelec, civecs = _hamiltonian(water, [1])
    pts = []
    for tau in (0.0, 1.0):
        pt = PCNEVPT2(eris, nelec, civecs=civecs)
        pt.verbose = 0
        pt.cvs = True
        pt.cvs_window = tau
        pt.kernel()
        pts.append(pt)
    assert pts[1].decay_window[1] > 0
    assert abs(pts[1].h2_decay_by_class['ccav'] - pts[0].h2_decay_by_class['ccav']).max() > 1e-8
    # the CVS part and the discrete channels do not depend on the window
    assert abs(pts[1].heff - pts[0].heff).max() < 1e-12
    for name in ('ccaa', 'caaa'):
        assert abs(pts[1].h2_decay_by_class[name] - pts[0].h2_decay_by_class[name]).max() < 1e-12


def test_inner_core_given_by_hand(water):
    '''cvs_core names the inner core; the automatic choice for O 2s holes is
    O 1s (index 0), giving it by hand changes nothing, an empty set drops
    every hole-filling label.'''
    eris, nelec, civecs = _hamiltonian(water, [1])
    auto = PCNEVPT2(eris, nelec, civecs=civecs)
    auto.verbose = 0
    auto.cvs = True
    auto.kernel()
    hand = PCNEVPT2(eris, nelec, civecs=civecs)
    hand.verbose = 0
    hand.cvs = True
    hand.cvs_core = list(auto.cvs_core)
    hand.kernel()
    assert abs(hand.heff - auto.heff).max() < 1e-12
    none = PCNEVPT2(eris, nelec, civecs=civecs)
    none.verbose = 0
    none.cvs = True
    none.cvs_core = []
    none.kernel()
    for name in DN_POSITIVE:
        assert abs(none.h2_by_class.get(name, numpy.zeros(1))).max() == 0
