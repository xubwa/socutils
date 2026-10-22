'''One-body symmetries of the active space: detection, adapted orbitals,
block diagonal sector Hamiltonians, and invariance of the NEVPT2 results.'''
import numpy
import pytest
from scipy.stats import unitary_group

from socutils.zmr.active import symmetry
from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.pc.nevpt2 import PCNEVPT2
from socutils.zmr.tests._helpers import (complete_levels, model_eris, random_hamiltonian,
                                spinfree_model_eris)
from socutils.zmr.uc.nevpt2 import UCNEVPT2

NELEC = 3


@pytest.fixture(scope='module')
def spin_model():
    '''Hidden spin symmetry: 3 spatial active orbitals, 6 spin orbitals.'''
    eris = spinfree_model_eris(1, 3, 2, seed=4)
    adapted, sym = symmetry.adapt(eris)
    return eris, adapted, sym


def test_random_hamiltonian_has_no_symmetry():
    h, eri = random_hamiltonian(6, seed=1)
    assert symmetry.find(h, eri) is None
    eris = model_eris(2, 4, 2, seed=2)
    same, sym = symmetry.adapt(eris)
    assert sym is None and same is eris and same.active_charges is None
    assert ActiveSpace.from_eris(eris).blocks(2)[1] == 1


def test_spin_symmetry_is_found(spin_model):
    eris, adapted, sym = spin_model
    # particle number and the three spin components
    assert sym.ngen == 4
    assert sym.error < 1e-10
    assert abs(sym.u.conj().T @ sym.u - numpy.eye(6)).max() < 1e-12
    # the generic generator is a spin component along some axis: +-m, three times each
    values, counts = numpy.unique(sym.charges, return_counts=True)
    assert len(values) == 2 and (counts == 3).all()
    assert abs(values.sum()) < 1e-10
    assert (adapted.active_charges == sym.charges).all()
    # without the random mixing nothing is hidden, the same algebra is found
    plain = spinfree_model_eris(1, 3, 2, seed=4, hide=False)
    h, eri, _ = plain.active_hamiltonian()
    assert symmetry.find(h, eri).ngen == 4


def test_adapted_hamiltonian_conserves_the_charge(spin_model):
    _, adapted, sym = spin_model
    h, eri, _ = adapted.active_hamiltonian()
    assert symmetry.violation(h, eri, sym.charges) < 1e-10
    assert abs(h - numpy.diag(h.diagonal())).max() < 1e-10       # heff stays diagonal
    hp, ep = symmetry.project(h, eri, sym.charges)
    assert symmetry.violation(hp, ep, sym.charges) == 0
    assert abs(hp - h).max() < 1e-10 and abs(ep - eri).max() < 1e-10


@pytest.mark.parametrize('nelec', [1, 2, 3, 4])
def test_sectors_are_block_diagonal(spin_model, nelec):
    eris, adapted, sym = spin_model
    dense = ActiveSpace.from_eris(eris)
    act = ActiveSpace.from_eris(adapted)
    assert act.symmetry_error < 1e-10
    assert dense.blocks(nelec)[1] == 1
    labels, nblocks = act.blocks(nelec)
    # one block per value of the total charge (spin projection)
    assert nblocks == min(nelec, 6 - nelec) + 1
    sec = act.sector(nelec)
    assert abs(sec.lam - dense.sector(nelec).lam).max() < 1e-10
    h = act.hmat(nelec)
    assert abs(h @ sec.vecs - sec.vecs * sec.lam).max() < 1e-10
    for k in range(sec.dim):
        assert len(set(labels[abs(sec.vecs[:, k]) > 0].tolist())) == 1
    # the lowest roots come from the block-wise spectrum
    e, c = ActiveSpace.from_eris(adapted).solve(nelec, 2, full=False)
    assert abs(e - sec.lam[:2] - act.e_core).max() < 1e-12


def test_charges_can_be_given_directly():
    h, eri = random_hamiltonian(4, seed=3, scale=0.1)
    charges = numpy.array([0.5, 0.5, -0.5, -0.5])
    act = ActiveSpace(h, eri, charges=charges)
    assert act.symmetry_error == pytest.approx(symmetry.violation(h, eri, charges))
    assert act.symmetry_error > 1e-3
    assert symmetry.violation(act.heff, act.aaaa, charges) == 0
    assert act.blocks(2)[1] == 3


def test_approximate_symmetry_is_used_only_within_tol():
    eris = spinfree_model_eris(1, 3, 2, seed=4, hide=False)
    h, eri, _ = eris.active_hamiltonian()
    kick, _ = random_hamiltonian(6, seed=9, scale=1e-5)
    assert symmetry.find(h + kick, eri) is None
    loose = symmetry.find(h + kick, eri, tol=1e-3)
    assert loose is not None and 1e-7 < loose.error < 1e-3


def _nstate(eris, nmin):
    lam = ActiveSpace.from_eris(eris).sector(NELEC).lam
    return complete_levels(lam, nmin)


@pytest.mark.parametrize('method', ['uc', 'pc_union', 'pc_union_rdm'])
@pytest.mark.parametrize('e0', ['state', 'average'])
def test_results_do_not_depend_on_the_adaptation(spin_model, method, e0):
    eris, adapted, _ = spin_model
    nstate = _nstate(eris, 3)

    def run(x):
        if method == 'uc':
            obj = UCNEVPT2(x, NELEC, nstate=nstate, e0=e0)
        else:
            obj = PCNEVPT2(x, NELEC, nstate=nstate, e0=e0,
                           amplitudes='rdm' if method.endswith('rdm') else 'spectral')
        obj.kernel()
        return obj

    a, b = run(eris), run(adapted)
    assert abs(a.energies - b.energies).max() < 1e-9
    for name in a.h2_by_class:
        assert abs(numpy.trace(a.h2_by_class[name]) - numpy.trace(b.h2_by_class[name])) < 1e-9, name
    # the adapted run really went block by block
    assert b.active_space.blocks(NELEC)[1] > 1


def test_mixed_model_states_span_several_blocks(spin_model):
    '''Model states that mix the blocks (a rotation inside the degenerate
    levels) merge the blocks they connect; the invariant results stay.'''
    _, adapted, _ = spin_model
    nstate = _nstate(adapted, 3)
    ref = PCNEVPT2(adapted, NELEC, nstate=nstate, e0='average')
    ref.kernel()
    u = unitary_group.rvs(nstate, random_state=5)
    rot = PCNEVPT2(adapted, NELEC, civecs=ref.civecs @ u, e0='average')
    rot.kernel()
    assert abs(rot.energies - ref.energies).max() < 1e-9
    assert abs(rot.h2 - u.conj().T @ ref.h2 @ u).max() < 1e-9


def test_state_specific_contraction_with_blocks(spin_model):
    '''contraction='state' is not rotation invariant, so it is compared in
    one orbital basis: block by block vs dense (charges ignored).'''
    _, adapted, _ = spin_model
    nstate = _nstate(adapted, 3)
    blocked = PCNEVPT2(adapted, NELEC, nstate=nstate, contraction='state')
    blocked.kernel()
    plain = adapted.rotate(numpy.eye(adapted.nmo))
    plain.active_charges = None
    dense = PCNEVPT2(plain, NELEC, civecs=blocked.civecs, contraction='state')
    dense.kernel()
    assert dense.active_space.blocks(NELEC)[1] == 1
    assert abs(blocked.heff - dense.heff).max() < 1e-9
