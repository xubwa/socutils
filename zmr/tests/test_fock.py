import numpy
import pytest

from socutils.zmr import fock
from socutils.zmr.interface import pyscf_nr
from socutils.zmr.tests import systems


def _spin_diagonal_gamma(nact_spatial, seed):
    '''Random Hermitian active 1-RDM without alpha-beta coherence and with
    different alpha and beta blocks.'''
    rng = numpy.random.default_rng(seed)
    g = numpy.zeros((2 * nact_spatial,) * 2)
    for s in range(2):
        a = rng.normal(size=(nact_spatial,) * 2)
        g[s::2, s::2] = (a + a.T) * 0.3 + numpy.eye(nact_spatial) * 0.5
    return g


@pytest.mark.parametrize('nfrozen', [0, 1])
def test_fock_matches_ao_jk(nfrozen):
    mc = systems.h2o_casci(4, 4)
    mf = mc._scf
    eris, _ = pyscf_nr.from_casci(mc, nfrozen=nfrozen)
    gamma = _spin_diagonal_gamma(mc.ncas, seed=3)
    f = fock.generalized_fock(eris, gamma)

    c = mc.mo_coeff
    cc = c[:, :mc.ncore]
    ca = c[:, mc.ncore:mc.ncore + mc.ncas]
    dm = [cc @ cc.T + ca @ gamma[s::2, s::2] @ ca.T for s in range(2)]
    vj, vk = mf.get_jk(mf.mol, numpy.array(dm))
    corr = c[:, nfrozen:]
    for s in range(2):
        f_ao = mf.get_hcore() + vj[0] + vj[1] - vk[s]
        ref = corr.T @ f_ao @ corr
        assert abs(f[s::2, s::2] - ref).max() < 1e-10
    # no alpha-beta coupling in a spin-free Hamiltonian with such a gamma
    assert abs(f[0::2, 1::2]).max() < 1e-12


def test_quasi_canonicalize():
    mc = systems.h2o_casci(4, 4, basis='6-31g')
    eris, _ = pyscf_nr.from_casci(mc, nfrozen=1)
    sp = eris.spaces
    gamma = _spin_diagonal_gamma(mc.ncas, seed=4)
    qc = fock.quasi_canonicalize(eris, gamma)
    f_new = fock.generalized_fock(qc.eris, gamma)
    assert abs(f_new - qc.fock).max() < 1e-10
    for label, eps in [('c', qc.eps_core), ('v', qc.eps_virt)]:
        s = sp.slice(label)
        assert abs(f_new[s, s] - numpy.diag(eps)).max() < 1e-10
    a = sp.slice('a')
    f_old = fock.generalized_fock(eris, gamma)
    assert abs(f_new[a, a] - f_old[a, a]).max() < 1e-12
    assert qc.eris.symmetry_error() < 1e-12


def test_shells_and_report_and_explicit_virtual_count():
    '''Shell table for choosing the frozen core and the kept virtuals by
    hand, and truncation to a given number of virtual orbitals.'''
    import io
    from socutils.zmr import fock
    from socutils.zmr.active.sectors import ActiveSpace
    from socutils.zmr.tests._helpers import model_eris
    assert fock.shells([-5.0, -4.999, -1.0, -0.999, -0.998, 0.5]) == [(0, 2), (2, 5), (5, 6)]
    assert fock.shells([]) == []
    eris = model_eris(3, 4, 5, seed=2)
    act = ActiveSpace.from_eris(eris)
    gamma = act.averaged_rdm1(act.solve(2, 2)[1], 2)
    out = io.StringIO()
    ec, ev = fock.orbital_report(eris, gamma, stdout=out)
    text = out.getvalue()
    assert 'core orbitals (3)' in text and 'virtual orbitals (5)' in text
    assert abs(ec - fock.canonical_core(eris, gamma)[0]).max() < 1e-12
    new, kept, dropped = fock.truncate_virtuals(eris, gamma, nkeep=3)
    assert new.spaces.nvirt == 3 and abs(kept - ev[:3]).max() < 1e-12 and len(dropped) == 2
    with pytest.raises(ValueError):
        fock.truncate_virtuals(eris, gamma)
    with pytest.raises(ValueError):
        fock.truncate_virtuals(eris, gamma, nkeep=9)
