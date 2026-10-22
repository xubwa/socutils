import numpy
import pytest
from scipy.stats import unitary_group

from socutils.zmr.interface import pyscf_nr
from socutils.zmr.reference import dets
from socutils.zmr.tests import systems


def _cas_spectrum_full(eris, nelec, nroots):
    sp = eris.spaces
    strs = dets.cas_strings(sp.ncore, sp.nact, nelec)
    hmat = dets.operator_matrix(dets.hamiltonian_operator(eris.h1, eris.eri_block('pppp'), eris.ecore),
                                strs).toarray()
    return numpy.linalg.eigvalsh(hmat)[:nroots]


def _cas_spectrum_active(eris, nelec, nroots):
    heff, aaaa, ecore = eris.active_hamiltonian()
    strs = dets.gen_strings(range(eris.spaces.nact), nelec)
    hmat = dets.operator_matrix(dets.hamiltonian_operator(heff, aaaa, ecore), strs).toarray()
    return numpy.linalg.eigvalsh(hmat)[:nroots]


@pytest.mark.parametrize('nfrozen', [0, 1, 2])
def test_casci_energy_matches_pyscf(nfrozen):
    mc = systems.h2o_casci(4, 4)
    eris, nelec = pyscf_nr.from_casci(mc, nfrozen=nfrozen)
    assert eris.spaces.ncore == 2 * (mc.ncore - nfrozen)
    assert eris.symmetry_error() < 1e-12
    e_full = _cas_spectrum_full(eris, nelec, 1)[0]
    e_act = _cas_spectrum_active(eris, nelec, 1)[0]
    assert abs(e_full - mc.e_tot) < 1e-10
    assert abs(e_act - mc.e_tot) < 1e-10


def test_rotate_matches_rotated_mo_coeff():
    mc = systems.h2o_casci(4, 4)
    eris, _ = pyscf_nr.from_casci(mc)
    sp = eris.spaces
    rng = numpy.random.default_rng(0)
    # real rotation inside the (spatial) core and virtual blocks
    nc, na, nv = mc.ncore, mc.ncas, mc.mo_coeff.shape[1] - mc.ncore - mc.ncas
    r = numpy.eye(nc + na + nv)
    for s, n in [(slice(0, nc), nc), (slice(nc + na, None), nv)]:
        q, _ = numpy.linalg.qr(rng.normal(size=(n, n)))
        r[s, s] = q
    eris2 = pyscf_nr.from_mo_coeff(mc.mol, mc.get_hcore(), mc.mo_coeff @ r, 0, nc, na)
    eris3 = eris.rotate(numpy.kron(r, numpy.eye(2)))
    assert abs(eris2.h1 - eris3.h1).max() < 1e-12
    assert abs(eris2.eri_block('pppp') - eris3.eri_block('pppp')).max() < 1e-12
    assert sp.nmo == eris3.nmo


def test_complex_active_rotation_leaves_cas_spectrum_invariant():
    mc = systems.h2o_casci(4, 4)
    eris, nelec = pyscf_nr.from_casci(mc)
    sp = eris.spaces
    u = numpy.eye(sp.nmo, dtype=complex)
    u[sp.slice('a'), sp.slice('a')] = unitary_group.rvs(sp.nact, random_state=1)
    eris2 = eris.rotate(u)
    assert eris2.symmetry_error() < 1e-12
    e1 = _cas_spectrum_active(eris, nelec, 10)
    e2 = _cas_spectrum_active(eris2, nelec, 10)
    assert abs(e1 - e2).max() < 1e-10


def test_eri_block_labels():
    mc = systems.h2o_casci(4, 4)
    eris, _ = pyscf_nr.from_casci(mc)
    sp = eris.spaces
    assert eris.eri_block('cvca').shape == (sp.ncore, sp.nvirt, sp.ncore, sp.nact)
    with pytest.raises(KeyError):
        eris.eri_block('cvcx')
