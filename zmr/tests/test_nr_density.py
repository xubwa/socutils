'''zmr.nr.density: the unrelaxed density of the spin-free PC-NEVPT2 of a singlet through
the spin-orbital route, against the spin-free energies and finite fields (frozen
orbitals and CI vector).'''
import numpy
import pytest
from pyscf import gto, scf, mcscf
from pyscf.df import incore


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='6-31g', verbose=0)
    cderi = incore.cholesky_eri(mol, auxbasis='weigend')
    mf = scf.RHF(mol).density_fit(auxbasis='weigend')
    mf.kernel()
    mc = mcscf.CASSCF(mf, 4, 4)
    mc.conv_tol = 1e-11
    mc.kernel()
    return mol, mc, cderi


def pyscf_nr_civecs(mc):
    from socutils.zmr.interface import pyscf_nr
    return pyscf_nr.nr_civecs(mc)


def _nr_e2(mol, mc, cderi, hcore, civecs=None):
    '''(E2 of the spin-free PC-NEVPT2 on the orbitals of mc, CI vectors): civecs None
    solves the CASCI of these integrals (zmr's solver, converged tightly).'''
    from socutils.zmr.interface import pyscf_nr
    from socutils.zmr.nr import PCNEVPT2
    from socutils.zmr.nr.active import ActiveSpace
    eris = pyscf_nr.nr_eris(mol, hcore, mc.mo_coeff, 0, mc.ncore, mc.ncas, cderi=cderi)
    nelec = (mc.nelecas[0], mc.nelecas[1])
    if civecs is None:
        civecs = ActiveSpace.from_eris(eris).solve(nelec, 1)[1]
    pt = PCNEVPT2(eris, nelec, civecs=civecs)
    pt.verbose = 0
    pt.kernel()
    return float(numpy.real(pt.heff[0, 0] - pt.hmodel[0, 0])), civecs


def test_spin_orbital_route(water):
    from socutils.zmr.nr import density
    mol, mc, cderi = water
    ref, d2, pt = density.make_rdm1(mc, cderi)
    e2_so = float(sum(v[0, 0].real for v in pt.h2_by_class.values()))
    h0 = mc.get_hcore()
    # both routes with the CI vector of the CASSCF
    c0 = pyscf_nr_civecs(mc)
    e2_nr, _ = _nr_e2(mol, mc, cderi, h0, c0)
    assert abs(e2_so - e2_nr) < 1e-11
    c = mc.mo_coeff
    dm_cas = c.T @ mol.intor('int1e_ovlp') @ mc.make_rdm1() @ mol.intor('int1e_ovlp') @ c
    assert abs(ref - dm_cas).max() < 1e-5
    assert abs(numpy.trace(d2)) < 1e-10
    assert abs(d2.imag).max() < 1e-10
    # finite field at fixed orbitals and CI vector, in the spin-free code
    s = mol.intor('int1e_ovlp')
    x = s @ c
    rng = numpy.random.default_rng(3)
    y = rng.normal(size=d2.shape)
    o = 0.5 * (y + y.T)
    fd = []
    for h in (2e-4, 1e-4):
        fd.append((_nr_e2(mol, mc, cderi, h0 + h * (x @ o @ x.T), c0)[0]
                   - _nr_e2(mol, mc, cderi, h0 - h * (x @ o @ x.T), c0)[0]) / (2 * h))
    fd = (4 * fd[1] - fd[0]) / 3
    analytic = float(numpy.sum(o * d2).real)
    assert abs(fd - analytic) < 1e-6 * max(1.0, abs(analytic)) + 1e-9, (fd, analytic)
