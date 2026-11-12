'''The relaxed density of the spin-free PC-NEVPT2 of a singlet (analytic orbital and CI
gradients of E2, Z-vector of the CASSCF, Lagrangian density) against finite fields with
the CASSCF and the NEVPT2 redone in the field (water, 6-31G, CAS(4,4), density fitting).'''
import numpy
import pytest
from pyscf import gto, scf, mcscf
from pyscf.df import incore

AUX = 'weigend'


@pytest.mark.slow
def test_relaxed_dipole():
    from socutils.zmr.interface import pyscf_nr
    from socutils.zmr.nr import PCNEVPT2 as NRPC
    from socutils.zmr.nr import density as nr_density
    from socutils.zmr.reference import ci_gradient as cg, orbital_gradient as og
    from socutils.zmr.reference import relaxed_fd as rf
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='6-31g', verbose=0)
    cderi = incore.cholesky_eri(mol, auxbasis=AUX)
    q = mol.atom_charges()
    with mol.with_common_orig((q[:, None] * mol.atom_coords()).sum(0) / q.sum()):
        op = mol.intor('int1e_r')[2]
    h0 = scf.RHF(mol).get_hcore()

    def casscf(h, mo0=None):
        mf = scf.RHF(mol).density_fit(auxbasis=AUX)
        mf.get_hcore = lambda *a: h
        mf.conv_tol = 1e-13
        mf.kernel()
        mc = mcscf.CASSCF(mf, 4, 4)
        mc.conv_tol = 1e-12
        mc.conv_tol_grad = 1e-6
        mc.max_cycle_macro = 300
        mc.kernel(mo0)
        return rf.polish(mc)

    def e_tot(mc, h):
        eris = pyscf_nr.nr_eris(mol, h, mc.mo_coeff, 0, mc.ncore, mc.ncas, cderi=cderi)
        pt = NRPC(eris, (2, 2), civecs=numpy.asarray(mc.ci).reshape(-1, 1))
        pt.verbose = 0
        pt.kernel()
        return mc.e_tot + float(sum(v[0, 0].real for v in pt.h2_by_class.values()))

    mc = casscf(h0)
    route = nr_density.spin_orbital_route(mc, cderi)
    pt = route.pt()
    g_qc, u = og.gradient(pt)
    g = route.spin_sum(numpy.conj(u) @ g_qc @ u.T).real
    z = rf.zvector(mc, numpy.concatenate([mc.pack_uniq_var(g - g.T),
                                          cg.ci_gradient(pt, route)]))
    _, d2, _ = nr_density.make_rdm1(mc, cderi)
    c = mc.mo_coeff
    dm = mc.make_rdm1() + c @ d2.real.T @ c.T + rf.lagrangian_density(mc, z)
    relaxed = numpy.einsum('ij,ji', op, dm)
    ff = []
    for sgn in (1, -1):
        m2 = casscf(h0 + sgn * 1e-3 * op, mc.mo_coeff)
        ff.append(e_tot(m2, h0 + sgn * 1e-3 * op))
    ff = (ff[0] - ff[1]) / 2e-3
    # a single step of 1e-3: second order in the field, ~1e-6
    assert abs(relaxed - ff) < 5e-6, (relaxed, ff)
    # the unrelaxed density is off by much more
    assert abs(numpy.einsum('ij,ji', op, mc.make_rdm1() + c @ d2.real.T @ c.T) - ff) > 1e-3
