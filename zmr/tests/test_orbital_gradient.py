'''zmr.reference.orbital_gradient: the orbital gradient of the PC-NEVPT2 second-order
energy at fixed CI vector (all classes, spin-orbital route) against finite differences in
the parameters of PySCF's CASSCF (water, 6-31G, CAS(4,4)).'''
import numpy
import pytest
from pyscf import gto, scf, mcscf
from pyscf.df import incore


@pytest.mark.slow
def test_orbital_gradient_all_classes():
    from socutils.zmr.interface import pyscf_nr
    from socutils.zmr.nr import PCNEVPT2 as NRPC
    from socutils.zmr.nr.active import ActiveSpace
    from socutils.zmr.pc.nevpt2 import PCNEVPT2
    from socutils.zmr.reference import orbital_gradient as og, relaxed_fd as rf
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='6-31g', verbose=0)
    cderi = incore.cholesky_eri(mol, auxbasis='weigend')
    mf = scf.RHF(mol).density_fit(auxbasis='weigend')
    mf.conv_tol = 1e-12
    mf.kernel()
    mc = mcscf.CASSCF(mf, 4, 4)
    mc.conv_tol = 1e-12
    mc.kernel()
    h = mc.get_hcore()

    def e2(m, c):
        eris = pyscf_nr.nr_eris(mol, h, m, 0, mc.ncore, mc.ncas, cderi=cderi)
        pt = NRPC(eris, (2, 2), civecs=numpy.asarray(c).reshape(-1, 1))
        pt.verbose = 0
        pt.kernel()
        return float(sum(v[0, 0].real for v in pt.h2_by_class.values()))

    fd = rf.fd_gradient(e2, mc)
    from socutils.zmr.nr import density as nr_density
    route = nr_density.spin_orbital_route(mc, cderi)
    pt = route.pt()
    g_qc, u = og.gradient(pt)
    # the Fock part through AO J/K builds against the one through MO three-index integrals
    g_b, _ = og.gradient(pt, fock='b')
    assert abs(g_qc - g_b).max() < 1e-11
    g = route.spin_sum(numpy.conj(u) @ g_qc @ u.T).real
    an = mc.pack_uniq_var(g - g.T)
    n = len(an)
    assert abs(fd[:n] - an).max() < 1e-7, abs(fd[:n] - an).max()


def test_ccvv_explicit_native():
    '''The explicit (integrals in v) part of the ccvv orbital gradient from the native
    kernel and the AO back-transformation against the reference with all amplitudes and
    MO three-index integrals.'''
    from socutils.zmr.nr import density as nr_density
    from socutils.zmr.reference import orbital_gradient as og
    from socutils.zmr.uc import density, gradient
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='6-31g', verbose=0)
    cderi = incore.cholesky_eri(mol, auxbasis='weigend')
    mf = scf.RHF(mol).density_fit(auxbasis='weigend')
    mf.kernel()
    mc = mcscf.CASSCF(mf, 4, 4)
    mc.kernel()
    route = nr_density.spin_orbital_route(mc, cderi)
    pt = route.pt()
    ctx = density.context(pt)
    e, dcc, dvv, g_fast = gradient.ccvv(ctx)
    e_ref, dcc_ref, dvv_ref = density.ccvv(ctx, kramers=False)
    assert abs(e - e_ref) < 1e-12
    assert abs(dcc - dcc_ref).max() < 1e-12 and abs(dvv - dvv_ref).max() < 1e-12
    eris = ctx.eris
    sp = eris.spaces
    gam = numpy.zeros((ctx.eris.cd_block('vc').shape[0], sp.nmo, sp.nmo))
    gam[:, sp.slice('v'), sp.slice('c')] = og.ccvv_explicit(ctx)
    bmo = og.mo_vectors(eris).real
    g_ref = numpy.einsum('Prs,Psq->rq', bmo, gam + gam.transpose(0, 2, 1))
    assert abs(g_fast - g_ref).max() < 1e-11, abs(g_fast - g_ref).max()
