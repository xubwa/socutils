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
    eris0 = pyscf_nr.nr_eris(mol, h, mc.mo_coeff, 0, mc.ncore, mc.ncas, cderi=cderi)
    c0 = ActiveSpace.from_eris(eris0).solve((2, 2), 1)[1][:, 0]
    mc.ci = c0.reshape(numpy.asarray(mc.ci).shape)

    def e2(m, c):
        eris = pyscf_nr.nr_eris(mol, h, m, 0, mc.ncore, mc.ncas, cderi=cderi)
        pt = NRPC(eris, (2, 2), civecs=numpy.asarray(c).reshape(-1, 1))
        pt.verbose = 0
        pt.kernel()
        return float(sum(v[0, 0].real for v in pt.h2_by_class.values()))

    fd = rf.fd_gradient(e2, mc)
    eris, nelec = pyscf_nr.from_casci(mc, cderi=cderi)
    pt = PCNEVPT2(eris, nelec, nstate=1)
    pt.verbose = 0
    pt.kernel()
    g_qc, u = og.gradient(pt)
    g_so = numpy.conj(u) @ g_qc @ u.T
    g = (g_so[0::2, 0::2] + g_so[1::2, 1::2]).real
    an = mc.pack_uniq_var(g - g.T)
    n = len(an)
    assert abs(fd[:n] - an).max() < 1e-7, abs(fd[:n] - an).max()
