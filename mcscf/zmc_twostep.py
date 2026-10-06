'''Two-step spinor CASSCF with the exact orbital Hessian.

A macro iteration solves the CI problem in the current orbitals and then takes
one orbital step for the resulting (state-averaged) density matrices: the
augmented-Hessian equations with the exact orbital Hessian inside a trust
region.  The CI coefficients are not coupled to the orbital rotations in the
step itself; the coupling is recovered over the macro iterations, optionally
accelerated by L-BFGS.

For fixed density matrices the energy is E(kappa) with C' = C exp(kappa) and

    E = E0 + 2 Re<g, x> + <x, H x> + O(x^3),      x = pack(kappa)

    g   = pack(A - A^+)
    H x = pack(A1 - A1^+ - 1/2 [A - A^+, kappa])

A is the generalised Fock matrix and A1 the same expression with the one-index
transformed integrals h1 = [h, kappa], (pq|rs)1.  The transformed two-electron
integrals need L_pm kappa_mi for every p, i.e. virtual-virtual Cholesky vectors;
they are not stored: zmc_ao2mo._CDERIS.rotated_pass half-transforms the AO
vectors with the rotated occupied orbitals once per H x.  An exact H x therefore
costs about one integral pass, against a handful of matrix products for the
super-CI Hessian of zmc_superci.

Conventions as in zmc_superci: solver density matrices gamma_pq = <p^+ q>,
Gamma_pqrs = <p^+ r^+ s q>; the MO density is D = gamma^T.
'''
import time
import numpy
from pyscf import lib
from socutils.mcscf import zmc_superci
from socutils.mcscf.zmc_superci import _tick


def gen_g_hop_exact(mc, mo, casdm1, casdm2, eris):
    '''Energy, orbital gradient and exact Hessian-vector product for fixed
    density matrices.

    Returns e, g, hop: e the energy of (casdm1, casdm2) in the orbitals mo,
    g the packed gradient, hop(x) the packed H x.
    '''
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    nmo = mo.shape[1]
    act = slice(ncore, nocc)
    dm = numpy.asarray(casdm1).T                         # D_tu = <u^+ t>
    h = lib.dot(mo.conj().T, lib.dot(mc.get_hcore(), mo))
    vc = lib.dot(mo.conj().T, lib.dot(eris.vj_core - eris.vk_core, mo))
    vja, vka = eris.get_jk_active_mo(dm)
    va = vja - vka
    cd_pa = eris.cd_pa                                    # L_P,pu  (naux, nmo, ncas)
    naux = cd_pa.shape[0]
    cd_up = getattr(eris, 'cd_up', None)                  # the same as (P, u, p)
    if cd_up is None:
        cd_up = numpy.ascontiguousarray(cd_pa.transpose(0, 2, 1))
    cd_up2 = cd_up.reshape(naux * ncas, nmo)
    dm2 = numpy.asarray(casdm2).reshape(ncas * ncas, ncas * ncas)
    cd_aa = eris.cd_aa.reshape(naux, ncas * ncas)

    def contract_dm2(l_aa):                               # T_P,tu = sum_vw L_P,vw Gamma_tuvw
        return lib.dot(l_aa, dm2.T).reshape(naux, ncas, ncas)

    def half_fock(l_up2, t):                              # sum_P,u L_P,pu T_P,tu -> (p, t)
        return lib.dot(l_up2.T, numpy.ascontiguousarray(t.transpose(0, 2, 1)).reshape(naux * ncas, ncas))

    def fock(h_, vc_, va_):
        a = numpy.zeros((nmo, nmo), dtype=complex)
        a[:, :ncore] = (h_ + vc_ + va_)[:, :ncore]
        a[:, act] = (h_ + vc_)[:, act] @ dm
        return a

    t0 = contract_dm2(cd_aa)
    a0 = fock(h, vc, va)
    a0[:, act] += half_fock(cd_up2, t0)
    g_mat = a0 - a0.conj().T
    e = (mc.energy_nuc() + numpy.trace(h[:ncore, :ncore]) + 0.5 * numpy.trace(vc[:ncore, :ncore])
         + numpy.einsum('tu,ut', (h + vc)[act, act], dm)
         + 0.5 * numpy.vdot(cd_aa.conj(), t0.reshape(naux, -1))).real

    def hop(x):
        _t = time.perf_counter()
        k = mc.unpack_uniq_var(x)
        jkc, jka, cd1 = eris.rotated_pass(k, dm)          # cd1[P, u, p] = sum_m L_P,pm k_mu
        _t = _tick('exact H x: AO pass', _t)
        h1 = lib.dot(h, k) - lib.dot(k, h)
        vc1 = lib.dot(vc, k) - lib.dot(k, vc) + jkc
        va1 = lib.dot(va, k) - lib.dot(k, va) + jka
        # L1_P,pu = sum_m L_pm k_mu - k_pm L_mu, kept in the (P, u, p) layout
        cd1 = cd1.reshape(naux * ncas, nmo)
        cd1 -= lib.dot(cd_up2, numpy.ascontiguousarray(k.T))
        l1_aa = numpy.ascontiguousarray(cd1.reshape(naux, ncas, nmo)[:, :, act].transpose(0, 2, 1))
        t1 = contract_dm2(l1_aa.reshape(naux, ncas * ncas))
        a1 = fock(h1, vc1, va1)
        a1[:, act] += half_fock(cd1, t0) + half_fock(cd_up2, t1)
        out = mc.pack_uniq_var(a1 - a1.conj().T - 0.5 * (lib.dot(g_mat, k) - lib.dot(k, g_mat)))
        _tick('exact H x: MO assembly', _t)
        return out

    return e, mc.pack_uniq_var(g_mat), hop


def gen_g_hop(mc, mo, casdm1, casdm2, eris):
    '''Gradient and exact Hessian-vector product, same return values as
    zmc_superci.gen_g_hop: g, h_diag, hop, sop, precond, mo.

    The metric is the unit one; the diagonal of the super-CI Hessian serves as
    the preconditioner of the Davidson iterations.'''
    sci = zmc_superci.gen_g_hop(mc, mo, casdm1, casdm2, eris)
    h_diag, mo = sci[1], sci[5]
    g, hop = gen_g_hop_exact(mc, mo, casdm1, casdm2, eris)[1:]
    return g, h_diag, hop, (lambda x: x), None, mo


def mcscf_twostep(mc, mo_coeff, **kwargs):
    '''Two-step CASSCF with the exact orbital Hessian.

    One CI solution and one orbital step per macro iteration: the augmented-
    Hessian equations with the exact orbital Hessian, solved by Davidson
    iterations (one exact H x each).  With bfgs=True the L-BFGS acceleration
    over the macro iterations accounts for the orbital-CI coupling (Kreplin,
    Knowles, Werner, JCP 152, 074102 (2020): the UC-AH method with L-BFGS).

    The macro iteration (trust region, L-BFGS, Kramers handling) is that of
    zmc_superci.mcscf_superci, used unchanged: see there for the arguments.
    Only the Hessian and the metric differ.
    '''
    return zmc_superci.mcscf_superci(mc, mo_coeff, gen_g_hop_fn=gen_g_hop, **kwargs)
