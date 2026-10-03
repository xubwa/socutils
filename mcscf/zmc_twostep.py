'''Two-step spinor CASSCF with the exact orbital Hessian.

A macro iteration solves the CI problem in the current orbitals and then
minimises the energy of the resulting (state-averaged) density matrices with
respect to the orbitals: a few Newton steps, each from the augmented-Hessian
equations with the exact orbital Hessian inside a trust region.  The CI
coefficients are not coupled to the orbital rotations; the coupling is
recovered by the macro iterations.

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
import numpy as np
from numpy.linalg import norm
from scipy.linalg import expm
from pyscf import lib
from pyscf.lib import logger
from socutils.mcscf import zmcscf, zmc_ao2mo, zmc_superci
from socutils.mcscf.zmc_superci import _tick, TIMINGS, print_timings, ensure_kramers, davidson


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
    dm2 = numpy.asarray(casdm2).reshape(ncas * ncas, ncas * ncas)
    cd_aa = eris.cd_aa.reshape(naux, ncas * ncas)

    def contract_dm2(l_aa):                               # T_P,tu = sum_vw L_P,vw Gamma_tuvw
        return lib.dot(l_aa, dm2.T).reshape(naux, ncas, ncas)

    def half_fock(l_pu, t):                               # sum_P,u L_P,pu T_P,tu -> (p, t)
        return lib.einsum('Lpu,Ltu->pt', l_pu, t)

    def fock(h_, vc_, va_):
        a = numpy.zeros((nmo, nmo), dtype=complex)
        a[:, :ncore] = (h_ + vc_ + va_)[:, :ncore]
        a[:, act] = (h_ + vc_)[:, act] @ dm
        return a

    t0 = contract_dm2(cd_aa)
    a0 = fock(h, vc, va)
    a0[:, act] += half_fock(cd_pa, t0)
    g_mat = a0 - a0.conj().T
    e = (mc.energy_nuc() + numpy.trace(h[:ncore, :ncore]) + 0.5 * numpy.trace(vc[:ncore, :ncore])
         + numpy.einsum('tu,ut', (h + vc)[act, act], dm)
         + 0.5 * numpy.vdot(cd_aa.conj(), t0.reshape(naux, -1))).real

    def hop(x):
        _t = time.perf_counter()
        k = mc.unpack_uniq_var(x)
        jkc, jka, cd1 = eris.rotated_pass(k, dm)
        _t = _tick('exact H x: AO pass', _t)
        h1 = lib.dot(h, k) - lib.dot(k, h)
        vc1 = lib.dot(vc, k) - lib.dot(k, vc) + jkc
        va1 = lib.dot(va, k) - lib.dot(k, va) + jka
        cd1 -= numpy.matmul(k, cd_pa)                     # L1_P,pu = sum_m L_pm k_mu - k_pm L_mu
        t1 = contract_dm2(numpy.ascontiguousarray(cd1[:, act, :]).reshape(naux, ncas * ncas))
        a1 = fock(h1, vc1, va1)
        a1[:, act] += half_fock(cd1, t0) + half_fock(cd_pa, t1)
        out = mc.pack_uniq_var(a1 - a1.conj().T - 0.5 * (lib.dot(g_mat, k) - lib.dot(k, g_mat)))
        _tick('exact H x: MO assembly', _t)
        return out

    return e, mc.pack_uniq_var(g_mat), hop


def mcscf_twostep(mc, mo_coeff, conv_tol=None, conv_tol_grad=None, max_cycle_micro=4, micro_grad_red=0.1,
                  trust_radius=0.2, trust_radius_max=2.0, davidson_maxiter=10, ah_root='yanai',
                  kramers_check=True, kramers_project=True, cderi=None, verbose=5):
    '''Two-step CASSCF with the exact orbital Hessian.

    max_cycle_micro   orbital Newton steps per CI solution.
    micro_grad_red    the orbital steps of a macro iteration stop once the
                      fixed-density gradient is below micro_grad_red times the
                      gradient at the start of the macro iteration.
    trust_radius      initial trust radius, the Frobenius norm of the rotation
                      matrix.  It follows rho = dE / Q with the fixed-density
                      energy: halved for rho < 0.25, doubled for rho > 0.75 at
                      the boundary.  A step that raises the fixed-density
                      energy is halved until it does not.
    kramers_check     require interleaved, phase +1 Kramers pairs at entry.
    kramers_project   project every rotation onto its time-reversal even part.

    Returns conv, e_tot, e_cas, fcivec, mo, None (as mcscf_superci).
    '''
    log = logger.new_logger(mc, verbose)
    t_start = time.perf_counter()
    if conv_tol is None:
        conv_tol = mc.conv_tol
    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(conv_tol)
    mol = mc.mol
    mo = mo_coeff
    ncas = mc.ncas
    if kramers_check:
        from socutils.tools import kramers
        s_ovlp = mol.intor('int1e_ovlp_spinor')
        dev = kramers.pairing_deviation(mol, mo, s_ovlp)
        log.info('Kramers pairing of the starting orbitals: max deviation %.2e', dev)
        if dev > 1e-6:
            raise RuntimeError('starting orbitals are not interleaved, phase +1 Kramers pairs '
                               '(max deviation %.2e); see socutils.tools.kramers.symmetrize' % dev)
    TIMINGS.clear()

    def integrals(mo_):
        _t = time.perf_counter()
        eris_ = zmc_ao2mo._CDERIS(mc, mo_, cderi=cderi, level=2)
        _tick('integral half-transform (_CDERIS)', _t)
        return eris_

    def solve_ci(mo_, eris_):
        _t = time.perf_counter()
        mci = zmcscf._fake_h_for_fast_casci(mc, mo_, eris_)
        e_tot, e_cas, fcivec = mci.kernel(mo_, ci0=None, verbose=verbose)
        _t = _tick('CI solver', _t)
        dm1, dm2 = mc.fcisolver.make_rdm12(fcivec, ncas, mc.nelecas)
        _tick('make_rdm12', _t)
        return e_tot, e_cas, fcivec, dm1, dm2

    def gradient(mo_, eris_, dm1, dm2):
        _t = time.perf_counter()
        out = gen_g_hop_exact(mc, mo_, dm1, dm2, eris_)
        _tick('gradient and fixed-density energy', _t)
        return out

    eris = integrals(mo)
    e_tot, e_cas, fcivec, casdm1, casdm2 = solve_ci(mo, eris)
    e_last = e_tot
    de = e_tot
    conv = False
    imacro = 0
    n_hx = n_micro = 0
    norm_rot = 0.0
    while imacro < mc.max_cycle_macro:
        e_fix, g, hop = gradient(mo, eris, casdm1, casdm2)
        norm_g0 = norm(g)
        print(f'Iter {imacro:3d}: E = {e_tot:20.15f}  dE = {de:12.10f}'
              f'  norm(grad) = {norm_g0:8.6f} step_size = {norm_rot:8.6f}', flush=True)
        log.debug('fixed-density energy - CI energy: %.2e', e_fix - numpy.asarray(e_tot).real)
        if abs(de) < conv_tol and norm_g0 < conv_tol_grad:
            conv = True
            break
        norm_rot = 0.0
        for imicro in range(max_cycle_micro):
            norm_g = norm(g)
            if imicro > 0 and norm_g < max(micro_grad_red * norm_g0, 0.3 * conv_tol_grad):
                break
            # the diagonal of the super-CI Hessian preconditions the Davidson
            # iterations; the Hessian itself is exact, the metric is the unit one
            _t = time.perf_counter()
            h_diag = zmc_superci.gen_g_hop(mc, mo, casdm1, casdm2, eris)[1]
            _t = _tick('super-CI diagonal (preconditioner)', _t)
            counter = [0]

            def hop_counted(v):
                counter[0] += 1
                return hop(v)
            x, eps, hx = davidson(hop_counted, g, h_diag, sop=lambda v: v, max_stepsize=trust_radius / np.sqrt(2.),
                                  mmax=davidson_maxiter, use_lambda=True, root=ah_root, with_hx=True)
            _tick('AH Davidson (total)', _t)
            n_hx += counter[0]
            dr = mc.unpack_uniq_var(x)
            if kramers_project:
                dr = ensure_kramers(dr)
            x_taken = mc.pack_uniq_var(dr)
            if norm(x_taken - x) > 1e-8 * max(norm(x), 1e-30):
                hx = hop(x_taken)        # the projection changed the step: its model needs H x again
                n_hx += 1
            scale = 1.0
            if norm(dr) > trust_radius:
                scale = trust_radius / norm(dr)
            for ibt in range(6):
                xs = scale * x_taken
                q_model = 2. * np.vdot(g, xs).real + scale**2 * np.vdot(x_taken, hx).real
                _t = time.perf_counter()
                mo_new = lib.dot(mo, expm(scale * dr))
                _tick('expm of the rotation', _t)
                eris_new = integrals(mo_new)
                e_new, g_new, hop_new = gradient(mo_new, eris_new, casdm1, casdm2)
                d_fix = e_new - e_fix
                rho = d_fix / q_model if q_model < 0 else -1.0
                step = scale * norm(dr)
                print(f'  micro {imicro}: |g| = {norm_g:.3e}  |dr| = {step:.4e}  trust radius = {trust_radius:.4e}'
                      f'  dE(fixed density) = {d_fix:.4e}  model = {q_model:.4e}  rho = {rho:.3f}'
                      f'  H x calls = {counter[0]}', flush=True)
                if d_fix <= conv_tol:
                    break
                scale *= 0.5
                trust_radius = 0.5 * step
                print(f'  fixed-density energy rose: step halved, trust radius -> {trust_radius:.4e}')
            if abs(q_model) >= conv_tol:
                if rho < 0.25:
                    trust_radius = 0.5 * step
                elif rho > 0.75 and step > 0.9 * trust_radius:
                    trust_radius = min(2. * trust_radius, trust_radius_max)
            mo, eris, e_fix, g, hop = mo_new, eris_new, e_new, g_new, hop_new
            norm_rot = np.sqrt(norm_rot**2 + step**2)
            n_micro += 1
        e_tot, e_cas, fcivec, casdm1, casdm2 = solve_ci(mo, eris)
        de = e_tot - e_last
        e_last = e_tot
        imacro += 1
        _t = time.perf_counter()
        lib.chkfile.save(mc.chkfile, f'mo_coeff_iter_{imacro}', mo)
        _tick('chkfile save of the orbitals', _t)
        if kramers_check:
            print('Kramers pairing deviation of the orbitals after macro iter %d: %.3e'
                  % (imacro, kramers.pairing_deviation(mol, mo, s_ovlp)))
        mc.e_tot, mc.e_cas = e_tot, e_cas
    mc.mo_coeff = mo
    print(f'mcscf_twostep: {imacro} macro iterations, {n_micro} orbital steps, {n_hx} exact H x')
    print_timings('mcscf_twostep', wall=time.perf_counter() - t_start)
    return conv, e_tot, e_cas, fcivec, mo, None
