'''SO-SCI: second-order treatment of the active orbital rotations, super-CI for the rest.

Kreplin, Knowles, Werner, J. Chem. Phys. 152, 074102 (2020), Sec. II C.  The
orbital rotations are split into

    A   rotations with an active index: virtual-active and active-core
    C   virtual-core rotations

and the Hessian of the augmented-Hessian equations is

    <x, H x> = <x_A, H_exact x_A> + <x_C, H_SCI x_C> + 2 Re <x_A, H_SCI x_C>

i.e. the exact orbital Hessian in the A-A block, the super-CI Hessian in the C-C
block and in the A-C coupling.  The metric is the unit one (for the C block that
is the super-CI overlap in the spinor normalisation).

The A-A block of the exact Hessian needs only two-electron integrals with two
active indices,

    J^tu_pq = (pq|tu)    from the pair operators eris.jtu_tril, transformed to the MO basis
    K^tu_pq = (pt|qu)    contracted on the fly from eris.cd_pa

so a product costs a few contractions over (naux, nmo, ncas) and (nmo, nmo,
ncas^2) arrays and no pass over the AO vectors, unlike the full exact Hessian
of zmc_twostep.

Deviation from the paper: the rotation is a single exponential exp(A + C), not
exp(C) exp(A); KKW report nearly identical convergence for the two.

Conventions as in zmc_superci: gamma_pq = <p^+ q>, Gamma_pqrs = <p^+ r^+ s q>,
MO density D = gamma^T.
'''
import time
import numpy
from pyscf import lib
from socutils.mcscf import zmc_superci
from socutils.mcscf.zmc_superci import _tick


def pair_coulomb_mo(eris):
    '''J[t, u, p, q] = (pq|tu) for all orbitals p, q, from the AO pair operators.'''
    if getattr(eris, 'jtu_tril', None) is None:
        raise NotImplementedError('SO-SCI needs the pair operators J^tu (eris.jtu_tril)')
    mo_sph = eris.mo_sph
    nao = mo_sph.shape[0] // 2
    nmo = mo_sph.shape[1]
    ncas = eris.nocc - eris.ncore
    jt = eris.jtu_tril.T                                   # (ncas^2, npair), symmetric in the AO pair
    jao = (lib.unpack_tril(numpy.ascontiguousarray(jt.real))
           + 1j * lib.unpack_tril(numpy.ascontiguousarray(jt.imag)))    # (ncas^2, nao, nao)
    ca, cb = mo_sph[:nao], mo_sph[nao:]
    jmo = (numpy.matmul(ca.conj().T, numpy.matmul(jao, ca))
           + numpy.matmul(cb.conj().T, numpy.matmul(jao, cb)))
    return jmo.reshape(ncas, ncas, nmo, nmo)


def gen_hop_active(mc, mo, casdm1, casdm2, eris):
    '''The A-A block of the exact orbital Hessian.

    Returns g, hop_aa, pack_a: the packed gradient, hop_aa(x) = P_A H_exact P_A x
    and the projector pack_a(x) = P_A x onto the rotations with an active index.
    '''
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    nmo = mo.shape[1]
    act = slice(ncore, nocc)
    core = slice(0, ncore)
    dm = numpy.asarray(casdm1).T
    h = lib.dot(mo.conj().T, lib.dot(mc.get_hcore(), mo))
    vc = lib.dot(mo.conj().T, lib.dot(eris.vj_core - eris.vk_core, mo))
    vja, vka = eris.get_jk_active_mo(dm)
    va = vja - vka
    lp = eris.cd_pa                                        # L_P,pu  (naux, nmo, ncas)
    naux = lp.shape[0]
    lup = getattr(eris, 'cd_up', None)                     # the same as (P, u, p)
    if lup is None:
        lup = numpy.ascontiguousarray(lp.transpose(0, 2, 1))
    # every contraction below is a matrix product in one of these layouts
    lp2 = lp.reshape(naux, nmo * ncas)                     # [P, (p u)]
    lup2 = lup.reshape(naux * ncas, nmo)                   # [(P u), p]
    lc = numpy.ascontiguousarray(lp[:, core, :])           # L_P,iu
    lc2 = lc.reshape(naux, ncore * ncas)                   # [P, (i u)]
    lcu2 = numpy.ascontiguousarray(lup[:, :, core]).reshape(naux * ncas, ncore)     # [(P u), i]
    dm2_mat = numpy.asarray(casdm2).reshape(ncas * ncas, ncas * ncas)
    cd_aa = eris.cd_aa.reshape(naux, ncas * ncas)

    def contract_dm2(l_aa):                                # T_P,tu = sum_vw L_P,vw Gamma_tuvw
        return lib.dot(l_aa, dm2_mat.T).reshape(naux, ncas, ncas)

    def half_fock(t):                                      # sum_P,u L_P,pu T_P,tu -> (p, t)
        return lib.dot(lup2.T, numpy.ascontiguousarray(t.transpose(0, 2, 1)).reshape(naux * ncas, ncas))

    t0 = contract_dm2(cd_aa)
    f2 = half_fock(t0)                                     # two-body part of the generalised Fock matrix
    a0 = numpy.zeros((nmo, nmo), dtype=complex)
    a0[:, core] = (h + vc + va)[:, core]
    a0[:, act] = (h + vc)[:, act] @ dm + f2
    g_mat = a0 - a0.conj().T

    # The pair operators stay in the AO basis.  Every use of (pq|tu) below is
    # contracted with a vector on one side, (pq|tu) v_q = C_p^+ J^tu (C v), so
    # a product with the AO matrix replaces the transformation of all J^tu to
    # the MO basis.  ja[(t mu), (u nu)] = J^tu_{mu nu} (spin free), and
    # wa[(t mu), (u nu)] = sum_vw Gamma_tuvw J^vw_{mu nu}.
    _t = time.perf_counter()
    if getattr(eris, 'jtu_tril', None) is None:
        raise NotImplementedError('SO-SCI needs the pair operators J^tu (eris.jtu_tril)')
    mo_sph = eris.mo_sph
    nao = mo_sph.shape[0] // 2
    cs = [numpy.ascontiguousarray(mo_sph[:nao]), numpy.ascontiguousarray(mo_sph[nao:])]      # alpha, beta
    cs_core = [numpy.ascontiguousarray(c[:, core]) for c in cs]
    jt = eris.jtu_tril.T                                   # (ncas^2, npair)
    wt = lib.dot(dm2_mat, numpy.ascontiguousarray(jt))

    def _pair_matrix(tril):                                # [(t mu), (u nu)] from the packed [(t u), (mu nu)]
        full = (lib.unpack_tril(numpy.ascontiguousarray(tril.real))
                + 1j * lib.unpack_tril(numpy.ascontiguousarray(tril.imag))).reshape(ncas, ncas, nao, nao)
        return numpy.ascontiguousarray(full.transpose(0, 2, 1, 3)).reshape(ncas * nao, ncas * nao)
    ja = _pair_matrix(jt)
    wa = _pair_matrix(wt)
    del jt, wt
    _tick('SO-SCI: pair operators J^tu, W^tu (AO)', _t)

    def _apply(mat, v):                                    # sum_(u nu) mat[(t mu), (u nu)] v[nu, u] -> [mu, t]
        return lib.dot(mat, numpy.ascontiguousarray(v.T).reshape(-1, 1)).reshape(ncas, nao).T

    def _apply_t(mat, v):                                  # the same with mat transposed
        return lib.dot(numpy.ascontiguousarray(v.T).reshape(1, -1), mat).reshape(ncas, nao).T

    mask = numpy.zeros((nmo, nmo))
    mask[act, core] = 1
    mask[nocc:, act] = 1
    mask_a = mc.pack_uniq_var(mask + 0j).real > 0.5

    def pack_a(x):
        return numpy.where(mask_a, x, 0)

    def hop_aa(x):
        _t = time.perf_counter()
        k = mc.unpack_uniq_var(pack_a(x))
        xtc = numpy.ascontiguousarray(k[act, core])        # kappa_ti
        kpa = numpy.ascontiguousarray(k[:, act])           # kappa_pu, p core or virtual
        y = kpa @ dm                                       # first-order active density, (p, u)

        # (J - K) of the first-order core density  dD[t,i] = kappa_ti, dD[i,t] = kappa_ti^*
        rho = 2 * lib.dot(lc2, xtc.T.reshape(-1, 1)).real.ravel()           # 2 Re sum_ti L_P,it kappa_ti
        m2 = numpy.ascontiguousarray(numpy.matmul(xtc, lc)).reshape(naux * ncas, ncas)   # [(P t), u]
        # v[nu, t] = sum_i C_nu,i kappa_ti^* (per spin):
        #   sum_ti kappa_ti^* (p i|t u) = C_p^+ sum_t J^tu v_t        -> [p, u]
        #   sum_ti kappa_ti   (i j|s t) = sum_t v_t^+ J^st C_j        -> [s, j]
        vs = [cc @ xtc.conj().T for cc in cs_core]
        k_pu = sum(c.conj().T @ _apply_t(ja, v) for c, v in zip(cs, vs))
        k_sj = sum(_apply(ja, v.conj()).T @ cc for cc, v in zip(cs_core, vs))
        jkc_pu = (lib.dot(rho[None, :] + 0j, lp2).reshape(nmo, ncas)
                  - lib.dot(lup2.T, m2)
                  - k_pu)
        j_ti = lib.dot(rho[None, :] + 0j, lc2).reshape(ncore, ncas).conj().T
        jkc_ti = (j_ti
                  - k_sj
                  - lib.dot(m2.T, lcu2).conj())
        # one-index transformed vectors with two active indices; the first
        # piece, sum_m L_P,vm kappa_mw, also gives the exchange of the
        # first-order active density
        l1 = lib.dot(lup2, kpa.conj()).conj().reshape(naux, ncas, ncas)     # [P, v, w]
        rho = 2 * lib.dot(lp2, y.conj().reshape(-1, 1)).real.ravel()        # 2 Re sum_pu L_P,up y_pu
        n2 = numpy.ascontiguousarray(numpy.matmul(l1, dm).transpose(0, 2, 1)).reshape(naux * ncas, ncas)   # [(P u), s]
        # w[nu, u] = sum_p C_nu,p y_pu:  sum_pu y_pu^* (p j|s u) = sum_u w_u^+ J^su C_j   -> [s, j]
        ws = [c @ y for c in cs]
        jka_ti = (lib.dot(rho[None, :] + 0j, lc2).reshape(ncore, ncas).conj().T
                  - lib.dot(n2.T, lcu2.conj())
                  - sum(_apply(ja, w.conj()).T @ cc for cc, w in zip(cs_core, ws)))
        l1 -= numpy.matmul(numpy.ascontiguousarray(k[act, :]), lp)
        t1 = contract_dm2(l1.reshape(naux, ncas * ncas))
        # sum_um W[t,u,p,m] kappa_mu = C_p^+ sum_u W^tu (C kappa)_u               -> [p, t]
        r12 = (sum(c.conj().T @ _apply(wa, c @ kpa) for c in cs) - k @ f2 + half_fock(t1))

        h1 = h @ k - k @ h
        vc1 = vc @ k - k @ vc
        va1 = va @ k - k @ va
        a1_pt = (h1 + vc1)[:, act] @ dm + jkc_pu @ dm + r12          # rows p: core and virtual are valid
        a1_ti = (h1 + vc1 + va1)[act, core] + jkc_ti + jka_ti
        gk = 0.5 * (g_mat @ k - k @ g_mat)
        out = numpy.zeros((nmo, nmo), dtype=complex)
        out[nocc:, act] = a1_pt[nocc:] - gk[nocc:, act]
        out[act, core] = a1_ti - a1_pt[core].conj().T - gk[act, core]
        out = pack_a(mc.pack_uniq_var(out))
        _tick('SO-SCI: exact active H x', _t)
        return out

    return mc.pack_uniq_var(g_mat), hop_aa, pack_a


def gen_g_hop(mc, mo, casdm1, casdm2, eris):
    '''Gradient and SO-SCI Hessian-vector product, same return values as
    zmc_superci.gen_g_hop: g, h_diag, hop, sop, precond, mo.

    The diagonal of the super-CI Hessian serves as the preconditioner.'''
    g_sci, h_diag, hop_sci, sop_sci, precond_sci, mo = zmc_superci.gen_g_hop(mc, mo, casdm1, casdm2, eris)
    g, hop_aa, pack_a = gen_hop_active(mc, mo, casdm1, casdm2, eris)

    def hop(x):
        xa = pack_a(x)
        return hop_sci(x) - pack_a(hop_sci(xa)) + hop_aa(xa)

    return g, h_diag, hop, (lambda x: x), None, mo


def mcscf_sosci(mc, mo_coeff, **kwargs):
    '''SO-SCI orbital optimisation.

    One orbital step per CI solution, as in the super-CI driver, whose macro
    iteration (trust region, L-BFGS acceleration with bfgs=True, Kramers
    handling) is used unchanged: see zmc_superci.mcscf_superci for the
    arguments.  Only the Hessian and the metric differ.
    '''
    return zmc_superci.mcscf_superci(mc, mo_coeff, gen_g_hop_fn=gen_g_hop, **kwargs)
