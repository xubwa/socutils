#
# Mixed CD / COSX integral provider of the two-component CASSCF (zmc_ao2mo._CDERIS, mc.core_exchange = 'cosx')
#
'''Core exchange of the spinor CASSCF: Cholesky factor for every term with an active index, COSX for the
core-core exchange alone.

With i, j core, t, u active and p, q general orbitals the energy functional is

    E = sum_i h_ii + 1/2 sum_i (J[D_c] - K^X[D_c])_ii + sum_tu (h + J[D_c] - K^L[D_c])_tu gamma_tu
        + 1/2 sum (tu|vw)_L Gamma_tuvw

J and every exchange integral with an active index from the factor L (K^L), the exchange of the core density
with itself, E_x^cc = -1/2 tr(D_c K^X[D_c]), from COSX (K^X).  The orbital gradient is the exact derivative of
this functional: the core columns of the generalised Fock matrix carry K^X[D_c] C_core, its active columns
K^L[D_c] C_act.  In the MO basis the core exchange operator the solvers see is therefore

    M[:, core] = K^X[D_c][:, core],   M[:, act] = K^L[D_c][:, act],   M[act, virt] = M[virt, act]^*,
    M elsewhere = K^X[D_c]      (only in the super-CI Hessian's Fock blocks: f_vv, f_oo, ...)

Hermitian except for the active-core blocks (M[act, core] COSX, M[core, act] factor).  zmc_ao2mo._CDERIS keeps
it as eris.vk_core (spinor AO basis, S C M C^+ S), so the CASCI core energy, h1eff, the super-CI gradient and
the Fock blocks of its Hessian need no further change.  K^L[D_c] C_act costs naux n (n n_act + n_core n_act)
(from the active half-transformed vectors of the pass); no core half transformation, no naux n^2 n_core.

K^X is the variational COSX exchange: the derivative of its own energy.  With the overlap fit (Q = P w phi,
P = S S_g^-1) the one-sided K = P K_plain[D] is not; the derivative of q(X) = <X, P K_plain[X]> is

    K^X[X] = 1/2 (P K_plain[X] + K_plain[P^T X])

(K_plain is self-adjoint for the Frobenius product), computed for the real pieces of the GHF density in one
multi-density COSX call (twice the pieces of the one-sided K: 8 for a Kramers core), so that
1/2 tr(D K^X[D]) is the energy whose gradient the solver uses.

Solvers.  zmc_superci: gradient exact, the super-CI Hessian built from the Fock blocks of M.  zmc_twostep:
exact Hessian of the mixed functional (_CDERIS.rotated_pass: J of the first-order core density from the factor,
its exchange by one multi-density COSX call in the core columns and from the factor in the active columns, plus
the commutator corrections of the non-Hermitian M); finite-difference exact (mcscf/test/test_core_cosx.py).
zmc_sosci: gradient exact; its active-block Hessian keeps the factor-only response (no COSX call per H x), so
the core-core exchange curvature there is the factor's (a difference of the order of the core exchange error).

The core-core exchange is where a three-index factor is least accurate: with the acCD auxiliary basis (tau
1e-3) at converged CASSCF orbitals, error of -1/2 tr(D_c K[D_c]) against the exact four-index value, CD / COSX
(GRIDX2, heavy-atom radial rule): HBr cc-pVDZ CAS(2,4) +1.0e-5 / -1.5e-6, AuH x2c-SVPall CAS(2,4) +8.86e-4 /
-3.6e-7 Eh; the converged mixed CASSCF energy minus the all-CD one is the difference of the two (HBr -1.15e-5,
AuH -8.866e-4 Eh; super-CI, SO-SCI and two-step reach the same mixed energy to 1e-10).  HgCl2 x2c-SVPall
CAS(4,6) (SO-SCI, mixed started from the all-CD orbitals): -1.129e-3 = -(CD +9.35e-4) + (COSX -1.95e-4).

Cost per macro iteration (hgcl2_lattice, x2c-SVPall, acCD factor, CAS(4,6), 24 cores of a Gold 6248R socket;
3360: c442 shared, factor terms from naux 722 times 32 = 23104, rough):
    nao_nr (n2c, ncore)       all-CD _CDERIS    mixed: factor + COSX         exact H x (two-step): all-CD / mixed
    1260 (2520, 1364)         190 s             61 + 211..231                 909 / 290 s
    3360 (6720, 3644)         ~7400 s           ~2050 + 3000..4400            out of memory (~13 h) / ~3200 s
(COSX of the core: 666160 points at 3360, 3000 s on c442, 4435 s on a dedicated node; the all-CD exact H x needs
4 x 240 x n_core x nao x 8 B = 94 GB twice for the core half-transform buffers of rotated_pass at 3360.)  The
all-CD core exchange grows as naux n^2 n_core ~ N^3.7 here, the COSX one as ~N^2.6: crossover near nao_nr ~
2000 - 2500 for these lattices; below it core_exchange = 'cd' (the default) is the cheaper one per macro
iteration, while the mixed exact H x is cheaper at every size measured.
'''
import numpy
from pyscf import lib
from pyscf.lib import logger

CORE_EXCHANGE = 'cd'    # default of mc.core_exchange: 'cd' (all from the factor) or 'cosx' (core-core exchange by COSX)
GRID_LEVEL = 2      # COSX grid of the core-core exchange (spinor_jk.grid_args: heavy-atom radial rule)
# Lebedev sizes of the two inner pruning regions (r < 0.4 R_Bragg) at least these.  The SCF grids' tables (26 / 50
# points there on GRIDX2) integrate the spherical closed core shells of an SCF density, not the anisotropic core
# densities that a fully relaxed CASSCF explores: rotations of core spinors into a fully occupied active pair are
# (nearly) free in the exact functional, and the mixed functional's minimiser walked into a density whose COSX
# core exchange was 0.159 Eh too low on every radial grid (HgCl2 x2c-SVPall CAS(4,6) SO-SCI, Hg 3d partly in the
# active space).  Same density, inner tables (26, 50) -> (50, 110) (+25 % points): error -1.6e-1 -> -1.9e-4 Eh,
# the HF core density's -2.0e-4 unchanged.
INNER_ANGULAR = (50, 110)
SUB_ROWS = 400      # MB of the unpacked AO vectors in the in-pass contraction sum_P L_P W_P


def use_cosx(mc):
    '''whether the core-core exchange of mc comes from COSX: mc.core_exchange (else CORE_EXCHANGE)'''
    m = getattr(mc, 'core_exchange', None)
    if m is None:
        m = CORE_EXCHANGE
    m = str(m).lower()
    if m not in ('cd', 'cosx'):
        raise ValueError('core_exchange %r: cd or cosx' % m)
    return m == 'cosx'


def grid_args(mol, level):
    '''the COSX grid of the core exchange: spinor_jk.grid_args(mol, level) with the inner two pruning regions
    raised to INNER_ANGULAR'''
    from socutils.scf import spinor_jk
    from socutils.dft import grids as sgrids
    args = spinor_jk.grid_args(mol, level)
    ang = args.get('angular_grid', sgrids.Grids.angular_grid)
    table = list(sgrids.ANGULAR_TABLES[int(ang)] if isinstance(ang, (int, numpy.integer)) else ang)
    for i, n in enumerate(INNER_ANGULAR or ()):
        table[i] = max(table[i], n)
    args['angular_grid'] = tuple(table)
    return args


def cosx_object(mc, level=None):
    '''the COSX object of the core exchange (cached on the SCF object): grid of grid_args, overlap fit, the
    basis-representation check of spinor_jk (fit_quality, up to COSX_GRID_RETRIES radial refinements)'''
    from socutils.scf import spinor_jk, cosx
    mf = mc._scf
    mol = mf.mol
    if level is None:
        level = getattr(mc, 'core_exchange_grid', None) or GRID_LEVEL
    cache = getattr(mf, '_core_cosx', None)
    if cache is None or cache[0] is not mol:
        cache = mf._core_cosx = (mol, {})
    key = (level, INNER_ANGULAR)
    if key in cache[1]:
        return cache[1][key]
    k = None
    for it in range(spinor_jk.COSX_GRID_RETRIES + 1):
        args = grid_args(mol, level)
        if it:
            args = dict(args, **spinor_jk.grid_args(mol, level, 2. * it))
            args['angular_grid'] = grid_args(mol, level)['angular_grid']
        k = cosx.COSX(mol, cosx.make_grids(mol, **args), fit=True,
                      tol=min(cosx.TOL, .1 * getattr(mf, 'conv_tol', 1e-9)))
        k.verbose = mf.verbose
        k.stdout = mf.stdout
        k.build()
        lo, hi = spinor_jk.fit_quality(k)
        logger.info(mf, 'core COSX: level %d, angular %s, %d points, fit eigenvalues [%.3g, %.3g]',
                    level, args['angular_grid'], k._weights.size, lo, hi)
        if spinor_jk.COSX_FIT_RANGE[0] < lo and hi < spinor_jk.COSX_FIT_RANGE[1]:
            break
        k = None
    if k is None:
        raise RuntimeError('core_exchange = cosx: no COSX grid of level %d represents the basis' % level)
    # the kernel's per-thread buffers (the pair integrals of a grid block, cosx.COSX.multi_mem) grow as nao^2 blk:
    # nao 3360 (HgCl2)32 on 24 threads needed ~100 GB at blk 112 and failed to allocate with 134 GB free.  Halve
    # the block when the 8 densities of the core K would not fit in half the free memory
    free = max(mol.max_memory - lib.current_memory()[0], 1.)
    per, fixed = k.multi_mem()
    if k.blk > 56 and fixed + 12 * per > .5 * free:
        k.blk = 56
        logger.info(mf, 'core COSX: %.0f MB of kernel buffers at %d points per block, %.0f MB free: 56 points per '
                    'block', fixed + 12 * per, cosx.BLK, free)
    cache[1][key] = k
    return k


def kvar_real(k, xs):
    '''variational COSX K of real matrices xs (m, nao, nao): 1/2 (P K_plain[X] + K_plain[P^T X]) (fit) or
    K_plain[X] (no fit), not symmetrised'''
    xs = numpy.ascontiguousarray(xs)
    if k._proj is None:
        return k.get_k(xs, hermi=0)
    m = len(xs)
    p = k._proj
    ptx = lib.einsum('ki,xkj->xij', p, xs)
    kp = k._plain_k(numpy.concatenate([xs, ptx]))
    return .5 * (lib.einsum('ij,xjk->xik', p, kp[:m]) + kp[m:])


def kvar_ghf(k, mol, dm):
    '''variational COSX K of Hermitian GHF densities dm ((n,) 2nao, 2nao) in the scalar-AO spin-block basis'''
    from socutils.scf import spinor_jk as sj
    dm = numpy.asarray(dm)
    shape = dm.shape
    nao = mol.nao_nr()
    dms = dm.reshape(-1, 2 * nao, 2 * nao)
    n = dms.shape[0]
    aa, ab = dms[:, :nao, :nao], dms[:, :nao, nao:]
    ba, bb = dms[:, nao:, :nao], dms[:, nao:, nao:]
    cplx = numpy.iscomplexobj(dms)
    kramers = sj._is_kramers_even(aa, ab, ba, bb)
    pieces = sj._pieces(aa, ab, ba, bb, cplx, kramers, 1)
    ks = kvar_real(k, sj._stack(pieces))
    ks = sj._project(pieces, list(ks.reshape(len(pieces), n, nao, nao)))
    vk = sj._assemble(pieces, ks, n, nao, dms.dtype, kramers, 1)
    return vk.reshape(shape)


def contract_lw(eri1, wt, nao, sub_mb=None):
    '''Z = sum_P L_P W_P (2nao, k) for a block of packed AO vectors eri1 (nb, npair) and per-P complex
    matrices wt (nb, k, 2nao) (W_P^T, spin-block AO rows alpha first); L_P acts on each spin block'''
    nb, kk = wt.shape[0], wt.shape[1]
    # real columns: (alpha re, alpha im, beta re, beta im) x k, rows (P, nu)
    wc = numpy.empty((nb, nao, 4, kk))
    wc[:, :, 0] = wt[:, :, :nao].real.transpose(0, 2, 1)
    wc[:, :, 1] = wt[:, :, :nao].imag.transpose(0, 2, 1)
    wc[:, :, 2] = wt[:, :, nao:].real.transpose(0, 2, 1)
    wc[:, :, 3] = wt[:, :, nao:].imag.transpose(0, 2, 1)
    wc = wc.reshape(nb, nao, 4 * kk)
    if sub_mb is None:
        sub_mb = SUB_ROWS
    sub = max(1, int(sub_mb * 1e6 / (8 * nao * nao)))
    z = numpy.zeros((nao, 4 * kk))
    for p0 in range(0, nb, sub):
        p1 = min(nb, p0 + sub)
        l2 = lib.unpack_tril(numpy.ascontiguousarray(eri1[p0:p1])).reshape((p1 - p0) * nao, nao)
        # L_P symmetric: rows (P, nu) of the stacked L_P are the columns nu
        lib.dot(l2.T, wc[p0:p1].reshape(-1, 4 * kk), beta=1, c=z)
    z = z.reshape(nao, 4, kk)
    out = numpy.empty((2 * nao, kk), dtype=complex)
    out[:nao] = z[:, 0] + 1j * z[:, 1]
    out[nao:] = z[:, 2] + 1j * z[:, 3]
    return out


def j_density_tril(dm_sph, nao):
    '''packed Re(D_aa + D_bb) with the off-diagonal elements doubled: rho_P = eri1 @ this'''
    d = (dm_sph[:nao, :nao] + dm_sph[nao:, nao:]).real
    d = d + d.T
    d[numpy.diag_indices(nao)] *= .5
    return lib.pack_tril(d)


def mixed_exchange_mo(kx_mo, kl_pa, ncore, ncas):
    '''the core exchange operator M of the mixed functional (MO basis, see the module doc)'''
    nocc = ncore + ncas
    m = numpy.array(kx_mo, dtype=complex)
    m[:, ncore:nocc] = kl_pa
    m[ncore:nocc, nocc:] = kl_pa[nocc:].conj().T
    return m
