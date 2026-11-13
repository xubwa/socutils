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
'''
import numpy
from pyscf import lib
from pyscf.lib import logger

CORE_EXCHANGE = 'cd'    # default of mc.core_exchange: 'cd' (all from the factor) or 'cosx' (core-core exchange by COSX)
GRID_LEVEL = 2      # COSX grid of the core-core exchange (spinor_jk.grid_args: heavy-atom radial rule)
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


def cosx_object(mc, level=None):
    '''the COSX object of the core exchange (cached on the SCF object, shared with its spinor COSX)'''
    from socutils.scf import spinor_jk
    mf = mc._scf
    if level is None:
        level = getattr(mc, 'core_exchange_grid', None) or GRID_LEVEL
    st = spinor_jk._state(mf, mf.mol)
    k = spinor_jk._kobj(mf, st, level)
    if k is None:
        raise RuntimeError('core_exchange = cosx: no COSX grid of level %d represents the basis' % level)
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
