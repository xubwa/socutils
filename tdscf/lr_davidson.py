#
# Author Xubo Wang <wangxubo0201@outlook.com>
#
# Paired-trial-vector Davidson solver for the (complex) RPA / linear-response
# eigenvalue problem
#
#     K z = omega z,   K = [[A, B], [-B*, -A*]],   A^H = A, B^T = B,
#
# following Olsen, Jensen, Jorgensen, J. Comput. Phys. 74, 265 (1988): every
# trial vector b = (x, y) enters the subspace together with its CT/paired
# partner p(b) = (y*, x*).  The projected pencil then inherits the full RPA
# structure -- the reduced Hamiltonian is the RPA Hessian and the reduced
# metric is the indefinite k-metric -- so the subspace problem is itself a
# small RPA problem, roots come in +/- pairs, and the Krein-space metric is
# represented correctly [cf. Furche & Chen, JCP 163, 174104 (2025); Z. Li,
# arXiv:2009.01136].
#
# Only matvecs with K on the b vectors are needed: the partner's matvec is
# free, since K p(b) = -p(K b)  (equivalently H p(b) = p(H b) with H = kK).
#
# ---------------------------------------------------------------------------
# Difference from pyscf.tdscf._lr_eig.eig (basis for a future pyscf PR)
# ---------------------------------------------------------------------------
# pyscf's _lr_eig.eig is the *same* Olsen paired Davidson; it is NOT a generic
# non-Hermitian solver.  The one substantive difference is the ROOT SELECTION.
# pyscf's pickeig (tdscf/ghf.py, tdscf/rhf.py) keeps an eigenpair when
#
#     |Im(omega)| < REAL_EIG_THRESHOLD  and  Re(omega) > positive_eig_threshold
#
# i.e. it is blind to the Krein norm  eta = |X|^2 - |Y|^2.  For a complex
# (relativistic / 2-component / GHF) reference the indefinite-metric subspace
# breeds spurious "eta-neutral" ghost Ritz roots with small positive real
# eigenvalues; pickeig admits them, they sort into the lowest-nroots window and
# displace the true (eta-positive, particle-branch) roots, so the run either
# never converges or aborts with "Not enough eigenvalues".  The physical
# excitations are exactly the eta-positive roots [Furche & Chen 2025; Li 2020],
# so the fix is to select on eta, not just on sign(Re omega), which this solver
# does *during* the iteration (eta_filter).
#
# Decisive evidence (cc-pVDZ H2O, 2c spinor B3LYP, identical lr_eig, only the
# pick changed): default pickeig admits a ghost root 0.3166 Ha and misses a
# true root -> err 1.3e-2 Ha; the same pickeig with an extra `eta > 0` test ->
# err 8.4e-10 Ha.  For real references the ghosts rarely appear (real RKS-TDDFT
# reproduces a dense Casida solve to ~1e-12 eV), which is why pickeig has been
# adequate there.  The minimal pyscf fix is ~5 lines inside pickeig.
#

import numpy as np
import scipy.linalg


def paired_eig(matvec, hdiag, nroots=3, x0=None, conv_tol=1e-6,
               max_cycle=100, max_space=None, pos_tol=1e-3, eta_filter=True,
               verbose=False):
    '''Solve K z = omega z for the lowest `nroots` positive real eigenvalues.

    Args:
        matvec : callable(zs) -> K zs, zs of shape (nz, 2n)
        hdiag : real array (2n,), diagonal approximation of K,
            i.e. hstack(e_ia, -e_ia)
        pos_tol : roots below this threshold are considered spurious
        eta_filter : if True, reject candidate Ritz vectors whose Krein norm
            eta = |X|^2 - |Y|^2 is not positive (the eta-neutral/antiparticle
            ghosts of the indefinite-metric problem).  Setting it False mimics
            an eta-blind 'positive real' selection (e.g. pyscf's pickeig).

    Returns:
        conv (bool array, nroots), e (real array, nroots, ascending),
        zs (complex array (nroots, 2n), normalized to z^H k z = 1,
        i.e. |X|^2 - |Y|^2 = 1), nmatvec
    '''
    hdiag = np.asarray(hdiag).real.ravel()
    n2 = hdiag.size
    n = n2 // 2
    # Track a few buffer roots beyond nroots so that roots discovered late
    # (e.g. after a subspace collapse) are not silently missed.
    ntrack = min(nroots + 3, n)

    def pair(z):
        '''CT partner: (x, y) -> (y*, x*).'''
        return np.concatenate([z[..., n:].conj(), z[..., :n].conj()], axis=-1)

    def kmul(z):
        w = np.array(z, copy=True)
        w[..., n:] *= -1
        return w

    if max_space is None:
        max_space = min(max(8*ntrack, 24), max(n, 2*ntrack))

    if x0 is None:
        ng = min(ntrack+2, n)
        idx = np.argsort(hdiag[:n])[:ng]
        x0 = np.zeros((ng, n2), dtype=complex)
        x0[np.arange(ng), idx] = 1.0
    else:
        x0 = np.atleast_2d(np.asarray(x0, dtype=complex))

    def ortho_add(Bmat, news):
        '''Orthonormalize candidate rows against the span of {b_i, p(b_i)}
        (and against each other plus their pairs); return accepted rows.'''
        out = []
        for d in news:
            nrm = np.linalg.norm(d)
            if nrm < 1e-14:
                continue
            d = d / nrm
            for _ in range(2):          # double MGS for stability
                if len(Bmat):
                    d = d - Bmat.T.dot(Bmat.conj().dot(d))
                    P = pair(Bmat)
                    d = d - P.T.dot(P.conj().dot(d))
                for a in out:
                    d = d - a*np.dot(a.conj(), d)
                    pa = pair(a)
                    d = d - pa*np.dot(pa.conj(), d)
            nrm = np.linalg.norm(d)
            if nrm > 1e-6:
                out.append(d/nrm)
        return out

    B = np.zeros((0, n2), dtype=complex)     # trial vectors (pairs implicit)
    U = np.zeros((0, n2), dtype=complex)     # K @ B
    news = x0
    conv = np.zeros(nroots, dtype=bool)
    e = np.zeros(nroots)
    zs = np.zeros((nroots, n2), dtype=complex)
    nmatvec = 0

    for icyc in range(max_cycle):
        add = ortho_add(B, news)
        if not add:
            break                            # subspace exhausted
        add = np.asarray(add)
        u = np.asarray(matvec(add)).reshape(len(add), n2)
        nmatvec += len(add)
        B = np.vstack([B, add])
        U = np.vstack([U, u])
        m = len(B)

        # Reduced pencil on the paired subspace W = [b_i, p(b_i)]:
        #   H w rows: H b = k(K b),  H p(b) = p(H b)
        W = np.vstack([B, pair(B)])
        HW = np.vstack([kmul(U), pair(kmul(U))])
        Hred = W.conj().dot(HW.T)            # inherits the RPA Hessian form
        Sred = W.conj().dot(kmul(W).T)       # inherits the k-metric form
        w, c = scipy.linalg.eig(Hred, Sred)

        finite = np.isfinite(w)
        realpos = (finite
                   & (np.abs(w.imag) <= 1e-6*np.maximum(1.0, np.abs(w.real)))
                   & (w.real > pos_tol))
        ncomplex = int(np.count_nonzero(finite & ~realpos
                                        & (np.abs(w.imag) > 1e-6)))
        cand = np.where(realpos)[0]
        cand = cand[np.argsort(w.real[cand])]

        zlist, klist, elist = [], [], []
        for i in cand:
            ci = c[:, i]
            z = W.T.dot(ci)
            eta = np.real(np.vdot(z, kmul(z)))   # Krein-metric norm |X|^2-|Y|^2
            if eta_filter and eta < 1e-10:
                continue                         # eta-neutral/negative ghost
            s = 1.0/np.sqrt(abs(eta)) if abs(eta) > 1e-14 else 1e7
            zlist.append(z*s)
            klist.append(kmul(HW.T.dot(ci))*s)   # = K z
            elist.append(w.real[i])
            if len(zlist) == ntrack:
                break
        nfound = len(zlist)
        if nfound == 0:
            news = x0
            continue
        zk = np.asarray(zlist)
        Kz = np.asarray(klist)
        ek = np.asarray(elist)
        resid = Kz - ek[:, None]*zk
        rnorm = np.linalg.norm(resid, axis=1)
        nret = min(nroots, nfound)
        conv[:] = False
        conv[:nret] = rnorm[:nret] < conv_tol
        e[:nret] = ek[:nret]
        zs[:nret] = zk[:nret]
        if verbose:
            print('paired davidson %3d  m=%3d  e=%s  |r|max=%.2e  nmv=%d%s'
                  % (icyc+1, m, np.array2string(ek, precision=6), rnorm.max(),
                     nmatvec, ('  (%d complex roots in subspace!)' % ncomplex)
                     if ncomplex else ''))
        # Converge the requested roots, but also require the tracked buffer
        # roots' residuals to be reasonably small so that a root discovered
        # late cannot displace an already "converged" higher root.
        if (nfound >= nroots and conv[:nroots].all()
                and (rnorm[nroots:] < max(conv_tol*1e3, 1e-4)).all()):
            break

        news = []
        for k in range(nfound):
            if rnorm[k] < conv_tol:
                continue
            den = hdiag - ek[k]
            den = np.where(np.abs(den) > 1e-4, den, 1e-4)
            news.append(resid[k]/den)
        if not news:                         # found < ntrack, all converged
            news = list(x0)
        if m + len(news) > max_space:        # collapse to current Ritz vectors
            B = np.zeros((0, n2), dtype=complex)
            U = np.zeros((0, n2), dtype=complex)
            news = list(zk) + news

    return conv, e, zs, nmatvec
