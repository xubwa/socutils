'''Stage 2 of the THC exchange (HANDOFF Open item 2): one-centre projection of the partners of the compact functions.

A compact function c on atom A times any function x (a CX pair) is confined to the compact region of A, where x is
smooth.  Projecting x onto the functions k of A in that region,
    x ~= sum_k a_xk phi_k,      a = argmin int w (x - sum_k a_k phi_k)^2,   w(r) = sum_{c on A} phi_c(r)^2 / |phi_c|^2
(atomic grid at A), turns every CX pair into one-centre pairs, rho_cx ~= sum_k a_xk rho_ck: the local fit, the SR
three-centre integrals and the LR rows of the CX pairs then only exist for the one-centre pairs (linear storage).
Here the projection is applied to the level-2 pieces (THCExchange.build_common: J3, local C, LR rows) and K (level-2
form) and J (RI-J of the total densities) are compared with the unprojected ones for a minao density.

usage: proj_check.py SYSTEM ALPHA_C [NRAD NANG]'''
import sys, time, runpy, numpy as np, scipy.linalg
from pyscf import gto, dft, lib
from pyscf.pbc import scf as pscf, gto as pgto
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import split_basis, RULES
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc.bench.hybrid_k import _inv_metric, k_from_factors
from socutils.pbc.thc import THCExchange

name, ac = sys.argv[1], float(sys.argv[2])
nrad, nang = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (100, 302)
import os
EXTRA = [tuple(float(v) if i > 1 else int(v) for i, v in enumerate(x.split(':')))
         for x in os.environ.get('PROJ_EXTRA', '').split(',') if x]              # e.g. "4:6:0.3:30,6:8:0.2:60"
PROJ_ONLY = bool(os.environ.get('PROJ_ONLY'))
sysm = cells.from_cif(name) if name.endswith('.cif') else cells.SYSTEMS[name]
if os.environ.get('SUPERCELL'):
    sysm = cells.supercell(sysm, [int(x) for x in os.environ['SUPERCELL'].split(',')])
B = dict(runpy.run_path(RULES)['BASIS'])
full = cells.make_cell(sysm, 1e-10, nucmod='G', basis=B, max_memory=150000)
cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=split_basis(B, ac), max_memory=150000)
els = sorted(set(full.atom_symbol(i) for i in range(full.natm)))
aux = {e: auxgen_from_basis({e: B[e]}, contract=False, prune_lmax=False, threshold=1e-8)[e] for e in els}
dm0 = pscf.RHF(full).get_init_guess(key='minao')
T = np.linalg.lstsq(cell.pbc_intor('int1e_ovlp'), pgto.intor_cross('int1e_ovlp', cell, full), rcond=1e-12)[0]
dm = T @ dm0 @ T.T
t = time.time()
th = THCExchange(cell, aux, ac, selector='kmeans')     # no point selection needed here
th.build_common()
print('%s alpha_c %g: split nao %d, CX pairs %d of %d (%.0f s)' % (name, ac, cell.nao, th.cx.sum(), len(th.cx), time.time() - t),
      flush=True)
nao = cell.nao
aoat = th.aoat
hard = np.zeros(nao, bool)
hard[np.setdiff1d(np.arange(nao), th.smooth)] = True
ti, tj = th.ti, th.tj
pidx = np.full((nao, nao), -1)
pidx[ti, tj] = np.arange(len(ti))
pidx[tj, ti] = np.arange(len(ti))

# projection coefficients per atom with compact functions: a^A (n_A, nao), A's functions k
t = time.time()
proj = {}
for A in (range(cell.natm) if not os.environ.get('CUT') else []):
    cA = np.where(hard & (aoat == A))[0]
    if len(cA) == 0:
        continue
    el = cell.atom_symbol(A)
    m = gto.M(atom=[(el, (0., 0., 0.))], basis={el: B[gto.mole._std_symbol(el)]}, unit='B', verbose=0, spin=None)
    g = dft.gen_grid.Grids(m)
    g.atom_grid = {el: (nrad, nang)}
    g.prune = None
    g.build()
    phi = cell.pbc_eval_gto('GTOval_sph', g.coords + cell.atom_coord(A))      # (npt, nao)
    pc = phi[:, cA]
    w = g.weights * np.einsum('ri,ri->r', pc, pc / np.einsum('ri,ri,r->i', pc, pc, g.weights))
    kA = np.where(aoat == A)[0]
    S = (phi[:, kA] * w[:, None]).T @ phi[:, kA]
    bk = (phi[:, kA] * w[:, None]).T @ phi                                    # (n_A, nao)
    e, v = scipy.linalg.eigh(S)
    keep = e > 1e-10 * e.max()
    a = (v[:, keep] / e[keep]) @ v[:, keep].T @ bk                            # (n_A, nao)
    res = phi - phi[:, kA] @ a
    # richer projector sets: A's functions + even-tempered primitives (l <= L, n exponents in [emin, emax])
    far_ = aoat != A
    for spec in EXTRA:
        L_, n_, e0, e1 = spec
        exps = np.geomspace(e0, e1, n_)
        pm = gto.M(atom=[(el, (0., 0., 0.))], basis={el: [[l, [x, 1.]] for l in range(L_ + 1) for x in exps]},
                   unit='B', verbose=0, spin=None)
        chi = np.hstack([phi[:, kA], pm.eval_gto('GTOval_sph', g.coords)])
        chi = chi / np.sqrt(np.einsum('ri,ri,r->i', chi, chi, w))          # equilibrated (unit weighted norm)
        S2 = (chi * w[:, None]).T @ chi
        e2, v2 = scipy.linalg.eigh(S2)
        k2 = e2 > 1e-13 * e2.max()
        a2 = (v2[:, k2] / e2[k2]) @ v2[:, k2].T @ ((chi * w[:, None]).T @ phi)
        r2 = phi - chi @ a2
        # absolute residual in the compact region, relative to the largest function there (what the pair
        # densities phi_c x carry into K and J)
        nrm = np.sqrt(np.einsum('ri,ri,r->i', phi, phi, w))
        rabs = np.sqrt(np.einsum('ri,ri,r->i', r2, r2, w))
        nf = nrm[far_].max()
        sig = far_ & (nrm >= 1e-2 * nf)
        # by the distance of x's atom from A (minimum image, A)
        dA = np.array([min(np.linalg.norm(cell.atom_coord(aoat[i]) - cell.atom_coord(A) + t_)
                           for t_ in np.array([[a_, b_, c_] for a_ in (-1, 0, 1) for b_ in (-1, 0, 1) for c_ in (-1, 0, 1)])
                           @ cell.lattice_vectors()) for i in range(nao)]) * 0.52917721092
        for r0, r1 in ((0.01, 2.), (2., 3.), (3., 4.), (4., 99.)):
            selr = sig & (dA >= r0) & (dA < r1)
            if selr.any():
                print('        x at %.0f-%.0f A (%d significant): |res|/|x| max %.1e, median %.1e'
                      % (r0, r1, selr.sum(), (rabs[selr] / nrm[selr]).max(), np.median(rabs[selr] / nrm[selr])))
        print('     + l <= %d, %d exponents [%g, %g] (%d projectors, %d kept): |res|/max|x_far|: max %.1e, 99%% %.1e; '
              'significant x (%d): |res|/|x| max %.1e, median %.1e' % (L_, n_, e0, e1, chi.shape[1], k2.sum(),
              (rabs[far_] / nf).max(), np.quantile(rabs[far_] / nf, .99), sig.sum(), (rabs[sig] / nrm[sig]).max(),
              np.median(rabs[sig] / nrm[sig])), flush=True)
    nrm0 = np.sqrt(np.einsum('ri,ri,r->i', phi, phi, w))
    err = np.sqrt(np.einsum('ri,ri,r->i', res, res, w)) / nrm0[aoat != A].max()
    far = aoat != A
    proj[A] = (kA, a)
    print('  atom %d %s: %d compact, %d one-centre functions; projection error of the other functions in the compact '
          'region (|res| / max|x_far|): max %.1e, median %.1e' % (A, el, len(cA), len(kA), err[far].max(), np.median(err[far])),
          flush=True)
print('projection %.0f s' % (time.time() - t), flush=True)
if PROJ_ONLY:
    sys.exit()
if os.environ.get('CUT'):
    # drop the CX pairs whose partner atom is beyond R (minimum image between the two atoms of the pair)
    A_ = np.asarray(cell.lattice_vectors())
    Ts = np.array([[a_, b_, c_] for a_ in (-1, 0, 1) for b_ in (-1, 0, 1) for c_ in (-1, 0, 1)]) @ A_
    X_ = cell.atom_coords()
    dat = np.array([[np.linalg.norm(X_[j] - X_[i] + Ts, axis=1).min() for j in range(cell.natm)]
                    for i in range(cell.natm)]) * 0.52917721092
    ew, U = np.linalg.eigh(th.M)
    Mh = (U * np.sqrt(np.maximum(ew, 0))) @ U.T
    v, e = _inv_metric(th.M)
    W = (v / np.sqrt(e)).T
    ss = th.ss

    def k_l2(J3, C, Llr):
        Y = Mh @ C
        Lss = Llr.copy(); Lss[:, th.cx] = 0.
        L = W @ J3
        Ls = np.zeros_like(L); Ls[:, ss] = L[:, ss]
        return (k_from_factors(Ls, Ls, dm, nao) + k_from_factors(C, J3, dm, nao) + k_from_factors(J3, C, dm, nao)
                - k_from_factors(Y, Y, dm, nao) + k_from_factors(Llr, Llr, dm, nao) - k_from_factors(Lss, Lss, dm, nao))

    def j_ri(J3, Llr):
        d = (dm + dm.T)[ti, tj]; d[ti == tj] *= .5
        L = W @ J3
        jp = L.T @ (L @ d) + Llr.T @ (Llr @ d)
        Jm = np.zeros((nao, nao)); Jm[ti, tj] = jp; Jm[tj, ti] = jp
        return Jm

    th.Y = th.phi = th.r = None                          # memory: only J3, C, LR rows are needed
    K0, J0 = k_l2(th.J, th.C, th.Llr), j_ri(th.J, th.Llr)
    pd = dat[aoat[ti], aoat[tj]]
    # decreasing R: the dropped set only grows, so the columns are zeroed in place (no copies)
    for R in sorted([float(x) for x in os.environ['CUT'].split(',')], reverse=True):
        drop = th.cx & (pd > R)
        th.J[:, drop] = 0.; th.C[:, drop] = 0.; th.Llr[:, drop] = 0.
        dK = T.T @ (k_l2(th.J, th.C, th.Llr) - K0) @ T
        dJ = T.T @ (j_ri(th.J, th.Llr) - J0) @ T
        print('R_cut %.1f A: CX pairs kept %d of %d; K max %.2e E_x %.2e Eh; J max %.2e E_J %.2e Eh'
              % (R, (th.cx & ~drop).sum(), th.cx.sum(), abs(dK).max(), -.25 * np.einsum('ij,ji', dm0, dK),
                 abs(dJ).max(), .5 * np.einsum('ij,ji', dm0, dJ)), flush=True)
    sys.exit()

# projected pair basis: column (c, x), c compact on A, x not on A -> sum_k a_xk column (c, k)
def project(F):
    G = F.copy()
    for p in np.where(th.cx)[0]:
        a_, b_ = ti[p], tj[p]
        c, x = (a_, b_) if hard[a_] else (b_, a_)
        A = aoat[c]
        if aoat[x] == A:
            continue
        kA, aa = proj[A]
        G[:, p] = F[:, pidx[c, kA]] @ aa[:, x]
    return G

t = time.time()
J3, C, Llr = th.J, th.C, th.Llr
J3p, Cp, Llrp = project(J3), project(C), project(Llr)
print('projected columns %.0f s' % (time.time() - t), flush=True)
ew, U = np.linalg.eigh(th.M)
Mh = (U * np.sqrt(np.maximum(ew, 0))) @ U.T
v, e = _inv_metric(th.M)
W = (v / np.sqrt(e)).T
ss = th.ss


def level2_k(J3, C, Llr):
    Y = Mh @ C
    Lss = Llr.copy(); Lss[:, th.cx] = 0.
    L = W @ J3
    Ls = np.zeros_like(L); Ls[:, ss] = L[:, ss]
    return (k_from_factors(Ls, Ls, dm, nao) + k_from_factors(C, J3, dm, nao) + k_from_factors(J3, C, dm, nao)
            - k_from_factors(Y, Y, dm, nao) + k_from_factors(Llr, Llr, dm, nao) - k_from_factors(Lss, Lss, dm, nao))


def ri_j(J3, Llr):
    d = (dm + dm.T)[ti, tj]; d[ti == tj] *= .5
    L = W @ J3
    jp = L.T @ (L @ d) + Llr.T @ (Llr @ d)
    Jm = np.zeros((nao, nao)); Jm[ti, tj] = jp; Jm[tj, ti] = jp
    return Jm


K2, K2p = level2_k(J3, C, Llr), level2_k(J3p, Cp, Llrp)
J2, J2p = ri_j(J3, Llr), ri_j(J3p, Llrp)
f = lambda K: T.T @ K @ T
print('projected - level 2: K max %.2e, E_x %.2e Eh; J max %.2e, E_J %.2e Eh (max |K| %.1e)'
      % (abs(f(K2p - K2)).max(), -.25 * np.einsum('ij,ji', dm0, f(K2p - K2)), abs(f(J2p - J2)).max(),
         .5 * np.einsum('ij,ji', dm0, f(J2p - J2)), abs(f(K2)).max()), flush=True)
