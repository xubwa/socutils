'''Level 2 of the compact/smooth exchange ladder (HANDOFF Open item 2): compact pairs fitted locally, smooth pairs
by a global fit, at the Gamma point; compared with level 1 (the global RSDF fit) and level 0 (exact SR, RSJK).

The basis is split at alpha_c (sr_locality.split_basis: compact = primitives >= alpha_c, smooth = the rest, each
part the span of its contraction columns), so every function is either compact or smooth and a pair p = (m n) is
SS (both smooth) or CX (at least one compact).  In RSDF lr='gspace' the kernel is
    v' = erfc(w r)/r  +  LR rows (exact, G != 0)  -  c0 S S,
and only the SR (erfc) part is fitted, so the levels differ only there.  With J = (P|p)_SR (lattice sums at Gamma,
RSDF keep_raw) and M = (P|Q)_SR:

  level 1:  (p|q) = J_p^T M^-1 J_q                                   all aux, all pairs
  level 2:  SS-SS  J_p^T M^-1 J_q                                     global fit (level 3 replaces it by THC)
            CX-SS  C_p^T J_q            C_p = M_DD^-1 J_D,p,  D = aux on the two atoms of p (Bloch sums)
            CX-CX  C_p^T J_q + J_p^T C_q - C_p^T M C_q                (robust: error quadratic in the fit errors)
  level 0:  exact SR (RSJK, SR kernel only, no dd exclusion)

(The SS fit restricted to the smooth aux, exponent < 2 alpha_c, is reported as a diagnostic.)  The CX-SS term uses the exact three-centre integrals on the SS side, as level 3 (THC for SS) will; the
CX treatment is shared.  K^SR is compared in the split basis for a minao density (and for any dm passed in).

usage: hybrid_k.py SYSTEM ALPHA_C [ALPHA_C ...]'''
import os, sys, time, runpy, numpy as np, scipy.linalg
from pyscf import lib
from pyscf.pbc import scf as pscf
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import split_basis, RULES
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc.rsdf import RSDF

OMEGA = 0.5


def _inv_metric(M, lindep=1e-12):
    '''M^-1 on the kept space of the unit-diagonal metric (relative eigenvalue cut).'''
    d = 1. / np.sqrt(M.diagonal())
    e, v = scipy.linalg.eigh(M * d[:, None] * d[None])
    k = e > lindep * e.max()
    v = v[:, k] * d[:, None]
    return v, e[k]                                   # M^-1 = v diag(1/e) v^T


def k_from_factors(A, Bf, dm, nao):
    '''K_ab = sum_P sum_cd A_P[a c] dm[c d] B_P[b d] for packed real factors A, B (nP, npair).'''
    K = np.zeros((nao, nao), dtype=np.result_type(dm, A))
    blk = max(1, int(2e9 / 8 / (nao * nao * 3)))
    for p0 in range(0, len(A), blk):
        a = lib.unpack_tril(A[p0:p0 + blk])
        b = a if Bf is A else lib.unpack_tril(Bf[p0:p0 + blk])
        K += np.einsum('pac,cd,pbd->ab', a, dm, b, optimize=True)
    return K


def build(cell, aux):
    r = RSDF(cell, aux, omega=OMEGA, precision=1e-10, lr='gspace')
    r.keep_raw = True
    r.cderi_in_memory = True
    r.verbose = 0
    r.build()
    return r


def levels(cell, aux, ac, dm):
    '''K^SR of levels 1 and 2 (split-basis cell) for the density dm.'''
    t = time.time()
    r = build(cell, aux)
    nao = cell.nao
    J = r.int3c_sr                                   # (naux, npair) raw SR 3c
    M = r.metric_sr
    naux = len(M)
    nkept = r.cderi.shape[0] - r.nlr
    K1 = k_from_factors(r.cderi[:nkept], r.cderi[:nkept], dm, nao)
    print('   RSDF (split cell): nao %d, naux %d, SR kept %d, LR rows %d (%.0f s)' % (nao, naux, nkept, r.nlr, time.time() - t),
          flush=True)
    t = time.time()
    # classes
    aohard = np.repeat([cell.bas_exp(i).min() >= ac for i in range(cell.nbas)], np.diff(cell.ao_loc_nr()))
    aoat = np.repeat([cell.bas_atom(i) for i in range(cell.nbas)], np.diff(cell.ao_loc_nr()))
    ti, tj = np.tril_indices(nao)
    cx = aohard[ti] | aohard[tj]
    from pyscf.df import addons as df_addons
    auxmol = df_addons.make_auxmol(cell, aux)        # the order of RSDF's cell-0 aux functions
    pexp = np.repeat([auxmol.bas_exp(i).min() for i in range(auxmol.nbas)], np.diff(auxmol.ao_loc_nr()))
    pat = np.repeat([auxmol.bas_atom(i) for i in range(auxmol.nbas)], np.diff(auxmol.ao_loc_nr()))
    assert len(pexp) == naux, (len(pexp), naux)
    ss = np.where(~cx)[0]
    ew, U = scipy.linalg.eigh(M)
    Mh = (U * np.sqrt(np.maximum(ew, 0))) @ U.T

    def k_ss(sel):
        # SS-SS pairs, global fit with the aux functions sel
        v, e = _inv_metric(M[np.ix_(sel, sel)])
        Ls = np.zeros((len(e), J.shape[1]))
        Ls[:, ss] = (v / np.sqrt(e)).T @ J[np.ix_(sel, ss)]
        return k_from_factors(Ls, Ls, dm, nao), len(e)

    def k_cx(C):
        # pairs with a compact function: one-sided C^T J against SS, robust between CX pairs
        Y = Mh @ C
        return k_from_factors(C, J, dm, nao) + k_from_factors(J, C, dm, nao) - k_from_factors(Y, Y, dm, nao)

    # CX: local fit on the pair's two atoms; and the global fit of the same pairs (diagnostic)
    C = np.zeros_like(J)
    pa, pb = aoat[ti], aoat[tj]
    for A in range(cell.natm):
        for B in range(A + 1):
            cols = np.where(cx & (((pa == A) & (pb == B)) | ((pa == B) & (pb == A))))[0]
            if len(cols) == 0:
                continue
            D = np.where((pat == A) | (pat == B))[0]
            vd, ed = _inv_metric(M[np.ix_(D, D)])
            C[np.ix_(D, cols)] = (vd / ed) @ (vd.T @ J[np.ix_(D, cols)])
    vg, eg = _inv_metric(M)
    Cg = np.zeros_like(J)
    cxi = np.where(cx)[0]
    Cg[:, cxi] = (vg / eg) @ (vg.T @ J[:, cxi])
    sm = np.where(pexp < 2 * ac)[0]
    Kss_sm, nsm = k_ss(sm)
    Kss_all, _ = k_ss(np.arange(naux))
    Kcx_loc, Kcx_glob = k_cx(C), k_cx(Cg)
    K2 = Kss_all + Kcx_loc
    diag = {'global SS + global CX (= level 1)': Kss_all + Kcx_glob,
            'smooth-aux SS + global CX': Kss_sm + Kcx_glob,
            'smooth-aux SS + local CX': Kss_sm + Kcx_loc}
    print('   level 2: smooth aux %d of %d (kept %d), CX pairs %d of %d (%.0f s)'
          % (len(sm), naux, nsm, cx.sum(), len(cx), time.time() - t), flush=True)
    for lab, K in diag.items():
        dK = K - K1
        print('   [%s] - level 1: max %.2e, d(E_x) %.2e Eh' % (lab, abs(dK).max(), -.25 * np.einsum('ij,ji', dm, dK)), flush=True)
    return K1, K2, r


def exact_sr(cell, dm, precision=None):
    '''Exact SR K (RSJK, SR kernel only, no dd exclusion).  precision: cell.precision for the image range and
    RSJK's direct_scf_tol (pyscf derives ~1e-15 from 1e-10; the level differences need far less).'''
    from pyscf.pbc.scf.rsjk import RangeSeparatedJKBuilder
    t = time.time()
    c = cell.copy()
    c.omega = -OMEGA
    if precision:
        c.precision = precision
    rs = RangeSeparatedJKBuilder(c, kpts=np.zeros((1, 3)))
    rs.exclude_dd_block = False
    rs.allow_drv_nodddd = False
    rs.verbose = 0
    rs.build(omega=OMEGA)
    vk = rs._get_jk_sr(dm[None, None], hermi=1, kpts=np.zeros((1, 3)), with_j=False, with_k=True)[-1]
    print('   exact SR K (RSJK, precision %.0e, direct_scf_tol %.1e): %.0f s'
          % (c.precision, rs.direct_scf_tol, time.time() - t), flush=True)
    return np.asarray(vk).reshape(dm.shape).real


def main(name, acs):
    sysm = cells.from_cif(name) if name.endswith('.cif') else cells.SYSTEMS[name]
    B = dict(runpy.run_path(RULES)['BASIS'])
    full = cells.make_cell(sysm, 1e-10, nucmod='G', basis=B, max_memory=150000)
    els = sorted(set(full.atom_symbol(i) for i in range(full.natm)))
    aux = {e: auxgen_from_basis({e: B[e]}, contract=False, prune_lmax=False, threshold=1e-8)[e] for e in els}
    dm0 = pscf.RHF(full).get_init_guess(key='minao')
    print('%s: %d atoms, nao %d' % (name, full.natm, full.nao), flush=True)
    # exact SR K once, in the original basis (every split basis spans it: K_full = T^T K_split T), cached
    K0 = None
    if not os.environ.get('HYB_NO_EXACT'):
        f = os.environ.get('HYB_EXACT_FILE', 'exact_%s.npy' % os.path.basename(name).split('.')[0])
        if os.path.exists(f):
            K0 = np.load(f)
            print('   exact SR K read from %s' % f, flush=True)
        else:
            prec = float(os.environ.get('HYB_EXACT_PREC', '1e-8'))
            K0 = exact_sr(full, dm0, prec)
            np.save(f, K0)
            if os.environ.get('HYB_EXACT_CHECK'):          # the same at pyscf's default precision
                Kt = exact_sr(full, dm0)
                print('   exact at precision %.0e vs default: max %.2e, d(E_x) %.2e Eh'
                      % (prec, abs(K0 - Kt).max(), -.25 * np.einsum('ij,ji', dm0, K0 - Kt)), flush=True)
    for ac in acs:
        cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=split_basis(B, ac), max_memory=150000)
        # exact transform of the density to the split basis: chi_full = chi_split T
        S_ss = cell.pbc_intor('int1e_ovlp')
        from pyscf.pbc import gto as pgto
        S_sf = pgto.intor_cross('int1e_ovlp', cell, full)
        T = np.linalg.lstsq(S_ss, S_sf, rcond=1e-12)[0]          # (nao_split, nao_full)
        resid = np.einsum('ij,ji->', T.T @ S_ss @ T - full.pbc_intor('int1e_ovlp'), np.eye(full.nao))
        dm = T @ dm0 @ T.T
        print('\n== alpha_c %g: split nao %d (representation check |tr(T^T S T - S)| %.1e)' % (ac, cell.nao, abs(resid)),
              flush=True)
        K1, K2, r = levels(cell, aux, ac, dm)
        K1, K2 = T.T @ K1 @ T, T.T @ K2 @ T                     # to the original basis
        if K0 is not None:
            for lab, K in (('level 1 (global fit)', K1), ('level 2 (compact local)', K2)):
                dK = K - K0
                print('   %s: max |K - K_exact| %.2e, |dK|_F %.2e, d(E_x) = -1/4 tr(D dK) %.2e Eh'
                      % (lab, abs(dK).max(), np.linalg.norm(dK), -.25 * np.einsum('ij,ji', dm0, dK)), flush=True)
        dK = K2 - K1
        print('   level 2 - level 1: max %.2e, d(E_x) %.2e Eh (max |K| %.2e)'
              % (abs(dK).max(), -.25 * np.einsum('ij,ji', dm0, dK), abs(K1).max()), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], [float(x) for x in sys.argv[2:]])
