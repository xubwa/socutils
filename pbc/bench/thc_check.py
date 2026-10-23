'''Validation of level 3 (pbc/thc.py: THC for the smooth-smooth pairs + level-2 local fit for the compact pairs)
against level 2 (global SS fit + local CX), level 1 (global RSDF) and level 0 (exact SR, RSJK), minao density.

For every alpha_c and N_mu (multiples of the number of smooth functions; pivoted-Cholesky prefixes) it reports
  - the per-pair ISDF interpolation error of sampled SS pair densities on the grid (max / median),
  - the THC (P|q)_SR of the SS pairs against the exact three-centre integrals,
  - the SS-SS block of K: THC and the level-2 global fit against the exact SS-SS block (RSJK on the smooth
    functions alone + the exact LR rows of the SS pairs),
  - the full K: level 3 - level 2, and levels 3 / 2 / 1 against exact (SR part, original basis),
with E_x = -1/4 tr(D dK), storage and timings.

usage: thc_check.py SYSTEM ALPHA_C[,ALPHA_C...] NMU_FACTOR[,NMU_FACTOR...]   (env THC_EXACT_FILE: cached exact SR K,
       original basis, as hybrid_k writes it)'''
import os, sys, time, runpy, numpy as np
from pyscf.pbc import scf as pscf, gto as pgto
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import split_basis, RULES
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc.bench.hybrid_k import _inv_metric, k_from_factors, exact_sr
from socutils.pbc.thc import THCExchange


def ex(dm, dK):
    return -.25 * np.einsum('ij,ji', dm, dK)


def main(name, acs, facs):
    sysm = cells.from_cif(name) if name.endswith('.cif') else cells.SYSTEMS[name]
    B = dict(runpy.run_path(RULES)['BASIS'])
    full = cells.make_cell(sysm, 1e-10, nucmod='G', basis=B, max_memory=150000)
    els = sorted(set(full.atom_symbol(i) for i in range(full.natm)))
    aux = {e: auxgen_from_basis({e: B[e]}, contract=False, prune_lmax=False, threshold=1e-8)[e] for e in els}
    dm0 = pscf.RHF(full).get_init_guess(key='minao')
    print('%s: %d atoms, nao %d' % (name, full.natm, full.nao), flush=True)
    f = os.environ.get('THC_EXACT_FILE')
    if os.environ.get('THC_NO_EXACT'):
        K0 = None
    elif f and os.path.exists(f):
        K0 = np.load(f)
    else:
        K0 = exact_sr(full, dm0, 1e-8)
        if f:
            np.save(f, K0)
    for ac in acs:
        cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=split_basis(B, ac), max_memory=150000)
        T = np.linalg.lstsq(cell.pbc_intor('int1e_ovlp'), pgto.intor_cross('int1e_ovlp', cell, full), rcond=1e-12)[0]
        dm = T @ dm0 @ T.T
        print('\n== alpha_c %g: split nao %d' % (ac, cell.nao), flush=True)
        th = THCExchange(cell, aux, ac).build_common()
        nao, sm, ss = th.nao, th.smooth, th.ss
        # level 2 (and the pieces of level 1) from the same RSDF raw data
        J, M, C, Y = th.J, th.M, th.C, th.Y
        v, e = _inv_metric(M)
        Ls = np.zeros((len(e), J.shape[1]))
        Ls[:, ss] = (v / np.sqrt(e)).T @ J[:, ss]
        Kdf_ss_sr = k_from_factors(Ls, Ls, dm, nao)
        Kcx_loc = k_from_factors(C, J, dm, nao) + k_from_factors(J, C, dm, nao) - k_from_factors(Y, Y, dm, nao)
        Llr = th.Llr
        Lss = np.zeros_like(Llr)
        Lss[:, ss] = Llr[:, ss]
        Klr_all = k_from_factors(Llr, Llr, dm, nao)
        Klr_ss = k_from_factors(Lss, Lss, dm, nao)
        K1_sr = k_from_factors(th.r.cderi[:th.nkept], th.r.cderi[:th.nkept], dm, nao)
        K2_sr = Kdf_ss_sr + Kcx_loc
        # exact SS-SS block: RSJK on the smooth functions alone (same shells, same order) + LR rows of SS pairs
        cs = cells.make_cell(sysm, 1e-10, nucmod='G', basis=split_basis(B, ac, keep='smooth'), max_memory=150000)
        Sdiff = abs(cs.pbc_intor('int1e_ovlp') - cell.pbc_intor('int1e_ovlp')[np.ix_(sm, sm)]).max()
        dss = dm[np.ix_(sm, sm)]
        Kex_ss = exact_sr(cs, dss, 1e-8) + Klr_ss[np.ix_(sm, sm)]
        Kdf_ss = (Kdf_ss_sr + Klr_ss)[np.ix_(sm, sm)]
        print('   smooth cell check |S_s - S_split[s,s]| %.1e; level-2 SS-SS block vs exact: max %.2e, E_x %.2e Eh'
              % (Sdiff, abs(Kdf_ss - Kex_ss).max(), ex(dss, Kdf_ss - Kex_ss)), flush=True)
        for lab, Kl in (('level 1', K1_sr), ('level 2', K2_sr)) if K0 is not None else ():
            d = T.T @ Kl @ T - K0
            print('   %s vs exact (SR, original basis): max %.2e, E_x %.2e Eh' % (lab, abs(d).max(), ex(dm0, d)), flush=True)
        ns = len(sm)
        for fac in facs:
            nmu = int(round(fac * ns))
            if nmu > len(th.piv):
                print('   N_mu %d > %d pivots, skipped' % (nmu, len(th.piv)))
                continue
            rank = th.build_thc(nmu)
            K3 = th.get_k(dm)
            K3_sr = K3 - Klr_all
            Pmu = th.phimu
            Kthc_ss = Pmu.T @ ((Pmu @ dss @ Pmu.T) * th.V) @ Pmu
            Jt = th.j_thc_ss()
            dJ = Jt - J[:, ss]
            d30 = T.T @ K3_sr @ T - K0 if K0 is not None else np.full((1, 1), np.nan)
            d32 = K3_sr - K2_sr
            print('   N_mu %d (%.1f x nao_s, rank %d): pair err max %.1e med %.1e | (P|SS) THC vs exact max %.1e (max |J| %.1e) | '
                  'SS block THC vs exact: max %.2e E_x %.2e | level 3 - level 2: max %.2e E_x %.2e | level 3 vs exact: E_x %.2e Eh'
                  % (nmu, fac, rank, th.pair_err[0], th.pair_err[1], abs(dJ).max(), abs(J[:, ss]).max(),
                     abs(Kthc_ss - Kex_ss).max(), ex(dss, Kthc_ss - Kex_ss), abs(d32).max(), ex(dm, d32), ex(dm0, d30) if K0 is not None else np.nan),
                  flush=True)
            st = th.storage()
            print('      storage (doubles): %s' % ', '.join('%s %.2e' % kv for kv in st.items()))
            print('      timings (s): %s' % ', '.join('%s %.1f' % kv for kv in th.timings.items()), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], [float(x) for x in sys.argv[2].split(',')], [float(x) for x in sys.argv[3].split(',')])
