'''Locality of the short-range (erfc) density fit -- the measurement behind HANDOFF Open item 2.

The SR kernel erfc(w r)/r is short-ranged, so its fitting problem is local and can be studied on a finite cluster
cut from the crystal (molecular integrals with omega < 0), without the Gamma-point restriction that lattice-summed
auxiliary functions cannot resolve a domain larger than the cell.  For a cluster of radius R_cl around a central atom:

  1. metric decay:      max |(M_SR^-1)_PQ| over aux Q at distance d from the central atom's aux P;
  2. coefficient decay: C = M_SR^-1 (P|mn)_SR for the AO pairs (m on the central atom, n within r_pair):
                        max |C_P,mn| and the norm fraction of C by distance of P from the pair's atoms;
  3. local-fit error:   for each pair, the SR self-repulsion error of the fit restricted to the aux functions within R
                        of the pair's atoms (a pair-specific domain, PARI; and a domain shared by all pairs of the
                        central atom), err(R) = (mn|mn) - B_D^T M_DD^-1 B_D >= 0 (variational, exact 4c reference),
                        against the global fit err(inf).  The fit error of (mn|ls) is bounded by sqrt(err_mn err_ls).

usage: python sr_locality.py SYSTEM CENTRE R_CLUSTER [OMEGA ...]
  SYSTEM: a name of cells.SYSTEMS or a CIF; CENTRE: element symbol of the central atom (first such atom); bases from
  the production rules (Eu ETB0T10, Y DYVDZ-SF, O PVDZ-X2C, Si/Cl PVDZ_X2C); aux: auxgen uncontracted tau 1e-8.'''
import os, sys, time, runpy, numpy as np, scipy.linalg
from pyscf import gto
from socutils.pbc.bench import cells
from socutils.pbc.bench.auxgen_util import auxgen_from_basis

BOHR = 0.52917721092
RULES = '/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/basis_rules.py'


def cluster(system, centre, rcl):
    A, atoms = cells.from_cif(system) if system.endswith('.cif') else cells.SYSTEMS[system]
    A = np.asarray(A)
    x0 = [x for e, x in atoms if e == centre][0]
    n = int(np.ceil(rcl / min(np.linalg.norm(A, axis=1)))) + 1
    T = np.array([[i, j, k] for i in range(-n, n + 1) for j in range(-n, n + 1) for k in range(-n, n + 1)]) @ A
    out = [(e, x + t - x0) for t in T for e, x in atoms if np.linalg.norm(x + t - x0) < rcl]
    out.sort(key=lambda a: np.linalg.norm(a[1]))
    return out


def main(system, centre, rcl, omegas):
    B = runpy.run_path(RULES)['BASIS']
    atoms = cluster(system, centre, rcl)
    els = sorted(set(e for e, _ in atoms))
    t = time.time()
    aux = auxgen_from_basis({e: B[e] for e in els}, contract=False, prune_lmax=False, threshold=1e-8)
    mol = gto.M(atom=[(e, tuple(x)) for e, x in atoms], basis={e: B[e] for e in els}, unit='A', spin=None,
                verbose=0, nucmod='G')
    auxm = gto.M(atom=[(e, tuple(x)) for e, x in atoms], basis={e: aux[e] for e in els}, unit='A', spin=None,
                 verbose=0)
    X = mol.atom_coords()
    pat = np.repeat([auxm.bas_atom(i) for i in range(auxm.nbas)], np.diff(auxm.ao_loc_nr()))
    aoat = np.repeat([mol.bas_atom(i) for i in range(mol.nbas)], np.diff(mol.ao_loc_nr()))
    dist = np.linalg.norm(X[:, None] - X[None], axis=2) * BOHR       # A
    print('%s: cluster of %d atoms (R %.1f A), nao %d, naux %d (auxgen %.0f s)' % (
        system, mol.natm, rcl, mol.nao, auxm.nao, time.time() - t), flush=True)
    # pairs: m on the central atom (0), n on atoms within 3 A of it
    near = np.where(dist[0] < float(os.environ.get('SRLOC_RPAIR', '3.0')))[0]
    msh = [i for i in range(mol.nbas) if mol.bas_atom(i) == 0]
    nsh = [i for i in range(mol.nbas) if mol.bas_atom(i) in set(near)]
    bins = np.array([0, 1e-3, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 8, 10, 99])
    for w in omegas:
        t = time.time()
        with auxm.with_range_coulomb(-w):
            M = auxm.intor('int2c2e')
        M = .5 * (M + M.T)
        prune = float(os.environ.get('SRLOC_PRUNE', '0'))
        if prune > 0:
            # pivoted Cholesky of the SR metric: keep the aux functions that are not (numerically) linear
            # combinations of the others (residual diagonal > prune * max diagonal)
            from scipy.linalg import lapack
            # on the unit-diagonal metric (the aux self-repulsions span many orders of magnitude, tight
            # functions of heavy atoms dominate an unscaled tolerance)
            dg = 1. / np.sqrt(M.diagonal())
            _, piv, rank, info = lapack.dpstrf(M * dg[:, None] * dg[None], lower=1, tol=prune)
            sel = np.sort(piv[:rank] - 1)
            print('   pruned aux (dpstrf on the normalised metric, tol %.0e): %d -> %d' % (prune, len(M), rank), flush=True)
        else:
            sel = np.arange(len(M))
        M = M[np.ix_(sel, sel)]
        e, v = scipy.linalg.eigh(M)
        keep = e > 1e-12 * e.max()
        Minv = (v[:, keep] / e[keep]) @ v[:, keep].T
        tm = time.time() - t
        # 1. metric decay from the central atom's aux functions
        patw = pat[sel]
        p0 = np.where(patw == 0)[0]
        dP = dist[0][patw]
        mi = np.abs(Minv[p0]).max(axis=0)
        print('\n omega %.2f: metric eigenvalues [%.1e, %.1e], kept %d/%d (%.0f s)' % (w, e[0], e.max(), keep.sum(), len(e), tm))
        print('   |M^-1|(centre aux, Q) max by distance of Q (A):',
              '  '.join('%s %.1e' % ('%.1f-%.1f' % (bins[i], bins[i + 1]), mi[(dP >= bins[i]) & (dP < bins[i + 1])].max() / mi.max())
                        for i in range(len(bins) - 1) if ((dP >= bins[i]) & (dP < bins[i + 1])).any()))
        # 2./3. pairs
        pm = gto.mole.conc_mol(mol, auxm)
        mlo = mol.ao_loc_nr()
        n0 = min(nsh)
        with pm.with_range_coulomb(-w):
            Bt = pm.intor('int3c2e', shls_slice=(msh[0], msh[-1] + 1, n0, max(nsh) + 1, mol.nbas, pm.nbas))
        m_idx = np.arange(mlo[msh[0]], mlo[msh[-1] + 1])
        n_all = np.arange(mlo[n0], mlo[max(nsh) + 1])
        nsel = np.isin(aoat[n_all], near)
        Bt = Bt[:, nsel]
        n_idx = n_all[nsel]
        Bp = Bt.reshape(-1, auxm.nao).T[sel]                              # (naux, npair), pair k = im * nn + jn
        pa_m = np.repeat(aoat[m_idx], len(n_idx)); pa_n = np.tile(aoat[n_idx], len(m_idx))
        # exact SR self-repulsion (mn|mn) of every pair (4c, omega < 0)
        posm = {i: k for k, i in enumerate(m_idx)}
        posn = {j: k for k, j in enumerate(n_idx)}
        selfrep = np.zeros(Bp.shape[1])
        with mol.with_range_coulomb(-w):
            for ish in msh:
                for jsh in nsh:
                    eri = mol.intor('int2e', shls_slice=(ish, ish + 1, jsh, jsh + 1, ish, ish + 1, jsh, jsh + 1))
                    d = np.einsum('ijij->ij', eri)
                    for a, i in enumerate(range(mlo[ish], mlo[ish + 1])):
                        for b2, j in enumerate(range(mlo[jsh], mlo[jsh + 1])):
                            if j in posn:
                                selfrep[posm[i] * len(n_idx) + posn[j]] = d[a, b2]
        big = selfrep > 1e-6 * selfrep.max()
        # C = M^-1 B in chunks of pairs (memory); statistics by distance bin
        nb = len(bins) - 1
        bmax, bsq = np.zeros(nb), np.zeros(nb)
        err_g = np.zeros(Bp.shape[1])
        chunk = max(1, int(4e9 / 8 / Minv.shape[0]))
        for k0 in range(0, Bp.shape[1], chunk):
            k1 = min(Bp.shape[1], k0 + chunk)
            Cc = Minv @ Bp[:, k0:k1]
            err_g[k0:k1] = selfrep[k0:k1] - np.einsum('Pk,Pk->k', Bp[:, k0:k1], Cc)
            dPpair = np.minimum(dist[pa_m[k0:k1]][:, patw], dist[pa_n[k0:k1]][:, patw]).T
            for i in range(nb):
                msk = (dPpair >= bins[i]) & (dPpair < bins[i + 1])
                if msk.any():
                    bmax[i] = max(bmax[i], np.abs(Cc[msk]).max())
                    bsq[i] += (Cc[msk] ** 2).sum()
            del Cc, dPpair
        line = ['%.1f-%.1f %.1e/%.1e' % (bins[i], bins[i + 1], bmax[i] / bmax.max(), np.sqrt(bsq[i] / bsq.sum()))
                for i in range(nb) if bsq[i] > 0]
        print('   C = M^-1 (P|mn): max|C|/max and norm fraction by distance of P from the pair atoms (A):')
        print('     ' + '  '.join(line))
        print('   global SR fit: err(mn|mn) max %.2e  rel to (mn|mn) max %.2e  (%d pairs)' % (
            err_g[big].max(), (err_g[big] / selfrep[big]).max(), big.sum()))
        # 3. local fits
        for R in (0., 2.5, 3.5, 4.5, 5.5, 6.5):
            if R > rcl - 1.5:
                break
            err_pair = np.zeros(Bp.shape[1])
            npd = 0
            for A_ in set(pa_n):                      # pair-specific domain: aux within R of 0 or of A_
                cols = np.where(pa_n == A_)[0]
                D = np.where((dist[0][patw] <= R + 1e-6) | (dist[A_][patw] <= R + 1e-6))[0]
                npd = max(npd, len(D))
                ee, vv = scipy.linalg.eigh(M[np.ix_(D, D)])
                kk = ee > 1e-12 * ee.max()
                ch = max(1, int(2e9 / 8 / max(len(D), 1)))
                for c0 in range(0, len(cols), ch):
                    cc = cols[c0:c0 + ch]
                    x = vv[:, kk].T @ Bp[np.ix_(D, cc)]
                    err_pair[cc] = selfrep[cc] - np.einsum('ik,ik->k', x, x / ee[kk][:, None])
            Ds = np.where(dist[0][patw] <= R + 3.0 + 1e-6)[0]   # shared domain: aux within R + 3 A of the centre
            ee, vv = scipy.linalg.eigh(M[np.ix_(Ds, Ds)])
            kk = ee > 1e-12 * ee.max()
            vk_, ek_ = vv[:, kk], ee[kk]
            err_sh = np.empty(Bp.shape[1])
            ch = max(1, int(2e9 / 8 / max(len(Ds), 1)))
            for k0 in range(0, Bp.shape[1], ch):
                x = vk_.T @ Bp[Ds, k0:k0 + ch]
                err_sh[k0:k0 + ch] = selfrep[k0:k0 + ch] - np.einsum('ik,ik->k', x, x / ek_[:, None])
            print('   domain R = %.1f A: pair-specific (PARI) err max %.2e rel %.2e (%d aux max) | shared (R+3 A, %d aux) err max %.2e rel %.2e' % (
                R, err_pair[big].max(), (err_pair[big] / selfrep[big]).max(), npd, len(Ds),
                err_sh[big].max(), (err_sh[big] / selfrep[big]).max()), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), [float(x) for x in sys.argv[4:]] or [0.5])
