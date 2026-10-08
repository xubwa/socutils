"""Which primitives should stay uncontracted?  Selection by the correlation energy.

For a closed-shell atom and an uncontracted set, all occupied orbitals are contracted and
primitives are released one at a time (greedy), each time the one that recovers most of the
MP2 correlation energy of a chosen group of electrons, relative to the fully uncontracted
set.  Spin-free X2C-1e RHF + MP2 with PySCF.

The occupied orbitals are the same in every contracted set (they are contracted functions),
so one SCF in the uncontracted set and one half-transformed integral array (i mu|j nu)
suffice: for a candidate set the virtual space is the Fock matrix diagonalised in the span
of the contracted functions, and its MP2 energy takes a second.

    sel = Selector('Yb', prims, nfrozen=18)          # correlate everything above orbital 18
    keep, hist = sel.greedy(group_frozen=23, target=0.99)
"""
import numpy as np
import scipy.linalg

from . import engine, contract, genbas

LNAMES = 'spdfghi'


class Selector:
    def __init__(self, element, prims, nfrozen, gau_nuc=True, verbose=1):
        from pyscf import gto, scf, ao2mo
        self.element, self.verbose = element, verbose
        self.prims = {l: sorted(v, reverse=True) for l, v in prims.items()}
        self.gau_nuc = gau_nuc
        basis = [[l, [e, 1.0]] for l in sorted(self.prims) for e in self.prims[l]]
        self.mol = gto.M(atom=element, basis={element: basis}, verbose=0,
                         nucmod='G' if gau_nuc else None)
        mf = scf.RHF(self.mol).sfx2c1e()
        mf.conv_tol = 1e-10
        mf.kernel()
        if not mf.converged:
            raise RuntimeError('RHF did not converge')
        self.e_hf = mf.e_tot
        self.nocc = self.mol.nelectron // 2
        self.nfrozen = nfrozen
        self.fock = mf.get_fock()
        self.ovlp = mf.get_ovlp()
        self.mo_occ = mf.mo_coeff[:, :self.nocc]
        self.e_occ = mf.mo_energy[:self.nocc]
        nao = self.mol.nao_nr()
        co = np.ascontiguousarray(self.mo_occ[:, nfrozen:])
        eye = np.eye(nao)
        g = ao2mo.general(self.mol, (co, eye, co, eye), compact=False)
        na = co.shape[1]
        self.g = g.reshape(na, nao, na, nao)             # (i mu | j nu)
        # atomic occupied shells per l, for the contraction
        self.pattern0 = contract.pattern_occupied(element, self.prims, {}, gau_nuc=gau_nuc)
        self._entry_cache = {}
        self.e2_ref = self.e2_of_span(np.eye(nao))

    # -- MP2 in the span of a set of functions given in the primitive AO basis
    def e2_of_span(self, b, group_frozen=None):
        """b: (nao, nb) functions; the span must contain the occupied orbitals.  Returns the
        MP2 energies {frozen: E2} for the stored group and, if asked, a smaller one."""
        s = b.T @ self.ovlp @ b
        f = b.T @ self.fock @ b
        e, c = scipy.linalg.eigh(f, s)
        cv = b @ c[:, self.nocc:]
        ev = e[self.nocc:]
        nf = self.nfrozen
        eo = self.e_occ[nf:]
        na = len(eo)
        i0 = 0 if group_frozen is None else group_frozen - nf
        e2 = 0.0
        tmp = np.tensordot(self.g[i0:], cv, axes=([3], [0]))        # (i mu | j b)
        for i in range(na - i0):
            iajb = np.tensordot(cv.T, tmp[i][:, i0:, :], axes=([1], [0]))   # (a, j, b)
            iajb = iajb.transpose(1, 0, 2)                                    # (j, a, b)
            d = (eo[i0 + i] + eo[i0:, None, None]) - ev[None, :, None] - ev[None, None, :]
            e2 += np.sum(iajb * (2 * iajb - iajb.transpose(0, 2, 1)) / d)
        return e2

    def span(self, keep):
        """Contracted functions (SF contraction of the occupied orbitals + the primitives
        `keep` = {l: [numbers counted from the smallest exponent]}) in the primitive AO basis."""
        from pyscf import gto
        key = tuple(sorted((l, tuple(sorted(v))) for l, v in keep.items() if v))
        if key in self._entry_cache:
            return self._entry_cache[key]
        if not hasattr(self, '_base'):
            self._base = contract.contract(self.element, self.prims, self.pattern0, 'sf',
                                           gau_nuc=self.gau_nuc, name='T')
        shells = {}
        for l, (exps, coef) in self._base.shells.items():
            n = len(exps)
            nc = self.pattern0[l][0]
            if nc == 0:
                shells[l] = (exps, coef)
                continue
            cols = [coef[:, :nc]]
            for k in sorted(keep.get(l, [])):
                col = np.zeros((n, 1)); col[n - k] = 1.0
                cols.append(col)
            shells[l] = (exps, np.hstack(cols))
        ent = genbas.Entry('X:T', '', shells)
        molc = gto.M(atom=self.element, basis={self.element: genbas.to_pyscf(ent)}, verbose=0,
                     nucmod='G' if self.gau_nuc else None)
        suc = gto.intor_cross('int1e_ovlp', self.mol, molc)
        b = np.linalg.solve(self.ovlp, suc)
        self._entry_cache[key] = b
        return b

    def fraction(self, keep, group_frozen=None):
        ref = self.e2_ref if group_frozen is None else self.e2_of_span(np.eye(self.mol.nao_nr()), group_frozen)
        return self.e2_of_span(self.span(keep), group_frozen) / ref

    def greedy(self, group_frozen=None, keep=None, target=0.99, max_add=40, ls=None):
        """Release primitives one at a time until `target` of the uncontracted MP2 energy of
        the group (orbitals above group_frozen) is recovered.  Returns (keep, history)."""
        keep = {l: list(v) for l, v in (keep or {}).items()}
        ls = [l for l in sorted(self.prims) if self.pattern0[l][0] > 0] if ls is None else ls
        ref = self.e2_ref if group_frozen is None else self.e2_of_span(np.eye(self.mol.nao_nr()), group_frozen)
        hist = []
        frac = self.e2_of_span(self.span(keep), group_frozen) / ref
        for _ in range(max_add):
            if frac >= target:
                break
            best = None
            for l in ls:
                for k in range(1, len(self.prims[l]) + 1):
                    if k in keep.get(l, []):
                        continue
                    trial = {**keep, l: keep.get(l, []) + [k]}
                    fr = self.e2_of_span(self.span(trial), group_frozen) / ref
                    if best is None or fr > best[0]:
                        best = (fr, l, k)
            frac, l, k = best
            keep.setdefault(l, []).append(k)
            hist.append((l, k, self.prims[l][-k], frac))
            if self.verbose:
                print('   + %s #%-2d (%.5g)  -> %.4f' % (LNAMES[l], k, self.prims[l][-k], frac), flush=True)
        return {l: sorted(v) for l, v in keep.items()}, hist


def keep_string(keep):
    return ';'.join('%s:%s' % (LNAMES[l], ','.join(str(k) for k in sorted(v))) for l, v in sorted(keep.items()) if v)
