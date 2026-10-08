"""Which primitives should stay uncontracted?  Selection by the correlation energy.

All occupied atomic orbitals are contracted (SF-X2C-1e contraction from the X2CAMF
configuration-average SCF) and primitives are released one at a time (greedy), each time
the one that recovers most of the MP2 correlation energy of a chosen group of electrons,
relative to the fully uncontracted set.

Non-canonical MP2 on a fixed reference: the reference determinant is built from the
contracted atomic orbitals themselves (closed shells doubly occupied, the open shell
high-spin), so the occupied orbitals are the same in every candidate set and no SCF is
needed per candidate.  One Fock build and one set of half-transformed integrals
(i mu|j nu) in the uncontracted primitives serve all candidates; for a candidate the
virtual space is the Fock matrix diagonalised in the span of its functions (semicanonical
orbitals), and
    E2 = singles sum |f_ia|^2 / (e_i - e_a)  +  same-spin and opposite-spin doubles.
For a closed-shell atom this is ordinary RMP2 (the contracted orbitals are the HF orbitals).
Spin-free X2C-1e Hamiltonian and integrals from PySCF.

    sel = Selector('S', prims, nfrozen=1)               # correlate everything above 1s
    keep, hist = sel.greedy(group_frozen=5, target=0.99)            # valence 3s3p
    keep, hist = sel.greedy(group_frozen=1, keep=keep, target=0.99) # + 2s2p
"""
import numpy as np
import scipy.linalg

from . import engine, contract, genbas

LNAMES = 'spdfghi'


class Selector:
    def __init__(self, element, prims, nfrozen, gau_nuc=True, verbose=1):
        from pyscf import gto, scf, ao2mo
        self.element, self.verbose, self.gau_nuc = element, verbose, gau_nuc
        self.prims = {l: sorted(v, reverse=True) for l, v in prims.items()}
        nucmod = 'G' if gau_nuc else None
        basis = [[l, [e, 1.0]] for l in sorted(self.prims) for e in self.prims[l]]
        z = engine.charge(element)
        self.mol = mol = gto.M(atom=element, basis={element: basis}, verbose=0, nucmod=nucmod,
                               spin=z % 2)
        nao = mol.nao_nr()
        self.ovlp = mol.intor_symmetric('int1e_ovlp')

        # occupied atomic orbitals = the contracted functions of the occupied-only set
        self.pattern0 = contract.pattern_occupied(element, self.prims, {}, gau_nuc=gau_nuc)
        self._base = contract.contract(element, self.prims, self.pattern0, 'sf',
                                       gau_nuc=gau_nuc, name='T')
        occ_shells = {l: (e, c[:, :self.pattern0[l][0]]) for l, (e, c) in self._base.shells.items()
                      if self.pattern0[l][0] > 0}
        mol_occ = gto.M(atom=element, basis={element: genbas.to_pyscf(genbas.Entry('X:T', '', occ_shells))},
                        verbose=0, nucmod=nucmod, spin=z % 2)
        b_occ = np.linalg.solve(self.ovlp, gto.intor_cross('int1e_ovlp', mol, mol_occ))
        # electrons per (n, l) from the atomic SCF; AO order of mol_occ: l, contraction, m
        nel = {}
        for n, l, two_j, f, e in self._base.scf.occupied():
            nel[(n, l)] = nel.get((n, l), 0.0) + f * (two_j + 1)
        occ_a, occ_b, col = [], [], 0
        for l in sorted(occ_shells):
            for k in range(self.pattern0[l][0]):
                ne = int(round(nel.get((k + l + 1, l), 0.0)))
                na, nb = min(ne, 2 * l + 1), max(ne - (2 * l + 1), 0)      # Hund: alpha first
                occ_a += list(range(col, col + na))
                occ_b += list(range(col, col + nb))
                col += 2 * l + 1
        self.nocc = (len(occ_a), len(occ_b))
        if sum(self.nocc) != z:
            raise RuntimeError('occupation count %s does not match Z = %d' % (self.nocc, z))
        self.nfrozen = nfrozen
        c_occ = [b_occ[:, occ_a], b_occ[:, occ_b]]

        # one Fock build with the reference determinant; semicanonical occupied orbitals
        mf = scf.UHF(mol).sfx2c1e()
        dm = [c @ c.T for c in c_occ]
        hcore = mf.get_hcore()
        veff = mf.get_veff(mol, dm)
        self.e_ref = mf.energy_tot(dm, hcore, veff)
        self.fock = [hcore + veff[0], hcore + veff[1]]
        self.c_occ, self.e_occ = [], []
        for s in (0, 1):
            e, u = np.linalg.eigh(c_occ[s].T @ self.fock[s] @ c_occ[s])
            self.c_occ.append(c_occ[s] @ u)
            self.e_occ.append(e)
        # half-transformed integrals over the active occupied orbitals (i >= nfrozen)
        act = [np.ascontiguousarray(self.c_occ[s][:, nfrozen:]) for s in (0, 1)]
        eye = np.eye(nao)
        self.g = {}
        for s, t in ((0, 0), (0, 1), (1, 1)):
            if act[s].shape[1] == 0 or act[t].shape[1] == 0:
                continue
            g = ao2mo.general(mol, (act[s], eye, act[t], eye), compact=False)
            self.g[(s, t)] = g.reshape(act[s].shape[1], nao, act[t].shape[1], nao)
        self._span_cache = {}
        self._ref_cache = {}

    # -- MP2 in the span of a set of functions given in the primitive AO basis
    def _virtuals(self, b, s):
        """Semicanonical virtual orbitals of spin s in span(b) minus the occupied space."""
        co = self.c_occ[s]
        bv = b - co @ (co.T @ (self.ovlp @ b))
        sv = bv.T @ self.ovlp @ bv
        w, x = np.linalg.eigh(sv)
        x = x[:, w > 1e-8] / np.sqrt(w[w > 1e-8])
        bv = bv @ x
        e, u = np.linalg.eigh(bv.T @ self.fock[s] @ bv)
        return bv @ u, e

    def e2_of_span(self, b, group_frozen=None):
        nf = self.nfrozen
        i0 = 0 if group_frozen is None else group_frozen - nf
        cv, ev = zip(*(self._virtuals(b, s) for s in (0, 1)))
        e2 = 0.0
        for s in (0, 1):                                      # singles
            eo = self.e_occ[s][nf + i0:]
            if len(eo) == 0:
                continue
            fia = self.c_occ[s][:, nf + i0:].T @ self.fock[s] @ cv[s]
            e2 += np.sum(fia ** 2 / (eo[:, None] - ev[s][None, :]))
        for (s, t), g in self.g.items():                      # doubles
            eo_s, eo_t = self.e_occ[s][nf + i0:], self.e_occ[t][nf + i0:]
            if len(eo_s) == 0 or len(eo_t) == 0:
                continue
            tmp = np.tensordot(g[i0:, :, i0:, :], cv[t], axes=([3], [0]))      # (i mu | j b)
            for i in range(len(eo_s)):
                iajb = np.tensordot(cv[s].T, tmp[i], axes=([1], [0])).transpose(1, 0, 2)   # (j, a, b)
                d = (eo_s[i] + eo_t[:, None, None]) - ev[s][None, :, None] - ev[t][None, None, :]
                if s == t:
                    e2 += 0.5 * np.sum(iajb * (iajb - iajb.transpose(0, 2, 1)) / d)
                else:
                    e2 += np.sum(iajb * iajb / d)
        return e2

    def e2_reference(self, group_frozen=None):
        """MP2 energy of the group in the fully uncontracted set."""
        if group_frozen not in self._ref_cache:
            self._ref_cache[group_frozen] = self.e2_of_span(np.eye(self.mol.nao_nr()), group_frozen)
        return self._ref_cache[group_frozen]

    def span(self, keep):
        """Contracted functions (occupied orbitals + the primitives `keep` = {l: [numbers
        counted from the smallest exponent]}) in the primitive AO basis."""
        from pyscf import gto
        key = tuple(sorted((l, tuple(sorted(v))) for l, v in keep.items() if v))
        if key in self._span_cache:
            return self._span_cache[key]
        shells = {}
        for l, (exps, coef) in self._base.shells.items():
            n, nc = len(exps), self.pattern0[l][0]
            if nc == 0:
                shells[l] = (exps, coef)
                continue
            cols = [coef[:, :nc]]
            for k in sorted(keep.get(l, [])):
                col = np.zeros((n, 1))
                col[n - k] = 1.0
                cols.append(col)
            shells[l] = (exps, np.hstack(cols))
        molc = gto.M(atom=self.element, basis={self.element: genbas.to_pyscf(genbas.Entry('X:T', '', shells))},
                     verbose=0, nucmod='G' if self.gau_nuc else None, spin=self.mol.spin)
        b = np.linalg.solve(self.ovlp, gto.intor_cross('int1e_ovlp', self.mol, molc))
        self._span_cache[key] = b
        return b

    def fraction(self, keep, group_frozen=None):
        return self.e2_of_span(self.span(keep), group_frozen) / self.e2_reference(group_frozen)

    def greedy(self, group_frozen=None, keep=None, target=0.99, max_add=40, ls=None):
        """Release primitives one at a time until `target` of the uncontracted MP2 energy of
        the group (orbitals above group_frozen) is recovered.  Returns (keep, history)."""
        keep = {l: list(v) for l, v in (keep or {}).items()}
        ls = [l for l in sorted(self.prims) if self.pattern0[l][0] > 0] if ls is None else ls
        ref = self.e2_reference(group_frozen)
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
            if best is None:
                break
            frac, l, k = best
            keep.setdefault(l, []).append(k)
            hist.append((l, k, self.prims[l][-k], frac))
            if self.verbose:
                print('   + %s #%-2d (%.5g)  -> %.4f' % (LNAMES[l], k, self.prims[l][-k], frac), flush=True)
        return {l: sorted(v) for l, v in keep.items()}, hist


def keep_string(keep):
    return ';'.join('%s:%s' % (LNAMES[l], ','.join(str(k) for k in sorted(v))) for l, v in sorted(keep.items()) if v)
