'''
Active-space Hamiltonian in the N, N+-1, N+-2 electron sectors.

    H_act = sum_tu heff_tu t^+ u + 1/2 sum_tuvw (tu|vw) t^+ v^+ w u

Each sector is diagonalized once, densely; lambda_k are the eigenvalues of
H_act without E_core (conventions.md section 8).  The same object applies
single creation / annihilation operators to batches of CI vectors, from which
all operator strings of the perturber classes are composed.
'''

import numpy

from socutils.zmr.active import strings as st
from socutils.zmr.active import symmetry
from socutils.zmr.lib import blocks as lb
from socutils.zmr.lib import kernels
from socutils.zmr.lib import linalg as la

_BATCH = 2**22  # max number of matrix-element contributions per batch


class Sector:
    '''Eigen-decomposition of H_act for a fixed electron number.

    Attributes:
        nelec: number of active electrons
        lam: (dim,) eigenvalues of H_act (E_core not included)
        vecs: (dim, dim) eigenvectors as columns, in string order
    '''

    def __init__(self, nelec, lam, vecs):
        self.nelec = nelec
        self.lam = lam
        self.vecs = vecs

    @property
    def dim(self):
        return len(self.lam)


class SectorSpace:
    '''Dense sector Hamiltonians of an active space, their block structure
    and spectra.  A sector is named by its electron number (an int for
    spinors, (n_alpha, n_beta) for spin-restricted orbitals); subclasses
    provide dim, target and make_hmat.'''

    def __init__(self):
        self._sectors = {}
        self._hmat = {}
        self._blocks = {}

    def dim(self, nelec):
        raise NotImplementedError

    def target(self, nelec, dn):
        '''Sector reached from nelec by an operator string that changes the
        electron number by dn.'''
        raise NotImplementedError

    def make_hmat(self, nelec):
        raise NotImplementedError

    def hmat(self, nelec):
        '''Dense H_act in the nelec sector (cached).'''
        if nelec not in self._hmat:
            self._hmat[nelec] = self.make_hmat(nelec)
        return self._hmat[nelec]

    def apply_h(self, nelec, q):
        '''H_act q for vectors of the nelec sector given as the columns of
        q (dim, m).'''
        return la.gemm(self.hmat(nelec), q)

    def blocks(self, nelec):
        '''(labels, nblocks): connected components of H_act in the nelec
        sector (cached); one block unless the Hamiltonian has exact zeros.'''
        if nelec not in self._blocks:
            self._blocks[nelec] = lb.components(self.hmat(nelec))
        return self._blocks[nelec]

    def has_sector(self, nelec):
        return nelec in self._sectors

    def sector(self, nelec):
        '''Full eigen-decomposition of H_act for nelec electrons (cached).'''
        if nelec not in self._sectors:
            lam, vecs = lb.eigh(self.hmat(nelec), *self.blocks(nelec))
            self._sectors[nelec] = Sector(nelec, lam, vecs)
        return self._sectors[nelec]

    def solve(self, nelec, nroots, full=True):
        '''Lowest nroots CASCI states: (total energies, CI vectors as columns).
        full=False computes only these roots unless the full spectrum of the
        sector is already available or cheap (block diagonal Hamiltonian).'''
        if full or self.has_sector(nelec) or self.blocks(nelec)[1] > 1:
            sec = self.sector(nelec)
            return sec.lam[:nroots] + self.e_core, sec.vecs[:, :nroots]
        lam, vecs = la.eigh_lowest(self.hmat(nelec), nroots)
        return lam + self.e_core, vecs


class ActiveSpace(SectorSpace):
    '''Active-space Hamiltonian (heff, aaaa) plus the constant E_core.

    charges: symmetry charges of the active orbitals (zmr.active.symmetry).
    The integrals that violate charge conservation are set to zero (their
    largest element is kept in symmetry_error), which makes the sector
    Hamiltonians exactly block diagonal; the blocks are found from the zero
    pattern and every sector is diagonalized block by block.'''

    # Sectors of up to dense_limit determinants keep a dense Hamiltonian.
    # Larger ones are matrix free: H_act is only applied to vectors
    # (apply_h), the model states come from a Lanczos solver, and neither the
    # full spectrum nor the block structure is available (the RDM route of
    # PC-NEVPT2 needs neither).
    dense_limit = 20000

    def is_dense(self, nelec):
        return self.dim(nelec) <= self.dense_limit

    def apply_h(self, nelec, q):
        if self.is_dense(nelec):
            return SectorSpace.apply_h(self, nelec, q)
        out = kernels.sigma(self.norb, nelec, self.dim(nelec), self.heff, self.aaaa, q)
        if out is None:
            raise NotImplementedError('matrix-free sectors need libzmr (run make)')
        return out

    def solve(self, nelec, nroots, full=True):
        if self.is_dense(nelec):
            return SectorSpace.solve(self, nelec, nroots, full)
        from scipy.sparse.linalg import LinearOperator, eigsh
        dim = self.dim(nelec)
        op = LinearOperator((dim, dim), dtype=numpy.complex128,
                            matvec=lambda x: self.apply_h(nelec, x.reshape(dim, 1))[:, 0],
                            matmat=lambda x: self.apply_h(nelec, x))
        lam, vecs = eigsh(op, k=nroots, which='SA', tol=1e-12,
                          ncv=min(dim, max(4 * nroots, 40)))
        order = numpy.argsort(lam)
        return lam[order] + self.e_core, numpy.ascontiguousarray(vecs[:, order])

    def __init__(self, heff, aaaa, e_core=0.0, charges=None):
        SectorSpace.__init__(self)
        self.norb = heff.shape[0]
        if self.norb > st.MAX_ORB:
            raise NotImplementedError('more than %d active orbitals' % st.MAX_ORB)
        self.heff = numpy.asarray(heff, dtype=numpy.complex128)
        self.aaaa = numpy.asarray(aaaa, dtype=numpy.complex128).reshape((self.norb,) * 4)
        self.e_core = float(e_core)
        self.charges = None
        self.symmetry_error = 0.0
        if charges is not None:
            self.charges = numpy.asarray(charges, dtype=float)
            self.symmetry_error = symmetry.violation(self.heff, self.aaaa, self.charges)
            self.heff, self.aaaa = symmetry.project(self.heff, self.aaaa, self.charges)
        self._tables = {}

    @classmethod
    def from_eris(cls, eris):
        return cls(*eris.active_hamiltonian(), charges=getattr(eris, 'active_charges', None))

    def target(self, nelec, dn):
        return nelec + dn

    def dim(self, nelec):
        return len(st.gen_strings(self.norb, nelec))

    def table(self, nelec, dagger):
        key = (nelec, dagger)
        if key not in self._tables:
            self._tables[key] = st.single_op_table(self.norb, nelec, dagger)
        return self._tables[key]

    # ------------------------------------------------------------------
    # operator actions

    def apply_des(self, vecs, nelec):
        '''vecs (..., dim_N) -> out (..., norb, dim_{N-1}),
        out[..., p, :] = a_p vecs'''
        return self._apply(vecs, nelec, False)

    def apply_cre(self, vecs, nelec):
        '''vecs (..., dim_N) -> out (..., norb, dim_{N+1}),
        out[..., p, :] = a_p^+ vecs'''
        return self._apply(vecs, nelec, True)

    def _apply(self, vecs, nelec, dagger):
        target, sign = self.table(nelec, dagger)
        ndst = self.dim(nelec + (1 if dagger else -1))
        return kernels.apply_op(target, sign, vecs, ndst)

    def apply_strings(self, vecs, nelec, blocks):
        '''Operator strings on CI vectors: vecs (..., dim_N) ->
        out (..., nX, dim_target), out[..., X, :] = O_X vecs.

        blocks: list of (daggers, ops); every block holds the strings
        c_1 ... c_n with c_k = a^+_{ops[x, k]} if daggers[k] else a_{ops[x, k]}
        (the rightmost operator acts first).  All blocks must lead to the
        same sector; their functions are concatenated along X.'''
        vecs = numpy.asarray(vecs)
        lead = vecs.shape[:-1]
        ntarget = {nelec + sum(1 if d else -1 for d in dag) for dag, _ in blocks}
        if len(ntarget) != 1:
            raise ValueError('operator blocks change the electron number differently')
        ops = [numpy.asarray(o, dtype=numpy.int64).reshape(len(o), len(dag)) for dag, o in blocks]
        nx = sum(len(o) for o in ops)
        v2 = numpy.ascontiguousarray(vecs, dtype=numpy.complex128).reshape(-1, vecs.shape[-1])
        out = numpy.empty((v2.shape[0], nx, self.dim(ntarget.pop())), dtype=numpy.complex128)
        x0 = 0
        for (dag, _), o in zip(blocks, ops):
            kernels.apply_strings(self.norb, nelec, dag, o, v2, out, x0)
            x0 += len(o)
        return out.reshape(lead + out.shape[1:])

    # ------------------------------------------------------------------
    # Hamiltonian

    def make_hmat(self, nelec):
        '''Dense H_act in the nelec sector (string order).'''
        if not self.is_dense(nelec):
            raise NotImplementedError(
                'sector with %d electrons has %d determinants (dense_limit = %d): its '
                'Hamiltonian is not stored; use PCNEVPT2(amplitudes="rdm")'
                % (nelec, self.dim(nelec), self.dense_limit))
        hmat = kernels.build_hmat(self.norb, nelec, self.dim(nelec), self.heff, self.aaaa)
        if hmat is None:
            hmat = self.make_hmat_numpy(nelec)
        return hmat

    def make_hmat_numpy(self, nelec):
        '''Reference implementation of make_hmat through intermediate N-1 and
        N-2 strings.'''
        n = self.norb
        dim = self.dim(nelec)
        hmat = numpy.zeros((dim, dim), dtype=numpy.complex128)
        if dim == 0 or nelec == 0:
            return hmat
        # one-body: <I|p^+ q|J> = sum_K <I|p^+|K> <K|q|J>, K in N-1
        tgt, sgn = self.table(nelec - 1, True)
        self._accumulate(hmat, tgt.T, sgn.T, lambda a, b: self.heff[a, b],
                         numpy.broadcast_to(numpy.arange(n), tgt.T.shape))
        if nelec < 2:
            return hmat
        # two-body: 1/2 (pq|rs) <I|p^+ r^+|K> <K|s q|J>, K in N-2, where
        # <I|p^+ r^+|K> = sigma_pr(K) and <K|s q|J> = sigma_qs(K) for
        # J = a_q^+ a_s^+ K
        t1, s1 = self.table(nelec - 2, True)          # a_r^+ : N-2 -> N-1
        t2, s2 = self.table(nelec - 1, True)          # a_p^+ : N-1 -> N
        nk = t1.shape[1]
        mid = t1.T                                    # (nK, r)
        tgt = numpy.where(mid[:, None, :] >= 0,
                          t2[numpy.arange(n)[None, :, None], numpy.maximum(mid, 0)[:, None, :]],
                          -1)                          # (nK, p, r)
        sgn = s1.T[:, None, :] * s2[numpy.arange(n)[None, :, None],
                                    numpy.maximum(mid, 0)[:, None, :]]
        sgn = numpy.where(tgt >= 0, sgn, 0)
        pair = numpy.arange(n * n).reshape(n, n)       # pair index (p, r)
        w = 0.5 * self.aaaa.transpose(0, 2, 1, 3).reshape(n * n, n * n)  # [(p,r),(q,s)] = (pq|rs)
        self._accumulate(hmat, tgt.reshape(nk, n * n), sgn.reshape(nk, n * n),
                         lambda a, b: w[a, b],
                         numpy.broadcast_to(pair.ravel(), (nk, n * n)))
        return hmat

    @staticmethod
    def _accumulate(hmat, tgt, sgn, coef, label):
        '''hmat[I_a, I_b] += sgn_a sgn_b coef(label_a, label_b) for every
        intermediate string K (rows of tgt/sgn/label) and every pair of
        valid entries a, b of that row.'''
        dim = hmat.shape[0]
        valid = tgt >= 0
        m = int(valid.sum(axis=1).max()) if valid.size else 0
        if m == 0:
            return
        # every intermediate string has the same number of valid entries
        order = numpy.argsort(~valid, axis=1, kind='stable')[:, :m]
        tgt = numpy.take_along_axis(tgt, order, axis=1)
        sgn = numpy.take_along_axis(sgn, order, axis=1)
        label = numpy.take_along_axis(numpy.asarray(label), order, axis=1)
        flat = hmat.ravel()
        step = max(1, _BATCH // (m * m))
        for k0 in range(0, len(tgt), step):
            k1 = min(len(tgt), k0 + step)
            t, s, lab = tgt[k0:k1], sgn[k0:k1], label[k0:k1]
            val = s[:, :, None] * s[:, None, :] * coef(lab[:, :, None], lab[:, None, :])
            numpy.add.at(flat, (t[:, :, None] * dim + t[:, None, :]).ravel(), val.ravel())

    # ------------------------------------------------------------------
    # model space helpers

    def trans_rdm1(self, bra, ket, nelec):
        '''t[p, q] = <bra| p^+ q |ket>'''
        db = self.apply_des(bra, nelec)
        dk = self.apply_des(ket, nelec)
        return db.conj() @ dk.T

    def averaged_rdm1(self, civecs, nelec, weights=None):
        '''sum_I w_I <Psi_I| p^+ q |Psi_I> for CI vectors given as columns.'''
        civecs = numpy.asarray(civecs)
        if civecs.ndim == 1:
            civecs = civecs[:, None]
        nstate = civecs.shape[1]
        if weights is None:
            weights = numpy.full(nstate, 1.0 / nstate)
        g = numpy.zeros((self.norb, self.norb), dtype=numpy.complex128)
        for w, c in zip(weights, civecs.T):
            g += w * self.trans_rdm1(c, c, nelec)
        return g
