'''
Density-matrix route of the spin-free PC-NEVPT2: metric and Koopmans
matrices of the perturber functions from reduced quantities of the model
states, with intermediate states in place of the 4-RDM (Chatterjee and
Sokolov, J. Chem. Theory Comput. 16, 6343 (2020), appendix).

For one class and one sector (n_alpha + da, n_beta + db) of its spin
patterns, a basis of operators W_w is chosen in which the active operators
A_(P,p) of all patterns p of that sector are short combinations,
A_(P,p) = sum_w W_w C_p[w, P] (coefficients):

    ccav        t^+_s
    cavv        t_s
    ccaa        t^+_s u^+_s'           (s, s') = (A, A), (B, B), (A, B)
    aavv        u_s' t_s               (s', s) = (A, A), (B, B), (B, A)
    cava        1, t^+_A u_A, t^+_B u_B     (sector unchanged)
                t^+_s u_s'                  (s != s')
    caaa        t^+_s,  t^+_s E_uv
    aaav        t_s,    E_uv t_s

(E_uv = sum_s u^+_s v_s).  The reduced quantities are

    G[I, J][w, w'] = <I| W_w^+ W_w' |J>
    K[I, J][w, w'] = <I| W_w^+ H_act W_w' |J>

and the spin-free matrices of the class are S = sum_p C_p^T G C_p,
K = sum_p C_p^T K C_p.  G is a spin-summed contraction of transition RDMs
of rank up to three (the Gram matrix of the vectors W_w |J>, at most
n + n^3 of them).  H_act is never applied to these vectors: with W = y_1 y_2
(y_1 the first factor of the table, y_2 the rest; signs for odd y_2)

    H y_1 y_2 |J> = y_1 (H y_2 |J>) +- y_2 ([H, y_1] |J>) + [[H, y_1], y_2} |J>

where H y_2 |J> and [H, y_1] |J> = H y_1 |J> - y_1 H |J> are the intermediate
states (2 n vectors per spin, each one product with H_act, and the n^2
vectors H E_uv |J>, which follow from them), and the double (anti)commutator is a combination of the
functions W_w |J> themselves with integrals as coefficients,

    [[H, y_1], y_2} |J> = sum_w W_w |J> M[w, (y_1 y_2)].

Hence K = W^T R + G M with R = y_1 (H y_2 |J>) +- y_2 ([H, y_1] |J>): one
matrix product of the cost of a 3-RDM, D n^6, for caaa and aaav, where the
explicit formula would need the 4-RDM.  No model-space invariance is assumed
(H |J> is used as a vector).  The matrices M are derived in
docs/derivations/nr-nevpt2.md section 6; zmr.reference.nr_koopmans has the
brute-force K (H_act applied to every function) for checks.
'''

import numpy
import scipy.sparse

from socutils.zmr.lib import kernels
from socutils.zmr.lib import linalg as la
from socutils.zmr.nr.active import ALPHA, BETA, so

SPINS = (ALPHA, BETA)


def _dn(spin, k):
    return (k, 0) if spin == ALPHA else (0, k)


def _add(*dns):
    return (sum(d[0] for d in dns), sum(d[1] for d in dns))


def _rows(*columns):
    '''Operator strings (rows) for all combinations of the given columns of
    spin-orbital indices, the first column being the slowest index.'''
    grids = numpy.meshgrid(*columns, indexing='ij')
    return numpy.stack([g.ravel() for g in grids], axis=1)


class ModelStates:
    '''The model states and their intermediate states (cached).'''

    def __init__(self, act, civecs, nelec):
        self.act = act
        self.nelec = tuple(nelec)
        self.n = act.norb
        self.c = numpy.ascontiguousarray(numpy.asarray(civecs).T)        # (J, D)
        self.nstate = self.c.shape[0]
        self._cache = {}
        self._buffers = {}

    def _cached(self, key, make):
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    def sector(self, dn):
        return self.act.target(self.nelec, dn)

    def buffer(self, name, shape):
        '''Uninitialized array for the vectors of a three-operator class.
        The memory is kept and reused by the next sector (first use of fresh
        pages is a visible part of the time).'''
        size = int(numpy.prod(shape))
        if self._buffers.get(name, numpy.empty(0)).size < size:
            self._buffers[name] = None
            self._buffers[name] = numpy.empty(size)
        return self._buffers[name][:size].reshape(shape)

    def orbitals(self, spin):
        return so(numpy.arange(self.n), spin, self.n)

    def apply(self, vecs, nelec, daggers, ops):
        '''O_x vecs for the operator strings ops (rows): (..., D) -> (..., nx, D').'''
        act = self.act
        ops = numpy.asarray(ops, dtype=numpy.int64).reshape(len(ops), len(daggers))
        target = act.target(nelec, act.spin_change(daggers, ops))
        if act.dim(nelec) == 0 or act.dim(target) == 0:
            return numpy.zeros(vecs.shape[:-1] + (len(ops), act.dim(target)))
        return act.apply_strings(vecs, nelec, [(daggers, ops)])

    def apply_e(self, vecs, nelec):
        '''E_uv vecs: (..., D) -> (..., n^2, D), (u, v) with u the slow index.'''
        out = 0
        for spin in SPINS:
            o = self.orbitals(spin)
            out = out + self.apply(vecs, nelec, (True, False), _rows(o, o))
        return out

    def apply_h(self, nelec, vecs):
        '''H_act on vectors (..., D).  The few intermediate states do not pay
        for a dense sector Hamiltonian: it is used only if it exists already.'''
        if vecs.size == 0:
            return vecs.copy()
        act = self.act
        q = numpy.ascontiguousarray(vecs.reshape(-1, vecs.shape[-1]).T)
        out = None
        if nelec not in act._hmat:
            out = kernels.sigma_ab(act.norb, nelec, act.heff, act.aaaa, q)
        if out is None:
            out = act.apply_h(nelec, q)
        return numpy.ascontiguousarray(out.T).reshape(vecs.shape)

    def apply_into(self, vecs, nelec, daggers, ops, out):
        '''out[v, x, :] = O_x vecs[v, :] for vecs (nvec, D) and a contiguous
        out (nvec, nx, D').'''
        act = self.act
        ops = numpy.asarray(ops, dtype=numpy.int64).reshape(len(ops), len(daggers))
        target = act.target(nelec, act.spin_change(daggers, ops))
        if not kernels.apply_strings_ab(act.norb, nelec, target, daggers, ops, vecs, out):
            out[:] = act.apply_strings(vecs, nelec, [(daggers, ops)])

    def apply_add(self, vecs, nelec, daggers, ops, dest, out):
        '''out[dest[v, x], :] += O_x vecs[v, :] for vecs (nvec, D) and a
        contiguous out (nrow, D'); the destinations must be distinct.'''
        act = self.act
        ops = numpy.asarray(ops, dtype=numpy.int64).reshape(len(ops), len(daggers))
        target = act.target(nelec, act.spin_change(daggers, ops))
        if not kernels.apply_strings_ab_add(act.norb, nelec, target, daggers, ops, vecs,
                                            dest, out):
            res = act.apply_strings(vecs, nelec, [(daggers, ops)])
            out[numpy.asarray(dest).ravel()] += res.reshape(-1, out.shape[1])

    def hj(self):
        '''H |J>'''
        return self._cached('hj', lambda: self.apply_h(self.nelec, self.c))

    def one(self, spin, dagger):
        '''y |J> for y = t^+_spin (dagger) or t_spin: (J, t, D')'''
        return self._cached(('one', spin, dagger), lambda: self.apply(
            self.c, self.nelec, (dagger,), self.orbitals(spin)[:, None]))

    def h_one(self, spin, dagger):
        '''H y |J>'''
        sec = self.sector(_dn(spin, 1 if dagger else -1))
        return self._cached(('h_one', spin, dagger),
                            lambda: self.apply_h(sec, self.one(spin, dagger)))

    def comm_one(self, spin, dagger):
        '''[H, y] |J> = H y |J> - y H |J>'''
        return self._cached(('comm_one', spin, dagger), lambda: (
            self.h_one(spin, dagger)
            - self.apply(self.hj(), self.nelec, (dagger,), self.orbitals(spin)[:, None])))

    def e(self):
        '''E_uv |J>: (J, n^2, D)'''
        return self._cached('e', lambda: self.apply_e(self.c, self.nelec))

    def h_e(self):
        '''H E_uv |J> from the 4 n one-operator intermediate states, without
        further products with H_act:

            H E_uv |J> = sum_s [u^+_s (H v_s |J>) - v_s ([H, u^+_s] |J>)]
                         + 2 h_vu |J> + sum_pq [2 (vu|pq) - (pu|vq)] E_pq |J>

        (the anticommutators {[H, u^+_s], v_s} of basis_cava, summed over s).'''
        return self._cached('h_e', self._h_e)

    def _h_e(self):
        n, h, eri = self.n, self.act.heff, self.act.aaaa
        e = self.e()
        m = (2 * eri.transpose(1, 0, 2, 3) - eri.transpose(1, 2, 0, 3)).reshape(n * n, n * n)
        out = numpy.empty_like(e)
        rows = numpy.arange(n * n).reshape(n, n)                       # [u, v]
        for j in range(self.nstate):
            out[j] = la.gemm(m, e[j])
            out[j] += 2 * h.T.reshape(n * n, 1) * self.c[j]
            for spin in SPINS:
                o = self.orbitals(spin)[:, None]
                # vector v, string u^+ -> row (u, v); vector u, string v -> row (u, v)
                self.apply_add(self.h_one(spin, False)[j], self.sector(_dn(spin, -1)),
                               (True,), o, rows.T, out[j])
                self.apply_add(-self.comm_one(spin, True)[j], self.sector(_dn(spin, 1)),
                               (False,), o, rows, out[j])
        return out

    def h_e_direct(self):
        '''H E_uv |J> by n^2 products with H_act (check of h_e).'''
        return self.apply_h(self.nelec, self.e())


class Basis:
    '''Operator basis of one class in one sector.

    w, r: vectors W_w |J> and R_w |J>, arrays (J, nW, D')
    m: matrix M (nW, nW) of the double commutators, or None
    leads: for every block of basis operators (daggers, ops): one operator
        string that occurs in no other basis operator; the coefficient of a
        function on a basis operator is that of its lead string.'''

    def __init__(self, w, r, m, leads):
        self.w, self.r, self.m, self.leads = w, r, m, leads

    def index(self):
        '''{(daggers, operator string): position in the basis}'''
        out = {}
        pos = 0
        for daggers, ops in self.leads:
            ops = numpy.asarray(ops).reshape(len(ops), len(daggers))
            for row in ops:
                out[(tuple(daggers), tuple(int(x) for x in row))] = pos
                pos += 1
        return out


def _swap(x):
    '''(J, a, b, D) -> (J, b, a, D), contiguous, flattened to (J, a b, D)'''
    j, a, b, d = x.shape
    return numpy.ascontiguousarray(x.transpose(0, 2, 1, 3)).reshape(j, a * b, d)


def _flat(x):
    j, a, b, d = x.shape
    return x.reshape(j, a * b, d)


def basis_one(ms, dn, dagger):
    '''ccav (t^+_s) and cavv (t_s): H is applied to the n functions.'''
    spin = ALPHA if dn[0] else BETA
    return Basis(ms.one(spin, dagger), ms.h_one(spin, dagger), None,
                 [((dagger,), ms.orbitals(spin)[:, None])])


def basis_ccaa(ms, dn):
    '''W_tu = t^+_s u^+_s';  H t^+ u^+ |J> = t^+ (H u^+ |J>) - u^+ [H, t^+] |J>
    + sum_pr (pt|ru) p^+ r^+ |J>'''
    s1, s2 = {(2, 0): (ALPHA, ALPHA), (0, 2): (BETA, BETA), (1, 1): (ALPHA, BETA)}[dn]
    n, eri = ms.n, ms.act.aaaa
    o1, o2 = ms.orbitals(s1), ms.orbitals(s2)
    w = ms.apply(ms.c, ms.nelec, (True, True), _rows(o1, o2))
    r = _swap(ms.apply(ms.h_one(s2, True), ms.sector(_dn(s2, 1)), (True,), o1[:, None]))
    r -= _flat(ms.apply(ms.comm_one(s1, True), ms.sector(_dn(s1, 1)), (True,), o2[:, None]))
    m = eri.transpose(0, 2, 1, 3).reshape(n * n, n * n)
    return Basis(w, r, m, [((True, True), _rows(o1, o2))])


def basis_aavv(ms, dn):
    '''W_ut = u_s' t_s;  H u t |J> = u (H t |J>) - t [H, u] |J>
    + sum_pr (up|tr) p_s' r_s |J>'''
    s2, s1 = {(-2, 0): (ALPHA, ALPHA), (0, -2): (BETA, BETA), (-1, -1): (BETA, ALPHA)}[dn]
    n, eri = ms.n, ms.act.aaaa
    o1, o2 = ms.orbitals(s1), ms.orbitals(s2)
    w = ms.apply(ms.c, ms.nelec, (False, False), _rows(o2, o1))
    r = _swap(ms.apply(ms.h_one(s1, False), ms.sector(_dn(s1, -1)), (False,), o2[:, None]))
    r -= _flat(ms.apply(ms.comm_one(s2, False), ms.sector(_dn(s2, -1)), (False,), o1[:, None]))
    m = eri.transpose(1, 3, 0, 2).reshape(n * n, n * n)
    return Basis(w, r, m, [((False, False), _rows(o2, o1))])


def basis_ccvv(ms, dn):
    return Basis(ms.c[:, None, :], ms.hj()[:, None, :], None,
                 [((), numpy.zeros((1, 0), dtype=int))])


def _particle_hole(ms, s1, s2):
    '''W_tu = t^+_s1 u_s2 and R_tu = t^+ (H u |J>) - u [H, t^+] |J>'''
    o1, o2 = ms.orbitals(s1), ms.orbitals(s2)
    w = ms.apply(ms.c, ms.nelec, (True, False), _rows(o1, o2))
    r = _swap(ms.apply(ms.h_one(s2, False), ms.sector(_dn(s2, -1)), (True,), o1[:, None]))
    r -= _flat(ms.apply(ms.comm_one(s1, True), ms.sector(_dn(s1, 1)), (False,), o2[:, None]))
    return w, r, ((True, False), _rows(o1, o2))


def basis_cava(ms, dn):
    '''{[H, t^+_s], u_s'} = d(s,s') [h_ut + sum_rq (ut|rq) E_rq]
    - sum_pq (pt|uq) p^+_s q_s' '''
    n, eri = ms.n, ms.act.aaaa
    if n == 0:
        return basis_ccvv(ms, dn)
    exch = -eri.transpose(0, 3, 1, 2).reshape(n * n, n * n)
    if dn != (0, 0):
        s1, s2 = (ALPHA, BETA) if dn == (1, -1) else (BETA, ALPHA)
        w, r, lead = _particle_hole(ms, s1, s2)
        return Basis(w, r, exch, [lead])
    parts = [_particle_hole(ms, s, s) for s in SPINS]
    w = numpy.concatenate([ms.c[:, None, :]] + [p[0] for p in parts], axis=1)
    r = numpy.concatenate([ms.hj()[:, None, :]] + [p[1] for p in parts], axis=1)
    nn = n * n
    m = numpy.zeros((1 + 2 * nn, 1 + 2 * nn))
    coul = eri.transpose(2, 3, 1, 0).reshape(nn, nn)
    for k in range(2):
        sl = slice(1 + k * nn, 1 + (k + 1) * nn)
        m[0, sl] = ms.act.heff.T.ravel()
        m[1:1 + nn, sl] = coul
        m[1 + nn:, sl] = coul
        m[sl, sl] += exch
    return Basis(w, r, m, [((), numpy.zeros((1, 0), dtype=int))] + [p[2] for p in parts])


def _three_operator_vectors(ms, spin, dagger):
    '''W = y_t, y_t E_uv and R = H y_t, y_t (H E_uv |J>) + E_uv [H, y_t] |J>
    for y = t^+_spin (dagger) or t_spin; arrays (J, n + n^3, D'), the
    three-operator part in the order (u, v, t).  The vectors are written
    straight into their place.'''
    n, nstate = ms.n, ms.nstate
    o = ms.orbitals(spin)
    target = ms.sector(_dn(spin, 1 if dagger else -1))
    dim = ms.act.dim(target)
    w = ms.buffer('w', (nstate, n + n**3, dim))
    r = ms.buffer('r', (nstate, n + n**3, dim))
    w[:, :n] = ms.one(spin, dagger)
    r[:, :n] = ms.h_one(spin, dagger)
    e, he, comm = ms.e(), ms.h_e(), ms.comm_one(spin, dagger)
    # E_uv on the vector t goes to the row (u, v, t)
    dest = numpy.arange(n * n)[None, :] * n + numpy.arange(n)[:, None]
    for j in range(nstate):
        ms.apply_into(e[j], ms.nelec, (dagger,), o[:, None], w[j, n:].reshape(n * n, n, dim))
        ms.apply_into(he[j], ms.nelec, (dagger,), o[:, None], r[j, n:].reshape(n * n, n, dim))
        for s2 in SPINS:
            o2 = ms.orbitals(s2)
            ms.apply_add(comm[j], target, (True, False), _rows(o2, o2), dest, r[j, n:])
    return w, r


def basis_caaa(ms, dn):
    '''W = t^+_s, t^+_s E_uv (order (u, v, t));
    H t^+ E_uv |J> = t^+ (H E_uv |J>) + E_uv [H, t^+] |J> + [[H, t^+], E_uv] |J>,
    [[H, t^+], E_uv] = - h_vt u^+ + sum_pr (pt|ru) p^+ E_rv - sum_pq (pt|vq) p^+ E_uq
                       - sum_rq (vt|rq) u^+ E_rq'''
    spin = ALPHA if dn[0] else BETA
    n, h, eri = ms.n, ms.act.heff, ms.act.aaaa
    o = ms.orbitals(spin)
    w, r = _three_operator_vectors(ms, spin, True)
    eye = numpy.eye(n)
    # rows (a, b, c) = a^+ E_bc, columns (t, u, v)
    m6 = (numpy.einsum('ptru,cv->prctuv', eri, eye)
          - numpy.einsum('ptvq,bu->pbqtuv', eri, eye)
          - numpy.einsum('vtrq,au->arqtuv', eri, eye))
    m1 = -numpy.einsum('vt,au->atuv', h, eye)
    m = numpy.zeros((n + n**3, n + n**3))
    m[:n, n:] = m1.transpose(0, 2, 3, 1).reshape(n, n**3)
    m[n:, n:] = m6.transpose(1, 2, 0, 4, 5, 3).reshape(n**3, n**3)
    lead3 = _rows(o, o, o)[:, [2, 0, 1]]                  # (u, v, t) -> t^+ u^+ v
    return Basis(w, r, m, [((True,), o[:, None]), ((True, True, False), lead3)])


def basis_aaav(ms, dn):
    '''W = t_s, E_uv t_s = t_s E_uv - d(u,t) v_s (order (u, v, t));
    H t E_uv |J> = t (H E_uv |J>) + E_uv [H, t] |J> + [[H, t], E_uv] |J>,
    [[H, t], E_uv] = - h_tu v - sum_qr (tu|qr) E_qr v - sum_pq (tp|qu) E_qv p
                     + sum_pr (tp|vr) E_ur p'''
    spin = ALPHA if dn[0] else BETA
    n, h, eri = ms.n, ms.act.heff, ms.act.aaaa
    o = ms.orbitals(spin)
    w, r = _three_operator_vectors(ms, spin, False)
    w5 = w[:, n:].reshape(ms.nstate, n, n, n, -1)
    r5 = r[:, n:].reshape(ms.nstate, n, n, n, -1)
    for u in range(n):
        w5[:, u, :, u] -= w[:, :n]
        r5[:, u, :, u] -= r[:, :n]
    eye = numpy.eye(n)
    # rows (a, b, c) = E_ab c, columns (u, v, t)
    m6 = (-numpy.einsum('tuqr,cv->qrcuvt', eri, eye)
          - numpy.einsum('tpqu,bv->qbpuvt', eri, eye)
          + numpy.einsum('tpvr,au->arpuvt', eri, eye))
    m1 = -numpy.einsum('tu,cv->cuvt', h, eye)
    m = numpy.zeros((n + n**3, n + n**3))
    m[:n, n:] = m1.reshape(n, n**3)
    m[n:, n:] = m6.reshape(n**3, n**3)
    return Basis(w, r, m, [((False,), o[:, None]), ((True, False, False), _rows(o, o, o))])


BASES = {
    'ccvv': basis_ccvv,
    'ccav': lambda ms, dn: basis_one(ms, dn, True),
    'cavv': lambda ms, dn: basis_one(ms, dn, False),
    'ccaa': basis_ccaa,
    'aavv': basis_aavv,
    'cava': basis_cava,
    'caaa': basis_caaa,
    'aaav': basis_aaav,
}


def _mirror(k):
    k = numpy.tril(k)
    k += numpy.tril(k, -1).T
    return k


def reduced_matrices(basis, check=False, diagonal=False):
    '''(G, K) as arrays (nstate, nW, nstate, nW) from the vectors of a basis.

    K = W^T R + G M is symmetric, so only its lower triangle is computed
    (half a matrix product) and mirrored.  check=True forms all of it and
    also returns the largest |K - K^T| relative to the largest |K|, a test
    of the commutator formulas.  diagonal=True: only the blocks K[J, :, J, :]
    (state-specific contraction), the others are zero.'''
    nstate, nw, dim = basis.w.shape
    w = basis.w.reshape(nstate * nw, dim)
    r = basis.r.reshape(nstate * nw, dim)
    g = la.herk(w).reshape(nstate, nw, nstate, nw)
    if diagonal and not check:
        k = numpy.zeros_like(g)
        for j in range(nstate):
            kj = la.gemmt_lower(basis.w[j], basis.r[j])
            if basis.m is not None:
                kj += la.gemm(numpy.ascontiguousarray(g[j, :, j, :]), basis.m)
            k[j, :, j, :] = _mirror(kj)
        return g, k
    k = la.gemm(w, r, 'N', 'T') if check else la.gemmt_lower(w, r)
    if basis.m is not None:
        k = k.reshape(nstate * nw * nstate, nw)
        k += la.gemm(numpy.ascontiguousarray(g.reshape(k.shape)), basis.m)
        k = k.reshape(nstate * nw, nstate * nw)
    if check:
        asymmetry = abs(k - k.T).max(initial=0.0) / max(abs(k).max(initial=0.0), 1e-300)
        return g, k.reshape(g.shape), asymmetry
    return g, _mirror(k).reshape(g.shape)


def coefficients(pattern, nparams, index):
    '''Sparse C (nW, nP): function P of the pattern = sum_w W_w C[w, P].'''
    rows, cols, vals = [], [], []
    offset = 0
    for size, terms in zip(nparams, pattern.terms):
        for coef, daggers, ops in terms:
            ops = numpy.asarray(ops).reshape(size, len(daggers))
            coef = numpy.broadcast_to(numpy.asarray(coef, dtype=float), (size,))
            for p in range(size):
                w = index.get((tuple(daggers), tuple(int(x) for x in ops[p])))
                if w is not None and coef[p] != 0:
                    rows.append(w)
                    cols.append(offset + p)
                    vals.append(coef[p])
        offset += size
    return scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(len(index), offset))


def _congruence(c, a):
    '''C^T a C for sparse C (nW, nP) and dense a (nstate, nW, nstate, nW),
    for all pairs of states at once: (nstate, nP, nstate, nP)'''
    nstate, nw = a.shape[:2]
    big = scipy.sparse.kron(scipy.sparse.identity(nstate), c, format='csr')
    out = big.T @ (big.T @ a.reshape(nstate * nw, nstate * nw)).T
    return numpy.asarray(out).T.reshape(nstate, c.shape[1], nstate, c.shape[1])


class Reduced:
    '''Reduced matrices of the classes for a set of model states, cached per
    (class name, sector change).'''

    def __init__(self, act, civecs, nelec, check=False, flip_tol=1e-9, diagonal=False):
        self.ms = ModelStates(act, civecs, nelec)
        # only K[J, :, J, :] (state-specific contraction)
        self.diagonal = diagonal
        self._sectors = {}
        self._parities = None
        # True: K is formed completely and its asymmetry recorded (largest
        # |K - K^T| relative to the largest |K|), instead of one triangle
        self.check = check
        self.asymmetry = 0.0
        # the matrices of two sectors that differ by the exchange of alpha
        # and beta are computed once if the CI vectors have that symmetry to
        # flip_tol (largest element of C^T -+ C; 0 switches this off).  The
        # matrices of the second sector are then off by that order.
        self.flip_tol = flip_tol
        self.flip_error = None
        # largest number of basis operators of a sector
        self.max_basis = 0

    def flip_parities(self):
        '''p_J = +-1 if every model state goes into p_J times itself when its
        alpha and beta strings are exchanged (n_alpha = n_beta, e.g. Ms = 0
        states of definite spin), else None.  The largest deviation
        max |C_J^T - p_J C_J| over the states is kept in flip_error; the
        symmetry is used if it is at most flip_tol.'''
        if self._parities is None:
            self._parities = False
            ms = self.ms
            if ms.nelec[0] == ms.nelec[1] and self.flip_tol > 0:
                c = ms.c.reshape(ms.nstate, int(round(numpy.sqrt(ms.c.shape[1]))), -1)
                ct = c.transpose(0, 2, 1)
                even = abs(ct - c).max(axis=(1, 2))
                odd = abs(ct + c).max(axis=(1, 2))
                self.flip_error = float(numpy.minimum(even, odd).max(initial=0.0))
                if self.flip_error <= self.flip_tol:
                    self._parities = numpy.where(even <= odd, 1.0, -1.0)
        return None if self._parities is False else self._parities

    def sector(self, name, dn):
        '''(G, K, {lead string: position}) of a class in the sector reached
        by dn.  The matrices of the sector with alpha and beta exchanged are
        the same up to the parities of the states (flip_parities) and are
        not computed twice.'''
        key = (name, tuple(dn))
        flipped = (name, (dn[1], dn[0]))
        if key not in self._sectors and flipped in self._sectors \
                and self.flip_parities() is not None:
            g, k, index = self._sectors[flipped]
            p = self.flip_parities()
            sign = p[:, None, None, None] * p[None, None, :, None]
            n = self.ms.n
            index = {(dag, tuple((x + n) % (2 * n) for x in ops)): pos
                     for (dag, ops), pos in index.items()}
            self._sectors[key] = (g * sign, k * sign, index)
        if key not in self._sectors:
            basis = BASES[name](self.ms, tuple(dn))
            if self.check:
                g, k, asymmetry = reduced_matrices(basis, True)
                self.asymmetry = max(self.asymmetry, asymmetry)
            else:
                g, k = reduced_matrices(basis, diagonal=self.diagonal)
            self.max_basis = max(self.max_basis, g.shape[1])
            self._sectors[key] = (g, k, basis.index())
        return self._sectors[key]

    def class_matrices(self, pclass):
        '''Spin-free S[I, P, J, Q] and K[I, P, J, Q] of a class
        (zmr.nr.classes), or None if no pattern has a target sector.'''
        act, ms = self.ms.act, self.ms
        nparams = pclass.nparams(act.norb)
        nx, nstate = sum(nparams), ms.nstate
        s = numpy.zeros((nstate, nx, nstate, nx))
        k = numpy.zeros_like(s)
        found = False
        for pattern in pclass.patterns(act.norb):
            if act.dim(ms.sector(pattern.dn)) == 0:
                continue
            found = True
            g, kk, index = self.sector(pclass.name, pattern.dn)
            c = coefficients(pattern, nparams, index)
            s += _congruence(c, g)
            k += _congruence(c, kk)
        return (s, k) if found else None


def contracted_space(smat, kmat, lindep):
    '''Orthonormal, H_act-diagonal combinations of the functions from their
    metric and Koopmans matrices: coefficients (nfunc, m) and energies.'''
    s, u = la.eigh(numpy.ascontiguousarray(0.5 * (smat + smat.T)))
    keep = s > lindep
    x = numpy.ascontiguousarray(u[:, keep] / numpy.sqrt(s[keep]))
    if x.shape[1] == 0:
        return x, numpy.zeros(0)
    kx = la.gemm(numpy.ascontiguousarray(0.5 * (kmat + kmat.T)), x)
    mu, wv = la.eigh(la.gemm(x, kx, 'T', 'N'))
    return la.gemm(x, wv), mu
