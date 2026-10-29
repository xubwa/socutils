'''
Product-space CI (SQD / QSCI spaces, ProductSpaceCI) with the CI vector
distributed by rows over MPI processes (lib/zgasci/PRODUCT_SPACE.md, section
"MPI").

The space is every alpha string of a list times every beta string of the
same list (nalpha = nbeta, spatial orbitals, no orbital symmetry), the
states those of even (spin_flip = +1) or odd (-1) spin, C[b, a] = spin_flip
C[a, b].  Process r holds the rows (alpha strings) rows[r] .. rows[r + 1] - 1
of C and of every vector, each row all beta strings.  With the identity of
zg_sigma_flip, sigma = Z + parity Z^T + Y:

* Z = the (BB,BB) terms applied to C: row-local; each process applies the
  kernel of a product plan of its own alpha strings and all beta strings,
  with the integrals of the beta-beta pairs only.
* Z^T: a distributed transpose (pairwise exchanges of blocks).
* Y = the (AB,AB) terms: the only part that couples rows of different
  processes, through the alpha single excitations of the own rows.  The rows
  of other processes that these reach (the halo; few, since the strings are
  sorted as integers and a single excitation mostly keeps the high bits) are
  found once and fetched at every product by pairwise exchanges;
  zg_sigma_ab_rows then applies the terms once, from the own rows and the
  halo, to the own rows.

The eigensolver is a LOBPCG-like iteration of one vector (X, the
preconditioned residual T, the previous direction P and their products with
H: six vectors of the own rows); dot products are summed over the
processes.  One process
per node with all its threads is the arrangement intended.

Run (example):
    mpirun -np 4 -ppn 1 python script.py
with, in script.py,
    from socutils.hci import gasci_mpi
    e, c_rows, info = gasci_mpi.kernel(h1, eri, norb, nelec, strs)
'''

import time

import numpy
from pyscf import ao2mo

from socutils.hci import gasci

# Largest MPI message in elements (counts are C ints); ZGASCI_DIST_CHUNK overrides (tests).
import os as _os
_CHUNK = int(_os.environ.get('ZGASCI_DIST_CHUNK', 1 << 28))


def _block(fn, src, dst):
    '''zg_transpose_d / zg_copy_block_d between two 2-d float64 arrays (views
    with unit column stride): dst = src.T or dst = src.'''
    if src.size == 0:
        return
    assert src.strides[1] == 8 and dst.strides[1] == 8
    nr, nc = src.shape
    fn(nr, nc, src.ctypes.data, src.strides[0] // 8, dst.ctypes.data, dst.strides[0] // 8)


def row_ranges(n, size, weights=None):
    '''Contiguous row ranges, rows[r] .. rows[r + 1] - 1 for process r, of
    about equal size (or weight).'''
    if weights is None:
        return numpy.array([(n * r) // size for r in range(size + 1)], dtype=numpy.int64)
    cum = numpy.concatenate([[0], numpy.cumsum(weights)])
    cuts = [0] + [int(numpy.searchsorted(cum, cum[-1] * r / size)) for r in range(1, size)] + [n]
    return numpy.array(cuts, dtype=numpy.int64)


class DistProductCI:
    '''See the module.  strs: alpha (= beta) strings as integers (pyscf
    cistring convention); they are sorted.'''

    def __init__(self, comm, h1, eri, norb, nelec, strs, spin_flip=1, rows=None):
        self.comm = comm
        self.rank, self.size = comm.Get_rank(), comm.Get_size()
        na, nb = nelec if isinstance(nelec, (tuple, list)) else (nelec // 2, nelec - nelec // 2)
        if na != nb:
            raise ValueError('nalpha = nbeta is required')
        if spin_flip not in (1, -1):
            raise ValueError('spin_flip must be 1 or -1')
        self.norb, self.nelec, self.parity = norb, (na, nb), spin_flip
        strs = numpy.unique(numpy.asarray(strs, dtype=numpy.int64))
        self.strs = strs
        self.n = n = len(strs)
        self.rows = row_ranges(n, self.size) if rows is None else numpy.asarray(rows, dtype=numpy.int64)
        self.lo, self.hi = int(self.rows[self.rank]), int(self.rows[self.rank + 1])
        self.nloc = self.hi - self.lo
        t0 = time.time()
        # The full plan (for Y) and the plan of the own rows (Z, diagonal).
        full = gasci.ProductSpaceCI(ci_strs=(strs, strs))
        self.plan = full.plan(norb, (na, nb))
        if self.plan.side_count(0) != n or self.plan.side_count(1) != n:
            raise ValueError('strings left out of the product plan')
        if self.nloc == n:  # one process: the full plan is the plan of the own rows
            self.lplan = self.plan
        else:
            own = gasci.ProductSpaceCI(ci_strs=(strs[self.lo:self.hi], strs))
            self.lplan = own.plan(norb, (na, nb)) if self.nloc > 0 else None
        eri = ao2mo.restore(1, numpy.asarray(eri, dtype=float), norb)
        self.v = full._hamiltonian(h1, eri, norb, na + nb, self.plan)
        a, b = gasci.pair_indices(2 * norb)
        bb = (a >= norb) & (b >= norb)
        self.v_bb = self.v * numpy.outer(bb, bb)
        self._halo_setup()
        self.t_setup = time.time() - t0
        self.stat = {'sigma': 0.0, 'z': 0.0, 'transpose': 0.0, 'halo': 0.0, 'ab': 0.0,
                     'nsigma': 0}

    def _halo_setup(self):
        '''The rows of other processes reached by alpha single excitations
        (with the diagonal) of the own rows, who sends what, and smap.'''
        comm, size, rows, strs, norb = self.comm, self.size, self.rows, self.strs, self.norb
        own = strs[self.lo:self.hi]
        reach = numpy.zeros(self.n, dtype=bool)
        one = numpy.int64(1)
        for i in range(norb):
            occ_i = (own >> i) & 1 == 1
            if not occ_i.any():
                continue
            for j in range(norb):
                if j == i:
                    continue
                m = occ_i & ((own >> j) & 1 == 0)
                if not m.any():
                    continue
                cand = own[m] ^ ((one << i) | (one << j))
                pos = numpy.searchsorted(strs, cand)
                pos[pos >= self.n] = 0
                hit = strs[pos] == cand
                reach[pos[hit]] = True
        reach[self.lo:self.hi] = False
        need = numpy.nonzero(reach)[0]
        owner = numpy.searchsorted(rows, need, side='right') - 1
        want = [need[owner == q] for q in range(size)]
        self.halo_recv = want                       # rows received from q, in order
        self.halo_send = comm.alltoall(want)        # rows q wants from me (global places)
        nh = len(need)
        self.nhalo = nh
        self.smap = numpy.full(self.n, -1, dtype=numpy.int64)
        self.smap[self.lo:self.hi] = numpy.arange(self.nloc)
        self.smap[need] = self.nloc + numpy.arange(nh)  # need is sorted, by owner
        self.halo_stats = comm.allgather((self.nloc, nh))

    def _halo_exchange(self, c, buf):
        '''buf = own rows of c, then the halo rows (in the order of smap).'''
        r, size = self.rank, self.size
        buf[:self.nloc] = c
        start = {}
        at = self.nloc
        for q in range(size):  # receiving places in buf, by owner (need was sorted)
            start[q] = at
            at += len(self.halo_recv[q])
        for k in range(1, size):
            dest, src = (r + k) % size, (r - k) % size
            send = numpy.ascontiguousarray(c[self.halo_send[dest] - self.lo])
            nr = len(self.halo_recv[src])
            recv = buf[start[src]:start[src] + nr]
            self._sendrecv_var(send, dest, recv, src)

    def _sendrecv_var(self, send, dest, recv, src):
        '''Send to dest and receive from src, buffers of any lengths, in
        chunks of _CHUNK elements, each chunk its own message tagged by its
        index: the two sides need not agree on a number of steps (the
        lengths of a pair differ, e.g. in the halo exchange).'''
        s = send.reshape(-1)
        rv = recv.reshape(-1)
        reqs = [self.comm.Irecv(rv[i:i + _CHUNK], source=src, tag=k)
                for k, i in enumerate(range(0, rv.size, _CHUNK))]
        reqs += [self.comm.Isend(s[i:i + _CHUNK], dest=dest, tag=k)
                 for k, i in enumerate(range(0, s.size, _CHUNK))]
        from mpi4py import MPI
        MPI.Request.Waitall(reqs)

    # -- distributed vectors -------------------------------------------------

    def zeros(self):
        return numpy.zeros((self.nloc, self.n))

    def dot(self, x, y):
        return self.comm.allreduce(float(numpy.vdot(x, y)))

    def transpose(self, x):
        '''The own rows of X^T, X distributed by rows (pairwise exchanges;
        packing and transposition of the blocks by zg_copy_block_d and
        zg_transpose_d, in parallel).'''
        lib_ = gasci._load()
        r, size, rows = self.rank, self.size, self.rows
        out = numpy.empty_like(x)
        lo, hi = self.lo, self.hi
        _block(lib_.zg_transpose_d, x[:, lo:hi], out[:, lo:hi])
        for k in range(1, size):
            dest, src = (r + k) % size, (r - k) % size
            send = numpy.empty((self.nloc, rows[dest + 1] - rows[dest]))
            _block(lib_.zg_copy_block_d, x[:, rows[dest]:rows[dest + 1]], send)
            recv = numpy.empty((rows[src + 1] - rows[src], hi - lo))
            self._sendrecv_var(send, dest, recv, src)
            _block(lib_.zg_transpose_d, recv, out[:, rows[src]:rows[src + 1]])
        return out

    def symmetrize(self, x):
        y = self.transpose(x)
        y *= self.parity
        y += x
        y *= 0.5
        return y

    def hdiag(self):
        if self.nloc == 0:
            return numpy.zeros((0, self.n))
        return self.lplan.hdiag(self.v).reshape(self.nloc, self.n)

    # -- H c ---------------------------------------------------------------

    def sigma(self, c):
        '''H c for c of the symmetry (C^T = parity C), distributed by rows.'''
        t0 = time.time()
        z = self.lplan.sigma(self.v_bb, c.ravel()).reshape(self.nloc, self.n) if self.nloc else self.zeros()
        t1 = time.time()
        zt = self.transpose(z)
        if self.parity < 0:
            zt *= -1
        z += zt
        del zt
        t2 = time.time()
        if self.nhalo:
            if not hasattr(self, '_hbuf'):
                self._hbuf = numpy.empty((self.nloc + self.nhalo, self.n))
            self._halo_exchange(c, self._hbuf)
            src = self._hbuf
        else:  # no halo (one process): the own rows in place
            src = numpy.ascontiguousarray(c)
        t3 = time.time()
        if self.nloc:  # z += Y
            self.plan.sigma_ab_rows(self.v, src, self.smap, z, self.lo, self.hi)
        t4 = time.time()
        st = self.stat
        st['z'] += t1 - t0
        st['transpose'] += t2 - t1
        st['halo'] += t3 - t2
        st['ab'] += t4 - t3
        st['sigma'] += time.time() - t0
        st['nsigma'] += 1
        return z

    # -- eigensolver -----------------------------------------------------------

    def kernel(self, ci0=None, tol=1e-8, tol_residual=1e-4, max_cycle=60, level_shift=1e-3,
               verbose=True, max_space=None):
        '''Lowest root of the symmetry: (e without ecore, own rows of c,
        info).  A locally optimal block (LOBPCG-like) iteration of one
        vector: X, its correction T and the previous direction P, with their
        products with H; six vectors of the own rows (max_space is ignored).
        ci0: own rows of a start vector (default: the lowest diagonal
        element, symmetrized).'''
        from scipy.linalg import blas, eigh
        axpy = blas.daxpy
        hd = self.hdiag()
        if ci0 is None:
            hs = hd
            if self.parity < 0:  # off the diagonal a = b, where the symmetrized vector vanishes
                hs = hd.copy()
                hs[numpy.arange(self.nloc), self.lo + numpy.arange(self.nloc)] = numpy.inf
            local_min = (float(hs.min()), self.rank, int(hs.argmin())) if self.nloc else (numpy.inf, self.rank, -1)
            _, owner, idx = min(self.comm.allgather(local_min))
            x = self.zeros()
            if self.rank == owner:
                x.flat[idx] = 1.0
        else:
            x = numpy.array(ci0, dtype=float).reshape(self.nloc, self.n)
        x = self.symmetrize(x)
        x /= numpy.sqrt(self.dot(x, x))
        hx = self.sigma(x)
        p = hp = None
        e_old = None
        info = {'cycles': 0, 'converged': False}
        t0 = time.time()
        xf = x.reshape(-1)
        for it in range(max_cycle):
            e = self.dot(x, hx)
            # T = -(HX - e X) / max(|hd - e|, level_shift), by rows (no full-length
            # temporary); positive definite, as LOBPCG needs (hd - e changes sign
            # e.g. with an S^2 penalty and odd parity)
            t = numpy.array(hx)
            axpy(xf, t.reshape(-1), a=-e)
            rnorm = numpy.sqrt(self.dot(t, t))
            de = None if e_old is None else e - e_old
            if verbose and self.rank == 0:
                print(f'  lobpcg {it:3d}  E {e:.12f}  dE {de if de is None else f"{de:.2e}"}  '
                      f'|r| {rnorm:.2e}  t {time.time() - t0:.1f}s', flush=True)
            info['cycles'] = it + 1
            if (de is not None and abs(de) < tol and rnorm < tol_residual) or rnorm < 1e-12:
                info['converged'] = True
                break
            e_old = e
            for r0 in range(0, self.nloc, 256):
                d = numpy.maximum(numpy.abs(hd[r0:r0 + 256] - e), level_shift)
                t[r0:r0 + 256] /= -d
            t = self.symmetrize(t)
            basis = [x, t] if p is None else [x, t, p]
            for q in basis[:1] + basis[2:]:  # T orthogonal to X and P
                axpy(q.reshape(-1), t.reshape(-1), a=-self.dot(q, t))
            nt = numpy.sqrt(self.dot(t, t))
            if nt < 1e-14:
                info['converged'] = True
                break
            t /= nt
            ht = self.sigma(t)
            hbasis = [hx, ht] if p is None else [hx, ht, hp]
            m = len(basis)
            hsub = numpy.empty((m, m))
            ssub = numpy.empty((m, m))
            for i_ in range(m):
                for j_ in range(i_, m):
                    hsub[i_, j_] = hsub[j_, i_] = self.dot(basis[i_], hbasis[j_])
                    ssub[i_, j_] = ssub[j_, i_] = self.dot(basis[i_], basis[j_])
            w, u = eigh(hsub, ssub)
            a, b = u[0, 0], u[1, 0]
            c_ = u[2, 0] if m == 3 else 0.0
            # X <- a X + b T + c P ;  P <- b T + c P  (and the products)
            x *= a
            axpy(t.reshape(-1), xf, a=b)
            hx *= a
            axpy(ht.reshape(-1), hx.reshape(-1), a=b)
            if p is None:
                p, hp = t, ht
                p *= b
                hp *= b
            else:
                axpy(p.reshape(-1), xf, a=c_)
                axpy(hp.reshape(-1), hx.reshape(-1), a=c_)
                p *= c_
                axpy(t.reshape(-1), p.reshape(-1), a=b)
                hp *= c_
                axpy(ht.reshape(-1), hp.reshape(-1), a=b)
                del t, ht
            nx = numpy.sqrt(self.dot(x, x))
            x /= nx
            hx /= nx
            np_ = numpy.sqrt(self.dot(p, p))
            if np_ > 0:
                p /= np_
                hp /= np_
        info['time'] = time.time() - t0
        info['residual'] = rnorm
        return e, x, info


def kernel(h1, eri, norb, nelec, strs, comm=None, spin_flip=1, **kwargs):
    '''Energy (without the core energy), own rows of the vector, info; see
    DistProductCI.'''
    if comm is None:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    solver = DistProductCI(comm, h1, eri, norb, nelec, strs, spin_flip=spin_flip)
    e, c, info = solver.kernel(**kwargs)
    info['stat'] = dict(solver.stat)
    info['t_setup'] = solver.t_setup
    info['rows'] = solver.rows
    return e, c, info


# ---- the C driver (lib/zgasci/dist, zgd_solve) as an SQD solver ----------

_dist_lib = None


def _load_dist():
    '''libzgasci_dist.so: SOCUTILS_ZGASCI_DIST_LIBRARY or
    socutils/lib/zgasci/dist (make there, Makefile).'''
    global _dist_lib
    if _dist_lib is None:
        import ctypes
        from socutils.lib import library_dir
        path = _os.environ.get('SOCUTILS_ZGASCI_DIST_LIBRARY') or _os.path.join(
            library_dir(), 'zgasci', 'dist', 'libzgasci_dist.so')
        lib = ctypes.CDLL(path)
        p = ctypes.c_void_p
        lib.zgd_solve.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, p, p, ctypes.c_int64,
                                  p, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                  ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  p, p, p, p]
        lib.zgd_solve.restype = ctypes.c_int
        lib.zgd_solve_ab.argtypes = ([ctypes.c_int] * 5 + [p, p, p, ctypes.c_int64, p,
                                     ctypes.c_int64, p, ctypes.c_int] + [ctypes.c_double] * 3
                                     + [ctypes.c_int] * 3 + [p] * 7)
        lib.zgd_solve_ab.restype = ctypes.c_int
        lib.zgd_set_eigensolver.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.zgd_set_eigensolver.restype = None
        lib.zgd_set_pspace.argtypes = [ctypes.c_int]
        lib.zgd_set_pspace.restype = None
        lib.zgd_promote.argtypes = lib.zgd_reset.argtypes = []
        lib.zgd_promote.restype = lib.zgd_reset.restype = None
        _dist_lib = lib
    return _dist_lib


def c_solve(h1, eri, norb, nelec, strs, comm=None, spin_flip=1, ss_penalty=0.0, tol=1e-8,
            rtol=1e-3, maxit=40, verbose=0, warm=0, strs_b=None, ovlp=None, eigensolver=None,
            max_space=12, pspace=None):
    '''The lowest state in strs x strs_b by the C driver on comm (collective:
    every process calls with the same arguments).  h1 (norb, norb), eri any
    pyscf form of (pq|rs), no core energy; or UHF integrals (different alpha
    and beta orbitals): h1 = (h1a, h1b), eri = ((aa|aa), (aa|bb), (bb|bb))
    (the pyscf direct_uhf order, (p_a q_a | r_b s_b) in the middle), with
    ovlp[p, q] = <p alpha | q beta> (Ca^T S Cb) for the S^2 penalty; no
    symmetry then.  nelec total (nalpha = nbeta) or (na, nb).  strs_b None: the same as strs (alpha strings).  spin_flip +-1
    (same lists, na = nb): only the states with C[b, a] = spin_flip C[a, b];
    None or 0: no symmetry (any lists, any na, nb).
    ss_penalty: H + ss_penalty (S^2 - ss), ss = |S_z| (|S_z| + 1) (the
    linear penalty of fix_spin_).  warm: bit 1, start from the solution kept
    in the last round (zgd_promote) if it has weight >= 1e-4 in this space;
    bit 2, keep this solution if it is the lowest of the round
    (zgasci_dist.c, zgd_solve_ab).  eigensolver: 'lobpcg' (6 vectors) or
    'davidson' (2 max_space + 5 vectors); None: as before (default lobpcg,
    or ZGASCI_DIST_SOLVER / ZGASCI_DIST_MAX_SPACE).  pspace: k, the
    P-space preconditioner on k alpha x k beta strings (H_PP exact, the rest
    diagonal; 0 off; None: as before, default off, or ZGASCI_DIST_PSPACE).
    Returns a dict: e (<H>), e_penalized, s2 (NaN without a penalty),
    iterations, converged, time, warm_weight (the weight of the last
    solution in this space, -1: no warm start); occ_a, occ_b (norb);
    maxabs_a, weight_a per string of strs (the largest |C| of its row,
    sum_b C[a, b]^2), maxabs_b, weight_b per string of strs_b (over its
    column); occ, maxabs, weight: the alpha ones.  The same on every
    process.'''
    if comm is None:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    if isinstance(nelec, (int, numpy.integer)):
        if nelec % 2:
            raise ValueError('nelec as (nalpha, nbeta) for nalpha != nbeta')
        nelec = (nelec // 2, nelec // 2)
    na, nb = int(nelec[0]), int(nelec[1])
    parity = int(spin_flip or 0)
    uhf = isinstance(eri, (tuple, list)) or numpy.ndim(h1) == 3
    if uhf:
        if len(h1) != 2 or len(eri) != 3:
            raise ValueError('uhf integrals: h1 = (h1a, h1b), eri = (aa, ab, bb)')
        parity = 0
        h1 = numpy.ascontiguousarray(numpy.stack([numpy.asarray(x, dtype=numpy.float64) for x in h1]))
        eri = numpy.ascontiguousarray(numpy.stack(
            [ao2mo.restore(1, numpy.asarray(x, dtype=numpy.float64), norb) for x in eri]))
    else:
        h1 = numpy.ascontiguousarray(h1, dtype=numpy.float64)
        eri = numpy.ascontiguousarray(ao2mo.restore(1, numpy.asarray(eri), norb),
                                      dtype=numpy.float64)
    if ovlp is not None:
        ovlp = numpy.ascontiguousarray(ovlp, dtype=numpy.float64)
    if uhf and ss_penalty and ovlp is None:
        raise ValueError('the S^2 penalty with uhf integrals needs ovlp')
    sa = numpy.ascontiguousarray(strs, dtype=numpy.int64)
    sb = sa if strs_b is None else numpy.ascontiguousarray(strs_b, dtype=numpy.int64)
    # the arrays of rank 0 on every process (e.g. orbitals from separate SCF runs may differ in
    # signs between processes, which would make the integrals inconsistent)
    h1, eri, sa, sb = (comm.bcast(x, root=0) if x.nbytes < 1 << 28 else _bcast_array(comm, x)
                       for x in (h1, eri, sa, sb))
    if ovlp is not None:
        ovlp = comm.bcast(ovlp, root=0)
    if parity and not (na == nb and numpy.array_equal(sa, sb)):
        raise ValueError('spin_flip needs the same alpha and beta strings and nalpha = nbeta')
    if eigensolver is not None:
        if eigensolver not in ('lobpcg', 'davidson'):
            raise ValueError('eigensolver: lobpcg or davidson')
        _load_dist().zgd_set_eigensolver(int(eigensolver == 'davidson'), int(max_space))
    if pspace is not None:
        _load_dist().zgd_set_pspace(int(pspace))
    out = numpy.zeros(7)
    occ_a, occ_b = numpy.zeros(norb), numpy.zeros(norb)
    maxabs_a, weight_a = numpy.zeros(len(sa)), numpy.zeros(len(sa))
    maxabs_b, weight_b = numpy.zeros(len(sb)), numpy.zeros(len(sb))
    _load_dist().zgd_solve_ab(comm.py2f(), norb, na, nb, int(uhf), h1.ctypes.data,
                              eri.ctypes.data, None if ovlp is None or not uhf else ovlp.ctypes.data,
                              len(sa), sa.ctypes.data, len(sb), sb.ctypes.data, parity,
                              float(ss_penalty), tol, rtol, maxit, verbose, int(warm),
                              out.ctypes.data, occ_a.ctypes.data, occ_b.ctypes.data,
                              maxabs_a.ctypes.data, maxabs_b.ctypes.data, weight_a.ctypes.data,
                              weight_b.ctypes.data)
    return dict(e=out[1], e_penalized=out[0], s2=out[2], iterations=int(out[3]),
                converged=bool(out[4]), time=out[5], warm_weight=out[6],
                occ_a=occ_a, occ_b=occ_b, maxabs_a=maxabs_a, maxabs_b=maxabs_b,
                weight_a=weight_a, weight_b=weight_b, occ=occ_a, maxabs=maxabs_a,
                weight=weight_a)


def _bcast_array(comm, x):
    '''x of rank 0 on every process (same shape and dtype everywhere).'''
    x = numpy.ascontiguousarray(x)
    comm.Bcast(x, root=0)
    return x


def _carryover(sa, sb, r, threshold, merge):
    '''The carryover of the default policy of qiskit-addon-sqd 0.14: the
    strings with max |C| >= threshold, by descending marginal weight; merge
    (symmetrize_spin): both sectors ranked together, repeats dropped, the
    same list for both.'''
    ka = numpy.nonzero(r['maxabs_a'] >= threshold)[0]
    kb = numpy.nonzero(r['maxabs_b'] >= threshold)[0]
    if merge:
        strs = numpy.concatenate([sa[ka], sb[kb]]).astype(numpy.int64)
        w = numpy.concatenate([r['weight_a'][ka], r['weight_b'][kb]])
        strs = strs[numpy.argsort(-w, kind='stable')]
        _, first = numpy.unique(strs, return_index=True)
        strs = strs[numpy.sort(first)]
        return strs, strs.copy()
    ca = sa[ka[numpy.argsort(-r['weight_a'][ka], kind='stable')]].astype(numpy.int64)
    cb = sb[kb[numpy.argsort(-r['weight_b'][kb], kind='stable')]].astype(numpy.int64)
    return ca, cb


def make_sci_solver(comm=None, carryover_threshold=1e-4, ss_penalty=0.0, spin_flip=1,
                    log=None, warm_start=True, symmetrize_spin=None, ovlp=None, **kwargs):
    '''An sci_solver for qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian
    (0.14) run collectively under MPI (every process runs the SQD script).
    Each subspace is solved by c_solve on all processes in turn: with
    spin_flip +-1 where its alpha and beta lists are the same and nalpha =
    nbeta (the symmetric path), otherwise without symmetry (any lists,
    nalpha != nbeta; spin_flip None forces it).  UHF integrals pass through
    diagonalize_fermionic_hamiltonian untouched: give one_body_tensor =
    (h1a, h1b), two_body_tensor = (aa, ab, bb) there (c_solve) and ovlp here
    for the S^2 penalty.  The
    results carry no state (sci_state None) but the carryover the default
    policy would choose (_carryover); symmetrize_spin: as given to
    diagonalize_fermionic_hamiltonian (None: whether the lists are the
    same).  warm_start: each call (an SQD iteration) starts its subspaces
    from the lowest solution of the previous call, restricted to the new
    space (kept on the C side).  log(info) is called on every result on the
    control process; kwargs go to c_solve.'''
    from qiskit_addon_sqd.fermion import SCIResult
    if comm is None:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

    lib = _load_dist()
    lib.zgd_reset()

    def solver(ci_strings, one_body, two_body, norb, nelec):
        lib.zgd_promote()
        res = []
        for sa, sb in ci_strings:
            sa, sb = numpy.asarray(sa), numpy.asarray(sb)
            same = (nelec[0] == nelec[1] and numpy.array_equal(sa, sb)
                    and not isinstance(two_body, (tuple, list)))
            r = c_solve(one_body, two_body, norb, nelec, sa, comm=comm, strs_b=sb,
                        spin_flip=spin_flip if same else None, ss_penalty=ss_penalty, ovlp=ovlp,
                        warm=3 if warm_start else 0, **kwargs)
            merge = numpy.array_equal(sa, sb) if symmetrize_spin is None else symmetrize_spin
            res.append(SCIResult(r['e'], None, orbital_occupancies=(r['occ_a'], r['occ_b']),
                                 carryover=_carryover(sa, sb, r, carryover_threshold, merge)))
            if log is not None and comm.Get_rank() == 0:
                r['nstr'], r['nstr_b'] = len(sa), len(sb)
                log(r)
        return res
    return solver
