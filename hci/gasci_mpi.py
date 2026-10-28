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
            local_min = (float(hd.min()), self.rank, int(hd.argmin())) if self.nloc else (numpy.inf, self.rank, -1)
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
            # T = -(HX - e X) / (hd - e), by rows (no full-length temporary)
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
                d = hd[r0:r0 + 256] - e
                d[numpy.abs(d) < level_shift] = level_shift
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
