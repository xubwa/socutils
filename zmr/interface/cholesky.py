'''
AO -> MO transformation and mean-field potentials from Cholesky (or
density-fitting) vectors of the Coulomb integrals,

    (mn|ls) = sum_L L_mn L_ls,        L real and symmetric in the spherical AOs.

For orbitals with spherical components c^s (alpha and beta for two-component
orbitals, a single component for real spatial orbitals)

    B[L, p, q] = sum_s sum_mn conj(c^s_mp) L_mn c^s_nq,
    (pq|rs) = sum_L B[L, p, q] B[L, r, s],

so a block costs two half transformations per vector, the first one with the
smaller of the two coefficient blocks, and one matrix product.  Nothing of
the size of the AO integrals is stored.

The layout follows socutils.mcscf.zmc_ao2mo._CDERIS: the vectors are read
block by block in packed form (lower triangle), from an array or from a
PySCF density-fitting object (with_df.loop, e.g. the with_df of the SCF that
points to cderi.chk), and the first half transformation L c is done by the
same C driver of PySCF (AO2MOnr_e2_drv on packed vectors), separately for
the real and imaginary parts of each spin component.

The mean-field potential of a density dm = sum_i w_i v_i v_i^+ (its
eigendecomposition; few non-zero w_i) is evaluated in the spherical AOs,

    J = sum_L L tr(L D),    D = sum_s sum_i w_i c^s_i c^s_i^+,
    K[s mu, s' nu] = sum_L sum_i w_i (L c^s_i)_mu conj((L c^s'_i)_nu),

and transformed back to the basis of the coefficients.

The purpose is to use the vectors of the orbital optimization (the modified
Cholesky decomposition of the socutils CASSCF), so that the perturbation
theory works with the Hamiltonian the orbitals and the reference energies
belong to.
'''

import ctypes

import numpy
from pyscf import lib as pyscf_lib
from pyscf.ao2mo import _ao2mo

from socutils.zmr.lib import linalg as la


def _same(x, y):
    return x is y or (x.shape == y.shape and numpy.array_equal(x, y))


def _identity(c):
    return (c,)


class Occupied:
    '''Mean-field quantities of the occupied orbitals (CholeskyAO2MO.occupied).'''

    def __init__(self, k_core, j_core, x_act):
        self.k_core = k_core
        self.j_core = j_core
        self.x_act = x_act


class CholeskyAO2MO:
    '''eri_fn(c1, c2, c3, c4) -> (pq|rs), and veff(dm) -> J[dm] - exchange K[dm].

    Args:
        cderi: real vectors, (naux, nao (nao + 1) / 2) (lower triangle, as
            pyscf.lib.pack_tril), (naux, nao * nao) or (naux, nao, nao); or a
            PySCF density-fitting object (loop(blksize) yielding packed
            blocks, get_naoaux()), which is read block by block
        nao: number of spherical AOs
        components: function mapping a coefficient block to the tuple of its
            spherical AO components, each (nao, n); default: the block itself
            (real spatial orbitals)
        exchange: factor of K in veff (1 for spin-orbitals / spinors, 1/2 for
            a spin-summed density of spatial orbitals)
        max_memory: size of the unpacked vectors handled at a time, in MB
    '''

    def __init__(self, cderi, nao, components=None, exchange=1.0, max_memory=200):
        npair = nao * (nao + 1) // 2
        self.with_df = None
        if hasattr(cderi, 'loop') and hasattr(cderi, 'get_naoaux'):
            self.with_df = cderi
            self.cderi = None
            self.naux = int(cderi.get_naoaux())
        else:
            cderi = numpy.asarray(cderi)
            if cderi.ndim == 3 or (cderi.ndim == 2 and cderi.shape[1] == nao * nao and nao > 1):
                cderi = pyscf_lib.pack_tril(cderi.reshape(-1, nao, nao))
            if cderi.ndim != 2 or cderi.shape[1] != npair:
                raise ValueError('cderi of shape %s does not match %d AOs' % (cderi.shape, nao))
            self.cderi = numpy.ascontiguousarray(cderi, dtype=numpy.float64)
            self.naux = cderi.shape[0]
        self.nao = nao
        self.components = _identity if components is None else components
        self.exchange = exchange
        self.max_memory = max_memory

    def loop(self, nhalf):
        '''(p0, p1, packed vectors p0:p1) in blocks such that the half
        transformed vectors (nhalf complex numbers per vector, two copies)
        fit max_memory.'''
        step = max(1, int(self.max_memory * 1e6 // (32 * max(nhalf, 1))))
        if self.with_df is None:
            for p0, p1 in pyscf_lib.prange(0, self.naux, step):
                yield p0, p1, self.cderi[p0:p1]
            return
        p0 = 0
        for block in self.with_df.loop(min(step, self.naux)):
            block = numpy.ascontiguousarray(block, dtype=numpy.float64)
            yield p0, p0 + block.shape[0], block
            p0 += block.shape[0]

    def _half(self, packed, comps):
        '''x[s, L, i, mu] = sum_nu L_(mu nu) c^s_(nu i) for a block of packed
        vectors, through PySCF's AO2MOnr_e2_drv (as _CDERIS._half_transform).'''
        nao = self.nao
        nb = packed.shape[0]
        n = comps[0].shape[1]
        real = not any(numpy.iscomplexobj(c) for c in comps)
        x = numpy.empty((len(comps), nb, n, nao),
                        dtype=numpy.float64 if real else numpy.complex128)
        buf = numpy.empty((nb * n, nao))
        lib = _ao2mo.libao2mo
        for s, c in enumerate(comps):
            for part in ('real',) if real else ('real', 'imag'):
                coeff = numpy.asfortranarray(getattr(c, part), dtype=numpy.float64)
                lib.AO2MOnr_e2_drv(lib.AO2MOtranse2_nr_s2, lib.AO2MOmmm_bra_nr_s2,
                                   buf.ctypes.data_as(ctypes.c_void_p),
                                   packed.ctypes.data_as(ctypes.c_void_p),
                                   coeff.ctypes.data_as(ctypes.c_void_p),
                                   ctypes.c_int(nb), ctypes.c_int(nao),
                                   (ctypes.c_int * 4)(0, n, 0, nao),
                                   pyscf_lib.c_null_ptr(), ctypes.c_int(0))
                if real:
                    x[s] = buf.reshape(nb, n, nao)
                else:
                    getattr(x[s], part)[:] = buf.reshape(nb, n, nao)
        return x

    def pair(self, c1, c2):
        '''B[L, p, q] for two coefficient blocks, as an array (naux, n1 * n2).'''
        comps1, comps2 = self.components(c1), self.components(c2)
        n1, n2 = c1.shape[1], c2.shape[1]
        ncomp, nao = len(comps1), self.nao
        real = not any(numpy.iscomplexobj(c) for c in tuple(comps1) + tuple(comps2))
        out = numpy.empty((self.naux, n1, n2), dtype=numpy.float64 if real else numpy.complex128)
        if n2 <= n1:
            # ket index first: x[s, L, q, mu], then all bra orbitals in one product
            left = numpy.concatenate(comps1, axis=0).conj()        # (s mu, p)
            for p0, p1, packed in self.loop(ncomp * nao * n2):
                x = self._half(packed, comps2)
                x = x.transpose(1, 2, 0, 3).reshape((p1 - p0) * n2, ncomp * nao)
                b = la.gemm(x, left)                               # (L q, p)
                out[p0:p1] = b.reshape(p1 - p0, n2, n1).transpose(0, 2, 1)
        else:
            right = numpy.concatenate(comps2, axis=0)              # (s nu, q)
            conj1 = tuple(c.conj() for c in comps1)
            for p0, p1, packed in self.loop(ncomp * nao * n1):
                x = self._half(packed, conj1)                      # (s, L, p, nu)
                x = x.transpose(1, 2, 0, 3).reshape((p1 - p0) * n1, ncomp * nao)
                out[p0:p1] = la.gemm(x, right).reshape(p1 - p0, n1, n2)
        return out.reshape(self.naux, n1 * n2)

    def __call__(self, c1, c2, c3, c4):
        shape = tuple(c.shape[1] for c in (c1, c2, c3, c4))
        if 0 in shape:
            real = not any(numpy.iscomplexobj(c) for c in (c1, c2, c3, c4))
            return numpy.zeros(shape, dtype=numpy.float64 if real else numpy.complex128)
        bra = self.pair(c1, c2)
        ket = bra if _same(c3, c1) and _same(c4, c2) else self.pair(c3, c4)
        return la.gemm(bra, ket, 'T', 'N').reshape(shape)

    # ------------------------------------------------------------------
    # mean field of the occupied orbitals in one streamed pass, as
    # socutils _CDERIS._build_cd_pa_and_vk: the vectors are read block by
    # block, half transformed with the core and active orbitals, and the
    # exchange matrix of the core is accumulated from the half-transformed
    # block itself (layout (L i, mu): no transposition, nothing of the size
    # of the three-index integrals kept).  Only the half-transformed active
    # vectors are stored; they give cd_pa for any orbitals and the active
    # exchange for any active density.

    def occupied(self, c_core, c_act, docc=1):
        '''Occupied(k_core, j_core, x_act) for core orbitals c_core (each
        occupied docc times) and active orbitals c_act: k_core and j_core in
        the spherical (spin) basis, x_act[L, t, (s mu)] = (L c_t^s)_mu.'''
        nao = self.nao
        cc, ca = self.components(c_core), self.components(c_act)
        ncomp = len(cc)
        nc, na = c_core.shape[1], c_act.shape[1]
        real = not any(numpy.iscomplexobj(x) for x in tuple(cc) + tuple(ca))
        dtype = numpy.float64 if real else numpy.complex128
        big = ncomp * nao
        kc = numpy.zeros((big, big), dtype=dtype)
        jpack = numpy.zeros(nao * (nao + 1) // 2)
        x_act = numpy.empty((self.naux, na, big), dtype=dtype)
        dpack = None
        if nc:
            dsph = sum((c @ c.conj().T).real for c in cc)
            dpack = pyscf_lib.pack_tril(dsph + dsph.T)
            diag = numpy.arange(nao)
            dpack[diag * (diag + 1) // 2 + diag] *= 0.5
        for p0, p1, packed in self.loop(ncomp * nao * (nc + na)):
            nb = p1 - p0
            if nc:
                jpack += la.gemm(la.gemm(dpack[None, :], packed, 'N', 'T'), packed)[0]
                x = self._half(packed, cc)                       # (s, L, i, mu)
                x = x.transpose(1, 2, 0, 3).reshape(nb * nc, big)
                kc += la.herk(x, 'C')                            # sum conj(x_m) x_n
            if na:
                x = self._half(packed, ca)
                x_act[p0:p1] = x.transpose(1, 2, 0, 3).reshape(nb, na, big)
        jc = docc * pyscf_lib.unpack_tril(jpack) if nc else numpy.zeros((nao, nao))
        # k[m, n] = sum x_m conj(x_n) is the transpose of the accumulated matrix
        return Occupied(numpy.ascontiguousarray(kc.T), jc, x_act)

    def cd_from_half(self, x, c):
        '''B[L, p, t] = sum_(s mu) conj(c^s_(mu p)) x[L, t, (s mu)]: the second
        half transformation of stored half-transformed vectors.'''
        naux, n2, big = x.shape
        left = numpy.concatenate(self.components(c), axis=0).conj()
        b = la.gemm(numpy.ascontiguousarray(x).reshape(naux * n2, big), left)
        return numpy.ascontiguousarray(b.reshape(naux, n2, -1).transpose(0, 2, 1))

    def fock(self, occ, gamma, c, c_act, kact=1.0):
        '''(J - K)[p, q] of the core plus the active density gamma[t, u] =
        <t^+ u> for the orbitals c; the active exchange is weighted by kact.'''
        nao = self.nao
        vj = occ.j_core
        vk = occ.k_core
        gamma = numpy.asarray(gamma)
        naux, na, big = occ.x_act.shape
        if na and abs(gamma).max(initial=0) > 0:
            xf = occ.x_act.reshape(naux * na, big)
            # charge of the active density: rho[L] = sum_tu gamma[t, u] B[L, t, u]
            baa = self.cd_from_half(occ.x_act, c_act)             # (L, t, u)
            rho = numpy.einsum('ltu,tu->l', baa, gamma)
            if abs(numpy.imag(rho)).max(initial=0) < 1e-14 * max(abs(rho).max(), 1e-300):
                rho = numpy.real(rho)
            jpack = 0
            for p0, p1, packed in self.loop(1):
                jpack = jpack + rho[p0:p1] @ packed
            parts = [jpack] if not numpy.iscomplexobj(jpack) else [jpack.real, jpack.imag]
            ja = [pyscf_lib.unpack_tril(numpy.ascontiguousarray(q)) for q in parts]
            vj = vj + (ja[0] if len(ja) == 1 else ja[0] + 1j * ja[1])
            # k[m, n] = sum_L sum_tu x_t[m] gamma[u, t] conj(x_u[n])
            z = numpy.matmul(gamma, occ.x_act).reshape(naux * na, big)
            vk = vk + kact * la.gemm(xf, z, 'C', 'N').T
        comps = [numpy.ascontiguousarray(q) for q in self.components(c)]
        stack = numpy.concatenate(comps, axis=0)
        out = -la.gemm(stack, la.gemm(numpy.ascontiguousarray(vk), stack), 'C', 'N')
        for cs in comps:
            out = out + la.gemm(cs, la.gemm(vj, cs), 'C', 'N')
        return out

    def coulomb(self, rho, c):
        '''J[p, q] = sum_L rho[L] B[L, p, q] for the orbitals c, for the charge
        rho[L] = sum_pq B[L, p, q] D[q, p] of a density given through
        three-index integrals.'''
        rho = numpy.asarray(rho)
        nao = self.nao
        real = not numpy.iscomplexobj(rho)
        jpack = numpy.zeros(nao * (nao + 1) // 2,
                            dtype=numpy.float64 if real else numpy.complex128)
        for p0, p1, packed in self.loop(1):
            jpack += rho[p0:p1] @ packed
        parts = [jpack] if real else [jpack.real, jpack.imag]
        jsph = [pyscf_lib.unpack_tril(numpy.ascontiguousarray(x)) for x in parts]
        jsph = jsph[0] if real else jsph[0] + 1j * jsph[1]
        out = 0
        for cs in self.components(c):
            cs = numpy.ascontiguousarray(cs)
            out = out + la.gemm(cs, la.gemm(jsph, cs), 'C', 'N')
        return out

    def veff(self, dm):
        '''J[dm] - exchange * K[dm] in the basis of the coefficients, with
        vj[m, n] = sum_ls (mn|ls) dm[s, l], vk[m, n] = sum_ls (ms|ln) dm[s, l].'''
        dm = numpy.asarray(dm)
        herm = 0.5 * (dm + dm.conj().T)
        anti = 0.5 * (dm - dm.conj().T)
        out = self._veff_hermitian(herm)
        if abs(anti).max(initial=0.0) > 1e-14 * max(abs(herm).max(initial=0.0), 1e-300):
            # dm = herm + i (anti / i), both parts Hermitian; J and K are linear
            out = out + 1j * self._veff_hermitian(anti / 1j)
        return out

    def _veff_hermitian(self, dm):
        n = dm.shape[0]
        nao = self.nao
        unit = numpy.concatenate(self.components(numpy.eye(n)), axis=0)
        ncomp = unit.shape[0] // nao
        real = not (numpy.iscomplexobj(dm) or numpy.iscomplexobj(unit))
        w, v = numpy.linalg.eigh(dm)
        keep = abs(w) > 1e-14 * max(abs(w).max(initial=0.0), 1e-300)
        w, v = w[keep], v[:, keep]
        r = len(w)
        if r == 0:
            return numpy.zeros((n, n), dtype=dm.dtype)
        comps = tuple(numpy.ascontiguousarray(c) for c in self.components(v))

        # J: only the real symmetric part of the spherical density contributes
        dsph = sum(((c * w) @ c.conj().T).real for c in comps)
        dpack = pyscf_lib.pack_tril(dsph + dsph.T)
        diag = numpy.arange(nao)
        dpack[diag * (diag + 1) // 2 + diag] *= 0.5
        jpack = numpy.zeros_like(dpack)

        ksph = numpy.zeros((ncomp * nao, ncomp * nao),
                           dtype=numpy.float64 if real else numpy.complex128)
        for p0, p1, packed in self.loop(ncomp * nao * r):
            jpack += la.gemm(la.gemm(dpack[None, :], packed, 'N', 'T'), packed)[0]
            if self.exchange:
                y = self._half(packed, comps)                      # (s, L, i, mu)
                y = numpy.ascontiguousarray(y.transpose(0, 3, 1, 2)).reshape(ncomp * nao, -1)
                ksph += la.gemm(y * numpy.tile(w, p1 - p0), y, 'N', 'C')
        jsph = pyscf_lib.unpack_tril(jpack)

        out = -self.exchange * la.gemm(unit, la.gemm(ksph, unit), 'C', 'N')
        for s in range(ncomp):
            us = numpy.ascontiguousarray(unit[s * nao:(s + 1) * nao])
            out = out + la.gemm(us, la.gemm(jsph, us), 'C', 'N')
        return out
