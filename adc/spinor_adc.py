#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Spinor ADC(2) (two-component, j-adapted Kramers-unrestricted) on top of a
:class:`socutils.scf.spinor_hf.SpinorSCF` reference.

Every spinor orbital is a full complex two-component spin-orbital, so the ADC
working equations are the bare spin-orbital expressions written with the
antisymmetrised physicist integrals ``<pq||rs> = <pq|rs> - <pq|sr>`` and the
MP1 doubles amplitudes ``t_ij^ab = <ij||ab> / (e_i + e_j - e_a - e_b)``.

Implemented strict ADC(2):

  IP  (electron detachment):  1h   + 2h1p space
  EA  (electron attachment):  1p   + 2p1h space
  EE  (neutral excitation):   1p1h + 2p2h space

IP additionally supports ADC(2)-x (``method='adc(2)-x'``), which adds the
first-order interaction in the 2h1p block (hole-hole ladder 0.5<mn||ij> plus
the particle-hole ring <ma||ei>, i.e. the EOM-IP Hbar(2h1p,2h1p) with t=0).
Because ADC(2)-x splits the doublet/quartet satellites, the spinor IP-ADC(2)-x
roots match PySCF UADC (spin-orbital), not RADC.

All three are solved by a single complex-Hermitian Davidson matvec over the
restricted (i<j / a<b) antisymmetric satellite space, so the cost is O(N^5)
per iteration rather than the O(N^9) of an explicit dense diagonalisation
(small satellite spaces fall back to a dense solve for robustness).  The EE
1p1h block is CIS plus the second-order self-energies and a non-separable
ph-ph term, the 1p1h<->2p2h coupling is first order and the 2p2h block is
diagonal (strict ADC(2)).  Orbital
energies are taken from the mean field (canonical); under a legal canonical
rotation (per-orbital phase or a unitary mix within an exactly degenerate /
Kramers block) the orbital energies are unchanged and the eigenvalues are
invariant while the eigenvectors become complex.

Two-electron integrals are the spinor Coulomb integrals ``int2e_spinor``,
matching the picture-change convention used across socutils.  For a
non-relativistic :class:`SpinorSCF` reference the IP/EA roots reproduce the
PySCF RADC IP/EA values (each spatial root appearing with its Kramers
multiplicity).
'''

import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger


def _davidson(matvec, diag, nroots, max_space=None, tol=1e-9, max_cycle=200):
    '''Lowest ``nroots`` eigenpairs of a complex Hermitian operator given as a
    matvec (acting on a flat complex vector) and its real diagonal.'''
    dim = diag.size
    if dim <= max(2000, 8 * nroots):
        # small (incl. the typical IP/EA satellite spaces): build the dense
        # matrix by probing the matvec and diagonalise directly -- robust and
        # avoids Davidson's missing-root failure mode for dense, near-
        # degenerate satellite spectra.
        H = np.empty((dim, dim), dtype=complex)
        e = np.zeros(dim, dtype=complex)
        for k in range(dim):
            e[k] = 1.0
            H[:, k] = matvec(e)
            e[k] = 0.0
        w = scipy.linalg.eigh(H, eigvals_only=True,
                              subset_by_index=[0, min(dim - 1, nroots - 1)])
        return np.sort(w.real)

    # iterative block Davidson; start from a generous set of guess vectors
    # (2*nroots) so near-degenerate low roots are not skipped.
    nguess = min(dim, max(2 * nroots, nroots + 12))
    if max_space is None:
        max_space = min(dim, max(nguess + 2 * nroots + 20, 10 * nroots))
    order = np.argsort(diag.real)
    # Preallocate the subspace (dim x max_space) and its conjugate, growing by
    # column index instead of np.column_stack (which reallocates the whole
    # dim x m block every iteration).  conj(V) is maintained incrementally so
    # the subspace projection does not re-conjugate all of V each cycle.
    V = np.zeros((dim, max_space), dtype=complex)
    AV = np.zeros((dim, max_space), dtype=complex)
    Vc = np.zeros((dim, max_space), dtype=complex)
    G = np.zeros((dim, nguess), dtype=complex)
    for i in range(nguess):
        G[order[i], i] = 1.0
    G, _ = np.linalg.qr(G)
    ncol = nguess
    V[:, :ncol] = G
    Vc[:, :ncol] = G.conj()
    for i in range(ncol):
        AV[:, i] = matvec(V[:, i])
    theta_old = np.zeros(nroots)
    for _ in range(max_cycle):
        Vv, AVv, Vcv = V[:, :ncol], AV[:, :ncol], Vc[:, :ncol]
        S = Vcv.T @ AVv
        S = (S + S.conj().T) * 0.5
        w, y = np.linalg.eigh(S)
        w = w[:nroots]
        y = y[:, :nroots]
        X = Vv @ y
        AX = AVv @ y
        res = AX - X * w[None, :]
        rnorm = np.linalg.norm(res, axis=0)
        if np.max(np.abs(w - theta_old)) < tol and np.max(rnorm) < np.sqrt(tol):
            return np.sort(w)
        theta_old = w
        # preconditioned new directions
        add = []
        for i in range(nroots):
            if rnorm[i] < np.sqrt(tol):
                continue
            d = diag.real - w[i]
            d[np.abs(d) < 1e-8] = 1e-8
            t = res[:, i] / d
            add.append(t)
        if not add:
            return np.sort(w)
        Vnew = np.column_stack(add)
        # orthonormalise against current space
        Vnew -= Vv @ (Vcv.T @ Vnew)
        Vnew -= Vv @ (Vcv.T @ Vnew)
        q, r = np.linalg.qr(Vnew)
        keep = np.abs(np.diag(r)) > 1e-7
        if not np.any(keep):
            return np.sort(w)
        q = q[:, keep]
        if ncol + q.shape[1] > max_space:
            # restart from the current Ritz vectors (q stays orthogonal to them,
            # since span(X) subset span(V) and q _|_ V)
            V[:, :nroots] = X
            AV[:, :nroots] = AX
            Vc[:, :nroots] = X.conj()
            ncol = nroots
        nq = q.shape[1]
        for j in range(nq):
            AV[:, ncol + j] = matvec(q[:, j])
        V[:, ncol:ncol + nq] = q
        Vc[:, ncol:ncol + nq] = q.conj()
        ncol += nq
    return np.sort(w)


class _SpinorADCERIs:
    '''Antisymmetrised physicist MO integral blocks ``<pq||rs>`` needed by
    spinor MP2/ADC(2)/ADC(2)-x (no vvvv: those methods never need it).

    Blocks are built lazily on first access and cached, so each method only
    pays for the blocks it touches (e.g. IP never forms the expensive
    three-virtual ``vovv`` block).  Each block is transformed from the real
    spherical AO integrals (int2e_sph) and recombined into the j-spinor MO
    basis by the C ``socutils.lib.ao2mo.nrr_fast`` driver (s4 AO permutational
    symmetry + iltj/igtj index ordering, ~4x faster than pyscf's s1-only
    ``nrr_outcore`` on the big virtual blocks) -- so the full complex
    ``int2e_spinor`` AO tensor and the all-virtual block are never formed.
    Unique chemist sub-blocks are cached and the
    particle-exchange symmetry ``(pq|rs)=(rs|pq)`` is reused (transpose only).
    '''

    def __init__(self, mol, mo_coeff, nocc):
        self.mol = mol
        self.o = mo_coeff[:, :nocc]
        self.v = mo_coeff[:, nocc:]
        self._chem_cache = {}
        self._blk_cache = {}
        self._half_cache = {}

    def _bra_half(self, A, B):
        # e1 (bra) half transform, cached by bra pair so kets sharing the same
        # bra reuse it (e.g. EE vovv & voov both need the bra=(v,v) half).
        k = (id(A), id(B))
        vin = self._half_cache.get(k)
        if vin is None:
            from socutils.lib.ao2mo import nrr_fast
            vin = nrr_fast.bra_half(self.mol, A, B, motype='j-spinor')
            self._half_cache[k] = vin
        return vin

    def _chem(self, A, B, C, D):
        cache = self._chem_cache
        key = (id(A), id(B), id(C), id(D))
        if key in cache:
            return cache[key]
        ex = (id(C), id(D), id(A), id(B))
        if ex in cache:
            out = cache[ex].transpose(2, 3, 0, 1)
            cache[key] = out
            return out
        from socutils.lib.ao2mo import nrr_fast
        g = np.asarray(nrr_fast.ket_transform(
            self.mol, self._bra_half(A, B), C, D, motype='j-spinor'))
        g = g.reshape(A.shape[1], B.shape[1], C.shape[1], D.shape[1])
        cache[key] = g
        return g

    def _blk(self, P, Q, R, S):
        # <PQ||RS> = (PR|QS) - (PS|QR)  (chemist -> antisym physicist)
        a = self._chem(P, R, Q, S).transpose(0, 2, 1, 3)
        b = self._chem(P, S, Q, R).transpose(0, 2, 3, 1)
        return a - b

    def _get(self, name, builder):
        if name not in self._blk_cache:
            self._blk_cache[name] = builder()
        return self._blk_cache[name]

    @property
    def oovv(self):                                # <ij||ab>
        return self._get('oovv', lambda: self._blk(self.o, self.o, self.v, self.v))

    @property
    def oooo(self):                                # <ij||kl>
        return self._get('oooo', lambda: self._blk(self.o, self.o, self.o, self.o))

    @property
    def ooov(self):                                # <ij||ka>
        return self._get('ooov', lambda: self._blk(self.o, self.o, self.o, self.v))

    @property
    def voov(self):                                # <ai||jb>
        return self._get('voov', lambda: self._blk(self.v, self.o, self.o, self.v))

    @property
    def vovv(self):                                # <ai||bc>
        return self._get('vovv', lambda: self._blk(self.v, self.o, self.v, self.v))

    @property
    def vvvv(self):                                # <ab||cd> (only EA/EE-ADC(2)-x)
        return self._get('vvvv', lambda: self._blk(self.v, self.v, self.v, self.v))

    @property
    def ovvo(self):                                # <ia||bj>
        return self._get('ovvo', lambda: self.voov.transpose(1, 0, 3, 2))

    @property
    def vvvo(self):                                # <bc||ak>
        return self._get('vvvo', lambda: self.vovv.conj().transpose(2, 3, 0, 1))

    @property
    def ovoo(self):                                # <ia||jk>
        return self._get('ovoo', lambda: self.ooov.conj().transpose(2, 3, 0, 1))


class SpinorADC(lib.StreamObject):
    '''Spinor ADC(2).

    Args:
        mf : a converged :class:`SpinorSCF` (or subclass) mean field.

    The MO coefficients/occupations are read from ``mf`` (or supplied), so
    injecting rotated orbitals (mf.copy() + overwrite mo_coeff) works without
    re-diagonalisation: the same canonical mo_energy is reused.
    '''

    def __init__(self, mf, mo_coeff=None, mo_occ=None):
        self._scf = mf
        self.mol = mf.mol
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.max_memory = mf.max_memory
        self.mo_coeff = mf.mo_coeff if mo_coeff is None else mo_coeff
        self.mo_occ = mf.mo_occ if mo_occ is None else mo_occ
        self.mo_energy = mf.mo_energy

        self._eris = None
        self._t2 = None
        self._sig_ip = None
        self._sig_ea = None

    @property
    def nocc(self):
        return int(np.count_nonzero(self.mo_occ > 0))

    @property
    def nmo(self):
        return self.mo_coeff.shape[1]

    # -- shared intermediates ------------------------------------------------
    def _build(self):
        if self._eris is not None:
            return
        no = self.nocc
        eris = _SpinorADCERIs(self.mol, np.asarray(self.mo_coeff), no)
        e = np.asarray(self.mo_energy)
        eo, ev = e[:no], e[no:]
        Voovv = eris.oovv                          # <ij||ab>
        d = (eo[:, None, None, None] + eo[None, :, None, None]
             - ev[None, None, :, None] - ev[None, None, None, :])
        t2 = Voovv / d
        # Static second-order self-energies (same as the IP/EA quasiparticle
        # corrections).  The t2.conj() placement makes the blocks transform
        # correctly under a complex orbital-phase gauge (holes ~ ph_i*).
        self._sig_ip = 0.25 * (lib.einsum('ikab,jkab->ij', t2.conj(), Voovv)
                               + lib.einsum('jkab,ikab->ij', t2, Voovv.conj()))
        self._sig_ea = 0.25 * (lib.einsum('ijac,ijbc->ab', t2.conj(), Voovv)
                               + lib.einsum('ijbc,ijac->ab', t2, Voovv.conj()))
        self._eris = eris
        self._t2 = t2

    # -- IP ------------------------------------------------------------------
    def ip_adc2(self, nroots=6, method='adc(2)'):
        '''Spinor IP ionization energies (lowest ``nroots``).

        ``method`` selects strict ADC(2) (default) or ADC(2)-x, which adds the
        first-order interaction in the 2h1p block (hole-hole ladder 0.5<mn||ij>
        plus the particle-hole ring <ma||ei>).  Solved by a complex-Hermitian
        Davidson matvec over the restricted (i<j) 2h1p space, so the cost is
        O(N^5) per iteration rather than the O(N^9) of a dense diagonalisation.
        '''
        self._build()
        no, nmo = self.nocc, self.nmo
        nv = nmo - no
        e = np.asarray(self.mo_energy)
        eo, ev = e[:no], e[no:]
        eris = self._eris

        Mss = -np.diag(eo).astype(complex) - self._sig_ip
        Wooov = eris.ooov                          # <kl||ia>
        WooovC = Wooov.conj()                       # precompute once (was per matvec)
        adc2x = (method == 'adc(2)-x')
        if not adc2x and method != 'adc(2)':
            raise NotImplementedError(method)
        if adc2x:
            Woooo = eris.oooo                      # <mn||ij>
            Wovvo = eris.ovvo                      # <ma||ei>

        oo = np.array([(k, l) for k in range(no) for l in range(k + 1, no)],
                      dtype=int).reshape(-1, 2)
        nop = len(oo)
        ok, ol = oo[:, 0], oo[:, 1]
        dim = no + nop * nv

        def unpack(rp):
            r2 = np.zeros((no, no, nv), dtype=complex)
            blk = rp.reshape(nop, nv)
            r2[ok, ol, :] = blk
            r2[ol, ok, :] = -blk
            return r2

        def pack(s2):
            return s2[ok, ol, :].reshape(-1)

        def matvec(x):
            r1 = x[:no]
            r2 = unpack(x[no:])
            s1 = Mss @ r1 + 0.5 * lib.einsum('klia,kla->i', Wooov, r2)
            s2 = (ev[None, None, :] - eo[:, None, None] - eo[None, :, None]) * r2
            s2 += lib.einsum('klia,i->kla', WooovC, r1)
            if adc2x:
                s2 += 0.5 * lib.einsum('mnij,mna->ija', Woooo, r2)
                t = lib.einsum('maei,mje->ija', Wovvo, r2)
                s2 += t - t.transpose(1, 0, 2)
            return np.concatenate([s1, pack(s2)])

        diag = np.empty(dim)
        diag[:no] = Mss.diagonal().real
        dd = ev[None, :] - (eo[ok] + eo[ol])[:, None]      # (nop, nv)
        if adc2x:
            # exact 2h1p diagonal: + <ij||ij> - <ia||ia> - <ja||ja>
            oooo_d = np.einsum('ijij->ij', eris.oooo).real     # <ij||ij>
            ovov_d = -np.einsum('iaai->ia', eris.ovvo).real    # <ia||ia>
            dd = (dd + oooo_d[ok, ol][:, None]
                  - ovov_d[ok, :] - ovov_d[ol, :])
        diag[no:] = dd.reshape(-1)
        return _davidson(matvec, diag, nroots)

    def ip_adc2x(self, nroots=6):
        '''Spinor IP-ADC(2)-x ionization energies.'''
        return self.ip_adc2(nroots, method='adc(2)-x')

    # -- EA ------------------------------------------------------------------
    def ea_adc2(self, nroots=6, method='adc(2)'):
        '''Spinor EA electron affinities (lowest ``nroots``).

        ``method`` selects strict ADC(2) (default) or ADC(2)-x, which adds the
        first-order interaction in the 2p1h block (particle-particle ladder
        0.5<ab||cd> plus the particle-hole ring <jb||ic>).  Davidson matvec over
        the restricted (a<b) 2p1h space.
        '''
        self._build()
        no, nmo = self.nocc, self.nmo
        nv = nmo - no
        e = np.asarray(self.mo_energy)
        eo, ev = e[:no], e[no:]
        eris = self._eris
        adc2x = (method == 'adc(2)-x')
        if not adc2x and method != 'adc(2)':
            raise NotImplementedError(method)

        Mpp = np.diag(ev).astype(complex) - self._sig_ea
        # <ai||cd> packed over the antisymmetric particle pair (c<d): the 2p1h
        # vector is already stored packed, so the dominant contraction runs
        # over c<d only (0.5 sum_full == sum_{c<d}) -- half the flops/storage.
        vv = np.array([(c, d) for c in range(nv) for d in range(c + 1, nv)],
                      dtype=int).reshape(-1, 2)
        nvp = len(vv)
        vc, vd = vv[:, 0], vv[:, 1]
        WvovvP = eris.vovv[:, :, vc, vd]           # <ai||(cd)>  [a,i,P]
        # Both 2p1h couplings are plain GEMMs, but lib.einsum('aiP,...') hides a
        # 47 MB transpose-copy per matvec.  Precompute the contiguous (a, P*i)
        # layout once so the matvec dots have no per-call transpose (~10-20x).
        WvovvP_Pi = np.ascontiguousarray(WvovvP.transpose(0, 2, 1)).reshape(nv, -1)
        WvovvPc_Pi = WvovvP_Pi.conj()
        ev_p = ev[vc] + ev[vd]
        dim = nv + nvp * no
        if adc2x:
            vvvv = eris.vvvv                        # <ab||cd>
            # packed pp-ladder matrix <(ab)||(cd)>  (nvp, nvp)
            Wvvvv_pp = vvvv[vc[:, None], vd[:, None], vc[None, :], vd[None, :]]
            Vjbic = -eris.voov.transpose(1, 0, 2, 3)   # <jb||ic> = -<bj||ic>

        def matvec(x):
            r1 = x[:nv]
            r2 = x[nv:].reshape(nvp, no)           # [P=(c<d), i]
            s1 = Mpp @ r1 + WvovvP_Pi @ r2.ravel()
            s2 = (ev_p[:, None] - eo[None, :]) * r2
            s2 += (r1 @ WvovvPc_Pi).reshape(nvp, no)
            if adc2x:
                s2 += Wvvvv_pp @ r2                # pp ladder
                r2f = np.zeros((no, nv, nv), dtype=complex)
                r2f[:, vc, vd] = r2.T
                r2f[:, vd, vc] = -r2.T
                t = -lib.einsum('jbic,jac->iab', Vjbic, r2f)   # ph ring
                ring = t - t.transpose(0, 2, 1)               # P(ab)
                s2 += ring[:, vc, vd].T
            return np.concatenate([s1, s2.reshape(-1)])

        diag = np.empty(dim)
        diag[:nv] = Mpp.diagonal().real
        dd = ev_p[:, None] - eo[None, :]                   # (nvp, no)
        if adc2x:
            # exact 2p1h diagonal: + <ab||ab> - <ia||ia> - <ib||ib>
            vvvv_d = Wvvvv_pp.diagonal().real              # <ab||ab>
            ovov_d = -np.einsum('iaai->ia', eris.ovvo).real  # <ia||ia>
            dd = (dd + vvvv_d[:, None]
                  - ovov_d.T[vc, :] - ovov_d.T[vd, :])
        diag[nv:] = dd.reshape(-1)
        return _davidson(matvec, diag, nroots)

    def ea_adc2x(self, nroots=6):
        '''Spinor EA-ADC(2)-x electron affinities.'''
        return self.ea_adc2(nroots, method='adc(2)-x')


    # -- EE ------------------------------------------------------------------
    def ee_adc2(self, nroots=6):
        '''Spinor EE-ADC(2) excitation energies (lowest ``nroots``).

        The 1p1h block is CIS + the second-order self-energies and the
        non-separable ph-ph term; the 1p1h<->2p2h coupling is first order and
        the 2p2h block is diagonal (strict ADC(2)).  Solved by a Davidson
        matvec (the 2p2h space is too large to store densely).
        '''
        self._build()
        no, nmo = self.nocc, self.nmo
        nv = nmo - no
        e = np.asarray(self.mo_energy)
        eo, ev = e[:no], e[no:]
        eris = self._eris
        t2 = self._t2
        Voovv = eris.oovv

        # ph-ph second-order non-separable term, with the first-order CIS-like
        # ring <aj||ib> folded in (same 'iajb,jb->ia' contraction) so the
        # matvec runs one einsum on a contiguous [i,a,j,b] block instead of two
        # (the separate <aj||ib> term used a transposed layout -> per-matvec copy)
        Lam = 0.5 * (lib.einsum('imae,jmbe->iajb', t2.conj(), Voovv)
                     + lib.einsum('jmbe,imae->iajb', t2, Voovv.conj()))
        Lam += eris.voov.transpose(2, 0, 1, 3)     # <aj||ib> -> [i,a,j,b]
        Wvovv = eris.vovv                          # <ak||bc>
        Wooov = eris.ooov                          # <jk||ib>
        Wovoo = eris.ovoo                          # <ib||jk>
        sig_ip, sig_ea = self._sig_ip, self._sig_ea
        Dijab = (-eo[:, None, None, None] - eo[None, :, None, None]
                 + ev[None, None, :, None] + ev[None, None, None, :])

        # restricted antisymmetric pair indices for the 2p2h vector
        oo = np.array([(j, k) for j in range(no) for k in range(j + 1, no)],
                      dtype=int).reshape(-1, 2)
        vv = np.array([(b, c) for b in range(nv) for c in range(b + 1, nv)],
                      dtype=int).reshape(-1, 2)
        nop, nvp = len(oo), len(vv)
        ns = no * nv
        ndp = nop * nvp
        dim = ns + ndp

        oj, ok = oo[:, 0], oo[:, 1]
        vb, vc = vv[:, 0], vv[:, 1]

        # Pack the antisymmetric particle pair (b<c) in the two dominant
        # O(no^2 nv^3) contractions (akbc, bcak); the smaller O(no^3 nv^2)
        # terms (jkib, ibjk) stay on the full virtual range.
        WvovvP = Wvovv[:, :, vb, vc]               # <ak||(bc)>  [a,k,P]
        # <(bc)||ak> = conj(<ak||(bc)>); reuse WvovvP.conj() instead of building
        # eris.vvvo and reshaping its [P,a,k] layout every matvec.
        WvovvPc = WvovvP.conj()
        # 2D (a, k*P) views: the (k,P) contractions below are plain GEMMs, but
        # lib.einsum('akP,...') hides a 47 MB transpose-copy per matvec -- do the
        # reshape+dot explicitly (~8x faster on these two dominant terms).
        WvovvP_m = WvovvP.reshape(nv, -1)
        WvovvPc_m = WvovvPc.reshape(nv, -1)
        D_hf = (ev[vb] + ev[vc])[None, None, :] - eo[:, None, None] \
            - eo[None, :, None]                    # [j,k,P]

        def unpack_holes(rp):                      # -> r2_hf[j,k,P]
            r2 = np.zeros((no, no, nvp), dtype=complex)
            blk = rp.reshape(nop, nvp)
            r2[oj, ok, :] = blk
            r2[ok, oj, :] = -blk
            return r2

        def pack_holes(s2):                        # r2_hf[j,k,P] -> packed
            return s2[oj, ok, :].reshape(-1)

        # full-virtual scratch for the jkib term, reused across matvecs (only
        # the (vb,vc)/(vc,vb) entries are ever written; the rest stay zero).
        r2f = np.zeros((no, no, nv, nv), dtype=complex)

        def matvec(x):
            r1 = x[:ns].reshape(no, nv)
            r2 = unpack_holes(x[ns:])              # [j,k,P], holes full
            s1 = (ev[None, :] - eo[:, None]) * r1
            s1 -= lib.einsum('ij,ja->ia', sig_ip, r1)
            s1 -= lib.einsum('ab,ib->ia', sig_ea, r1)
            s1 += lib.einsum('iajb,jb->ia', Lam, r1)    # incl. <aj||ib>
            s1 += r2.reshape(no, -1) @ WvovvP_m.T                 # akbc (packed)
            # jkib term needs the full virtual range (a free, b summed)
            r2f[:, :, vb, vc] = r2
            r2f[:, :, vc, vb] = -r2
            s1 -= 0.5 * lib.einsum('jkib,jkab->ia', Wooov, r2f)
            s2 = D_hf * r2                                        # diagonal
            tA = (r1 @ WvovvPc_m).reshape(no, no, nvp)           # bcak (packed)
            s2 += tA - tA.transpose(1, 0, 2)
            tB = lib.einsum('ibjk,ic->jkbc', Wovoo, r1)
            tB = tB - tB.transpose(0, 1, 3, 2)
            s2 += tB[:, :, vb, vc]
            return np.concatenate([s1.ravel(), pack_holes(s2)])

        diag = np.empty(dim)
        diag[:ns] = (ev[None, :] - eo[:, None]).ravel().real
        dd = Dijab[oj[:, None], ok[:, None], vb[None, :], vc[None, :]].real
        diag[ns:] = dd.reshape(-1)
        return _davidson(matvec, diag, nroots)


if __name__ == '__main__':
    from pyscf import gto, scf, adc
    from socutils.scf import spinor_hf
    mol = gto.M(atom='Ne 0 0 0', basis='cc-pvdz', verbose=0)
    mf = spinor_hf.SpinorSCF(mol); mf.verbose = 0; mf.kernel()
    myadc = SpinorADC(mf)
    print('spinor IP', np.round(np.unique(np.round(myadc.ip_adc2(8), 6)), 5)[:3])
    print('spinor EA', np.round(np.unique(np.round(myadc.ea_adc2(8), 6)), 5)[:3])
    print('spinor EE', np.round(np.unique(np.round(myadc.ee_adc2(8), 6)), 5)[:3])
    rmf = scf.RHF(mol).run()
    for t in ('ip', 'ea'):
        a = adc.ADC(rmf); a.method = 'adc(2)'; a.method_type = t; a.verbose = 0
        print('pyscf ', t, np.round(np.sort(np.asarray(a.kernel(nroots=4)[0])), 5))
    umf = scf.UHF(mol).run()
    a = adc.ADC(umf); a.method = 'adc(2)'; a.method_type = 'ee'; a.verbose = 0
    print('pyscf  ee', np.round(np.unique(np.round(a.kernel(nroots=8)[0], 6)), 5)[:3])
