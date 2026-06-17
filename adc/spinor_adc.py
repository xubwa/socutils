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


def _davidson(matvec, diag, nroots, max_space=None, tol=1e-9, max_cycle=200,
              return_vecs=False):
    '''Lowest ``nroots`` eigenpairs of a complex Hermitian operator given as a
    matvec (acting on a flat complex vector) and its real diagonal.  Returns
    the sorted eigenvalues, or ``(eigenvalues, eigenvectors)`` (columns) when
    ``return_vecs``.

    Small spaces are diagonalised densely; larger ones use pyscf's robust
    Davidson (``lib.davidson_nosym1`` with ``pick_real_eigs`` -- the same
    solver pyscf's own ADC/EOM use, which handles complex vectors, root
    following and restarts).'''
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
        nr = min(dim, nroots)
        w, V = scipy.linalg.eigh(H, subset_by_index=[0, nr - 1])
        if return_vecs:
            return w.real, V
        return np.sort(w.real)

    diag = np.asarray(diag).real

    def aop(xs):
        return [matvec(np.asarray(x, dtype=complex)) for x in xs]

    def precond(dx, e0, x0):
        d = diag - e0
        d[abs(d) < 1e-8] = 1e-8
        return dx / d

    # generous guess set (2*nroots) from the lowest diagonal entries so
    # near-degenerate low roots are not skipped
    nguess = min(dim, max(2 * nroots, nroots + 12))
    order = np.argsort(diag)
    x0 = np.zeros((nguess, dim), dtype=complex)
    for i in range(nguess):
        x0[i, order[i]] = 1.0
    if max_space is None:
        max_space = max(40, 12 + 8 * nroots)
    conv, e, v = lib.davidson_nosym1(
        aop, list(x0), precond, tol=tol, tol_residual=1e-7,
        nroots=nroots, max_cycle=max_cycle, max_space=max_space, verbose=0)
    e = np.asarray(e)
    if return_vecs:
        srt = np.argsort(e.real)
        V = np.asarray(v).T[:, srt]            # (dim, nroots)
        return e.real[srt], V
    return np.sort(e.real)


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

    @property
    def vvoo(self):                                # <ab||ij>  (EA-ADC(3))
        return self._get('vvoo', lambda: self.oovv.conj().transpose(2, 3, 0, 1))

    @property
    def ovvv(self):                                # <ia||bc>  (EA-ADC(3))
        return self._get('ovvv', lambda: self._blk(self.o, self.v, self.v, self.v))


# EE-ADC(3) third-order ph-ph block M^(3)_{ia,jb}, the neutral-excitation static
# self-energy.  Each entry is (coef, einsum-subscript, [(amp,block,conj),...],
# delta) for the *ket* half; the block is Hermitised as  m + m^dagger  and the
# delta tag broadcasts a Kronecker delta on the spectator leg ('vir' -> i j * d_ab,
# 'occ' -> a b * d_ij, 'none' -> full i a j b).  Amplitudes: T2=t2, T2b=t2^(2),
# T1b=t1^(2) (with .conj() where conj==1).  This is the *complete* third-order
# 1p1h secular increment: it reproduces pyscf's full ph-ph block (the static
# uadc_ee.get_imds part *plus* the t2.t2 contributions pyscf computes on the fly
# in its matvec) to ~1e-15 over general molecules.  Per-tensor conjugations are
# pinned by full complex orbital-rotation gauge covariance (phase + degenerate
# mixing, Gate-2 ~1e-14): t2^(2), t1^(2) are built on t2.conj() (transform as
# conjugated amplitudes) so they enter un-conjugated with the interaction block
# conjugated; the t2.t2.V ladders use (t2.conj, t2, V.conj).
_EE_M3_TERMS = [
    (-0.25, 'ikbc,bcjk->ij', [('T2b','oovv',0),('V','vvoo',1)], 'vir'),
    (-0.25, 'jkac,bcjk->ab', [('T2b','oovv',0),('V','vvoo',1)], 'occ'),
    (0.5, 'ikac,bcjk->iajb', [('T2b','oovv',0),('V','vvoo',1)], 'none'),
    (-1.0, 'kb,ibjk->ij', [('T1b','ov',0),('V','ovoo',1)], 'vir'),
    (-1.0, 'jc,bcja->ab', [('T1b','ov',0),('V','vvov',1)], 'occ'),
    (1.0, 'ka,ibjk->iajb', [('T1b','ov',0),('V','ovoo',1)], 'none'),
    (1.0, 'ic,bcja->iajb', [('T1b','ov',0),('V','vvov',1)], 'none'),
    (-0.125, 'ikbc,bclm,lmjk->ij', [('T2','oovv',1),('T2','vvoo',0),('V','oooo',1)], 'vir'),
    (-0.5, 'klbc,bdjk,icld->ij', [('T2','oovv',1),('T2','vvoo',0),('V','ovov',1)], 'vir'),
    (-0.25, 'klbc,bdkl,icjd->ij', [('T2','oovv',1),('T2','vvoo',0),('V','ovov',1)], 'vir'),
    (0.25, 'klbc,bckm,imjl->ij', [('T2','oovv',1),('T2','vvoo',0),('V','oooo',1)], 'vir'),
    (-0.125, 'jkac,dejk,bcde->ab', [('T2','oovv',1),('T2','vvoo',0),('V','vvvv',1)], 'occ'),
    (0.25, 'jkcd,cejk,bdae->ab', [('T2','oovv',1),('T2','vvoo',0),('V','vvvv',1)], 'occ'),
    (-0.5, 'jkcd,bcjl,ldka->ab', [('T2','oovv',1),('T2','vvoo',0),('V','ovov',1)], 'occ'),
    (-0.25, 'jkcd,cdjl,lbka->ab', [('T2','oovv',1),('T2','vvoo',0),('V','ovov',1)], 'occ'),
    (0.25, 'ikac,dejk,bcde->iajb', [('T2','oovv',1),('T2','vvoo',0),('V','vvvv',1)], 'none'),
    (0.5, 'ikac,cdjl,lbkd->iajb', [('T2','oovv',1),('T2','vvoo',0),('V','ovov',1)], 'none'),
    (0.5, 'klac,bdjk,icld->iajb', [('T2','oovv',1),('T2','vvoo',0),('V','ovov',1)], 'none'),
    (-1.0, 'klac,cdjk,ibld->iajb', [('T2','oovv',1),('T2','vvoo',0),('V','ovov',1)], 'none'),
    (0.25, 'klac,bcjm,imkl->iajb', [('T2','oovv',1),('T2','vvoo',0),('V','oooo',1)], 'none'),
    (0.25, 'klac,bdkl,icjd->iajb', [('T2','oovv',1),('T2','vvoo',0),('V','ovov',1)], 'none'),
    (-0.25, 'klac,cdkl,ibjd->iajb', [('T2','oovv',1),('T2','vvoo',0),('V','ovov',1)], 'none'),
    (-0.5, 'klac,bckm,imjl->iajb', [('T2','oovv',1),('T2','vvoo',0),('V','oooo',1)], 'none'),
    (-0.5, 'ikcd,cejk,bdae->iajb', [('T2','oovv',1),('T2','vvoo',0),('V','vvvv',1)], 'none'),
    (0.25, 'ikcd,cdjl,lbka->iajb', [('T2','oovv',1),('T2','vvoo',0),('V','ovov',1)], 'none'),
    (-0.25, 'ikcd,cdkl,lbja->iajb', [('T2','oovv',1),('T2','vvoo',0),('V','ovov',1)], 'none'),
]


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
        self._t1_2_cache = None
        self._t2_2_cache = None
        self._t1_2p_cache = None
        self._sig_ip3_cache = None
        self._sig_ea3_cache = None
        self._ee_lam3_cache = None

    def _t1_2(self):
        # second-order singles (only for spectroscopic amplitudes; built lazily
        # so plain IP/EA/EE never pay for the vovv block it needs).  t2.conj()
        # so the Dyson amplitudes transform correctly under a complex
        # orbital-phase gauge (holes ~ ph_i*), like sig_ip.
        if self._t1_2_cache is None:
            no = self.nocc
            e = np.asarray(self.mo_energy)
            eo, ev = e[:no], e[no:]
            t2 = self._t2
            eris = self._eris
            self._t1_2_cache = -(
                0.5 * lib.einsum('akcd,ikcd->ia', eris.vovv, t2.conj())
                - 0.5 * lib.einsum('klic,klac->ia', eris.ooov, t2.conj())
            ) / (eo[:, None] - ev[None, :])
        return self._t1_2_cache

    def _t2_2(self):
        '''Second-order doubles t2^(2) = R / D, where R is the same second-order
        doubles residual (pp + hh ladders + ph ring on t2) used by the MP3
        energy.  Cached; needed by the ADC(3) self-energies.  Built on t2 (the
        residual is contracted against t2.conj() on the i-side in the
        self-energy, which is where the gauge phase is fixed).'''
        if getattr(self, '_t2_2_cache', None) is None:
            no = self.nocc
            e = np.asarray(self.mo_energy)
            eo, ev = e[:no], e[no:]
            eris = self._eris
            t2 = self._t2
            d = (eo[:, None, None, None] + eo[None, :, None, None]
                 - ev[None, None, :, None] - ev[None, None, None, :])
            # residual on t2.conj() (gauge: holes ~ ph*), so t2^(2) is covariant
            # under a complex orbital rotation -- the same convention as the
            # validated MP3 residual in energy_mp3.
            tc = t2.conj()
            R = (0.5 * lib.einsum('klij,klab->ijab', eris.oooo, tc)
                 + 0.5 * lib.einsum('abcd,ijcd->ijab', eris.vvvv, tc))
            tr = lib.einsum('kbcj,ikac->ijab', eris.ovvo, tc)
            R += (tr - tr.transpose(1, 0, 2, 3)
                  - tr.transpose(0, 1, 3, 2) + tr.transpose(1, 0, 3, 2))
            self._t2_2_cache = R / d
        return self._t2_2_cache

    def _t1_2_pyscf(self):
        '''Second-order singles in the pyscf/ADC(3) sign convention
        ``t1^(2)_ia = (0.5 <ai||cd> t_ikcd - 0.5 <kl||ic> t_klac) / (e_i-e_a)``
        (no leading minus, unlike :meth:`_t1_2` which carries the
        spectroscopic-amplitude gauge sign).  Used only by the ADC(3)
        self-energy Group-A term.'''
        if getattr(self, '_t1_2p_cache', None) is None:
            no = self.nocc
            e = np.asarray(self.mo_energy)
            eo, ev = e[:no], e[no:]
            t2 = self._t2
            eris = self._eris
            # built on t2.conj() so t1^(2) is covariant under a complex orbital
            # rotation (transforms as e^{+i theta_occ - i theta_vir}); reduces
            # to the pyscf real t1^(2) for real orbitals.
            tc = t2.conj()
            self._t1_2p_cache = (
                0.5 * lib.einsum('akcd,ikcd->ia', eris.vovv, tc)
                - 0.5 * lib.einsum('klic,klac->ia', eris.ooov, tc)
            ) / (eo[:, None] - ev[None, :])
        return self._t1_2p_cache

    def _sig_ip3(self):
        '''Third-order IP self-energy (1h-1h block), the static spin-orbital
        Sigma^(3)_ij.  Built from t2, the second-order doubles t2^(2) and the
        second-order singles t1^(2) and the antisymmetrised integrals.  Derived
        by transcribing the spin-blocked pyscf UADC M_ij^(3) to the unified
        spin-orbital (antisymmetrised) form and validated to ~1e-10 against
        ``uadc_ip.get_imds`` (full matrix) across H2O/HF/N2/CO/LiH/BH.  The
        i-side amplitude is conjugated (gauge: holes ~ ph*, like _sig_ip) and
        the result is Hermitised.'''
        if getattr(self, '_sig_ip3_cache', None) is None:
            self._build()
            eris = self._eris
            t2 = self._t2
            tc = t2.conj()
            t2_2 = self._t2_2()
            t1_2 = self._t1_2_pyscf()
            oovv, oooo = eris.oovv, eris.oooo
            ovvo, ooov = eris.ovvo, eris.ooov
            S = np.zeros((self.nocc, self.nocc), dtype=complex)
            # Group A : second-order singles (P + P^dag with the covariant t1^(2))
            A1 = -lib.einsum('ld,ljid->ij', t1_2, ooov)
            S += A1 + A1.conj().T
            # Group B : second-order doubles, Sigma^(2) structure (P + P^dag)
            B1 = 0.25 * lib.einsum('ilde,jlde->ij', t2_2, oovv)
            S += B1 + B1.conj().T
            # Group O : two-amplitude hole-ladder (oooo) -- covariant as written
            S += 0.125 * lib.einsum('lmde,jnde,lmin->ij', tc, t2, oooo)
            S += 0.125 * lib.einsum('inde,lmde,jnlm->ij', tc, t2, oooo)
            S += -0.5 * lib.einsum('lnde,lmde,jnim->ij', tc, t2, oooo)
            # Group L : double particle-hole ladder (P + P^dag)
            Mt = lib.einsum('lmde,jldf->mejf', tc, t2)
            L1 = -0.5 * lib.einsum('mejf,mfei->ij', Mt, ovvo)
            S += L1 + L1.conj().T
            # Group S : single particle-hole ladder -- covariant as written
            S += -0.5 * lib.einsum('lmdf,lmde,jefi->ij', tc, t2, ovvo)
            self._sig_ip3_cache = 0.5 * (S + S.conj().T)
        return self._sig_ip3_cache

    def _sig_ea3(self):
        '''Third-order EA self-energy (1p-1p block), the static spin-orbital
        Sigma^(3)_ab.  The particle-hole mirror (o<->v) of the IP Sigma^(3):
        built from the o<->v dual amplitudes (t2, t2^(2), t1^(2) transposed
        with the denominator sign flip) and the dual antisym blocks (vvvv,
        vvoo, voov, vvvo).  Validated to ~3e-8 against ``uadc_ea.get_imds``
        (full matrix) over LiH/H2O/N2/HF.  i-side amplitudes conjugated (gauge:
        particles ~ ph), Hermitised.'''
        if getattr(self, '_sig_ea3_cache', None) is None:
            self._build()
            eris = self._eris
            t2 = self._t2
            nv = self.nmo - self.nocc
            vvvv, vvoo, voov, vvvo = eris.vvvv, eris.vvoo, eris.voov, eris.vvvo
            # o<->v dual amplitudes (carry the -1 from the flipped MP denominator)
            t2d = -t2.transpose(2, 3, 0, 1)
            tcd = -t2.conj().transpose(2, 3, 0, 1)
            t1_2d = -self._t1_2_pyscf().T
            t2_2d = -self._t2_2().transpose(2, 3, 0, 1)
            # interaction blocks are conjugated where the o<->v mirror flips the
            # bra/ket sense (so each term is covariant under a complex rotation);
            # for a real reference conj is the identity, so the validated value
            # is unchanged.  Which blocks carry .conj() is pinned by Gate-2.
            S = np.zeros((nv, nv), dtype=complex)
            # Group A : second-order singles
            A1 = lib.einsum('dl,dbal->ab', t1_2d, vvvo.conj())
            S += A1 + A1.conj().T
            # Group B : second-order doubles (Sigma^(2)-EA structure)
            B1 = 0.25 * lib.einsum('adlm,bdlm->ab', t2_2d, vvoo.conj())
            S += B1 + B1.conj().T
            # Group O/L/S : two-amplitude particle ladders (vvvv) and ph ladders
            S += -0.125 * lib.einsum('delm,bflm,deaf->ab', tcd, t2d, vvvv.conj())
            S += -0.125 * lib.einsum('aflm,delm,bfde->ab', tcd, t2d, vvvv.conj())
            O3 = 0.25 * lib.einsum('dflm,delm,bfae->ab', tcd, t2d, vvvv.conj())
            S += O3 + O3.conj().T
            Mtd = lib.einsum('delm,bdln->embn', tcd, t2d)
            L1 = lib.einsum('embn,enma->ab', Mtd, voov.conj())
            S += 0.5 * L1 + 0.5 * L1.conj().T
            S1 = 0.25 * lib.einsum('deln,delm,bmna->ab', tcd, t2d, voov.conj())
            S += S1 + S1.conj().T
            self._sig_ea3_cache = 0.5 * (S + S.conj().T)
        return self._sig_ea3_cache

    def _ee_lam3(self):
        '''Third-order ph-ph block increment M^(3)_{ia,jb} for EE-ADC(3) -- the
        neutral-excitation static self-energy (the full third-order increment of
        the 1p1h block, i.e. it already contains the third-order IP/EA self-energy
        legs and the genuine ph-ph coupling).  Returned as an [i,a,j,b] tensor to
        be folded into the second-order ph-ph term ``Lam``.  Built from the term
        table ``_EE_M3_TERMS`` (see its comment); Hermitised and gauge-covariant
        (Gate-2).'''
        if getattr(self, '_ee_lam3_cache', None) is None:
            self._build()
            eris = self._eris
            no, nv = self.nocc, self.nmo - self.nocc
            t2 = self._t2
            t2_2 = self._t2_2()
            t1_2 = self._t1_2_pyscf()
            o, v = eris.o, eris.v
            Vget = {'vvoo': eris.vvoo, 'ovoo': eris.ovoo, 'oooo': eris.oooo,
                    'vvvv': eris.vvvv, 'ovov': eris._blk(o, v, o, v),
                    'vvov': eris._blk(v, v, o, v)}
            Io, Iv = np.eye(no), np.eye(nv)

            def amp(kind, ix, conj):
                if kind == 'T2':
                    a = t2 if ix == 'oovv' else t2.transpose(2, 3, 0, 1)
                elif kind == 'T2b':
                    a = t2_2 if ix == 'oovv' else t2_2.transpose(2, 3, 0, 1)
                elif kind == 'T1b':
                    a = t1_2 if ix == 'ov' else t1_2.T
                else:
                    a = Vget[ix]
                return a.conj() if conj else a

            M = np.zeros((no, nv, no, nv), dtype=complex)
            for coef, sub, tens, delta in _EE_M3_TERMS:
                G = coef * lib.einsum(sub, *[amp(*t) for t in tens])
                if delta == 'none':
                    m = G
                elif delta == 'vir':
                    m = lib.einsum('ij,ab->iajb', G, Iv)
                else:
                    m = lib.einsum('ab,ij->iajb', G, Io)
                M += m + m.conj().transpose(2, 3, 0, 1)
            self._ee_lam3_cache = M
        return self._ee_lam3_cache

    def energy_mp3(self):
        '''Spinor MP2 + MP3 ground-state correlation energy.  The MP3 piece uses
        the second-order doubles residual (pp + hh ladders + ph ring on t2) --
        the same intermediate the IP/EA/EE-ADC(3) self-energies are built from.
        Matches pyscf adc(3) e_corr to machine precision.'''
        self._build()
        no = self.nocc
        e = np.asarray(self.mo_energy)
        eo, ev = e[:no], e[no:]
        eris = self._eris
        t2 = self._t2
        e_mp2 = 0.25 * lib.einsum('ijab,ijab->', eris.oovv.conj(), t2).real
        # second-order doubles residual R built on t2.conj() (gauge: holes ~
        # ph*); MP3 = 0.25 <t2 | R>.  Conjugation pinned by Gate-2 (rotation
        # invariance) -- matching pyscf alone does not fix it.
        tc = t2.conj()
        R = (0.5 * lib.einsum('klij,klab->ijab', eris.oooo, tc)
             + 0.5 * lib.einsum('abcd,ijcd->ijab', eris.vvvv, tc))
        tr = lib.einsum('kbcj,ikac->ijab', eris.ovvo, tc)
        R += (tr - tr.transpose(1, 0, 2, 3)
              - tr.transpose(0, 1, 3, 2) + tr.transpose(1, 0, 3, 2))
        e_mp3 = 0.25 * lib.einsum('ijab,ijab->', t2, R).real
        return e_mp2, e_mp3

    def _ip_trans_moments(self, no, nv, oo):
        '''IP-ADC(2) spectroscopic-amplitude (Dyson) vectors T_p for every
        spinor p, in the packed (1h + 2h1p[i<j]) basis -- shape (nmo, dim).'''
        t2 = self._t2
        t1_2 = self._t1_2()
        ok, ol = oo[:, 0], oo[:, 1]
        nop = len(oo)
        dim = no + nop * nv
        nmo = no + nv
        T = np.zeros((nmo, dim), dtype=complex)
        # occupied p: 1h-dominated, 2h1p part zero at this order
        for p in range(no):
            T[p, p] = 1.0
            T[p, :no] += -0.25 * lib.einsum('kcd,ikcd->i', t2[p], t2.conj())
        # virtual p: 2nd-order 1h (t1_2) + the t2 amplitudes in 2h1p
        for a in range(nv):
            p = no + a
            T[p, :no] = t1_2[:, a]
            T[p, no:] = t2.conj()[ok, ol, a, :].reshape(-1)
        return T

    def _ea_trans_moments(self, no, nv, vv):
        '''EA-ADC(2) attachment (Dyson) vectors T_p for every spinor p, in the
        packed (1p + 2p1h[a<b]) basis -- shape (nmo, dim).'''
        t2 = self._t2
        t1_2 = self._t1_2()
        vc, vd = vv[:, 0], vv[:, 1]
        nvp = len(vv)
        dim = nv + nvp * no
        nmo = no + nv
        T = np.zeros((nmo, dim), dtype=complex)
        # virtual p: 1p-dominated, 2p1h part zero at this order
        for a in range(nv):
            p = no + a
            T[p, a] = 1.0
            T[p, :nv] += -0.25 * lib.einsum('klc,klbc->b', t2[:, :, a, :], t2.conj())
        # occupied p: 2nd-order 1p (t1_2) + the t2 amplitudes in 2p1h
        # (1p and 2p1h parts carry the same relative sign -- as in IP)
        for i in range(no):
            T[i, :nv] = t1_2[i, :]
            T[i, nv:] = t2.conj()[:, i, :, :][:, vc, vd].T.reshape(-1)   # [(c<d), j]
        return T

    # -- IP ------------------------------------------------------------------
    def ip_adc2(self, nroots=6, method='adc(2)', ncvs=None, with_spec=False):
        '''Spinor IP ionization energies (lowest ``nroots``).

        With ``with_spec`` (strict ADC(2), full space) also returns the
        spectroscopic factors (pole strengths) ``P_n = sum_p |<N-1,n|a_p|0>|^2``.

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

        if ncvs is not None:
            # Core-valence separation: project onto 1h with a core hole and
            # 2h1p with at least one core hole (ncvs lowest spinors are core).
            # Same matrix elements, restricted space -- matches pyscf CVS.
            keep = list(range(ncvs))                       # 1h: i in core
            for P in range(nop):
                if ok[P] < ncvs:                           # >=1 core hole
                    base = no + P * nv
                    keep.extend(range(base, base + nv))
            keep = np.asarray(keep)

            def cvs_matvec(xr):
                xf = np.zeros(dim, dtype=complex)
                xf[keep] = xr
                return matvec(xf)[keep]

            return _davidson(cvs_matvec, diag[keep], nroots)

        if with_spec:
            if method != 'adc(2)':
                raise NotImplementedError('spec factors only for adc(2)')
            w, V = _davidson(matvec, diag, nroots, return_vecs=True)
            T = self._ip_trans_moments(no, nv, oo)         # (nmo, dim)
            P = np.sum(np.abs(T.conj() @ V) ** 2, axis=0)  # (nroots,)
            srt = np.argsort(w)
            return w[srt], P[srt]
        return _davidson(matvec, diag, nroots)

    def ip_adc2_spec(self, nroots=6):
        '''Spinor IP-ADC(2) energies and spectroscopic factors (pole
        strengths).  Returns ``(energies, pole_strengths)``.'''
        return self.ip_adc2(nroots, with_spec=True)

    def ip_cvs_adc2(self, nroots=6, ncvs=2, method='adc(2)'):
        '''Spinor CVS-IP-ADC(2) core-ionization energies.  ``ncvs`` is the
        number of core *spinors* (2x the spatial core count for a non-rel
        Kramers-paired reference).'''
        return self.ip_adc2(nroots, method=method, ncvs=ncvs)

    def ip_adc2x(self, nroots=6):
        '''Spinor IP-ADC(2)-x ionization energies.'''
        return self.ip_adc2(nroots, method='adc(2)-x')

    def ip_adc3(self, nroots=6):
        '''Spinor IP-ADC(3) ionization energies (lowest ``nroots``).

        The 1h-1h block is the effective Hamiltonian through third order
        (``-eps - Sigma^(2) - Sigma^(3)``); the 1h<->2h1p coupling is through
        second order (the first-order ``<kl||ia>`` plus the t2-dressed
        ``<ai||bc>``/``<kl||ia>`` pieces); the 2h1p block is first order (the
        ADC(2)-x hole-hole ladder ``0.5<mn||ij>`` + particle-hole ring
        ``<ma||ei>``).  Reproduces pyscf IP-UADC(3)/RADC(3) eigenvalues to
        ~1e-6 and is invariant under a legal complex orbital rotation.'''
        self._build()
        no, nmo = self.nocc, self.nmo
        nv = nmo - no
        e = np.asarray(self.mo_energy)
        eo, ev = e[:no], e[no:]
        eris = self._eris
        t2 = self._t2

        Mss = -np.diag(eo).astype(complex) - self._sig_ip - self._sig_ip3()
        Wooov = eris.ooov                              # <kl||ia>
        WooovC = Wooov.conj()
        Woooo = eris.oooo                              # <mn||ij>
        Wovvo = eris.ovvo                              # <ma||ei>
        Wvovv = eris.vovv                              # <ai||bc>
        WvovvC = Wvovv.conj()

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
            r2 = unpack(x[no:])                         # [k,l,a]
            # 1h <- 1h
            s1 = Mss @ r1
            # 1h <- 2h1p : first order + second order (t2-dressed) coupling.
            # r2 contracts with un-conjugated amplitudes (like the first-order
            # <kl||ia>) and conjugated interaction integrals, so the coupling is
            # covariant under a complex orbital rotation (pinned by Gate-2).
            s1 += 0.5 * lib.einsum('klia,kla->i', Wooov, r2)
            s1 += -0.25 * lib.einsum('jkbc,jka,aibc->i', t2, r2, WvovvC)
            s1 += 0.5 * lib.einsum('jlab,jka,likb->i', t2, r2, WooovC)
            s1 += 0.5 * lib.einsum('klab,kja,lijb->i', t2, r2, WooovC)
            # 2h1p <- 1h : Hermitian conjugate of the above coupling
            s2 = (ev[None, None, :] - eo[:, None, None] - eo[None, :, None]) * r2
            s2 += lib.einsum('klia,i->kla', WooovC, r1)
            s2 += -0.5 * lib.einsum('jkbc,aibc,i->jka', t2.conj(), Wvovv, r1)
            t = lib.einsum('jlab,likb,i->jka', t2.conj(), Wooov, r1)
            s2 += (t - t.transpose(1, 0, 2))
            # 2h1p <- 2h1p : ADC(2)-x block
            s2 += 0.5 * lib.einsum('mnij,mna->ija', Woooo, r2)
            tr = lib.einsum('maei,mje->ija', Wovvo, r2)
            s2 += tr - tr.transpose(1, 0, 2)
            return np.concatenate([s1, pack(s2)])

        diag = np.empty(dim)
        diag[:no] = Mss.diagonal().real
        dd = ev[None, :] - (eo[ok] + eo[ol])[:, None]
        oooo_d = np.einsum('ijij->ij', eris.oooo).real     # <ij||ij>
        ovov_d = -np.einsum('iaai->ia', eris.ovvo).real    # <ia||ia>
        dd = (dd + oooo_d[ok, ol][:, None]
              - ovov_d[ok, :] - ovov_d[ol, :])
        diag[no:] = dd.reshape(-1)
        return _davidson(matvec, diag, nroots)

    # -- EA ------------------------------------------------------------------
    def ea_adc2(self, nroots=6, method='adc(2)', with_spec=False):
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

        if with_spec:
            if adc2x:
                raise NotImplementedError('spec factors only for adc(2)')
            w, V = _davidson(matvec, diag, nroots, return_vecs=True)
            T = self._ea_trans_moments(no, nv, vv)         # (nmo, dim)
            P = np.sum(np.abs(T.conj() @ V) ** 2, axis=0)
            srt = np.argsort(w)
            return w[srt], P[srt]
        return _davidson(matvec, diag, nroots)

    def ea_adc2x(self, nroots=6):
        '''Spinor EA-ADC(2)-x electron affinities.'''
        return self.ea_adc2(nroots, method='adc(2)-x')

    def ea_adc3(self, nroots=6):
        '''Spinor EA-ADC(3) electron affinities (lowest ``nroots``).

        The particle-hole mirror of :meth:`ip_adc3`.  The 1p-1p block is the
        effective Hamiltonian through third order (``eps - Sigma^(2) +
        Sigma^(3)``; note the EA self-energy enters with the opposite sign to
        IP, mirroring the EA<->IP eigenvalue convention); the 1p<->2p1h coupling
        is through second order (the first-order ``<ai||bc>`` plus the t2-dressed
        ``<kl||ia>``/``<ia||bc>`` pieces); the 2p1h block is first order (the
        ADC(2)-x particle-particle ladder ``0.5<ab||cd>`` + particle-hole ring).
        Reproduces pyscf EA-UADC(3) eigenvalues to ~1e-6 and is invariant under
        a legal complex orbital rotation.'''
        self._build()
        no, nmo = self.nocc, self.nmo
        nv = nmo - no
        e = np.asarray(self.mo_energy)
        eo, ev = e[:no], e[no:]
        eris = self._eris
        t2 = self._t2

        Mpp = np.diag(ev).astype(complex) - self._sig_ea + self._sig_ea3()
        Wvovv = eris.vovv                              # <ai||bc>
        WvovvC = Wvovv.conj()
        Wooov = eris.ooov                              # <kl||ia>
        WooovC = Wooov.conj()
        Wovvv = eris.ovvv                              # <ia||bc>
        WovvvC = Wovvv.conj()
        vvvv = eris.vvvv                               # <ab||cd>
        Vjbic = -eris.voov.transpose(1, 0, 2, 3)       # <jb||ic> = -<bj||ic>

        vv = np.array([(c, d) for c in range(nv) for d in range(c + 1, nv)],
                      dtype=int).reshape(-1, 2)
        nvp = len(vv)
        vc, vd = vv[:, 0], vv[:, 1]
        Wvvvv_pp = vvvv[vc[:, None], vd[:, None], vc[None, :], vd[None, :]]
        ev_p = ev[vc] + ev[vd]
        dim = nv + nvp * no

        def unpack(rp):
            r2 = np.zeros((no, nv, nv), dtype=complex)
            blk = rp.reshape(nvp, no)
            r2[:, vc, vd] = blk.T
            r2[:, vd, vc] = -blk.T
            return r2

        def pack(s2):
            return s2[:, vc, vd].T.reshape(-1)

        def matvec(x):
            r1 = x[:nv]
            r2 = unpack(x[nv:])                         # [i,c,d]
            # 1p <- 1p
            s1 = Mpp @ r1
            # 1p <- 2p1h : first order + second order (t2-dressed) coupling.
            # r2 contracts with un-conjugated amplitudes and conjugated
            # interaction integrals (covariant; pinned by Gate-2).
            s1 += 0.5 * lib.einsum('aicd,icd->a', Wvovv, r2)
            s1 += -0.25 * lib.einsum('lmcd,icd,lmia->a', t2, r2, WooovC)
            s1 += lib.einsum('jlwd,jzw,lzda->a', t2, r2, WovvvC)
            # 2p1h <- 1p : Hermitian conjugate of the above coupling
            s2 = (ev[None, :, None] + ev[None, None, :] - eo[:, None, None]) * r2
            s2 += lib.einsum('aicd,a->icd', WvovvC, r1)
            s2 += -0.5 * lib.einsum('lmcd,lmia,a->icd', t2.conj(), Wooov, r1)
            t = lib.einsum('jlwd,lzda,a->jzw', t2.conj(), Wovvv, r1)
            s2 += t - t.transpose(0, 2, 1)
            # 2p1h <- 2p1h : ADC(2)-x block (pp ladder + ph ring)
            s2 += 0.5 * lib.einsum('abcd,icd->iab', vvvv, r2)
            tr = -lib.einsum('jbic,jac->iab', Vjbic, r2)
            s2 += tr - tr.transpose(0, 2, 1)
            return np.concatenate([s1, pack(s2)])

        diag = np.empty(dim)
        diag[:nv] = Mpp.diagonal().real
        dd = ev_p[:, None] - eo[None, :]
        vvvv_d = Wvvvv_pp.diagonal().real
        ovov_d = -np.einsum('iaai->ia', eris.ovvo).real
        dd = (dd + vvvv_d[:, None] - ovov_d.T[vc, :] - ovov_d.T[vd, :])
        diag[nv:] = dd.reshape(-1)
        return _davidson(matvec, diag, nroots)

    def ea_adc2_spec(self, nroots=6):
        '''Spinor EA-ADC(2) energies and spectroscopic factors (pole
        strengths).  Returns ``(energies, pole_strengths)``.'''
        return self.ea_adc2(nroots, with_spec=True)


    # -- EE ------------------------------------------------------------------
    def ee_adc2(self, nroots=6, method='adc(2)'):
        '''Spinor EE excitation energies (lowest ``nroots``).

        The 1p1h block is CIS + the second-order self-energies and the
        non-separable ph-ph term; the 1p1h<->2p2h coupling is first order.  For
        strict ADC(2) the 2p2h block is diagonal; ADC(2)-x adds the first-order
        2p2h interaction (pp ladder 0.5<ab||cd>, hh ladder 0.5<kl||ij>, and the
        ph ring P(ij)P(ab)<kb||cj>).  Solved by a Davidson matvec.
        '''
        self._build()
        no, nmo = self.nocc, self.nmo
        nv = nmo - no
        e = np.asarray(self.mo_energy)
        eo, ev = e[:no], e[no:]
        eris = self._eris
        t2 = self._t2
        Voovv = eris.oovv
        adc3 = (method == 'adc(3)')
        adc2x = (method == 'adc(2)-x') or adc3   # adc(3) keeps the ADC(2)-x 2p2h block
        if not adc2x and method != 'adc(2)':
            raise NotImplementedError(method)

        # ph-ph second-order non-separable term, with the first-order CIS-like
        # ring <aj||ib> folded in (same 'iajb,jb->ia' contraction) so the
        # matvec runs one einsum on a contiguous [i,a,j,b] block instead of two
        # (the separate <aj||ib> term used a transposed layout -> per-matvec copy)
        Lam = 0.5 * (lib.einsum('imae,jmbe->iajb', t2.conj(), Voovv)
                     + lib.einsum('jmbe,imae->iajb', t2, Voovv.conj()))
        Lam += eris.voov.transpose(2, 0, 1, 3)     # <aj||ib> -> [i,a,j,b]
        if adc3:
            # third-order ph-ph self-energy (full M^(3) increment: third-order
            # IP/EA legs + ph-ph coupling), folded into the same [i,a,j,b] term
            Lam = Lam + self._ee_lam3()
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
        if adc2x:
            Wvvvv = eris.vvvv                       # <ab||cd>
            Woooo = eris.oooo                       # <ij||kl>
            Wkbcj = eris.ovvo                       # <kb||cj>  [k,b,c,j]

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
            if adc2x:
                # 2p2h-2p2h first-order block on the full-virtual r2f[i,j,a,b]
                add = 0.5 * lib.einsum('abcd,ijcd->ijab', Wvvvv, r2f)   # pp
                add += 0.5 * lib.einsum('klij,klab->ijab', Woooo, r2f)  # hh
                tr = lib.einsum('kbcj,ikac->ijab', Wkbcj, r2f)          # ph ring
                add += (tr - tr.transpose(1, 0, 2, 3)
                        - tr.transpose(0, 1, 3, 2) + tr.transpose(1, 0, 3, 2))
                s2 += add[:, :, vb, vc]
            return np.concatenate([s1.ravel(), pack_holes(s2)])

        diag = np.empty(dim)
        diag[:ns] = (ev[None, :] - eo[:, None]).ravel().real
        dd = Dijab[oj[:, None], ok[:, None], vb[None, :], vc[None, :]].real
        if adc2x:
            # exact 2p2h diagonal: + <ab||ab> + <ij||ij>
            #                      - <ia||ia> - <ib||ib> - <ja||ja> - <jb||jb>
            vvvv_d = np.einsum('abab->ab', eris.vvvv).real
            oooo_d = np.einsum('ijij->ij', eris.oooo).real
            ovov_d = -np.einsum('iaai->ia', eris.ovvo).real        # <ia||ia>
            dd = (dd + vvvv_d[vb, vc][None, :]
                  + oooo_d[oj, ok][:, None]
                  - ovov_d[oj][:, vb] - ovov_d[oj][:, vc]
                  - ovov_d[ok][:, vb] - ovov_d[ok][:, vc])
        diag[ns:] = dd.reshape(-1)
        return _davidson(matvec, diag, nroots)

    def ee_adc2x(self, nroots=6):
        '''Spinor EE-ADC(2)-x excitation energies.'''
        return self.ee_adc2(nroots, method='adc(2)-x')

    def ee_adc3(self, nroots=6):
        '''Spinor EE-ADC(3) excitation energies.  The 1p1h block carries the
        third-order ph-ph self-energy (:meth:`_ee_lam3`); the 1p1h<->2p2h
        coupling is first order and the 2p2h block is the ADC(2)-x interaction.'''
        return self.ee_adc2(nroots, method='adc(3)')


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
