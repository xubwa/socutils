#
# Author Xubo Wang <wangxubo0201@outlook.com>
#
# TDA (Tamm-Dancoff) for a two-component (j-adapted spinor) reference.
#
# A spinor mean field is, structurally, a GHF/GKS reference with a single
# complex n2c x n2c MO coefficient matrix.  The linear-response sigma vector
# (pyscf.tdscf.ghf) is basis agnostic -- it only needs mf.gen_response, which
# returns (J - c_x K + f_xc) acting on the (complex) transition densities.  The
# spinor reference supplies that through the GHF response template (bound as
# SpinorSCF.gen_response): J/K from the spinor get_jk, f_xc from the
# 2-component SpinorNumInt2C (which also makes the collinear f_xc work with
# complex transition densities).
#
# TDA sets B=0, so its response matrix A is genuinely Hermitian and the Davidson
# solve is robust.  The full RPA/TDDFT (Casida) problem is k-Hermitian
# (non-Hermitian) on a Krein space with an indefinite metric [Furche & Chen,
# J. Chem. Phys. 163, 174104 (2025)]; on a general complex reference TDRPA uses
# the paired Davidson (lr_davidson).  On a Kramers-restricted closed shell the
# time-reversal adaptation makes A +- B real symmetric and TDRPA solves
# (A+B)P = wQ, (A-B)Q = wP in real arithmetic (lr_locked.rpa_eig; see TDRPA).
#
# Validated vs pyscf's 2-component GKS-TDA (non-relativistic integrals, no SOC)
# to machine precision (LDA / GGA / hybrid), and vs ReSpect (1e-X2C PBE TDA of
# Zn: 3P0,1,2 and 1P1 within 3e-8 a.u.).  The response kernel defaults to the
# non-collinear one (td.collinear = 'auto': 'ncol' for closed shells, 'mcol'
# for open shells) even on a collinear ground state: the collinear kernel has
# no spin-flip channel and scrambles spin-orbit fine structure.
#

import contextlib
import numpy
from pyscf import lib
from pyscf.tdscf import ghf as _ghf


def _build_ab_jk_cderi(td, blksize=None):
    '''Dense Coulomb+exact-exchange block A_JK[ia,jb] of the spinor TDA matrix,
    built MO-driven from the density-fitted (Cholesky) ERIs:

        A_JK[ia,jb] = sum_P (L_ia^P)* L_jb^P  -  c_x sum_P (L_ij^P)* L_ab^P
        L_pq^P = sum_{mu nu} (C^sph_p)*_mu  V^P_{mu nu}  C^sph_q,nu   (V = sph cderi)

    The (vir,vir) half-transform L_ab is the memory bottleneck (n2c can give
    hundreds of GB if held whole), so the cderi is consumed one aux block at a
    time and contracted straight into the (nocc,nvir,nocc,nvir) accumulators --
    nothing larger than one block's half-transform is ever resident.  Validated
    against the AO-driven gen_response to machine precision (incl. all complex
    conjugations) on a genuinely-complex C1 reference.

    Returns (A_JK, foo, fvv, orbo, orbv, hyb).  c_x = hyb is the exact-exchange
    fraction (1 for HF, the hybrid coefficient for a KS reference).

    ``foo``/``fvv`` are the occupied-occupied / virtual-virtual blocks of the
    MO-basis Fock matrix.  The TDA "orbital" term is ``F_ab d_ij - F_ij d_ab``,
    *not* ``(e_a - e_i) d_ij d_ab``: the orbital-energy-difference form is only
    correct for **canonical** orbitals (where F is diagonal in the MO basis and
    its diagonal *is* mo_energy).  Returning the full blocks makes the response
    correct for non-canonical references too (localized / rotated / natural
    orbitals); for canonical orbitals foo/fvv are diagonal = mo_energy and the
    result is identical.
    '''
    from pyscf.scf import hf
    mf = td._scf
    mol = mf.mol
    mask = td.get_frozen_mask()
    mo = mf.mo_coeff[:, mask]
    occ = mf.mo_occ[mask]
    occidx = numpy.where(occ == 1)[0]
    viridx = numpy.where(occ == 0)[0]
    orbo = mo[:, occidx]
    orbv = mo[:, viridx]
    nocc, nvir = orbo.shape[1], orbv.shape[1]

    # MO Fock occ/vir blocks (general, non-canonical-safe).  One AO Fock build
    # at setup -- negligible next to the per-aux-block A_JK contraction below,
    # and it avoids per-matvec Fock builds entirely.  For canonical orbitals
    # foo/fvv are diagonal (= mo_energy) and this reduces to e_a - e_i.
    fock_ao = mf.get_fock(dm=mf.make_rdm1())
    foo = orbo.conj().T @ fock_ao @ orbo
    fvv = orbv.conj().T @ fock_ao @ orbv

    nao = mol.nao_nr()
    c2 = numpy.vstack(mol.sph2spinor_coeff())          # (2*nao, n2c)
    oo = c2 @ orbo                                       # (2*nao, nocc)
    ov = c2 @ orbv                                       # (2*nao, nvir)
    ooa, oob = oo[:nao], oo[nao:]                        # alpha/beta spin blocks
    ova, ovb = ov[:nao], ov[nao:]

    if isinstance(mf, hf.KohnShamDFT):
        hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, mol.spin)[2]
    else:
        hyb = 1.0

    A_J = numpy.zeros((nocc, nvir, nocc, nvir), dtype=complex)
    A_K = numpy.zeros((nocc, nvir, nocc, nvir), dtype=complex)
    if blksize is None:
        # keep one block's (blk, nao, nvir) half-transform to ~2 GB
        blksize = max(1, int(2e8 / max(1, nao * nvir)))
    from pyscf.lib import logger
    log = logger.new_logger(td)
    naux = mf.with_df.get_naoaux()
    log.info('[mo_driven] building dense A_JK: naux=%d nocc=%d nvir=%d (A %d^2) blksize=%d',
             naux, nocc, nvir, nocc * nvir, blksize)
    t_ajk = (logger.process_clock(), logger.perf_counter())
    done = 0
    for eri1 in mf.with_df.loop(blksize):
        V = lib.unpack_tril(numpy.asarray(eri1))        # (blk, nao, nao) real, sph
        # spin-free cderi acts within the alpha and beta blocks identically
        Wva = lib.einsum('Pmn,na->Pma', V, ova)         # ket = vir, alpha
        Wvb = lib.einsum('Pmn,na->Pma', V, ovb)         # ket = vir, beta
        Lov = (lib.einsum('Pma,mi->Pia', Wva, ooa.conj()) +
               lib.einsum('Pma,mi->Pia', Wvb, oob.conj()))     # (blk,nocc,nvir)
        Lvv = (lib.einsum('Pmb,ma->Pab', Wva, ova.conj()) +
               lib.einsum('Pmb,ma->Pab', Wvb, ovb.conj()))     # (blk,nvir,nvir)
        # L_ab^P = sum_mu (C^vir_a)*_mu (V C^vir_b)_mu : bra a conjugated, ket b
        Woa = lib.einsum('Pmn,nj->Pmj', V, ooa)
        Wob = lib.einsum('Pmn,nj->Pmj', V, oob)
        Loo = (lib.einsum('Pmj,mi->Pij', Woa, ooa.conj()) +
               lib.einsum('Pmj,mi->Pij', Wob, oob.conj()))     # (blk,nocc,nocc)
        A_J += lib.einsum('Pia,Pjb->iajb', Lov.conj(), Lov)
        A_K -= hyb * lib.einsum('Pij,Pab->iajb', Loo.conj(), Lvv)
        done += V.shape[0]
        log.info('[mo_driven]   A_JK aux %d/%d (%.0f%%)', done, naux, 100.*done/naux)
    A_JK = (A_J + A_K).reshape(nocc * nvir, nocc * nvir)
    log.timer('[mo_driven] A_JK build', *t_ajk)
    return A_JK, foo, fvv, orbo, orbv, hyb


def picture_change_dipole(mf, origin=(0., 0., 0.)):
    '''Picture-change-corrected electric dipole matrices (3, n2c, n2c) in the
    contracted spinor basis, for an X2C mean field (``mf.with_x2c``) -- a
    thin wrapper over the central property-integral layer
    :func:`socutils.prop.int1e.prop_int1e`.

    ``x_response=False`` gives the plain picture-change operator
    [U+ D U]_++ of the field-free decoupling.  The X-response term it drops
    is small -- 1.8e-7 relative against 7.4e-5 for the picture change
    itself, measured on Sm3+ -- so the choice is a detail here; what would
    be wrong is skipping the picture change altogether.
    '''
    from socutils.prop.int1e import prop_int1e
    return prop_int1e(mf, 'r', x_response=False, origin=origin)


class _KernelXCMixin:
    '''Allow the response (A) xc kernel to differ from the ground-state xc.

    Set ``xc_kernel`` to e.g. ``'LDA,VWN'`` to build the TDA response with an
    (A)LDA kernel on top of a GGA/hybrid ground state.  The whole response xc --
    the local f_xc *and* the exact-exchange fraction -- then comes from
    ``xc_kernel`` (so an 'LDA,VWN' kernel carries no HF exchange; keep the exact
    exchange with e.g. ``'0.2*HF + 0.8*LDA, VWN'``), evaluated on the
    ground-state density.  Leave it None to use the ground-state functional.
    '''
    xc_kernel = None

    # Spin treatment of the response kernel (the ground state is unchanged).
    # 'auto': a collinear ground state gets 'ncol' for a closed shell (m = 0 by
    # Kramers symmetry; the isotropic kernel, identical to 'mcol' there) and
    # 'mcol' for an open shell; an 'ncol'/'mcol' ground state keeps its scheme.
    # The collinear kernel has no spin-flip channel and breaks the fine
    # structure of spin-orbit-coupled excitations; set 'col' only to compare.
    collinear = 'auto'

    # The grid of the response xc kernel (f_xc): None keeps the SCF grid; 'cpscf' the coarse CP-SCF grid of
    # the analytic Hessian (socutils.hessian.rhf_fast.CPSCF_GRID_ARGS, socutils.dft.grids); or a Grids object.
    fxc_grids = None

    def _fxc_grids(self):
        g = self.fxc_grids
        if g is None:
            return None
        if isinstance(g, str):
            if g != 'cpscf':
                raise ValueError('fxc_grids %r' % g)
            cache = getattr(self, '_fxc_grids_cache', None)
            if cache is None or cache.mol is not self._scf.mol:
                from socutils.dft.grids import hessian_grids
                from socutils.hessian.rhf_fast import CPSCF_GRID_ARGS
                cache = hessian_grids(self._scf.mol, **CPSCF_GRID_ARGS)
                cache.build(with_non0tab=True)
                lib.logger.info(self, 'response xc kernel on the CP-SCF grid: %d points (SCF grid %d)',
                                cache.weights.size, getattr(self._scf.grids.weights, 'size', 0))
                self._fxc_grids_cache = cache
            return cache
        if g.coords is None:
            g.build(with_non0tab=True)
        return g

    def response_collinear(self):
        ni = getattr(self._scf, '_numint', None)
        if ni is None or not hasattr(ni, 'collinear'):
            return None
        if self.collinear != 'auto':
            return self.collinear
        if ni.collinear[0] != 'c':
            return ni.collinear
        mol = self._scf.mol
        return 'ncol' if mol.spin == 0 and mol.nelectron % 2 == 0 else 'mcol'

    @contextlib.contextmanager
    def _response_env(self):
        # _gen_ghf_response returns a vind closure that reads mf.xc and the
        # numint's collinear scheme lazily (at Davidson time), so both must
        # stay set for the whole solve, not just while the response is built.
        with contextlib.ExitStack() as stack:
            if self.xc_kernel is not None:
                stack.enter_context(lib.temporary_env(self._scf, xc=self.xc_kernel))
            col = self.response_collinear()
            if col is not None:
                lib.logger.info(self, 'response xc kernel: collinear = %s', col)
                stack.enter_context(lib.temporary_env(self._scf._numint, collinear=col))
            g = self._fxc_grids() if hasattr(self._scf, 'grids') else None
            if g is not None:
                stack.enter_context(lib.temporary_env(self._scf, grids=g))
            yield
    # For an X2C reference (mf.with_x2c present) compute transition dipoles
    # with the picture-change-corrected operator (picture_change_dipole).
    # The bare int1e_r_spinor is exact for a nonrelativistic spinor/GHF
    # reference and is used automatically when there is no with_x2c.
    # Measured PCE of the bare operator (HBr/cc-pVDZ, x2camf): valence f
    # off by <~1e-3 relative, but K/L-edge (CVS) main lines off by 0.1-4%
    # and weak SOC satellites by tens of percent.
    picture_change = True

    def kernel(self, *args, **kwargs):
        with self._response_env():
            return super().kernel(*args, **kwargs)

    def cvs(self, core):
        '''Core-valence separation for core-excitation spectra.

        Keep only the ``core`` occupied orbitals active as holes and **freeze
        every other occupied orbital** (all virtuals stay active).  This is the
        inverse selection of the usual frozen-core: it sets ``self.frozen`` to
        the complement of ``core`` within the occupied space and returns self,
        so it chains:

            td = mf.TDA().cvs([0, 1])      # holes restricted to the 1s pair
            es = td.kernel()
            td.xc_kernel = 'LDA,VWN'       # optional: ALDA response kernel
            es_alda = td.kernel()

        ``core`` is a list of occupied spinor-orbital indices (a deep core is a
        Kramers pair, e.g. ``[0, 1]``).  Builds on pyscf's ``frozen`` mechanism
        (get_frozen_mask), so it composes with everything else here.
        '''
        import numpy
        core = set(int(i) for i in numpy.atleast_1d(core))
        occ = numpy.where(self._scf.mo_occ == 1)[0]
        self.frozen = [int(i) for i in occ if i not in core]
        return self

    def x0_from_chk(self, chkfile=None, nstates=None):
        '''Davidson initial guess (x0) from civectors saved by a previous run.

        pyscf's kernel() already *writes* ``tddft/e`` and ``tddft/xy`` to
        ``self.chkfile`` whenever it is set -- no extra save step is needed.
        This reads ``tddft/xy`` back so a later, larger-nstates solve warm-starts
        from the converged roots instead of cold unit guesses (a big saving:
        each spinor matvec is a grid f_xc evaluation).  Requested states beyond
        those saved are padded with the standard hdiag unit guesses:

            td.chkfile = 'tddft.chk'; td.nstates = 10; td.kernel()   # saves xy
            # later -- resume / extend, reusing the converged roots:
            td2 = mf.TDA().cvs(core); td2.chkfile = 'tddft.chk'
            td2.kernel(nstates=30, x0=td2.x0_from_chk(nstates=30))
        '''
        import numpy
        chkfile = chkfile or self.chkfile
        xy = lib.chkfile.load(chkfile, 'tddft/xy')
        x0 = [numpy.asarray(x).ravel() for x, y in xy]
        if nstates is None:
            nstates = self.nstates
        if len(x0) < nstates:                      # pad with fresh unit guesses
            extra = self.get_init_guess(self._scf, nstates)
            if isinstance(extra, tuple):
                extra = extra[0]
            x0 = x0 + [numpy.asarray(g).ravel() for g in extra[len(x0):nstates]]
        return x0

    def _dipole_ints(self):
        '''Electric-dipole matrices (3, n2c, n2c) in the spinor AO basis, in
        the picture of the reference: picture-change corrected for an X2C
        reference (see ``picture_change_dipole`` / the ``picture_change``
        flag), bare int1e_r_spinor otherwise (exact there).  Gauge origin
        (0,0,0), same as pyscf's int1e_r; transition moments between
        orthogonal states are origin independent.'''
        mol = self._scf.mol
        n2c = mol.nao_2c()
        if self.picture_change and getattr(self._scf, 'with_x2c', None) is not None:
            ints = picture_change_dipole(self._scf)
        else:
            ints = mol.intor('int1e_r_spinor', comp=3)
        return ints.reshape(3, n2c, n2c)

    def _contract_multipole(self, ints, hermi=True, xy=None):
        '''Transition multipole (e.g. dipole) for a spinor (2-component)
        reference -- pyscf's GHF stub raises NotImplementedError.

        ``oscillator_strength`` calls this with the spin-free electric-dipole
        integrals built in the *spherical* AO basis; the spinor MOs live in the
        n2c basis, so rebuild the operator there (``_dipole_ints``) and
        contract with the X (and Y) amplitudes over the active occupied/
        virtual spinors.  No factor of two: unlike closed-shell RHF the spinor
        basis carries no spin degeneracy to sum over.
        '''
        import numpy
        if xy is None:
            xy = self.xy
        nstates = len(xy)
        pol_shape = ints.shape[:-2]          # (3,) for the dipole
        ints = self._dipole_ints()

        mask = self.get_frozen_mask()
        mo = self._scf.mo_coeff[:, mask]
        occ = self._scf.mo_occ[mask]
        orbo = mo[:, occ == 1]
        orbv = mo[:, occ == 0]
        # MO transition dipole <i|r|a>  (active occ x active vir)
        ints = lib.einsum('xpq,pi,qj->xij', ints, orbo.conj(), orbv)
        pol = numpy.array([numpy.einsum('xij,ij->x', ints, x) for x, y in xy])
        if isinstance(xy[0][1], numpy.ndarray):    # TDHF/RPA has de-excitations
            # the Y part of the transition density is C_o Y C_v^H (pyscf's TDHF vind): <a|r|i> = <i|r|a>^*
            ymo = numpy.array([numpy.einsum('xij,ij->x', ints.conj(), y) for x, y in xy])
            pol = pol + ymo if hermi else pol - ymo
        return pol.reshape((nstates,) + pol_shape)

    def _couple_ref_transition_dipole(self):
        '''<Psi_0|r|Psi_k> for the reference-coupled solve: full matrix
        elements of the (picture-changed) dipole in the model space
        span{Phi_0, Phi_i^a}, between the *relaxed ground state* (root 0) and
        every root k.  Uses

            <Phi_0 |r| Phi_jb>  = r_jb
            <Phi_ia|r| Phi_jb>  = d_ij r_ab - d_ab r_ji + d_ij d_ab mu_ref

        (mu_ref = <Phi_0|r|Phi_0>, summed over *all* occupied spinors incl.
        frozen), plus the c0 reference-amplitude cross terms.  For a
        stationary reference (c0_0 = 1, X_0 = 0) this reduces exactly to the
        plain singles contraction sum_ia X^k_ia r_ia.  Requires the c0
        amplitudes stored by ``_direct_solve`` (couple_reference=True).'''
        import numpy
        xy = self.xy
        c0 = self.c0
        D = self._dipole_ints()
        mask = self.get_frozen_mask()
        mo = self._scf.mo_coeff[:, mask]
        occ = self._scf.mo_occ[mask]
        orbo = mo[:, occ == 1]
        orbv = mo[:, occ == 0]
        d_ov = lib.einsum('xpq,pi,qa->xia', D, orbo.conj(), orbv)
        d_vv = lib.einsum('xpq,pa,qb->xab', D, orbv.conj(), orbv)
        d_oo = lib.einsum('xpq,pi,qj->xij', D, orbo.conj(), orbo)
        occ_all = self._scf.mo_coeff[:, self._scf.mo_occ == 1]
        mu_ref = lib.einsum('xpq,pk,qk->x', D, occ_all.conj(), occ_all)
        X = numpy.asarray([x for x, y in xy])
        X0, c00 = X[0], c0[0]
        s0 = lib.einsum('xia,ia->x', d_ov, X0)      # <Phi_0|r|singles(Psi_0)>
        mu = (numpy.multiply.outer(numpy.conj(c00) * c0, mu_ref)
              + numpy.conj(c00) * lib.einsum('xia,kia->kx', d_ov, X)
              + numpy.multiply.outer(c0, s0.conj())
              + lib.einsum('ia,xab,kib->kx', X0.conj(), d_vv, X)
              - lib.einsum('ia,xji,kja->kx', X0.conj(), d_oo, X)
              + numpy.multiply.outer(lib.einsum('ia,kia->k', X0.conj(), X),
                                     mu_ref))
        return mu

    def oscillator_strength(self, e=None, xy=None, gauge='length', order=0,
                            e_ground=None):
        '''f = (2/3) * dE * |mu|^2.  The spinor transition dipole is complex, so
        use |mu|^2 = mu . mu* -- pyscf's base routine forms mu . mu (real-dipole
        assumption), which underestimates and can even go negative here.

        For a reference-coupled solve (couple_reference=True) the transition
        moment is <Psi_0|r|Psi_k> with Psi_0 the *relaxed ground state* (root
        0), evaluated with the full model-space matrix elements
        (``_couple_ref_transition_dipole``), and the prefactor is the physical
        dE = e_k - e_0.  The plain-CIS formula (bra = the reference
        determinant, singles term only) is a first-order-in-|F_ia| error for a
        non-stationary (CAHF / core-hole) reference -- on Ar L-edge CVS it is
        off by factors 0.05-3.5 per line.

        ``e_ground``: pass a separately-converged ground-state total energy to
        use the *physical* transition energy dE = E_tot(state) - e_ground for the
        prefactor, instead of self.e (which is relative to the -- possibly
        ill-defined -- reference determinant E0).  Essential for a relaxed /
        core-hole reference without couple_reference: the dipole
        mu = <Phi_0|r|Psi_k> is the right (state-specific) transition moment,
        but the intensity needs the true dE.
        '''
        import numpy
        if e is None:
            e = self.e
        e = numpy.asarray(e).real
        couple = (getattr(self, 'couple_reference', False)
                  and getattr(self, 'c0', None) is not None
                  and (xy is None or xy is self.xy))
        if couple:
            de = e - e[0]
            if e_ground is not None:
                de = self.total_energy(e) - e_ground
            if gauge == 'length' and order == 0:
                mu = self._couple_ref_transition_dipole()
                return (2. / 3. * numpy.einsum('s,sx,sx->s', de, mu,
                                               mu.conj())).real
            raise NotImplementedError(
                'couple_reference oscillator strength: length gauge only')
        if e_ground is not None:
            e = self.total_energy(e) - e_ground       # physical transition energy
        if gauge == 'length' and order == 0:
            mu = self.transition_dipole(xy)
            return (2. / 3. * numpy.einsum('s,sx,sx->s', e, mu, mu.conj())).real
        return super().oscillator_strength(e=e, xy=xy, gauge=gauge, order=order)

    def reference_energy(self):
        '''Energy of the single reference determinant the CIS is built on,
        <Phi_0|H|Phi_0>, evaluated from the current mo_coeff/mo_occ.

        Note this is the determinant defined by mf.mo_occ -- not necessarily
        mf.e_tot, if mo_coeff was replaced (e.g. by CAHF / core-hole orbitals).
        The CIS excitation energies are relative to *this* number, so the
        absolute state energies are reference_energy() + e.'''
        mf = self._scf
        return mf.energy_tot(dm=mf.make_rdm1())

    def total_energy(self, e=None):
        '''Absolute total energies of the excited states (Hartree):
        E_ref(determinant) + excitation energy.'''
        import numpy
        if e is None:
            e = self.e
        return self.reference_energy() + numpy.asarray(e).real

    @property
    def e_tot(self):
        '''Absolute total energies E0(reference determinant) + e.

        Overrides the pyscf base property, which returns mf.e_tot + e -- that is
        wrong whenever mo_coeff was replaced (relaxed / core-hole orbitals), as
        mf.e_tot still caches the *original* SCF determinant.  Here E0 is the
        energy of the determinant the CIS is actually built on.'''
        import numpy
        if self.e is None:
            return None
        return self.reference_energy() + numpy.asarray(self.e).real

    def _finalize(self):
        '''Dump results.  Replaces the base "Excitation energies (eV)" line --
        which is misleading for a non-stationary reference, where e is relative
        to an ill-defined E0 -- with E0, the Brillouin residual, and both the
        relative (dE vs E0) and absolute (E_tot) energies.'''
        import numpy
        from pyscf.lib import logger
        log = logger.new_logger(self)
        au2ev = 27.211386245988
        if not all(self.converged):
            log.note('TD states %s not converged.',
                     [i for i, x in enumerate(self.converged) if not x])
        try:
            e0 = self.reference_energy()
            bril = self.brillouin_norm()
        except Exception:
            log.note('Excitation energies (eV)\n%s',
                     numpy.asarray(self.e).real * au2ev)
            return self
        log.note('reference determinant E0 = %.10f Hartree '
                 '(Brillouin |F_ia|max = %.2e)', e0, bril)
        if bril > 1e-4:
            log.note('NOTE: the reference is non-stationary (|F_ia| not ~0). '
                     'The "dE vs E0" column below is relative to this '
                     'ill-defined E0 and is NOT a physical excitation energy -- '
                     'use the absolute E_tot and subtract a separately '
                     'converged ground-state energy.')
        refw = self.reference_weight
        head = '\n%-5s %14s %18s' % ('state', 'dE vs E0 (eV)', 'E_tot (Hartree)')
        log.note(head + ('%10s' % 'ref.wt' if refw is not None else ''))
        for k, ek in enumerate(numpy.asarray(self.e).real):
            line = '%-5d %14.4f %18.10f' % (k, ek * au2ev, e0 + ek)
            if refw is not None:
                line += '%10.3f' % refw[k]
            log.note(line)
        return self

    def brillouin_norm(self):
        '''max |F_ia| over the active occ-vir Fock block -- the Brillouin
        residual.  ~0 for a converged HF reference (or its internal rotations);
        markedly nonzero for a non-stationary reference (CAHF / core-hole),
        which is exactly when stripping the reference determinant from CIS is an
        approximation [the F_ia coupling to |Phi_0> is dropped].'''
        import numpy
        mf = self._scf
        mask = self.get_frozen_mask()
        mo = mf.mo_coeff[:, mask]
        occ = mf.mo_occ[mask]
        orbo = mo[:, occ == 1]
        orbv = mo[:, occ == 0]
        fock = mf.get_fock(dm=mf.make_rdm1())
        fov = orbo.conj().T @ fock @ orbv
        return abs(fov).max()

    def print_leading(self, nleading=3, thresh=0.1, xy=None, e=None, osc=None,
                      absolute=False, e_ground=None):
        '''Print the leading occ->vir amplitudes of each excited state.

        For every root, the dominant single-orbital transitions are listed by
        weight |X_ia|^2 (+ |Y_ia|^2 for RPA).  Orbital labels are the *absolute*
        spinor MO indices (so they survive a frozen/CVS active space).  Up to
        ``nleading`` transitions per state are shown, stopping early once the
        weight drops below ``thresh`` (the leading one is always printed).

        ``e_ground``: a separately-converged ground-state total energy.  When
        given, the oscillator strength uses the physical dE = E_tot - e_ground
        (not self.e), and a "dE phys (eV)" column is added -- the right way to
        report a relaxed / core-hole reference spectrum.
        '''
        import numpy
        from pyscf.lib import logger
        log = logger.new_logger(self)
        au2ev = 27.211386245988
        if xy is None:
            xy = self.xy
        if e is None:
            e = self.e
        mask = self.get_frozen_mask()
        occ = self._scf.mo_occ[mask]
        abs_idx = numpy.where(mask)[0]            # absolute MO index of each active orb
        occ_abs = abs_idx[occ == 1]               # active holes
        vir_abs = abs_idx[occ == 0]               # active particles
        nocc, nvir = len(occ_abs), len(vir_abs)
        e_phys = None if e_ground is None else (self.total_energy(e) - e_ground)
        if osc is None:
            try:
                osc = self.oscillator_strength(e=e, xy=xy, e_ground=e_ground)
            except Exception:
                osc = [None] * len(xy)
        refw = self.reference_weight       # |c0|^2 per state, or None
        rcol = '' if refw is None else '%8s' % 'ref.wt'
        pcol = '' if e_phys is None else '%14s' % 'dE phys (eV)'
        if absolute:
            e_abs = self.total_energy(e)
            eref = self.reference_energy()
            log.note('reference determinant energy E0 = %.10f Hartree '
                     '(Brillouin |F_ia|max = %.2e)', eref, self.brillouin_norm())
            log.note('\n%-5s %18s %12s%s %12s%s   leading occ -> vir (weight)',
                     'state', 'E_tot (Hartree)', 'E (eV)', pcol, 'osc.str.', rcol)
        else:
            log.note('\n%-5s %12s%s %12s%s   leading occ -> vir (weight)',
                     'state', 'E (eV)', pcol, 'osc.str.', rcol)
        for k, (x, y) in enumerate(xy):
            w = (numpy.asarray(x).reshape(nocc, nvir).conj() *
                 numpy.asarray(x).reshape(nocc, nvir)).real
            if isinstance(y, numpy.ndarray):       # RPA de-excitation weight
                w = w + (numpy.asarray(y).reshape(nocc, nvir).conj() *
                         numpy.asarray(y).reshape(nocc, nvir)).real
            order = numpy.argsort(w.ravel())[::-1]
            ostr = '' if osc[k] is None else '%12.4e' % osc[k]
            rstr = '' if refw is None else '%8.3f' % refw[k]
            pstr = '' if e_phys is None else '%14.4f' % (e_phys[k] * au2ev)
            if absolute:
                head = '%-5d %18.10f %12.4f%s %12s%s' % (
                    k, e_abs[k], e[k].real * au2ev, pstr, ostr, rstr)
            else:
                head = '%-5d %12.4f%s %12s%s' % (
                    k, e[k].real * au2ev, pstr, ostr, rstr)
            for n, idx in enumerate(order):
                wia = w.ravel()[idx]
                if n and wia < thresh:
                    break
                i, a = divmod(int(idx), nvir)
                log.note('%s   %4d -> %-4d (%.3f)', head, occ_abs[i], vir_abs[a], wia)
                head = ' ' * len(head)              # only label the first line
                if n + 1 >= nleading:
                    break


class TDA(_KernelXCMixin, _ghf.TDA):
    def Gradients(self):
        '''Excited-state nuclear gradient (socutils.grad.tdghf_grad); a spinor
        reference goes through its GHF / GKS representation.'''
        from socutils.grad import tdghf_grad
        from socutils.scf import spinor_hf, spinor_ghf
        if isinstance(self._scf, spinor_hf.SCF):
            return spinor_ghf._td_gradients(self)
        return tdghf_grad.Gradients(self)
    nuc_grad_method = Gradients

    # When True, gen_vind builds the response MO-driven: the Coulomb + exact
    # exchange come from a dense, precomputed A_JK block (cderi half-transformed
    # to the active occ/vir space once), and only the local f_xc is evaluated
    # per matvec on the grid.  Every Davidson matvec then avoids the AO Fock /
    # cderi build entirely -- essential for core-excitation (CVS) spectra, where
    # the occupied (hole) space is tiny but many roots/iterations are needed and
    # the AO-driven cost is one full Fock build *per trial vector*.
    mo_driven = False

    # When True, kernel() skips the Davidson solver and diagonalizes the full
    # dense A matrix directly (numpy.linalg.eigh).  Only sensible when the CIS
    # dimension nocc*nvir is small (it forms an nov x nov matrix and returns ALL
    # roots, then keeps the lowest nstates).  It reuses the same gen_vind matvec
    # as Davidson, so the two paths are guaranteed consistent.
    direct = False

    # When True, the reference determinant |Phi_0> is included in the response
    # space: the (1 + nov) x (1 + nov) matrix [[0, g^H],[g, A]] is diagonalized,
    # with g_ia = <Phi_i^a|H-E0|Phi_0> = F_ia* (the occ-vir Fock block coupling).
    # For a converged HF reference F_ia = 0 and this reduces to {0} + plain CIS;
    # for a NON-stationary reference (CAHF / core-hole, F_ia != 0) it restores
    # the coupling that ordinary CIS silently drops -- the lowest root is then
    # the variationally relaxed ground state.  Requires the dense path (forces
    # direct diagonalization).
    couple_reference = False
    reference_weight = None        # |c0|^2 (reference character) per state
    c0 = None                      # reference amplitude per state (complex)

    # The Davidson solver: 'locked' (socutils.tdscf.lr_locked: residuals of the nroots Ritz vectors only, converged
    # roots locked, one new vector per unconverged root, per-root preconditioning) or 'pyscf'
    # (pyscf.tdscf._lr_eig.eigh: MAX_SPACE_INC vectors every iteration, residuals of up to max_space Ritz vectors).
    davidson = 'locked'
    max_space = None            # lr_locked's subspace bound (None: max(8 nroots, 40))
    # Time-reversal adaptation of a Kramers-restricted closed-shell reference: A commutes with the antiunitary
    # time reversal K (K^2 = +1 on particle-hole excitations), so every root has a TR-even eigenvector z = K z and
    # the solve runs in the real space of TR-even vectors.  Their transition densities are TR-even (D_bb = D_aa^*,
    # D_ba = -D_ab^*: n real, m imaginary): half the f_xc channels, J densities and K pieces (socutils.dft.fxc_2c,
    # scf.spinor_jk).  'auto': where the reference's orbitals are Kramers paired (checked); True / False force.
    kramers = 'auto'
    KRAMERS_TOL = 1e-6
    nbuffer = None              # extra roots of the locked solve (None: max(2, nstates // 2))

    def kernel(self, x0=None, nstates=None):
        if not (self.direct or self.couple_reference):
            if self.davidson == 'locked':
                with self._response_env():
                    return self._locked_solve(x0, nstates)
            return super().kernel(x0=x0, nstates=nstates)
        # response xc / collinear scheme, as in _KernelXCMixin.kernel
        with self._response_env():
            return self._direct_solve(nstates)

    def _tr_operator(self):
        '''(K, (nocc, nvir)): time reversal on TDA amplitudes z (..., nocc, nvir), K z = T_o^* z^* T_v^T with
        K phi_q = sum_p phi_p T_pq on the reference orbitals; None when the orbitals are not Kramers paired (T
        not block diagonal / unitary in the occupied and virtual spaces) or the reference is not a spinor one'''
        mf = self._scf
        mol = mf.mol
        if not hasattr(mol, 'sph2spinor_coeff') or mf.mo_coeff is None or mol.spin != 0:
            return None
        mask = self.get_frozen_mask()
        C = numpy.asarray(mf.mo_coeff)[:, mask]
        occ = numpy.asarray(mf.mo_occ)[mask]
        nao = mol.nao_nr()
        try:
            U = numpy.vstack(mol.sph2spinor_coeff())
        except Exception:
            return None
        if C.shape[0] != U.shape[1]:
            return None
        Cg = U @ C
        KCg = numpy.vstack([-Cg[nao:].conj(), Cg[:nao].conj()])
        s = mol.intor_symmetric('int1e_ovlp')
        SKC = numpy.vstack([s @ KCg[:nao], s @ KCg[nao:]])
        T = Cg.conj().T @ SKC
        o, v = occ > 0, occ == 0
        To, Tv = T[o][:, o], T[v][:, v]
        tol = self.KRAMERS_TOL
        if (abs(T[o][:, v]).max(initial=0) > tol or abs(T[v][:, o]).max(initial=0) > tol
                or abs(To @ To.conj().T - numpy.eye(len(To))).max() > tol
                or abs(Tv @ Tv.conj().T - numpy.eye(len(Tv))).max() > tol):
            return None
        Toc = numpy.ascontiguousarray(To.conj())
        TvT = numpy.ascontiguousarray(Tv.T)

        def k(z):
            return lib.einsum('ji,xia,ab->xjb', Toc, numpy.asarray(z).conj(), TvT)
        return k, (int(o.sum()), int(v.sum()))

    def _locked_solve(self, x0=None, nstates=None):
        '''pyscf's TDA.kernel with the locking Davidson of socutils.tdscf.lr_locked'''
        from pyscf.lib import logger
        from socutils.tdscf import lr_locked
        cpu0 = (logger.process_clock(), logger.perf_counter())
        mf = self._scf
        if mf.mo_energy is None:
            mf.run()
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        log = logger.Logger(self.stdout, self.verbose)
        vind, hdiag = self.gen_vind(mf)

        def precond(r, e):
            d = hdiag - (e - self.level_shift)
            d[abs(d) < 1e-8] = 1e-8
            return r / d

        thr = self.positive_eig_threshold

        def pick(w, v):
            idx = numpy.where(w > thr)[0]
            return w[idx], v[:, idx]

        tr = self._tr_operator() if self.kramers else None
        if self.kramers is True and tr is None:
            raise RuntimeError('kramers=True: the reference orbitals are not Kramers paired')
        project = None
        if tr is not None:
            log.info('TDA: time-reversal adapted (TR-even trial vectors, real subspace)')
            nocc_, nvir_ = tr[1]

            def project(xs):
                z = numpy.asarray(xs).reshape(-1, nocc_, nvir_)
                return ((z + tr[0](z)) * .5).reshape(len(z), -1)
        if x0 is None:
            if tr is not None:
                # P(e) and P(i e) span the TR-even space of the unit vectors e of the guess
                g = numpy.asarray(self.get_init_guess(mf, 2 * nstates + 4), dtype=numpy.complex128)
                x0 = numpy.vstack([g, 1j * g])
            else:
                # a wider guess than nstates: a minimal one can miss roots (AuH x2c-SVPall: the 6th root)
                x0 = self.get_init_guess(mf, 2 * nstates + 4)
        # buffer roots, converged too and dropped: with locking a root missing from the subspace would stay
        # missing once the lowest nstates Ritz roots converge (AuH x2c-SVPall PBE0: the 3.826 eV root under
        # the 3.877 eV pair at nstates = 6; pyscf's lr_eigh misses the 3.803 eV pair at nstates = 4)
        nbuf = self.nbuffer if self.nbuffer is not None else max(2, nstates // 2)
        conv, e, x1 = lr_locked.eigh(vind, x0, precond, nstates + nbuf, tol=self.conv_tol, lindep=self.lindep,
                                     max_cycle=self.max_cycle, max_space=self.max_space, pick=pick, log=log,
                                     project=project, real=tr is not None)
        conv, e, x1 = conv[:nstates], e[:nstates], x1[:nstates]
        self.converged, self.e = conv, e
        mo_occ = mf.mo_occ[self.get_frozen_mask()]
        nocc = int((mo_occ > 0).sum())
        nvir = mo_occ.size - nocc
        self.xy = [(xi.reshape(nocc, nvir), 0) for xi in x1]
        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)
        log.timer('TDA (locked Davidson)', *cpu0)
        self._finalize()
        return self.e, self.xy

    def _coupling_vector(self):
        '''g_ia = <Phi_i^a|H-E0|Phi_0> in this module's amplitude convention.

        Equals the *conjugate* occ-vir Fock block F_ia* (= F_ai); the conjugate
        matches the foo.conj() hole-term convention and is fixed by requiring the
        augmented (reference + singles) matrix to be invariant under occ/vir
        rotations of a non-stationary reference.'''
        mf = self._scf
        mask = self.get_frozen_mask()
        mo = mf.mo_coeff[:, mask]
        occ = mf.mo_occ[mask]
        orbo = mo[:, occ == 1]
        orbv = mo[:, occ == 0]
        fock = mf.get_fock(dm=mf.make_rdm1())
        fov = orbo.conj().T @ fock @ orbv
        return fov.conj().ravel()

    def _direct_solve(self, nstates=None):
        '''Build the full TDA matrix A (via the gen_vind matvec on the identity)
        and diagonalize it densely.  For small nocc*nvir this is faster and more
        robust than Davidson, and returns every root.  With couple_reference the
        reference determinant is appended as an extra basis vector.'''
        import numpy
        from pyscf.lib import logger
        log = logger.new_logger(self)
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        vind, hdiag = self.gen_vind(self._scf)
        nov = hdiag.size
        mask = self.get_frozen_mask()
        mo_occ = self._scf.mo_occ[mask]
        nocc = int(numpy.count_nonzero(mo_occ == 1))
        nvir = mo_occ.size - nocc
        # vind(e_j) returns A @ e_j = column j of A, so vind(I) gives A^T
        A = vind(numpy.eye(nov)).reshape(nov, nov).T

        if self.couple_reference:
            g = self._coupling_vector()
            log.info('TDA direct (reference-coupled): dense %d x %d', nov + 1, nov + 1)
            M = numpy.zeros((nov + 1, nov + 1), dtype=complex)
            M[1:, 1:] = A
            M[1:, 0] = g
            M[0, 1:] = g.conj()
            M = (M + M.conj().T) * .5
            w, c = numpy.linalg.eigh(M)
            nstates = min(nstates, nov + 1)
            self.e = w[:nstates]
            self.c0 = c[0, :nstates].copy()   # reference amplitude per state
            self.reference_weight = (self.c0.conj() * self.c0).real
            self.xy = [(c[1:, i].reshape(nocc, nvir), 0)
                       for i in range(nstates)]
        else:
            log.info('TDA direct: dense diagonalization of the full %d x %d A',
                     nov, nov)
            A = (A + A.conj().T) * .5        # Hermitize against round-off
            w, c = numpy.linalg.eigh(A)
            nstates = min(nstates, nov)
            self.e = w[:nstates]
            self.c0 = None
            self.reference_weight = None
            self.xy = [(c[:, i].reshape(nocc, nvir), 0)
                       for i in range(nstates)]
        self.converged = [True] * nstates
        log.timer('TDA direct diagonalization', *cpu0)
        self._finalize()
        return self.e, self.xy

    def gen_vind(self, mf=None):
        if not self.mo_driven:
            out = self._gen_vind_fxc_mo(mf)
            return out if out is not None else super().gen_vind(mf)
        if mf is None:
            mf = self._scf
        from pyscf.scf import hf
        A_JK, foo, fvv, orbo, orbv, hyb = _build_ab_jk_cderi(self)
        nocc, nvir = orbo.shape[1], orbv.shape[1]
        nov = nocc * nvir
        # diagonal of the Fock "orbital" term, F_aa - F_ii (= e_a - e_i when
        # the orbitals are canonical).  Davidson preconditioner only.
        hdiag = (fvv.diagonal().real[None, :] -
                 foo.diagonal().real[:, None]).ravel().copy()

        from pyscf.lib import logger
        log = logger.new_logger(self)
        ksdft = isinstance(mf, hf.KohnShamDFT)
        if ksdft:
            ni = mf._numint
            xc = self.xc_kernel if self.xc_kernel is not None else mf.xc
            log.info('[mo_driven] caching xc kernel (collinear=%s, spin_samples=%s) ...',
                     getattr(ni, 'collinear', '?'), getattr(ni, 'spin_samples', '?'))
            rho0, vxc, fxc = ni.cache_xc_kernel(
                mf.mol, mf.grids, xc, mf.mo_coeff, mf.mo_occ, 1)
            # process the f_xc on chunks of trial vectors, not the whole Davidson
            # block at once: bounds the (chunk, n2c, n2c) transition-density
            # memory AND makes each grid f_xc evaluation visible (the dominant
            # per-matvec cost; J/K is already in A_JK).
            n2c = mf.mol.nao_2c()
            chunk = getattr(self, 'fxc_chunk', None) or max(
                1, int(2e9 / (n2c * n2c * 16)))
            log.info('[mo_driven] xc kernel cached; Davidson matvec = A_JK@x + '
                     'grid f_xc (in chunks of %d vectors)', chunk)

        counter = [0]

        def vind(zs):
            zs = numpy.asarray(zs).reshape(-1, nov)
            # Coulomb/exchange from the dense A_JK + the Fock "orbital" term.
            # The orbital term is the full F_ab d_ij - F_ij d_ab (non-canonical
            # safe), i.e. z @ F_vv^T - F_oo @ z, not just (e_a - e_i) * z.
            z = zs.reshape(-1, nocc, nvir)
            v = (zs @ A_JK.T).reshape(-1, nocc, nvir)
            # vir block + Sum_b F_ab z_ib ; hole block - Sum_j F_ij* z_ja.
            # The hole term carries foo.conj(): the transition density conjugates
            # the occupied index (orbo.conj()), so the occ-occ Fock block must be
            # contracted as F_ij* to stay consistent with the two-electron A_JK.
            # (For canonical orbitals foo is real-diagonal, conj is a no-op.)
            v += lib.einsum('xib,ab->xia', z, fvv)
            v -= lib.einsum('ij,xja->xia', foo.conj(), z)
            if ksdft:
                nvec = z.shape[0]
                for s in range(0, nvec, chunk):
                    t0 = (logger.process_clock(), logger.perf_counter())
                    zc = z[s:s + chunk]
                    dm1 = lib.einsum('xov,pv,qo->xpq', zc, orbv, orbo.conj())
                    vfxc = ni.get_fxc(mf.mol, mf.grids, xc, None, dm1, 0, 0, 0,
                                      rho0, vxc, fxc)
                    v[s:s + chunk] += lib.einsum('xpq,qo,pv->xov', vfxc,
                                                 orbo, orbv.conj())
                    counter[0] += 1
                    log.timer('[mo_driven]   f_xc chunk %d (%d/%d vectors of '
                              'this matvec)' % (counter[0], min(s + chunk, nvec),
                                                nvec), *t0)
            return v.reshape(-1, nov)

        return vind, hdiag


    def _gen_vind_fxc_mo(self, mf=None):
        '''the TDA matvec with the response J / K AO-driven (mf.get_jk: RIJ + multi-density COSX) and the f_xc
        in the MO form fused on the grid (socutils.dft.fxc_2c.mo_response: no transition density matrices for
        the xc part); None where it does not apply (fxc_2c.use_mo) and pyscf's operation runs'''
        from pyscf.scf import hf
        from pyscf.lib import logger
        from socutils.dft import fxc_2c
        if mf is None:
            mf = self._scf
        if not isinstance(mf, hf.KohnShamDFT) or getattr(self, 'wfnsym', None) is not None:
            return None
        ni = mf._numint
        if not hasattr(ni, 'U') or ni.collinear[0] not in ('n', 'm') or mf.do_nlc():
            return None
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)
        xctype = ni._xc_type(mf.xc)
        if omega != 0 or xctype not in ('LDA', 'GGA'):
            return None
        mol = mf.mol
        mask = self.get_frozen_mask()
        mo_coeff = mf.mo_coeff[:, mask]
        mo_energy = mf.mo_energy[mask]
        mo_occ = mf.mo_occ[mask]
        occidx = numpy.where(mo_occ == 1)[0]
        viridx = numpy.where(mo_occ == 0)[0]
        nocc, nvir = len(occidx), len(viridx)
        if not fxc_2c.use_mo(mol.nao, nocc, nvir):
            return None
        orbo, orbv = mo_coeff[:, occidx], mo_coeff[:, viridx]
        t0 = (logger.process_clock(), logger.perf_counter())
        rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc, mf.mo_coeff, mf.mo_occ, 1)
        iso = fxc_2c.isotropic_kernel(fxc, xctype)
        if iso is None:
            return None
        rho0 = vxc = fxc = None
        logger.timer(self, 'response xc kernel (isotropic, MO form)', *t0)
        # the GHF (scalar AO x spin) orbitals of the grid work
        orbo_g = numpy.ascontiguousarray(ni.U @ orbo)
        orbv_g = numpy.ascontiguousarray(ni.U @ orbv)
        e_ia = mo_energy[viridx] - mo_energy[occidx, None]
        hdiag = e_ia.ravel().real
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)
        log = logger.new_logger(self)

        def vind(zs):
            zs = numpy.asarray(zs).reshape(-1, nocc, nvir)
            dms = lib.einsum('xov,pv,qo->xpq', zs, orbv, orbo.conj())
            if hybrid:
                vj, vk = mf.get_jk(mol, dms, hermi=0)
                v1ao = vj - hyb * vk
            else:
                v1ao = mf.get_j(mol, dms, hermi=0)
            v1mo = lib.einsum('xpq,qo,pv->xov', v1ao, orbo, orbv.conj())
            t1 = (logger.process_clock(), logger.perf_counter())
            v1mo += fxc_2c.mo_response(ni, mol, mf.grids, iso, orbo_g, orbv_g, zs, xctype, mf.max_memory)
            log.timer_debug1('f_xc (MO form, %d vectors)' % zs.shape[0], *t1)
            v1mo += zs * e_ia
            return v1mo.reshape(v1mo.shape[0], -1)

        return vind, hdiag


CIS = TDA


class _PairedRPA(_ghf.TDHF):
    '''Full RPA/TDDFT solved with the paired-trial-vector Davidson
    (socutils.tdscf.lr_davidson), which represents the indefinite Krein
    metric correctly for complex orbitals.  Experimental -- TDA remains the
    recommended default.'''

    conv_tol = 1e-6
    reference_weight = None     # read by _KernelXCMixin._finalize (the reference-coupled TDA only)
    c0 = None

    def kernel(self, x0=None, nstates=None):
        import numpy as np
        from pyscf.lib import logger
        from socutils.tdscf import lr_davidson
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        log = logger.Logger(self.stdout, self.verbose)

        vind, hdiag = self.gen_vind(self._scf)
        conv, e, zs, nmv = lr_davidson.paired_eig(
            vind, hdiag, nroots=nstates, x0=x0, conv_tol=self.conv_tol,
            max_cycle=self.max_cycle,
            pos_tol=getattr(self, 'positive_eig_threshold', 1e-3),
            verbose=self.verbose > 4)
        log.debug('TDRPA paired Davidson: %d matvecs', nmv)
        self.converged = conv
        self.e = e

        mask = self.get_frozen_mask()
        mo_occ = self._scf.mo_occ[mask]
        nocc = int(np.count_nonzero(mo_occ > 0))
        nvir = mo_occ.size - nocc
        # zs are normalized to |X|^2 - |Y|^2 = 1 already
        self.xy = [(z[:nocc*nvir].reshape(nocc, nvir),
                    z[nocc*nvir:].reshape(nocc, nvir)) for z in zs]
        log.timer('TDRPA (paired Davidson)', *cpu0)
        self._finalize()
        return self.e, self.xy


class TDRPA(_KernelXCMixin, _PairedRPA):
    '''Full RPA / TDDFT.  On a Kramers-restricted closed-shell reference (kramers = 'auto' / True) it is solved in
    real arithmetic on the time-reversal even amplitudes (davidson = 'ssf', socutils.tdscf.lr_locked.rpa_eig);
    otherwise (or davidson = 'paired') with the paired complex Davidson of socutils.tdscf.lr_davidson.

    Time-reversal structure.  With the antilinear B^(z) = B z^* the complex RPA equations A X + B Y = w X,
    B^* X + A^* Y = -w Y read (A + B^) P = w Q, (A - B^) Q = w P for P = X + Y^*, Q = X - Y^*; A +- B^ are
    real-linear and symmetric in Re <u, v>.  Both commute with the time reversal K z = T_o^* z^* T_v^T (K^2 = +1),
    so P and Q can be taken TR even, a real space of dimension nocc nvir on which A +- B^ are real symmetric
    matrices (the TR-odd half i P gives the same roots again).  The transition density of (X, Y) is
    D(X) + D(Y^*)^H, D(z) = C_v z^T C_o^H, TR even for TR-even P, Q:

        A + B^ acts on the Hermitian part D(P) + D(P)^H:        TR even and Hermitian: the charge density n
                                                                (real), m = 0 -> J, exact exchange, f_nn
        A - B^ acts on the anti-Hermitian part D(Q) - D(Q)^H:   n = 0, the magnetisation i m (m real) ->
                                                                no J, exact exchange, f_mm

    so, unlike the nonrelativistic singlet case, A - B is not the orbital-energy difference for a pure functional
    (the spin-orbit-mixed magnetisation response f_mm enters), and exact exchange enters both.  One response of the
    density D(u) + D(v)^H (u = (p + q) / 2, v = (p - q) / 2) gives (A + B^) p and (A - B^) q at once (the top and
    the conjugated bottom block of the complex RPA matvec, top +- bottom^*): a TR-even density with Im n = 0, Re m = 0,
    the half channels / K pieces of the TR-adapted TDA, one TDA-sized response per (p, q) pair.  The same response
    operator as the complex path (exact, RIJ, RIJ + COSX), so both give the same roots.

    A +- B^ positive definite <=> the reference is stable; otherwise (lr_locked.RPAInstability) a warning and,
    with instability = 'tda', the TR-adapted TDA roots instead (self.unstable = True), 'raise' re-raises.'''
    davidson = 'ssf'            # 'ssf' (TR-adapted real solve where kramers applies) | 'paired'
    kramers = 'auto'
    KRAMERS_TOL = 1e-6
    nbuffer = None              # extra roots of the locked solve (None: max(2, nstates // 2))
    max_space = None            # lr_locked's subspace bound (None: max(8 nroots, 40))
    instability = 'tda'         # A +- B not positive definite: 'tda' (warn, TDA roots) | 'raise'
    unstable = None

    # The RPA structure (B^T = B, A^H = A; the paired solver's free partner matvec, this solver's one response
    # per (p, q) pair) needs a response with V(D^H) = V(D)^H.  Exact J / K, RIJ and f_xc have it; the COSX
    # response K (the symmetric bilinear form of the overlap fit, spinor_jk.RESPONSE_SYMMETRIC) does not, to
    # 5e-5 of |V| (AuH x2c-SVPall), so with COSX the paired and the TR-adapted solves differ by ~1e-8 Eh.
    # hsym_response = True takes 1/2 (V(D) + V(D^H)^H) (twice the response cost; for validation): both solvers
    # then solve the same RPA.
    hsym_response = False

    _tr_operator = TDA._tr_operator

    def gen_vind(self, mf=None):
        vind, hdiag = super().gen_vind(mf)
        if not self.hsym_response:
            return vind, hdiag
        nov = hdiag.size // 2

        def pair(z):
            return numpy.hstack([z[:, nov:].conj(), z[:, :nov].conj()])

        def vind_h(xys):
            # K p(b) = -p(K b) for the RPA matrix K = [[A, B], [-B^*, -A^*]], p(x, y) = (y^*, x^*)
            z = numpy.asarray(xys).reshape(-1, 2 * nov)
            return (numpy.asarray(vind(z)).reshape(-1, 2 * nov) - pair(numpy.asarray(vind(pair(z)))
                                                                         .reshape(-1, 2 * nov))) * .5
        return vind_h, hdiag

    def kernel(self, x0=None, nstates=None):
        with self._response_env():
            if self.davidson == 'ssf' and self.kramers:
                tr = self._tr_operator()
                if tr is None and self.kramers is True:
                    raise RuntimeError('kramers=True: the reference orbitals are not Kramers paired')
                if tr is not None:
                    return self._ssf_solve(tr, x0, nstates)
            return _PairedRPA.kernel(self, x0=x0, nstates=nstates)

    def gen_vind_pm(self, mf=None):
        '''(aop, hdiag): aop(ps, qs) -> ((A + B^) ps, (A - B^) qs) for rows of amplitudes (n, nocc nvir), one
        response density D(u) + D(v)^H per pair through the complex RPA matvec; hdiag = e_a - e_i'''
        vind, hdiag = self.gen_vind(mf if mf is not None else self._scf)
        nov = hdiag.size // 2
        hdiag = numpy.asarray(hdiag[:nov]).real

        def aop(ps, qs):
            ps = numpy.asarray(ps).reshape(-1, nov)
            qs = numpy.asarray(qs).reshape(-1, nov)
            u = (ps + qs) * .5
            v = (ps - qs) * .5
            hx = numpy.asarray(vind(numpy.hstack([u, v.conj()]))).reshape(-1, 2 * nov)
            top = hx[:, :nov]                   # A X + B Y
            bot = -hx[:, nov:].conj()           # (B^* X + A^* Y)^* = B X^* + A Y^*
            return top + bot, top - bot
        return aop, hdiag

    def _ssf_solve(self, tr, x0=None, nstates=None):
        from pyscf.lib import logger
        from socutils.tdscf import lr_locked
        cpu0 = (logger.process_clock(), logger.perf_counter())
        mf = self._scf
        if mf.mo_energy is None:
            mf.run()
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        log = logger.Logger(self.stdout, self.verbose)
        log.info('TDRPA: time-reversal adapted (TR-even P = X + Y*, Q = X - Y*, real A +- B, SSF Davidson)')
        aop, hdiag = self.gen_vind_pm(mf)
        k, (nocc, nvir) = tr

        def project(xs):
            z = numpy.asarray(xs).reshape(-1, nocc, nvir)
            return ((z + k(z)) * .5).reshape(len(z), -1)

        def precond(r, e):
            d = hdiag - (e - self.level_shift)
            d[abs(d) < 1e-8] = 1e-8
            return r / d

        thr = self.positive_eig_threshold

        def pick(w):
            return numpy.where(w > thr)[0]

        if x0 is None:
            ng = min(hdiag.size, 2 * nstates + 4)
            idx = numpy.argsort(hdiag, kind='stable')[:ng]
            g = numpy.zeros((ng, hdiag.size), dtype=numpy.complex128)
            g[numpy.arange(ng), idx] = 1
            x0 = numpy.vstack([g, 1j * g])
        else:
            x0 = numpy.asarray(x0)
            x0 = x0.reshape(-1, x0.shape[-1])[:, :hdiag.size]
        nbuf = self.nbuffer if self.nbuffer is not None else max(2, nstates // 2)
        nroots = min(nstates + nbuf, hdiag.size)
        try:
            conv, e, P, Q = lr_locked.rpa_eig(aop, x0, precond, nroots, tol=self.conv_tol, lindep=self.lindep,
                                              max_cycle=self.max_cycle, max_space=self.max_space, pick=pick,
                                              log=log, project=project)
        except lr_locked.RPAInstability as err:
            self.unstable = True
            log.warn('TDRPA: reference instability (%s): the RPA has imaginary / non-positive roots here', err)
            if self.instability == 'raise':
                raise
            log.warn('TDRPA: returning the time-reversal adapted TDA roots instead (td.instability = "raise" '
                     'to stop)')
            return self._tda_fallback(nstates)
        self.unstable = False
        conv, e, P, Q = conv[:nstates], e[:nstates], P[:nstates], Q[:nstates]
        self.converged, self.e = conv, e
        X = (P + Q) * .5
        Y = ((P - Q) * .5).conj()
        self.xy = [(x.reshape(nocc, nvir), y.reshape(nocc, nvir)) for x, y in zip(X, Y)]
        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)
        log.timer('TDRPA (TR-adapted SSF Davidson)', *cpu0)
        self._finalize()
        return self.e, self.xy

    def _tda_fallback(self, nstates):
        td = TDA(self._scf)
        for key in ('frozen', 'conv_tol', 'max_cycle', 'lindep', 'level_shift', 'positive_eig_threshold',
                    'xc_kernel', 'collinear', 'picture_change', 'kramers', 'KRAMERS_TOL', 'nbuffer', 'max_space',
                    'verbose', 'stdout', 'chkfile'):
            if hasattr(self, key):
                setattr(td, key, getattr(self, key))
        e, xy = td.kernel(nstates=nstates)
        self.converged, self.e = td.converged, e
        self.xy = [(x, numpy.zeros_like(x)) for x, _ in xy]
        return self.e, self.xy
