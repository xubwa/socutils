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
# Only TDA is exposed here.  TDA sets B=0, so its response matrix A is genuinely
# Hermitian and the Davidson solve (lr_eig.eigh) is robust.  The full RPA/TDDFT
# (Casida) problem is k-Hermitian (non-Hermitian) on a Krein space with an
# indefinite metric [Furche & Chen, J. Chem. Phys. 163, 174104 (2025)]; its
# iterative solution is delicate near eta-neutral roots and is left aside for
# now.
#
# Validated vs pyscf's 2-component GKS-TDA (non-relativistic integrals, no SOC)
# to machine precision (LDA / GGA / hybrid).  With SOC integrals the orbitals
# are genuinely complex; set mf._numint.collinear='mcol' for the spin-flip
# (non-collinear) response.
#

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

    # For an X2C reference (mf.with_x2c present) compute transition dipoles
    # with the picture-change-corrected operator (picture_change_dipole).
    # The bare int1e_r_spinor is exact for a nonrelativistic spinor/GHF
    # reference and is used automatically when there is no with_x2c.
    # Measured PCE of the bare operator (HBr/cc-pVDZ, x2camf): valence f
    # off by <~1e-3 relative, but K/L-edge (CVS) main lines off by 0.1-4%
    # and weak SOC satellites by tens of percent.
    picture_change = True

    def kernel(self, *args, **kwargs):
        # _gen_ghf_response returns a vind closure that reads mf.xc lazily (at
        # Davidson time), so the kernel xc must stay set for the whole solve,
        # not just while the response is generated.
        if self.xc_kernel is None:
            return super().kernel(*args, **kwargs)
        with lib.temporary_env(self._scf, xc=self.xc_kernel):
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
            ymo = numpy.array([numpy.einsum('xij,ij->x', ints, y) for x, y in xy])
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

    def kernel(self, x0=None, nstates=None):
        if not (self.direct or self.couple_reference):
            return super().kernel(x0=x0, nstates=nstates)
        # honour xc_kernel for the response (same as _KernelXCMixin.kernel)
        if self.xc_kernel is None:
            return self._direct_solve(nstates)
        with lib.temporary_env(self._scf, xc=self.xc_kernel):
            return self._direct_solve(nstates)

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
            return super().gen_vind(mf)
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


CIS = TDA


class _PairedRPA(_ghf.TDHF):
    '''Full RPA/TDDFT solved with the paired-trial-vector Davidson
    (socutils.tdscf.lr_davidson), which represents the indefinite Krein
    metric correctly for complex orbitals.  Experimental -- TDA remains the
    recommended default.'''

    conv_tol = 1e-6

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
    pass
