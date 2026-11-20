#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''Where a wavefunction hands over its density matrices.

One adapter per wavefunction family, and property kernels that loop over
whatever comes out.  A property does not need to know what produced the
density, and a wavefunction does not need a method per property -- otherwise
every new method costs one function per property and every new property costs
one method per method.

The adapter is also the *only* gate that knows which basis a density lives
in, and that matters more than it looks.  ``mol.nao_2c() == 2 * mol.nao_nr()``
exactly, so a spinor density and a spin-orbital one have identical shapes and
contract with each other in silence -- the measured -0.43 a.u. dipole where 0
was correct.  Shape cannot discriminate; the wavefunction's class can.  So
the basis is decided here, from the object, and everything downstream is in
the spinor (or 4c) basis by construction rather than by convention: a
spin-orbital density is rotated on the way out, and a scalar
(non-relativistic) reference is refused outright.

Every adapter yields the same 4-tuple::

    (label, dm, mo_coeff, meta)

``mo_coeff is None`` marks ``dm`` as an AO-basis density -- a mean field has
nothing else to give -- and :func:`socutils.prop.contract.contract_dm` takes
its AO branch on exactly that signal.  Otherwise ``dm`` is an MO-basis
``gamma[p,q] = <p^dagger q>`` spanning ``mo_coeff``, which is what a CI
wavefunction hands over: one full-space ``gamma`` per root, all roots, with
the root's energy and its degeneracy group in ``meta``.

Always an iterator, even for a single density.  A caller that has to ask "is
this one density or several?" ends up writing that test in every property
kernel, and in this basis it cannot even be answered from the shape.  Squeeze
at the *end*, where a human reads the number, not here.

Which density is a wavefunction's own business: relaxed or unrelaxed, a state
or a transition.  This module asks for what the object can give and passes it
on; it does not judge, and it has no ``relaxed=`` switch.
'''

import numpy

from pyscf import lib
from pyscf.scf import dhf as _dhf

__all__ = ['densities', 'level_densities', 'dm_meta']


def dm_meta(basis, kind='state', **kw):
    '''The provenance travelling with one density.  Descriptive only -- no
    consumer branches on it; it exists so a printed number can say what it
    came from.'''
    meta = dict(basis=basis, kind=kind)
    meta.update(kw)
    return meta


def _is_4c(mf):
    return isinstance(mf, _dhf.DHF)


def _basis_of(mf):
    '''Which basis this mean field's density matrix lives in, decided from the
    object rather than from any array shape.

    Duck-typed on purpose.  ``scf.x2camf_hf.x2camf_ghf`` builds its object
    on the fly and pyscf's own ``x2c.SCF`` is a spinor reference outside
    socutils' class tree, so a single isinstance chain misses cases that are
    plainly fine; the helper's class name is asked first.
    '''
    if _is_4c(mf):
        return '4c'

    x2c = getattr(mf, 'with_x2c', None)
    if x2c is not None:
        name = type(x2c).__name__
        if 'SpinOrbital' in name:
            return 'spin_orbital'
        if 'Spinor' in name or 'X2CAMF' in name or 'X2CMP' in name:
            return 'spinor'

    # No helper (a bare, non-relativistic spinor mean field) or an
    # unrecognised one: fall back to the class, then to the AO dimension.
    from socutils.scf import spinor_hf
    if isinstance(mf, spinor_hf.SpinorSCF):
        return 'spinor'
    from pyscf.scf import ghf as _ghf
    if isinstance(mf, _ghf.GHF):
        return 'spin_orbital'

    mo = getattr(mf, 'mo_coeff', None)
    if mo is not None and numpy.asarray(mo).shape[0] == mf.mol.nao_2c() and \
       numpy.iscomplexobj(mo):
        # a 2-component reference this module has not been taught about
        return 'spinor'
    return 'scalar'


def _check_nelec(mf, dm, label, basis):
    '''``Tr(S D)`` must be the electron count.  Cheap, and decisive about the
    one confusion shapes cannot catch: the spinor and spin-orbital overlaps
    are genuinely different matrices (measured max difference 0.76 for
    H2O/6-31G), so contracting a density against the wrong one misses the
    electron count by a wide margin -- 28.3 against 10 in that test.
    '''
    mol = mf.mol
    if basis == '4c':
        n2c = mol.nao_2c()
        s = numpy.zeros((n2c * 2, n2c * 2), dtype=complex)
        s[:n2c, :n2c] = mol.intor_symmetric('int1e_ovlp_spinor')
        c = lib.param.LIGHT_SPEED
        s[n2c:, n2c:] = mol.intor_symmetric('int1e_spsp_spinor') * (.5 / c) ** 2
    else:
        s = mol.intor_symmetric('int1e_ovlp_spinor')
    if s.shape != dm.shape:
        raise ValueError(
            '%s: density is %r but the %s overlap is %r'
            % (label, dm.shape, basis, s.shape))
    n = numpy.einsum('ij,ji->', s, dm).real
    nref = float(numpy.sum(mf.mo_occ)) if getattr(mf, 'mo_occ', None) is not None \
        else float(mol.nelectron)
    if abs(n - nref) > 1e-6 * max(nref, 1.0):
        raise ValueError(
            '%s: Tr(S D) = %.10f but this reference holds %.10f electrons. '
            'The density and the overlap are not in the same basis.'
            % (label, n, nref))


def _mean_field(mf):
    '''One AO-basis density from a mean field.

    A spin-orbital (GHF) density is rotated into the j-adapted spinor basis
    so that everything downstream is spinor: ``spinor2sph`` and
    ``sph2spinor`` are a dual pair, and the one for a *density* is the
    inverse of the one for an operator.  Doing it here, once, is what makes
    "spinor only" a fact about the data rather than a rule in a docstring.
    '''
    basis = _basis_of(mf)
    if basis == 'scalar':
        raise TypeError(
            'this property lane is for 2-component (spinor / GHF) and '
            '4-component references; %s is a scalar one. Its scalar '
            'counterparts live in socutils.prop.efg.rhf and friends.'
            % type(mf).__name__)

    dm = mf.make_rdm1()
    if basis == 'spin_orbital':
        from socutils.scf.spinor_hf import sph2spinor
        dm = sph2spinor(mf.mol, numpy.asarray(dm))
        basis = 'spinor'

    dm = numpy.asarray(dm)
    if dm.ndim != 2:
        raise ValueError('expected one (n,n) density from %s, got shape %r'
                         % (type(mf).__name__, dm.shape))
    _check_nelec(mf, dm, type(mf).__name__, basis)

    meta = dm_meta(basis, kind='state',
                   source=type(mf).__name__,
                   converged=bool(getattr(mf, 'converged', False)),
                   nelec=float(numpy.sum(mf.mo_occ)))
    yield ('scf', dm, None, meta)


def _roots(mc):
    '''``[(index, ci, energy), ...]`` over the roots a CASCI/CASSCF holds.

    The solvers do not agree on what a "CI vector" is -- zfci hands back a
    coefficient array (a list of them for several roots), SHCI an integer
    index (a ``range`` for several) that names a Dice rdm file -- and a
    state-averaged object keeps one averaged ``e_tot`` beside per-root
    ``e_states``.  This is the one place that knows all of that; the rest of
    the adapter sees a flat list.
    '''
    ci = mc.ci
    if ci is None:
        raise ValueError('%s has no CI vector: run .kernel() first' % type(mc).__name__)
    cis = list(ci) if isinstance(ci, (list, tuple, range)) else [ci]

    e = getattr(mc, 'e_states', None)
    e = numpy.atleast_1d(numpy.asarray(mc.e_tot if e is None else e)).real
    if len(e) != len(cis):
        if len(e) == 1:
            e = numpy.repeat(e, len(cis))
        else:
            raise ValueError('%s holds %d CI roots but %d energies'
                             % (type(mc).__name__, len(cis), len(e)))
    return [(i, c, float(ei)) for i, (c, ei) in enumerate(zip(cis, e))]


def _degeneracy(energies, tol):
    '''Group consecutive roots within ``tol`` Hartree: ``(level, degeneracy)``
    per root.  Descriptive -- a Kramers doublet or a (2J+1) multiplet shows
    up here as one level with d > 1, which is exactly the case where a single
    root's expectation value of a non-scalar operator is not by itself
    meaningful (see the module doc of socutils.prop.contract).'''
    levels = []
    lvl = -1
    e_prev = None
    for e in energies:
        if e_prev is None or abs(e - e_prev) > tol:
            lvl += 1
        levels.append(lvl)
        e_prev = e
    degen = numpy.bincount(levels)
    return [(l, int(degen[l])) for l in levels]


def _cas_frame(mc):
    '''Orbitals and partition of a CASCI/CASSCF, after the basis check.'''
    mf = mc._scf
    basis = _basis_of(mf)
    if basis == 'scalar':
        raise TypeError(
            'this property lane is for 2-component (spinor) and 4-component '
            'references; the CASCI is built on a scalar %s.' % type(mf).__name__)
    if basis == 'spin_orbital':
        raise TypeError(
            'a CASCI on a spin-orbital (GHF) reference is not handled: its '
            'orbitals would have to be rotated into the spinor basis before '
            'the MO contraction, and no such CASCI exists in socutils yet.')
    mo = numpy.asarray(mc.mo_coeff)
    nelecas = mc.nelecas
    nelec_cas = int(nelecas) if numpy.isscalar(nelecas) else int(sum(nelecas))
    return basis, mo, mc.ncore, mc.ncas, nelecas, nelec_cas


def _full_space(casdm1, nmo, ncore, ncas, diagonal, dtype):
    '''Embed an active-space (transition) density in the full MO space: the
    core block is the identity for a state density and zero for a transition
    density between different states.'''
    gamma = numpy.zeros((nmo, nmo), dtype=dtype)
    if diagonal:
        gamma[:ncore, :ncore] = numpy.eye(ncore)
    gamma[ncore:ncore + ncas, ncore:ncore + ncas] = casdm1
    return gamma


def _cas(mc, degen_tol=1e-6):
    '''Every root of a CASCI/CASSCF, as a full-space MO-basis ``gamma``.

    ``gamma`` spans all of ``mc.mo_coeff``: identity on the core block, the
    solver's active-space ``<p^dagger q>`` on the active block, zero on the
    virtuals.  Full-space rather than core/active split because the
    contraction is one einsum either way and the reader sees one matrix per
    state.  Which convention the solver uses was measured, not read: zfci
    (both the string and the determinant-list kernels) and pyscf's
    ``fci_dhf_slow`` give ``gamma[p,q] = <p^dagger q>`` for a
    ``(|p> + i|q>)/sqrt2`` test state, which is what
    :func:`socutils.prop.contract.contract_dm` contracts as
    ``sum_pq O_pq gamma_pq``.  (pyscf's docstring says ``<q^dagger p>``; the
    code does not.)  SHCI's ``read_rdm1`` conjugates the Dice h5 on the way
    in to land on the same convention.
    '''
    basis, mo, ncore, ncas, nelecas, nelec_cas = _cas_frame(mc)
    nmo = mo.shape[1]

    roots = _roots(mc)
    degen = _degeneracy([e for _, _, e in roots], degen_tol)
    for (i, ci, e), (level, d) in zip(roots, degen):
        casdm1 = numpy.asarray(mc.fcisolver.make_rdm1(ci, ncas, nelecas))
        if casdm1.shape != (ncas, ncas):
            raise ValueError('active-space rdm1 of root %d is %r, expected (%d, %d)'
                             % (i, casdm1.shape, ncas, ncas))
        gamma = _full_space(casdm1, nmo, ncore, ncas, True, numpy.result_type(casdm1, mo))

        n = gamma.trace().real
        nref = ncore + nelec_cas
        if abs(n - nref) > 1e-6 * max(nref, 1):
            raise ValueError('root %d: Tr(gamma) = %.10f but the CAS holds %d '
                             'electrons (%d core + %d active)'
                             % (i, n, nref, ncore, nelec_cas))

        meta = dm_meta(basis, kind='state',
                       source=type(mc).__name__,
                       solver=type(mc.fcisolver).__name__,
                       root=i, energy=e, level=level, degeneracy=d,
                       converged=bool(getattr(mc, 'converged', False)),
                       nelec=float(nref))
        yield ('state_%d' % i, gamma, mo, meta)


def densities(wfn, **kwargs):
    '''Iterate ``(label, dm, mo_coeff, meta)`` over whatever ``wfn`` can give.

    Dispatch is on the object, never on an array shape.  Anything this module
    does not recognise raises rather than being guessed at -- a density in the
    wrong basis contracts silently and the wrong number looks reasonable.
    '''
    if hasattr(wfn, 'fcisolver') and hasattr(wfn, 'ncas'):
        return _cas(wfn, **kwargs)
    if hasattr(wfn, 'make_rdm1') and hasattr(wfn, 'mo_occ'):
        return _mean_field(wfn, **kwargs)

    raise TypeError(
        'no density adapter for %s. Mean-field and CASCI/CASSCF references '
        'are supported; linear-response (TDA/TDDFT) and coupled-cluster '
        'adapters are not written yet.' % type(wfn).__name__)


def level_densities(mc, degen_tol=1e-6):
    '''Every degenerate level of a CASCI/CASSCF as the ``d x d`` matrix of
    state and transition densities ``gamma[i, j][p, q] = <i|p^dagger q|j>``
    over the roots of that level, in the full MO space (core identity on
    the diagonal entries only).

    This is what a time-odd operator needs: inside a Kramers doublet or a
    multiplet the solver's roots are an arbitrary unitary mixture, so a
    single root's ``<mu>`` can be anything between the two eigenvalues of
    the ``d x d`` block, while the block itself -- and every invariant built
    from it -- is basis independent.  The transition densities come from
    the solver's ``trans_rdm1`` in the same ``<p^dagger q>`` convention as
    its ``make_rdm1`` (measured for zfci; the ``i == j`` entry equals
    ``make_rdm1`` to round-off).

    Yields ``('level_%d', gammas, mo, meta)`` with ``gammas`` of shape
    ``(d, d, nmo, nmo)`` -- ``d^2`` full matrices resident at once, which is
    fine for a doublet on a few hundred orbitals and something to revisit
    for large multiplets on large bases.
    '''
    basis, mo, ncore, ncas, nelecas, nelec_cas = _cas_frame(mc)
    nmo = mo.shape[1]
    roots = _roots(mc)
    degen = _degeneracy([e for _, _, e in roots], degen_tol)
    dtype = numpy.result_type(mo, numpy.complex128)
    pos = 0
    while pos < len(roots):
        level, d = degen[pos]
        members = roots[pos:pos + d]
        gammas = numpy.zeros((d, d, nmo, nmo), dtype=dtype)
        for a, (i, ci, _) in enumerate(members):
            for b, (j, cj, _) in enumerate(members):
                t = numpy.asarray(mc.fcisolver.trans_rdm1(ci, cj, ncas, nelecas))
                if t.shape != (ncas, ncas):
                    raise ValueError('transition rdm1 <%d|%d> is %r, expected (%d, %d)'
                                     % (i, j, t.shape, ncas, ncas))
                gammas[a, b] = _full_space(t, nmo, ncore, ncas, a == b, dtype)
        e = numpy.array([ei for _, _, ei in members])
        meta = dm_meta(basis, kind='level',
                       source=type(mc).__name__,
                       solver=type(mc.fcisolver).__name__,
                       level=level, degeneracy=d,
                       roots=tuple(i for i, _, _ in members),
                       energy=float(e.mean()), energy_spread=float(e.max() - e.min()),
                       converged=bool(getattr(mc, 'converged', False)),
                       nelec=float(ncore + nelec_cas))
        yield ('level_%d' % level, gammas, mo, meta)
        pos += d
