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
``gamma[p,q] = <p^dagger q>`` spanning ``mo_coeff``.

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

__all__ = ['densities', 'dm_meta']


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

    Duck-typed on purpose.  ``somf.snso.SCF`` and ``scf.spinor_hf.SpinorSCF``
    are both spinor references with no common base below pyscf's ``hf.SCF``,
    and ``scf.x2camf_hf.x2camf_ghf`` builds its object on the fly, so a single
    isinstance chain misses cases that are plainly fine.
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


def densities(wfn, **kwargs):
    '''Iterate ``(label, dm, mo_coeff, meta)`` over whatever ``wfn`` can give.

    Dispatch is on the object, never on an array shape.  Anything this module
    does not recognise raises rather than being guessed at -- a density in the
    wrong basis contracts silently and the wrong number looks reasonable.
    '''
    if hasattr(wfn, 'make_rdm1') and hasattr(wfn, 'mo_occ') and \
       not hasattr(wfn, 'fcisolver'):
        return _mean_field(wfn, **kwargs)

    raise TypeError(
        'no density adapter for %s. Mean-field references are supported; '
        'CI/CASCI and linear-response adapters are not written yet.'
        % type(wfn).__name__)
