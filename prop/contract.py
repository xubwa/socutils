#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''The one place a property operator meets a density.

Every property expectation value in socutils is the same contraction, and it
is written once, here.  Five copies of it grew in ``prop/`` because each
kernel also built its own operator and fetched its own density; with those
two jobs given to their own layers, what is left is small enough that a
second copy would be indefensible.

**MO basis by default.**  The AO form is mathematically identical --
``Tr(O_AO D_AO) == Tr(O_MO gamma_MO)`` -- but the AO route carries two
ambiguities that no cheap check can catch, and the MO route carries neither:

  * The AO density convention.  ``C gamma^T C^dagger`` and ``C gamma
    C^dagger`` are both Hermitian and both give ``Tr(S D) = N``, so nothing
    short of a reference number separates them; picking wrong turns a
    Kramers-forbidden transition moment into a signal seven orders of
    magnitude too large.  In the MO basis no ``C`` appears and
    ``einsum('pq,pq->', O, gamma)`` is the only thing one can write.
  * The AO basis identity.  ``mol.nao_2c() == 2 * mol.nao_nr()`` exactly, so
    a spinor operator and a spin-orbital density have matching dimensions
    and contract silently -- the measured -0.43 a.u. dipole where 0 was
    correct.  An MO-basis gamma has dimension nmo whichever AO basis built
    it.

The AO entry point is kept because a mean field has nothing else to offer:
``mf.make_rdm1()`` is an AO density.  Its conventions are the caller's
responsibility, as with anything read from :mod:`socutils.prop.cfour`.

**No core/active split.**  ``gamma`` spans the whole MO space.  The core
block is contracted like any other -- ``Tr(O_core)`` would be the same
number only when the core density is the identity, and it is not always:
CAHF fractional occupations, :func:`socutils.scf.addons.even_occ` and
``lo/atomic_ref`` all produce cores that are not.

**Bare atomic units in, bare atomic units out.**  No electron-charge sign,
no nuclear term, no isotope factor, no unit conversion -- those belong to
the observable layer, which is where CFOUR also leaves them.

Degeneracy.  A single-state expectation value means what one expects only
when the level is non-degenerate, or when the operator's block over the
degenerate level is proportional to the identity.  Within a Kramers doublet
that holds for every time-even operator and fails for every time-odd one;
within a (2J+1) multiplet it holds only for scalars, so a free atom's EFG
depends on which M_J the solver happened to return.  This module does not
police that -- it cannot see the spectrum -- but :func:`expect` accepts and
passes through a ``degeneracy`` note so the observable layer can say so.
'''

import numpy

__all__ = ['expect', 'expect_ao', 'to_mo']


def to_mo(op, mo_coeff, mo_bra=None):
    '''Transform an AO-basis operator to the MO basis: ``C_bra^dag O C_ket``.

    ``op`` may carry a leading component axis; the result keeps it.  Pass
    ``mo_bra`` only when bra and ket orbitals differ.
    '''
    op = numpy.asarray(op)
    ket = numpy.asarray(mo_coeff)
    bra = ket if mo_bra is None else numpy.asarray(mo_bra)
    if op.shape[-2] != bra.shape[0] or op.shape[-1] != ket.shape[0]:
        raise ValueError(
            'operator is (..., %d, %d) but the orbitals are (%d, ...) and '
            '(%d, ...); the operator and the orbitals are not in the same AO '
            'basis' % (op.shape[-2], op.shape[-1], bra.shape[0], ket.shape[0]))
    return numpy.einsum('...pq,pi,qj->...ij', op, bra.conj(), ket,
                        optimize=True)


def _check_real(val, hermi, tol, what):
    '''Split a contraction into the part the operator's symmetry allows and
    the part it forbids, and complain if the forbidden part is not small.

    A Hermitian operator contracted with a state density gives a real number;
    an anti-Hermitian one (CFOUR's SOLZ stores i*L_z) gives an imaginary one.
    The discarded part is checked on its absolute value -- several legacy
    sites test a signed ``> tol`` and let a large negative imaginary part
    through in silence.
    '''
    val = numpy.asarray(val)
    keep, drop = (val.real, val.imag) if hermi >= 0 else (val.imag, -val.real)
    scale = max(float(abs(numpy.asarray(keep)).max()), 1.0)
    bad = float(abs(numpy.asarray(drop)).max())
    if tol is not None and bad > tol * scale:
        raise ValueError(
            '%s: the component forbidden by the operator symmetry is %.3e '
            '(relative %.3e), which is not numerical noise. For a %s '
            'operator the expectation value must be %s. Check that the '
            'density and the operator are in the same basis and that the '
            'density is the one you meant.'
            % (what, bad, bad / scale,
               'Hermitian' if hermi >= 0 else 'anti-Hermitian',
               'real' if hermi >= 0 else 'imaginary'))
    return keep


def expect(op, gamma, mo_coeff=None, mo_bra=None, hermi=1, tol=1e-6,
           complex_out=False, what='property'):
    '''Expectation value ``<O> = sum_pq O_pq gamma_pq`` in the MO basis.

    Args:
        op: operator matrix, ``(nao, nao)`` or ``(comp, nao, nao)``.  In the
            AO basis when ``mo_coeff`` is given, in the MO basis otherwise.
        gamma: one-particle density ``gamma[p,q] = <p^dagger q>`` over the
            full MO space -- a state density, or a transition density
            ``<i|p^dagger q|j>``.
        mo_coeff: orbitals to transform ``op`` with.  Omit if ``op`` is
            already in the MO basis.
        mo_bra: bra-side orbitals, when they differ from the ket's.
        hermi: +1 for a Hermitian operator (the value is real), -1 for an
            anti-Hermitian one (the value is imaginary; CFOUR's ``SOLZ``
            stores ``i*L_z``).
        tol: relative size above which the symmetry-forbidden component is
            an error rather than round-off.  ``None`` disables the check.
        complex_out: return the raw complex value instead of the allowed
            real part.  For a transition density between different states
            the value is genuinely complex and only its modulus is
            meaningful, so this is the right form there.

    Returns:
        scalar, or ``(comp,)`` -- in bare atomic units.
    '''
    op = numpy.asarray(op)
    gamma = numpy.asarray(gamma)
    if gamma.ndim != 2 or gamma.shape[0] != gamma.shape[1]:
        raise ValueError('gamma must be a square (nmo, nmo) matrix, got %r'
                         % (gamma.shape,))
    if mo_coeff is not None:
        op = to_mo(op, mo_coeff, mo_bra=mo_bra)
    if op.shape[-2:] != gamma.shape:
        raise ValueError(
            'operator is (..., %d, %d) but gamma is (%d, %d). If the operator '
            'is still in the AO basis, pass mo_coeff=; if it is in the MO '
            'basis, it must span the same orbitals as gamma.'
            % (op.shape[-2], op.shape[-1], gamma.shape[0], gamma.shape[1]))

    val = numpy.einsum('...pq,pq->...', op, gamma, optimize=True)
    if complex_out:
        return val
    out = _check_real(val, hermi, tol, what)
    return out


def expect_ao(op, dm, hermi=1, tol=1e-6, complex_out=False, what='property'):
    '''Expectation value against an AO-basis density -- the escape hatch.

    ``Tr(O D) = einsum('...pq,qp->...', O, D)``, the convention documented in
    :func:`socutils.fci.addons.trans_rdm1_ao`, i.e. ``D`` must satisfy
    ``<i|O|j> = einsum('uv,vu->', O_ao, D)``.

    Use this when the density is only available in the AO basis -- a mean
    field's ``make_rdm1()`` is exactly that case.  Everything else should go
    through :func:`expect`: in the AO basis neither the density's transpose
    convention nor its underlying basis (spinor vs spin-orbital, which have
    identical dimensions) can be checked here, and both have produced silent
    wrong answers in this tree.
    '''
    op = numpy.asarray(op)
    dm = numpy.asarray(dm)
    if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
        raise ValueError('dm must be a square (nao, nao) matrix, got %r'
                         % (dm.shape,))
    if op.shape[-2:] != dm.shape:
        raise ValueError('operator is (..., %d, %d) but dm is (%d, %d)'
                         % (op.shape[-2], op.shape[-1], dm.shape[0],
                            dm.shape[1]))
    val = numpy.einsum('...pq,qp->...', op, dm, optimize=True)
    if complex_out:
        return val
    return _check_real(val, hermi, tol, what)
