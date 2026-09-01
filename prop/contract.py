#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''Contract a one-electron property operator with a density matrix.

This is the one place in socutils where an operator meets a density, and it
is deliberately the whole of what this module does.  Which density -- a state
density, a transition density, relaxed or unrelaxed -- is the wavefunction's
business, not this layer's: it contracts what it is handed.

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

The AO route (``mo_coeff=None``) stays because a mean field has nothing else
to offer -- ``mf.make_rdm1()`` is an AO density -- and its conventions are
then the caller's responsibility.

Bare atomic units in, bare atomic units out: no electron-charge sign, no
nuclear contribution, no isotope factor, no unit conversion.  Those belong to
the property kernels, which is also where CFOUR leaves them.
'''

import numpy

__all__ = ['contract_dm', 'to_mo']


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
            '(%d, ...); operator and orbitals are not in the same AO basis'
            % (op.shape[-2], op.shape[-1], bra.shape[0], ket.shape[0]))
    return numpy.einsum('...pq,pi,qj->...ij', op, bra.conj(), ket,
                        optimize=True)


def contract_dm(op, dm, mo_coeff=None, mo_bra=None, hermi=1, tol=1e-6,
                complex_out=False, label='property'):
    '''Contract a property operator with a density matrix, in bare a.u.

    Args:
        op: operator matrix ``(nao, nao)`` or ``(comp, nao, nao)`` -- in the
            AO basis when ``mo_coeff`` is given, in the same basis as ``dm``
            otherwise.
        dm: one-particle density matrix.  With ``mo_coeff``, an MO-basis
            ``gamma[p,q] = <p^dagger q>`` (state or transition), contracted
            as ``sum_pq O_pq gamma_pq``.  Without it, an AO-basis density in
            the ``<i|O|j> = einsum('uv,vu->', O_ao, dm)`` convention of
            :func:`socutils.fci.addons.trans_rdm1_ao`, contracted as
            ``Tr(O dm)``.
        mo_coeff: orbitals to transform ``op`` with; ``None`` selects the AO
            route.
        mo_bra: bra-side orbitals, when they differ from the ket's.
        hermi: +1 for a Hermitian operator (the result is real), -1 for an
            anti-Hermitian one (the result is imaginary -- CFOUR's ``SOLZ``
            stores ``i*L_z``).  The forbidden component is checked, which is
            a free test that operator and density really are in the same
            basis.
        tol: relative size above which that forbidden component is an error
            rather than round-off.  ``None`` disables the check.
        complex_out: return the raw complex value.  Right for a transition
            density between different states, where the value is genuinely
            complex and only its modulus means anything.

    Returns:
        scalar, or ``(comp,)``.
    '''
    op = numpy.asarray(op)
    dm = numpy.asarray(dm)
    if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
        raise ValueError('dm must be a square matrix, got %r' % (dm.shape,))

    if mo_coeff is None:
        if op.shape[-2:] != dm.shape:
            raise ValueError(
                'operator is (..., %d, %d) but dm is (%d, %d). If the '
                'operator is in the AO basis and dm in the MO basis, pass '
                'mo_coeff=.' % (op.shape[-2], op.shape[-1], *dm.shape))
        val = numpy.einsum('...pq,qp->...', op, dm, optimize=True)
    else:
        op = to_mo(op, mo_coeff, mo_bra=mo_bra)
        if op.shape[-2:] != dm.shape:
            raise ValueError(
                'operator spans %d orbitals but dm spans %d; they must cover '
                'the same MO space' % (op.shape[-1], dm.shape[0]))
        val = numpy.einsum('...pq,pq->...', op, dm, optimize=True)

    if complex_out:
        return val

    keep, drop = (val.real, val.imag) if hermi >= 0 else (val.imag, -val.real)
    if tol is not None:
        scale = max(float(abs(numpy.asarray(keep)).max()), 1.0)
        bad = float(abs(numpy.asarray(drop)).max())
        if bad > tol * scale:
            raise ValueError(
                '%s: the component forbidden by the operator symmetry is '
                '%.3e (relative %.3e), not round-off. A %s operator must '
                'give a %s value; check that operator and density are in the '
                'same basis and that hermi=%+d is right for this operator.'
                % (label, bad, bad / scale,
                   'Hermitian' if hermi >= 0 else 'anti-Hermitian',
                   'real' if hermi >= 0 else 'imaginary', hermi))
    return keep
