#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''Observables: an operator store meets a wavefunction.

Each driver here is the same six lines with a different operator name:
pull the operator(s) from the store, loop over every density the
wavefunction can give (:func:`socutils.prop.densities.densities`), contract
(:func:`socutils.prop.contract.contract_dm`), apply the sign and nuclear
term the store's taxonomy records for that file, and return one bare-a.u.
array with a leading root axis that is squeezed away when there was only one
density.  Nothing here stores a wavefunction, converts units, or knows an
isotope.

The store is passed explicitly.  Today that is a
:class:`socutils.prop.cfour.CfourPropStore`; anything with the same
``op(name, atom=, comp=)`` / ``meta(name)`` / ``mol`` surface will do.  A
missing file is the store's error to raise, not something to work around.

Frame: the operators are whatever the store holds, so the nuclear term is
built in the *store's* ``mol`` -- if CFOUR reoriented the molecule, that is
the frame the numbers come out in.

Every root is reported.  For a degenerate level the per-root values of a
non-scalar operator depend on which basis the solver happened to pick inside
the level; the log table prints the level and its degeneracy next to each
row so that is visible, and the d x d treatment of such levels is a later
increment.
'''

import numpy
from pyscf.lib import logger

from socutils.prop.densities import densities
from socutils.prop.contract import contract_dm

__all__ = ['dipole', 'nuclear_dipole', 'efg', 'efg_principal', 'nuclear_density']


def _check_same_system(store, wfn):
    a, b = store.mol, wfn.mol
    if a.nao_2c() != b.nao_2c() or a.natm != b.natm or \
       not numpy.array_equal(a.atom_charges(), b.atom_charges()):
        raise ValueError(
            "the operator store's molecule (%d atoms, n2c=%d) is not the "
            "wavefunction's (%d atoms, n2c=%d)"
            % (a.natm, a.nao_2c(), b.natm, b.nao_2c()))


def _collect(wfn, ops, spec, label, nuclear=0.0, **kwargs):
    '''``sign * <ops> + nuclear`` for every density of ``wfn``; returns the
    stacked values and the metas, in the adapter's order.'''
    vals, metas = [], []
    for name, dm, mo, meta in densities(wfn, **kwargs):
        el = contract_dm(ops, dm, mo_coeff=mo, hermi=spec['hermi'],
                         label='%s (%s)' % (label, name))
        vals.append(spec['sign'] * el + nuclear)
        metas.append(meta)
    return numpy.asarray(vals), metas


def _log_table(log, title, metas, values, columns):
    '''One row per density: root, energy, level/degeneracy, then ``columns``.'''
    log.note('%s (a.u.)', title)
    head = '%6s %18s %6s ' % ('root', 'energy', 'level')
    log.note(head + ' '.join('%14s' % c for c in columns))
    for meta, row in zip(metas, values):
        e = meta.get('energy')
        lvl = meta.get('level')
        d = meta.get('degeneracy', 1)
        log.note('%6s %18s %6s ' % (
            meta.get('root', '-'),
            '%.10f' % e if e is not None else '-',
            ('%d(x%d)' % (lvl, d)) if lvl is not None else '-')
            + ' '.join('%14.8f' % v for v in numpy.atleast_1d(row)))


def _squeeze_roots(values, metas):
    return values[0] if len(metas) == 1 else values


def nuclear_dipole(mol):
    '''``sum_A Z_A R_A`` about the coordinate origin, in a.u.'''
    return numpy.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())


def dipole(wfn, store, verbose=None, **kwargs):
    '''Electric dipole moment ``-<r> + sum_A Z_A R_A`` for every density
    ``wfn`` can give, from the store's ``SOX2CDPX/Y/Z``.

    Returns:
        ``(3,)`` for a single density, ``(nroots, 3)`` otherwise; bare a.u.,
        Cartesian x, y, z.
    '''
    _check_same_system(store, wfn)
    log = logger.new_logger(wfn, verbose)
    names = ('SOX2CDPX', 'SOX2CDPY', 'SOX2CDPZ')
    spec = store.meta(names[0])
    ops = numpy.array([store.op(n) for n in names])
    nuc = nuclear_dipole(store.mol) if spec['nuclear'] == 'dipole' else 0.0

    vals, metas = _collect(wfn, ops, spec, 'dipole', nuclear=nuc, **kwargs)
    log.info('nuclear dipole (a.u.): %s', nuc)
    _log_table(log, 'Electric dipole moment', metas,
               numpy.c_[vals, numpy.linalg.norm(vals, axis=1)],
               ('x', 'y', 'z', '|mu|'))
    return _squeeze_roots(vals, metas)


# --- electric field gradient ----------------------------------------------

#: CFOUR's slot order for SOEFG / EFGNUC and where each slot sits in the 3x3
#: tensor (efgdiag.f90): zz, yy, xx, zx, zy, yx.
_EFG_SLOTS = ((2, 2), (1, 1), (0, 0), (0, 2), (1, 2), (0, 1))


def _efg_tensor(six):
    '''``(..., 6)`` in slot order -> ``(..., 3, 3)`` symmetric tensor.'''
    six = numpy.asarray(six)
    v = numpy.zeros(six.shape[:-1] + (3, 3), dtype=six.dtype)
    for k, (i, j) in enumerate(_EFG_SLOTS):
        v[..., i, j] = six[..., k]
        v[..., j, i] = six[..., k]
    return v


def efg_principal(v):
    '''Principal components of an EFG tensor, CFOUR's way: eigenvalues sorted
    by ascending ``|V|`` so the last one is ``V_zz``; returns
    ``(eigenvalues(..., 3), eta)`` with ``eta = (V_xx - V_yy) / V_zz``.'''
    v = numpy.asarray(v)
    w = numpy.linalg.eigvalsh(v)
    order = numpy.argsort(abs(w), axis=-1)
    w = numpy.take_along_axis(w, order, axis=-1)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        eta = (w[..., 0] - w[..., 1]) / w[..., 2]
    return w, eta


def efg(wfn, store, verbose=None, **kwargs):
    '''Electric field gradient at every nucleus, ``-<V_ab> + V_ab^nuc``, from
    the store's ``SOEFG`` and its ``EFGNUC``.

    Returns:
        ``(natm, 3, 3)`` for a single density, ``(nroots, natm, 3, 3)``
        otherwise; bare a.u.  Feed a tensor to :func:`efg_principal` for
        ``V_zz`` and ``eta``.
    '''
    _check_same_system(store, wfn)
    log = logger.new_logger(wfn, verbose)
    spec = store.meta('SOEFG')
    natm = store.mol.natm
    ops = numpy.array([store.op('SOEFG', atom=a, comp=k)
                       for a in range(natm) for k in range(6)])
    nuc = store.read_real('EFGNUC').ravel() if spec['nuclear'] == 'efg' else 0.0

    six, metas = _collect(wfn, ops, spec, 'EFG', nuclear=nuc, **kwargs)
    six = six.reshape(len(metas), natm, 6)
    vals = _efg_tensor(six)
    w, eta = efg_principal(vals)
    for a in range(natm):
        _log_table(log, 'EFG at %s%d' % (store.mol.atom_symbol(a), a), metas,
                   numpy.c_[six[:, a], w[:, a], eta[:, a]],
                   ('zz', 'yy', 'xx', 'zx', 'zy', 'yx', 'V1', 'V2', 'Vzz', 'eta'))
    return _squeeze_roots(vals, metas)


# --- densities at the nucleus ---------------------------------------------

#: One observable, three CFOUR formulations; the taxonomy carries each
#: file's sign and picture-change convention.
_NUCLEAR_DENSITY_FILES = {'contact': 'SODEN2C',
                          'contact_derivative': 'SODEN2CG',
                          'effective': 'SOMOSS2C'}


def nuclear_density(wfn, store, formulation='contact', verbose=None, **kwargs):
    '''Electron density at every nucleus from ``SODEN2C`` (``'contact'``),
    ``SODEN2CG`` (``'contact_derivative'``) or ``SOMOSS2C``
    (``'effective'``, the Moessbauer operator).

    Returns:
        ``(natm,)`` for a single density, ``(nroots, natm)`` otherwise; bare a.u.
    '''
    _check_same_system(store, wfn)
    log = logger.new_logger(wfn, verbose)
    name = _NUCLEAR_DENSITY_FILES[formulation]
    spec = store.meta(name)
    natm = store.mol.natm
    ops = numpy.array([store.op(name, atom=a) for a in range(natm)])

    vals, metas = _collect(wfn, ops, spec, name, **kwargs)
    _log_table(log, 'Electron density at the nuclei, %s' % formulation, metas, vals,
               tuple('%s%d' % (store.mol.atom_symbol(a), a) for a in range(natm)))
    return _squeeze_roots(vals, metas)
