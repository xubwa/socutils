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

__all__ = ['dipole', 'nuclear_dipole']


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
