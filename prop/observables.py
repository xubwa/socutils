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

__all__ = ['dipole', 'nuclear_dipole', 'efg', 'efg_principal', 'nuclear_density',
           'level_block', 'pseudo_spin_tensor', 'pseudo_spin_frame', 'g_tensor', 'hyperfine_tensor']


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


# --- time-odd operators: Kramers-level blocks and pseudo-spin tensors --------

#: Pauli matrices, the pseudo-spin-1/2 basis of a Kramers doublet block.
_PAULI = numpy.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])

_G_FILES = {'UT': ('UTM10X', 'UTM10Y', 'UTM10Z'), 'KB': ('SOM10X', 'SOM10Y', 'SOM10Z')}
_A_FILES = {'UT': 'UTM01', 'KB': 'SOM01'}


def level_block(wfn, store, ops, spec, verbose=None, label='operator', **kwargs):
    '''``M[a, i, j] = <i|O_a|j>`` over the roots of every degenerate level.

    ``ops`` is ``(ncomp, n2c, n2c)`` in the store's AO basis and ``spec`` the
    taxonomy row that goes with it (its sign is applied).  Returns
    ``(blocks, metas)`` with one ``(ncomp, d, d)`` complex array per level.
    The block is Hermitian in ``(i, j)`` for a Hermitian operator, which is
    checked -- it is the same free test of "operator and density in one
    basis" that :func:`socutils.prop.contract.contract_dm` runs.
    '''
    from socutils.prop.densities import level_densities
    from socutils.prop.contract import to_mo
    ops = numpy.asarray(ops)
    blocks, metas = [], []
    for name, gammas, mo, meta in level_densities(wfn, **kwargs):
        omo = to_mo(ops, mo)
        M = spec['sign'] * numpy.einsum('apq,ijpq->aij', omo, gammas, optimize=True)
        dev = abs(M - spec['hermi'] * M.conj().transpose(0, 2, 1)).max()
        scale = max(abs(M).max(), 1.0)
        if dev > 1e-6 * scale:
            raise ValueError('%s (%s): block is not %s (deviation %.3e); operator '
                             'and transition densities are not in one basis'
                             % (label, name, 'Hermitian' if spec['hermi'] > 0
                                else 'anti-Hermitian', dev))
        blocks.append(M)
        metas.append(meta)
    return blocks, metas


def pseudo_spin_tensor(block):
    '''``T[a, b] = Tr(M_a sigma_b)`` for a ``(3, 2, 2)`` Kramers-doublet
    block: the operator written as ``M_a = (1/2) sum_b T_ab sigma_b`` in the
    solver's basis of the doublet.  A change of that basis multiplies ``T``
    from the right by one rotation common to every operator of the level,
    so ``T T^T`` and products ``T_1 T_2^T`` are invariant.'''
    block = numpy.asarray(block)
    if block.shape[-2:] != (2, 2):
        raise ValueError('pseudo-spin 1/2 needs a doublet, got a %dx%d block' % block.shape[-2:])
    return numpy.einsum('aij,bji->ab', block, _PAULI).real


def pseudo_spin_frame(t_g):
    '''Fix the pseudo-spin frame on the g tensor.

    Convention: left polar decomposition ``T_g = P R``; ``P = sqrt(T T^T)``
    is symmetric positive semidefinite and becomes ``g/2``, ``R`` is the
    rotation that carries every other tensor of the level into the same
    frame (``T' = T R^T``).  Principal g values are therefore reported
    non-negative and the sign of a hyperfine tensor is *relative to g*; the
    overall sign of g (``det``) is not recoverable from ``G = g g^T`` and is
    not claimed.  When ``T_g`` is rank deficient (``g_perp = 0``, e.g. a
    case-(a) 2Pi_3/2 level) the rotation about the pseudo-spin axis is
    undetermined and only the parallel components of other tensors are
    pinned to g.  Returns ``(P, R)``.
    '''
    from scipy.linalg import polar
    R, P = polar(t_g, side='left')
    return P, R


def _principal(t):
    '''Principal values (descending |v|) and axes of a 3x3 tensor's symmetric part.'''
    w, v = numpy.linalg.eigh(0.5 * (t + t.T))
    order = numpy.argsort(-abs(w))
    return w[order], v[:, order]


def _log_tensor_table(log, title, metas, tensors, labels):
    log.note('%s (a.u.)', title)
    log.note('%6s %18s %4s %8s ' % ('level', 'energy', 'deg', 'roots')
             + ' '.join('%14s' % l for l in labels))
    for meta, t in zip(metas, tensors):
        w, _ = _principal(t) if not numpy.isnan(t).any() else (numpy.full(3, numpy.nan), None)
        log.note('%6d %18.10f %4d %8s ' % (meta['level'], meta['energy'], meta['degeneracy'],
                                           ','.join(map(str, meta['roots'])))
                 + ' '.join('%14.8f' % x for x in numpy.r_[w, t.ravel()]))


def _g_blocks(wfn, store, balance, **kwargs):
    names = _G_FILES[balance]
    spec = store.meta(names[0])
    ops = numpy.array([store.op(n) for n in names])
    return level_block(wfn, store, ops, spec, label='g (%s)' % balance, **kwargs)


def g_tensor(wfn, store, balance='UT', verbose=None, **kwargs):
    '''g tensor of every Kramers doublet of ``wfn`` from the store's
    magnetic-moment operators (``UTM10*`` or ``SOM10*``).

    ``g = 2 P`` with ``P`` from :func:`pseudo_spin_frame`: the factor 2 is
    ``1/mu_B`` in atomic units and makes a free electron give ``g_e``.
    Levels that are not doublets get a NaN tensor and a log line.

    Returns:
        ``(3, 3)`` for a single level, ``(nlevel, 3, 3)`` otherwise;
        symmetric, principal values non-negative (see the frame convention).
    '''
    _check_same_system(store, wfn)
    log = logger.new_logger(wfn, verbose)
    blocks, metas = _g_blocks(wfn, store, balance, **kwargs)
    out = []
    for M, meta in zip(blocks, metas):
        if meta['degeneracy'] != 2:
            log.info('level %d has degeneracy %d: no pseudo-spin-1/2 g tensor',
                     meta['level'], meta['degeneracy'])
            out.append(numpy.full((3, 3), numpy.nan))
            continue
        P, _ = pseudo_spin_frame(pseudo_spin_tensor(M))
        out.append(2 * P)
    out = numpy.array(out)
    _log_tensor_table(log, 'g tensor (%s)' % balance, metas, out,
                      ('g1', 'g2', 'g3') + tuple('g%s%s' % (a, b) for a in 'xyz' for b in 'xyz'))
    return _squeeze_roots(out, metas)


def hyperfine_tensor(wfn, store, balance='KB', g_balance='UT', verbose=None, **kwargs):
    '''Hyperfine tensor ``A`` of every Kramers doublet at every nucleus, in
    the pseudo-spin frame fixed by the g tensor of the same level.

    ``A_ab = Tr(M_a sigma_b)`` in that frame, so that the effective
    Hamiltonian is ``sum_ab A_ab I_a S~_b`` with pseudo-spin ``S~ = 1/2``;
    bare a.u. per unit nuclear magnetic moment -- no isotope factor.  The
    file's sign (CFOUR reports ``-Tr``) is applied; the sign of ``A`` is
    relative to the non-negative g of :func:`g_tensor`.

    Returns:
        ``(natm, 3, 3)`` for a single level, ``(nlevel, natm, 3, 3)`` otherwise.
    '''
    _check_same_system(store, wfn)
    log = logger.new_logger(wfn, verbose)
    natm = store.mol.natm
    name = _A_FILES[balance]
    spec = store.meta(name)
    ops = numpy.array([store.op(name, atom=a, comp=k) for a in range(natm) for k in range(3)])
    blocks, metas = level_block(wfn, store, ops, spec, label='hyperfine (%s)' % balance, **kwargs)
    gblocks, _ = _g_blocks(wfn, store, g_balance, **kwargs)
    out = []
    for M, Mg, meta in zip(blocks, gblocks, metas):
        if meta['degeneracy'] != 2:
            log.info('level %d has degeneracy %d: no pseudo-spin-1/2 hyperfine tensor',
                     meta['level'], meta['degeneracy'])
            out.append(numpy.full((natm, 3, 3), numpy.nan))
            continue
        _, R = pseudo_spin_frame(pseudo_spin_tensor(Mg))
        out.append(numpy.array([pseudo_spin_tensor(M[3 * a:3 * a + 3]) @ R.T for a in range(natm)]))
    out = numpy.array(out)
    for a in range(natm):
        _log_tensor_table(log, 'hyperfine tensor (%s) at %s%d, frame from g (%s)'
                          % (balance, store.mol.atom_symbol(a), a, g_balance), metas, out[:, a],
                          ('A1', 'A2', 'A3') + tuple('A%s%s' % (i, j) for i in 'xyz' for j in 'xyz'))
    return _squeeze_roots(out, metas)
