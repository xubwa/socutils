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
           'level_block', 'pseudo_spin_tensor', 'pseudo_spin_frame', 'spin_invariant',
           'g_tensor', 'g_principal', 'hyperfine_tensor', 'hyperfine_principal',
           'hyperfine_along_g', 'g_tensor_ff', 'hyperfine_tensor_ff', 'hyperfine_step']


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
    t = numpy.einsum('aij,bji->ab', block, _PAULI)
    # Tr(M sigma) is real for a Hermitian block; a finite imaginary part means
    # the block is not Hermitian, i.e. operator and densities disagree
    scale = max(abs(t.real).max(), 1.0)
    if abs(t.imag).max() > 1e-8 * scale:
        raise ValueError('Tr(M_a sigma_b) has an imaginary part %.3e (relative %.3e); '
                         'the block is not Hermitian'
                         % (abs(t.imag).max(), abs(t.imag).max() / scale))
    return t.real


def spin_invariant(block):
    '''``g g^T`` of a multiplet, from the block alone.

    Inside a manifold of ``d = 2S + 1`` states a time-odd operator is
    exactly linear in the pseudo-spin, ``M_a = (1/2) sum_b g_ab S_b`` (for
    ``S <= 1`` no higher odd rank exists, and for larger ``S`` this is the
    rank-1 part), so ``Tr(M_a M_b) = (K/4) (g g^T)_ab`` with
    ``K = S(S+1)(2S+1)/3``.  The trace of a product of two blocks is
    invariant under *any* unitary mixing of the manifold, which is what the
    CI solver leaves arbitrary -- so this works for a Kramers doublet and a
    spin triplet alike, and needs no identification of which state is which
    ``M_S``.  What it cannot give is a sign: ``g g^T`` is even in ``g``.
    '''
    block = numpy.asarray(block)
    d = block.shape[-1]
    if block.shape[-2] != d or d < 2:
        raise ValueError('expected a (ncomp, d, d) block with d >= 2, got %r' % (block.shape,))
    # The identity part of the block shifts the whole manifold instead of
    # splitting it, so it is not a Zeeman term: project it out.  It vanishes
    # exactly for a time-odd operator on a Kramers manifold and is otherwise
    # the size of the solver's degeneracy error.
    tr = numpy.einsum('aii->a', block) / d
    block = block - tr[:, None, None] * numpy.eye(d)
    S = (d - 1) / 2.0
    K = S * (S + 1) * (2 * S + 1) / 3
    return (4 / K) * numpy.einsum('aij,bji->ab', block, block).real


def _sym_sqrt(g2):
    '''Symmetric positive semidefinite square root.'''
    w, v = numpy.linalg.eigh(0.5 * (g2 + g2.T))
    return (v * numpy.sqrt(numpy.clip(w, 0, None))) @ v.T


def pseudo_spin_frame(t_g):
    '''Fix the pseudo-spin frame on the g tensor of a *doublet*.

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

    ``det R = sign(det g)``: a change of the doublet basis is an SU(2)
    rotation of the sigma index, which is proper, so the sign of ``det T_g``
    is a physical invariant of the level; forcing ``P >= 0`` pushes it into
    ``R``.  ``det R = -1`` therefore means the g tensor has an odd number of
    negative principal values -- real physics, not an error, and worth
    knowing because the sign of a hyperfine tensor is reported relative to
    this frame.  :func:`g_tensor` logs it.

    Only a doublet has a frame this simple: for ``d > 2`` a general unitary
    mixing of the manifold is not a rotation of the spin matrices, so the
    frame is not recoverable from one operator and only
    :func:`spin_invariant` is used there.
    '''
    from scipy.linalg import polar
    # scipy always returns (unitary, positive); with side='left', t_g = P @ R,
    # so the rotation is the FIRST element -- unpacking these the wrong way
    # round leaves g's principal values intact but scrambles the frame that
    # hyperfine_tensor rotates into, which is why both are checked here.
    R, P = polar(t_g, side='left')
    if abs(P @ R - t_g).max() > 1e-8 * max(abs(t_g).max(), 1.0):
        raise ValueError('polar decomposition did not reproduce T_g; scipy returned '
                         'the factors in an unexpected order')
    return P, R


def _principal(t):
    '''Principal values (descending |v|) and axes of a *symmetric* 3x3 tensor.
    The symmetrisation only removes round-off: g is symmetric by the
    convention of :func:`pseudo_spin_frame`.  Not for a hyperfine tensor,
    whose two indices live in different spaces -- see :func:`hyperfine_principal`.'''
    w, v = numpy.linalg.eigh(0.5 * (t + t.T))
    order = numpy.argsort(-abs(w))
    return w[order], v[:, order]


def g_principal(g):
    '''Principal g values and axes of one or more g tensors from :func:`g_tensor`.

    Returns ``(values, axes)``: ``values[..., 3]`` descending, ``axes[..., 3, 3]``
    with the principal directions as *columns* in the laboratory frame
    (``g = axes @ diag(values) @ axes.T``).  Levels reported as ``nan``
    (non-degenerate) stay ``nan``.
    '''
    g = numpy.asarray(g, dtype=float)
    out_w = numpy.full(g.shape[:-1], numpy.nan)
    out_v = numpy.full(g.shape, numpy.nan)
    for idx in numpy.ndindex(g.shape[:-2]):
        if numpy.isnan(g[idx]).any():
            continue
        out_w[idx], out_v[idx] = _principal(g[idx])
    return out_w, out_v


def hyperfine_principal(a):
    '''Principal values of hyperfine tensors from :func:`hyperfine_tensor`.

    ``A_ab`` couples a nuclear-spin direction ``a`` (laboratory frame) to a
    pseudo-spin direction ``b``; a rotation of the laboratory acts on ``a``
    and a change of the doublet basis on ``b``, so ``A`` is not symmetric
    and neither its eigenvalues nor those of its symmetric part are
    invariants.  What is invariant under the basis change is ``A A^T``: its
    eigenvalues are the squared principal values and its eigenvectors the
    nuclear-side principal axes.  Numerically this is the SVD
    ``A = U diag(values) V^T``.

    Returns ``(values, nuclear_axes, spin_axes)``: ``values[..., 3]``
    descending and non-negative (the sign of ``det A`` relative to the g
    frame is not folded in -- take ``numpy.linalg.det`` of the tensor
    itself); ``nuclear_axes[..., 3, 3]`` the columns of ``U`` (laboratory
    frame) and ``spin_axes[..., 3, 3]`` the columns of ``V`` (pseudo-spin
    frame fixed by g).  ``nan`` levels stay ``nan``.
    '''
    a = numpy.asarray(a, dtype=float)
    out_w = numpy.full(a.shape[:-1], numpy.nan)
    out_u = numpy.full(a.shape, numpy.nan)
    out_v = numpy.full(a.shape, numpy.nan)
    for idx in numpy.ndindex(a.shape[:-2]):
        if numpy.isnan(a[idx]).any():
            continue
        u, w, vt = numpy.linalg.svd(a[idx])
        out_w[idx], out_u[idx], out_v[idx] = w, u, vt.T
    return out_w, out_u, out_v


def hyperfine_along_g(a, g):
    '''Hyperfine coupling seen with the field along each principal axis of g.

    In the high-field limit the pseudo-spin quantises along ``g^T n`` for a
    field along ``n``, so the nucleus sees the field ``A g^T n / |g^T n|``
    and the observed coupling is its norm.  Along a principal axis ``e`` of
    the (symmetric) g this reduces to ``K = |A e|`` -- the norm of the column
    of ``A`` in the g frame, which equals the diagonal element only when
    ``A e`` is parallel to ``e``, i.e. when A and g share that axis.

    ``a`` is ``(..., natm, 3, 3)`` and ``g`` the matching ``(..., 3, 3)`` (one
    g per level, shared by the nuclei of that level).  Returns ``(k, axes)``:
    ``k[..., natm, 3]`` ordered like the principal g values (descending) and
    ``axes[..., 3, 3]`` the g principal axes used.
    '''
    a = numpy.asarray(a, dtype=float)
    g = numpy.asarray(g, dtype=float)
    _, axes = g_principal(g)
    # (..., natm, 3, 3) @ (..., 1, 3, 3) -> columns are A e_X, A e_Y, A e_Z
    cols = a @ axes[..., None, :, :]
    k = numpy.linalg.norm(cols, axis=-2)
    return k, axes


def _log_tensor_table(log, title, metas, tensors, labels, principal=_principal):
    '''``principal(t)`` must return the three principal values first; the
    symmetric-tensor default is right for g, the hyperfine table passes the
    singular values.'''
    log.note('%s (a.u.)', title)
    log.note('%6s %18s %4s %8s ' % ('level', 'energy', 'deg', 'roots')
             + ' '.join('%14s' % l for l in labels))
    for meta, t in zip(metas, tensors):
        w = principal(t)[0] if not numpy.isnan(t).any() else numpy.full(3, numpy.nan)
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

    ``g`` is the symmetric positive semidefinite square root of
    :func:`spin_invariant`'s ``g g^T``, which for a doublet is the ``2 P``
    of the frame convention in :func:`pseudo_spin_frame` (the factor 2 is
    ``1/mu_B`` in atomic units, so a free electron gives ``g_e``) and for a
    larger multiplet -- a spin triplet, say -- is the same construction
    without a frame.  The sign of ``g`` is not claimed.  A non-degenerate
    level gets a NaN tensor and a log line.

    Returns:
        ``(3, 3)`` for a single level, ``(nlevel, 3, 3)`` otherwise;
        symmetric, principal values non-negative.
    '''
    _check_same_system(store, wfn)
    log = logger.new_logger(wfn, verbose)
    blocks, metas = _g_blocks(wfn, store, balance, **kwargs)
    out = []
    for M, meta in zip(blocks, metas):
        if meta['degeneracy'] < 2:
            log.info('level %d is not degenerate: no pseudo-spin g tensor', meta['level'])
            out.append(numpy.full((3, 3), numpy.nan))
            continue
        if meta['degeneracy'] == 2:
            # the doublet has a frame, and its polar decomposition goes
            # through an SVD -- no squaring, so a vanishing g_perp comes out
            # at 1e-11 rather than the 1e-8 of a square root of g g^T
            P, R = pseudo_spin_frame(pseudo_spin_tensor(M))
            log.info('level %d: sign(det g) = %+d', meta['level'],
                     1 if numpy.linalg.det(R) > 0 else -1)
            out.append(2 * P)
        else:
            out.append(_sym_sqrt(spin_invariant(M)))
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
    if balance != g_balance:
        log.info('hyperfine from the %s files, pseudo-spin frame from the %s g tensor: '
                 'the two balance schemes are different treatments of the same picture '
                 'change, so the frame is consistent only to the difference between them '
                 '(measured 5e-4 on the Sm 4f expectation values)', balance, g_balance)
    out = []
    for M, Mg, meta in zip(blocks, gblocks, metas):
        if meta['degeneracy'] != 2:
            log.info('level %d has degeneracy %d: the pseudo-spin frame is fixed by g '
                     'only for a doublet, so no hyperfine tensor is reported',
                     meta['level'], meta['degeneracy'])
            out.append(numpy.full((natm, 3, 3), numpy.nan))
            continue
        _, R = pseudo_spin_frame(pseudo_spin_tensor(Mg))
        out.append(numpy.array([pseudo_spin_tensor(M[3 * a:3 * a + 3]) @ R.T for a in range(natm)]))
    out = numpy.array(out)
    for a in range(natm):
        _log_tensor_table(log, 'hyperfine tensor (%s) at %s%d, frame from g (%s); '
                          'A1-A3 are the singular values (A is not symmetric)'
                          % (balance, store.mol.atom_symbol(a), a, g_balance), metas, out[:, a],
                          ('A1', 'A2', 'A3') + tuple('A%s%s' % (i, j) for i in 'xyz' for j in 'xyz'),
                          principal=hyperfine_principal)
    return _squeeze_roots(out, metas)


# ---------------------------------------------------------------------------
# Finite field
# ---------------------------------------------------------------------------

_E3 = numpy.eye(3)
_PAIRS = ((0, 1), (0, 2), (1, 2))
_SIX = [_E3[a] for a in range(3)] + [(_E3[a] + _E3[b]) / 2**.5 for a, b in _PAIRS]
_CHECK = [numpy.array([sa, sb, 1.0]) / 3**.5 for sa in (1, -1) for sb in (1, -1)]


def _one_body_ci(mc, ops, spec):
    '''CI matrices of the one-electron operator ``ops`` (``(ncomp, n2c, n2c)``
    in the store's AO basis; the taxonomy sign is applied) over the active
    space of ``mc``: the operator put into ``make_hmat`` with zero
    two-electron integrals.'''
    from socutils.prop.densities import _cas_frame
    from socutils.prop.contract import to_mo
    _, mo, ncore, ncas, nelecas, _ = _cas_frame(mc)
    omo = spec['sign'] * to_mo(numpy.asarray(ops), mo[:, ncore:ncore + ncas])
    zero = numpy.zeros((ncas,) * 4, dtype=numpy.result_type(omo, numpy.complex128))
    return numpy.array([mc.fcisolver.make_hmat(o, zero, ncas, nelecas) for o in omo])


class _FiniteField:
    '''The active-space CI matrix of a CASCI/CASSCF rebuilt from its orbitals,
    the levels of its roots, and the splittings of every adjacent pair of
    roots under a one-electron perturbation.

    The rebuilt matrix must reproduce the wavefunction's own energies (to
    1e-8 Hartree), which is the check that orbitals, integrals and CI are
    from one calculation.  Every perturbation ``V`` is applied as ``+V`` and
    ``-V`` and the squared splittings averaged: a residual time-reversal
    breaking of the zero-field states acts like a small internal field and
    would otherwise enter linearly.
    '''

    def __init__(self, wfn, degen_tol, log):
        from scipy.linalg import eigh
        from socutils.prop.densities import _roots, _degeneracy, _cas_frame
        if not (hasattr(wfn, 'fcisolver') and hasattr(wfn, 'ncas')):
            raise TypeError('the finite-field route needs a CASCI/CASSCF, got %s'
                            % type(wfn).__name__)
        if not hasattr(wfn.fcisolver, 'make_hmat'):
            raise TypeError('%s cannot build the CI matrix explicitly (no make_hmat); '
                            'the finite-field route needs it' % type(wfn.fcisolver).__name__)
        roots = _roots(wfn)
        e = numpy.array([ei for _, _, ei in roots])
        degen = _degeneracy(e, degen_tol)
        self.metas, pos = [], 0
        while pos < len(roots):
            level, d = degen[pos]
            self.metas.append(dict(level=level, degeneracy=d, roots=tuple(range(pos, pos + d)),
                                   energy=float(e[pos:pos + d].mean())))
            pos += d
        _, mo, ncore, ncas, nelecas, _ = _cas_frame(wfn)
        h1, ecore = wfn.get_h1eff(mo)
        eri = numpy.asarray(wfn.get_h2eff(mo)).reshape((ncas,) * 4)
        self.H0 = wfn.fcisolver.make_hmat(h1, eri, ncas, nelecas)
        self.nroots = len(roots)
        w0 = eigh(self.H0, eigvals_only=True, subset_by_index=[0, self.nroots - 1])
        dev = abs(w0 + float(numpy.real(ecore)) - e).max()
        if dev > 1e-8:
            raise ValueError('the rebuilt active-space Hamiltonian does not reproduce the '
                             "wavefunction's energies (max deviation %.3e Hartree): orbitals "
                             'and CI are not from one calculation' % dev)
        log.debug('finite field: CI space %d, %d roots reproduced to %.1e Hartree',
                  self.H0.shape[0], self.nroots, dev)

    def delta2(self, V):
        '''Squared splitting of every adjacent root pair under ``+V`` and ``-V``, averaged.'''
        from scipy.linalg import eigh
        out = 0
        for s in (1, -1):
            w = eigh(self.H0 + s * V, eigvals_only=True, subset_by_index=[0, self.nroots - 1])
            out = out + 0.5 * numpy.diff(w)**2
        return out

    def quadratic_form(self, M, scale):
        '''``T`` of ``Delta^2 = scale^2 n^T T n`` for ``V = scale n.M``, per
        adjacent root pair, from the six directions of the module; four more
        directions give the relative residual of that quadratic form.'''
        dirs = numpy.array(_SIX + _CHECK)
        q = numpy.array([self.delta2(scale * numpy.einsum('a,aij->ij', n, M)) for n in dirs])
        q /= scale**2
        T = numpy.zeros((q.shape[1], 3, 3))
        for a in range(3):
            T[:, a, a] = q[a]
        for k, (a, b) in enumerate(_PAIRS):
            T[:, a, b] = T[:, b, a] = q[3 + k] - 0.5 * (q[a] + q[b])
        model = numpy.einsum('cn,pnm,cm->cp', dirs[6:], T, dirs[6:])
        resid = abs(q[6:] - model).max(axis=0) / numpy.maximum(numpy.einsum('pii->p', T), 1e-300)
        return T, resid

    def doublets(self):
        '''``(meta, first_root)`` for every level; first_root is ``None`` unless a doublet.'''
        return [(m, m['roots'][0] if m['degeneracy'] == 2 else None) for m in self.metas]


def _moment_ci(wfn, store, balance):
    names = _G_FILES[balance]
    return _one_body_ci(wfn, [store.op(n) for n in names], store.meta(names[0]))


def g_tensor_ff(wfn, store, field=0.1, balance='UT', verbose=None, degen_tol=1e-6):
    '''g tensor of every Kramers doublet of a CASCI/CASSCF by finite field.

    The active-space Hamiltonian is rebuilt from ``wfn``'s orbitals, the
    Zeeman term ``-B.mu`` (the same magnetic-moment operator as
    :func:`g_tensor`) is added to its one-electron part, and the CI problem is
    solved again in the field.  The splitting of a doublet along a unit
    direction ``n`` obeys ``Delta(n) = (B/2) sqrt(n^T G n)`` in atomic units
    (``mu_B = 1/2``), so the six directions ``x, y, z`` and
    ``(e_a + e_b)/sqrt2`` give the six elements of ``G = g g^T``.  No
    pseudo-spin is defined: only the energies enter, so -- like an EPR
    rotation -- this yields ``G`` and nothing about the sign of ``g``.

    Every direction is solved at ``+B`` and ``-B`` and ``Delta^2`` averaged
    (see :class:`_FiniteField`).  The leading remaining error is the
    second-order Zeeman mixing with other levels, of relative size
    ``(mu_B B / gap)^2``.

    Spin-orbit coupling is whatever the wavefunction's Hamiltonian contains
    (for X2CAMF it is already variational); only the Zeeman term is added.
    Orbitals are frozen, so the CI response to the field is complete within
    the active space and the orbital response is not included.

    Args:
        field: field strength in tesla (default 0.1 T).

    Returns:
        ``(3, 3)`` for a single level, ``(nlevel, 3, 3)`` otherwise, in the
        layout of :func:`g_tensor`: the symmetric positive semidefinite
        square root of ``G``.  Levels that are not doublets are NaN.
    '''
    from pyscf.data import nist
    _check_same_system(store, wfn)
    log = logger.new_logger(wfn, verbose)
    B = field / nist.AU2TESLA
    ff = _FiniteField(wfn, degen_tol, log)
    T, resid = ff.quadratic_form(-_moment_ci(wfn, store, balance), B)
    out = []
    for meta, i in ff.doublets():
        if i is None:
            log.info('level %d has degeneracy %d: the finite-field route here is for '
                     'Kramers doublets only', meta['level'], meta['degeneracy'])
            out.append(numpy.full((3, 3), numpy.nan))
            continue
        G = 4 * T[i]
        gaps = [abs(meta['energy'] - m['energy']) for m in ff.metas if m is not meta]
        split = 0.5 * B * numpy.sqrt(max(numpy.linalg.eigvalsh(G).max(), 0))
        log.info('level %d: finite field %.3g T, quadratic-form residual %.1e, '
                 'splitting / nearest gap %.1e', meta['level'], field, resid[i],
                 split / min(gaps) if gaps else 0.0)
        out.append(_sym_sqrt(G))
    out = numpy.array(out)
    _log_tensor_table(log, 'g tensor, finite field (%s, %.3g T)' % (balance, field), ff.metas, out,
                      ('g1', 'g2', 'g3') + tuple('g%s%s' % (a, b) for a in 'xyz' for b in 'xyz'))
    return _squeeze_roots(out, ff.metas)


def _sqrt_principal(t):
    '''Square roots of the eigenvalues of a PSD tensor, descending: |A| for A A^T.'''
    w, v = numpy.linalg.eigh(0.5 * (t + t.T))
    order = numpy.argsort(-w)
    return numpy.sqrt(numpy.clip(w[order], 0, None)), v[:, order]


def hyperfine_tensor_ff(wfn, store, atoms=None, field=0.1, moment=None, balance='KB',
                        g_balance='UT', verbose=None, degen_tol=1e-6):
    '''Hyperfine invariants of every Kramers doublet by finite field.

    Besides the Zeeman term ``-B.mu`` a classical nuclear magnetic moment
    ``m`` is coupled to the electrons through the store's hyperfine
    operator ``h`` (``H += m.h``).  For a doublet the squared splitting is

        Delta^2 = (B/2)^2 n^T G n + m^2 u^T (A A^T) u - B m n^T (g A^T) u

    for a field along ``n`` and a moment along ``u``, with three tensors that
    are all independent of any pseudo-spin frame:

    * ``G = g g^T`` -- the field alone, as in :func:`g_tensor_ff`;
    * ``A A^T`` -- the moment alone: its eigenvalues are the squared
      principal hyperfine couplings, its eigenvectors the nuclear-side axes;
    * ``g A^T`` -- field along ``e_a`` and moment along ``e_b`` together;
      the orientation *and sign* of the hyperfine coupling relative to g.

    ``A`` here is in the convention of :func:`hyperfine_tensor` (``H =
    sum A_ab I_a S_b``, bare a.u. per unit nuclear moment, no isotope
    factor), and the mapping route gives the same ``A A^T`` and ``g A^T``
    directly.  The coupling the nucleus sees with the field along ``n`` is
    ``|(g A^T)^T n| / sqrt(n^T G n)`` -- see :func:`hyperfine_step`.

    Every combination is solved at ``+`` and ``-`` (both fields reversed
    together) and ``Delta^2`` averaged.  The moment is a parameter, not a
    physical size: by default it is chosen so that the hyperfine splitting
    matches the Zeeman one at ``field``, from a first pass at ``m = B``.

    Args:
        atoms: indices of the nuclei (default all).
        field: Zeeman field in tesla (default 0.1 T).
        moment: nuclear-moment parameter in a.u. (default: see above).

    Returns:
        ``(G, AAt, gAt)``: ``G`` of shape ``(nlevel, 3, 3)``, ``AAt`` and ``gAt`` of
        shape ``(nlevel, natoms, 3, 3)`` (the level axis squeezed for a single
        level).  Levels that are not doublets are NaN.
    '''
    from pyscf.data import nist
    _check_same_system(store, wfn)
    log = logger.new_logger(wfn, verbose)
    B = field / nist.AU2TESLA
    ff = _FiniteField(wfn, degen_tol, log)
    Mmu = -_moment_ci(wfn, store, g_balance)
    Tg, _ = ff.quadratic_form(Mmu, B)
    Gp = 4 * Tg                                                  # per adjacent root pair
    lev = ff.doublets()
    first = [i for _, i in lev if i is not None]
    if not first:
        raise ValueError('no Kramers doublet among the roots of %s' % type(wfn).__name__)
    name = _A_FILES[balance]
    spec = store.meta(name)
    atoms = list(range(store.mol.natm)) if atoms is None else list(atoms)

    AAts, gAts = [], []
    for atom in atoms:
        Mh = _one_body_ci(wfn, [store.op(name, atom=atom, comp=k) for k in range(3)], spec)
        m = moment
        if m is None:
            AAt_trial, _ = ff.quadratic_form(Mh, B)
            i0 = first[0]
            kmax = numpy.linalg.eigvalsh(AAt_trial[i0]).max()
            gmax = numpy.linalg.eigvalsh(Gp[i0]).max()
            m = 0.5 * B * numpy.sqrt(gmax / kmax) if kmax > 1e-30 * max(gmax, 1) else B
        AAt, resid = ff.quadratic_form(Mh, m)
        gAt = numpy.zeros_like(AAt)
        for a in range(3):
            for b in range(3):
                d2 = ff.delta2(B * Mmu[a] + m * Mh[b])
                gAt[:, a, b] = ((0.5 * B)**2 * Gp[:, a, a] + m**2 * AAt[:, b, b] - d2) / (B * m)
        for meta, i in lev:
            if i is not None:
                log.info('level %d, atom %d: moment %.3e a.u., A A^T quadratic-form residual %.1e',
                         meta['level'], atom, m, resid[i])
        AAts.append(AAt)
        gAts.append(gAt)
    AAts, gAts = numpy.array(AAts), numpy.array(gAts)                 # (natoms, npairs, 3, 3)

    G_out, AAt_out, gAt_out = [], [], []
    nan = numpy.full((3, 3), numpy.nan)
    for meta, i in lev:
        if i is None:
            G_out.append(nan)
            AAt_out.append(numpy.full((len(atoms), 3, 3), numpy.nan))
            gAt_out.append(numpy.full((len(atoms), 3, 3), numpy.nan))
            continue
        G_out.append(Gp[i])
        AAt_out.append(AAts[:, i])
        gAt_out.append(gAts[:, i])
    G_out, AAt_out, gAt_out = numpy.array(G_out), numpy.array(AAt_out), numpy.array(gAt_out)
    for k, atom in enumerate(atoms):
        _log_tensor_table(log, 'hyperfine A A^T, finite field (%s) at %s%d; first three are |A|'
                          % (balance, store.mol.atom_symbol(atom), atom), ff.metas, AAt_out[:, k],
                          ('|A|1', '|A|2', '|A|3') + tuple('AA%s%s' % (i, j) for i in 'xyz' for j in 'xyz'),
                          principal=_sqrt_principal)
        _log_tensor_table(log, 'hyperfine g A^T, finite field (%s) at %s%d; field index first'
                          % (balance, store.mol.atom_symbol(atom), atom), ff.metas, gAt_out[:, k],
                          ('s1', 's2', 's3') + tuple('gA%s%s' % (i, j) for i in 'xyz' for j in 'xyz'),
                          principal=lambda t: (numpy.linalg.svd(t, compute_uv=False),))
    return (_squeeze_roots(G_out, ff.metas), _squeeze_roots(AAt_out, ff.metas),
            _squeeze_roots(gAt_out, ff.metas))


def hyperfine_step(G, gAt, n, scale=1.0):
    '''Field of the first hyperfine level crossing inside a Kramers doublet.

    With the field along the unit vector ``n`` the doublet splits by
    ``mu_B B g_eff``, ``g_eff = sqrt(n^T G n)``, and the nucleus sees the
    coupling ``A_eff = |scale gAt^T n| / g_eff``.  To first order the nuclear
    sublevels of the two branches cross at ``B_k = k A_eff / (2 mu_B
    g_eff)``, ``k = 1 .. 2I``; with ``mu_B = 1/2``

        B_1 = |scale gAt^T n| / (n^T G n)       (atomic units of field)

    and ``B_k = k B_1``.  ``G`` and ``gAt`` are from :func:`hyperfine_tensor_ff`
    (or ``g g^T`` and ``g A^T`` from the mapping); ``scale`` turns the bare
    per-unit-moment coupling into energy per unit nuclear spin (Hartree) --
    the isotope factor, which this module does not know.  Quadrupole,
    nuclear-Zeeman and non-secular hyperfine terms split each ``B_k`` into a
    narrow cluster and are not in this first-order value.  Multiply by
    ``pyscf.data.nist.AU2TESLA`` for tesla.
    '''
    n = numpy.asarray(n, dtype=float)
    n = n / numpy.linalg.norm(n)
    return abs(scale) * numpy.linalg.norm(numpy.asarray(gAt).T @ n) / (n @ numpy.asarray(G) @ n)
