#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Analytic nuclear Hessian of GHF (two-component, spin-orbital basis), with an
X2C spin-orbit hcore (socutils.scf.ghf.GHF(...).x2camf() / .x2cmp()).

The structure follows pyscf.hessian.rhf -- partial Hessian at fixed density +
first-order Fock matrices + coupled-perturbed GHF + response contraction --
with the two-electron terms written for complex, spin-off-diagonal density
matrices:

    E_J = 1/2 sum g(ab|cd) rho_ba rho_dc,        rho = Re(D_aa + D_bb)
    E_K = -1/2 sum_st sum g(ab|cd) D^st_bc D^ts_da
        = -1/2 sum_st [K(Re D^st, Re D^ts) - K(Im D^st, Im D^ts)]

(the imaginary part of the total density is antisymmetric and drops out of
E_J; the imaginary cross terms of E_K cancel between the st and ts blocks).
Each piece is a bilinear form sum g(ab|cd) A[..] B[..] of *real*, not
necessarily symmetric matrices, whose first and second nuclear derivatives
are built below term by term, one term per derivative position, with the
symmetry-free s1 kernels of libcvhf.

X2C hcore: the second derivative is a central finite difference of the
analytic first derivative (socutils.somf.x2c_grad); everything else is
analytic.  The atomic mean-field spin-orbit term of X2CAMF is a sum of
one-centre blocks that move rigidly with their atom and contributes neither
to the gradient nor to the Hessian.
'''

import numpy
from pyscf import lib, gto
from pyscf.lib import logger
from pyscf.scf import _vhf, cphf
from pyscf.hessian import rhf as rhf_hess
import socutils.grad.ghf_grad  # noqa: F401  (GHF.Gradients, X2C hcore derivative)

# electron pairs of the labels of g(ab|cd)
_PARTNER = {'a': 'b', 'b': 'a', 'c': 'd', 'd': 'c'}
_BRA = ('a', 'b')


def _same_electron(p, q):
    return (p in _BRA) == (q in _BRA)


def _labels_map(p, q):
    '''Relabel g(ab|cd) (real AOs: a<->b, c<->d, (ab)<->(cd) symmetric) so that
    the derivative positions land where the libcint integral carries them.'''
    if q is None:                                   # (nabla i j|k l)
        others = [x for x in 'abcd' if x not in (p, _PARTNER[p])]
        return 'int2e_ip1', 3, {p: 'i', _PARTNER[p]: 'j', others[0]: 'k', others[1]: 'l'}
    if p == q:                                      # (nabla nabla i j|k l)
        others = [x for x in 'abcd' if x not in (p, _PARTNER[p])]
        return 'int2e_ipip1', 9, {p: 'i', _PARTNER[p]: 'j', others[0]: 'k', others[1]: 'l'}
    if _same_electron(p, q):                        # (nabla i nabla j|k l)
        others = [x for x in 'abcd' if x not in (p, q)]
        return 'int2e_ipvip1', 9, {p: 'i', q: 'j', others[0]: 'k', others[1]: 'l'}
    # (nabla i j|nabla k l)
    return 'int2e_ip1ip2', 9, {p: 'i', _PARTNER[p]: 'j', q: 'k', _PARTNER[q]: 'l'}


# integrals symmetric in (k, l), and the scripts libcvhf provides for aosym s2kl
_KL_SYMMETRIC = ('int2e_ip1', 'int2e_ipip1', 'int2e_ipvip1')
_S2KL_SCRIPTS = ('ji->s1kl', 'jk->s1il', 'li->s1kj', 'lk->s1ij')


def _to_s2kl(script):
    '''(script', transpose dm, transpose output) with script' one of the s2kl
    scripts: k <-> l relabelling (a symmetry of the integral), dm index order
    (A -> A^T) and output index order (transpose).'''
    dm, out = script.split('->s1')
    swap = {'k': 'l', 'l': 'k', 'i': 'i', 'j': 'j'}
    for relabel in (False, True):
        d, o = (dm, out) if not relabel else (''.join(swap[x] for x in dm), ''.join(swap[x] for x in out))
        for td in (False, True):
            for to in (False, True):
                s = '%s->s1%s' % (d[::-1] if td else d, o[::-1] if to else o)
                if s in _S2KL_SCRIPTS:
                    return s, td, to
    return None


# Density-based integral screening (lib/vhf/hess_screen.c, SOCVHFderiv_prescreen):
# q_cond[0] for the pair carrying the derivatives, q_cond[1] for the other,
# from pyscf's estimates; dm_cond from all densities of the contraction.
_Q_COND = {'int2e_ip1': ('pp', 'int2e_ip1ip2', False),
           'int2e_ipip1': ('pppp', 'int2e_ipip1ipip2', False),
           'int2e_ipvip1': ('pppp', 'int2e_ipvip1ipvip2', False),
           'int2e_ip1ip2': ('pp', 'int2e_ip1ip2', True)}       # derivatives on both pairs
_q_cond_cache = {}
_prescreen_ptr = None
SCREEN_TOL = 1e-14


def _deriv_prescreen():
    global _prescreen_ptr
    if _prescreen_ptr is None:
        import ctypes
        import _ctypes
        from socutils.lib import load_library
        try:
            lib_ = load_library('libsocvhf', 'SOCUTILS_SOCVHF_LIBRARY')
            _prescreen_ptr = ctypes.c_void_p(_ctypes.dlsym(lib_._handle, 'SOCVHFderiv_prescreen'))
        except (OSError, AttributeError):
            _prescreen_ptr = False
    return _prescreen_ptr


def _q_cond(mol, intor):
    import ctypes
    key = (id(mol), mol._bas.tobytes(), mol._env.tobytes(), intor)
    if key in _q_cond_cache:
        return _q_cond_cache[key]
    libcvhf = _vhf.libcvhf
    kind, vhf_intor, both = _Q_COND[intor]
    ao_loc = mol.ao_loc_nr()
    nbas = mol.nbas
    q = numpy.empty((2, nbas, nbas))
    fq = libcvhf.CVHFnr_int2e_pppp_q_cond if kind == 'pppp' else libcvhf.CVHFnr_int2e_pp_q_cond
    args = (ao_loc.ctypes, mol._atm.ctypes, ctypes.c_int(mol.natm),
            mol._bas.ctypes, ctypes.c_int(nbas), mol._env.ctypes)
    with mol.with_integral_screen(SCREEN_TOL**2):
        fq(getattr(libcvhf, mol._add_suffix(vhf_intor)), lib.c_null_ptr(), q[0].ctypes, *args)
        if both:
            q[1] = q[0]
        else:
            libcvhf.CVHFnr_int2e_q_cond(getattr(libcvhf, mol._add_suffix('int2e')),
                                        lib.c_null_ptr(), q[1].ctypes, *args)
    _q_cond_cache.clear()
    _q_cond_cache[key] = q
    return q


def _dm_cond(mol, mats):
    '''shell-pair maxima of |M| over the matrices, symmetrized (nbas, nbas)'''
    loc = mol.ao_loc_nr()[:-1]
    m = numpy.zeros((mol.nao_nr(),) * 2)
    for M in mats:
        numpy.maximum(m, abs(M), out=m)
    d = numpy.maximum.reduceat(numpy.maximum.reduceat(m, loc, axis=0), loc, axis=1)
    return numpy.ascontiguousarray(numpy.maximum(d, d.T))


def _deriv_vhfopt(mol, intor, mats):
    '''vhfopt screening the derivative integrals intor (canonical: derivatives
    on the first pair, and for ip1ip2 also on the second) contracted with the
    densities mats; None if lib/libsocvhf.so is not built.'''
    ptr = _deriv_prescreen()
    if not ptr or intor not in _Q_COND:
        return None
    opt = _vhf._VHFOpt(mol, intor, 'CVHFnoscreen', direct_scf_tol=SCREEN_TOL)
    opt._this.fprescreen = ptr
    opt.q_cond = _q_cond(mol, intor)
    opt.dm_cond = _dm_cond(mol, mats)
    return opt


def _contract(mol, intor, comp, scripts, dms, shls_slice, vhfopt=None):
    '''One integral pass, several density contractions.  Integrals symmetric
    in (k, l) with the same k and l shells use aosym s2kl (half the
    integrals), the scripts mapped onto those libcvhf provides.  vhfopt:
    density-based screening (_deriv_vhfopt).'''
    mapped = None
    if intor in _KL_SYMMETRIC and tuple(shls_slice[4:6]) == tuple(shls_slice[6:8]):
        mapped = [_to_s2kl(s) for s in scripts]
        if any(m is None for m in mapped):
            mapped = None
    if mapped is None:
        vs = _vhf.direct_bindm(mol._add_suffix(intor), 's1', scripts, dms, comp,
                               mol._atm, mol._bas, mol._env, shls_slice=shls_slice,
                               vhfopt=vhfopt)
        if isinstance(vs, numpy.ndarray) and len(scripts) == 1:
            vs = [vs]
        return vs
    s2 = [m[0] for m in mapped]
    d2 = [numpy.ascontiguousarray(d.T) if m[1] else d for d, m in zip(dms, mapped)]
    vs = _vhf.direct_bindm(mol._add_suffix(intor), 's2kl', s2, d2, comp,
                           mol._atm, mol._bas, mol._env, shls_slice=shls_slice,
                           vhfopt=vhfopt)
    if isinstance(vs, numpy.ndarray) and len(scripts) == 1:
        vs = [vs]
    return [v.swapaxes(-1, -2) if m[2] else v for v, m in zip(vs, mapped)]


def _nonzero(m, thresh=1e-14):
    return abs(m).max() > thresh


def _slices(mol, restrict):
    '''restrict: {letter: atom}.  Returns shls_slice and AO ranges per letter.'''
    aoslices = mol.aoslice_by_atom()
    shls = []
    aorange = {}
    for x in 'ijkl':
        if x in restrict:
            s0, s1, p0, p1 = aoslices[restrict[x]]
            shls += [s0, s1]
            aorange[x] = slice(p0, p1)
        else:
            shls += [0, mol.nbas]
            aorange[x] = slice(None)
    return tuple(shls), aorange


def deriv1_matrices(mol, terms, atmlst=None):
    '''d/dR_X of M_{out} = sum g(ab|cd) A[A_pair] for a list of terms
    (A, A_pair, out_pair) with real A; all terms share each integral pass.

    Exchange K[A]_ad: A_pair='bc', out_pair='ad'.  Coulomb J[A]_ab: A_pair='dc',
    out_pair='ab'.  Returns a list (one per term) of {atom: (3, nao, nao)}.
    '''
    if atmlst is None:
        atmlst = range(mol.natm)
    nao = mol.nao_nr()
    terms = [(numpy.asarray(A, dtype=numpy.double, order='C'), ap, op) for A, ap, op in terms]
    res = [{ia: numpy.zeros((3, nao, nao)) for ia in atmlst} for _ in terms]
    live = [t for t, (A, ap, op) in enumerate(terms) if _nonzero(A)]
    if not live:
        return res
    vhfopt = _deriv_vhfopt(mol, 'int2e_ip1', [terms[t][0] for t in live])
    for ia in atmlst:
        for p in 'abcd':
            intor, comp, pi = _labels_map(p, None)
            shls_slice, rng = _slices(mol, {'i': ia})
            scripts, dms, outs = [], [], []
            for t in live:
                A, ap, op = terms[t]
                la, lb = pi[ap[0]], pi[ap[1]]
                lo1, lo2 = pi[op[0]], pi[op[1]]
                scripts.append('%s%s->s1%s%s' % (la, lb, lo1, lo2))
                dms.append(numpy.ascontiguousarray(A[rng[la], rng[lb]]))
                outs.append((t, rng[lo1], rng[lo2]))
            vs = _contract(mol, intor, comp, scripts, dms, shls_slice, vhfopt)
            for (t, r1, r2), v in zip(outs, vs):
                # d/dR of an AO = -nabla
                res[t][ia][:, r1, r2] -= v.reshape(3, *v.shape[-2:])
    return res


def deriv1_matrix(mol, A, A_pair, out_pair, atmlst=None):
    return deriv1_matrices(mol, [(A, A_pair, out_pair)], atmlst)[0]


# label permutations, symmetries of g(ab|cd) for real AOs, that bring the
# derivative position to 'a'
_TO_A = {'a': {'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd'},
         'b': {'a': 'b', 'b': 'a', 'c': 'c', 'd': 'd'},
         'c': {'a': 'c', 'b': 'd', 'c': 'a', 'd': 'b'},
         'd': {'a': 'd', 'b': 'c', 'c': 'b', 'd': 'a'}}


def deriv1_energies(mol, terms):
    '''First nuclear derivatives of E = sum_t coef_t sum g(ab|cd) A_t[A_pair]
    B_t[B_pair] at fixed real A_t, B_t, (natm, 3).  The four derivative
    positions are mapped onto the first by the permutational symmetry of g, so
    all terms share one unsliced int2e_ip1 pass.'''
    nao = mol.nao_nr()
    terms = [(numpy.asarray(A, dtype=numpy.double, order='C'), ap,
              numpy.asarray(B, dtype=numpy.double, order='C'), bp, c)
             for A, ap, B, bp, c in terms]
    terms = [t for t in terms if t[4] != 0 and _nonzero(t[0]) and _nonzero(t[2])]
    g = numpy.zeros((3, nao))
    if terms:
        intor, comp, pi = _labels_map('a', None)
        scripts, dms, outs = [], [], []
        for A, ap, B, bp, c in terms:
            for p in 'abcd':
                s = _TO_A[p]
                ap1, bp1 = s[ap[0]] + s[ap[1]], s[bp[0]] + s[bp[1]]
                # contract the pair without the derivative index in the
                # integral, keep the derivative index 'i' in the output
                (D, dp), (O, op) = ((A, ap1), (B, bp1)) if 'a' in bp1 else ((B, bp1), (A, ap1))
                scripts.append('%s%s->s1%s%s' % (pi[dp[0]], pi[dp[1]], pi[op[0]], pi[op[1]]))
                dms.append(D)
                outs.append((c, O, op[0] == 'a'))
        vhfopt = _deriv_vhfopt(mol, intor, [t[0] for t in terms] + [t[2] for t in terms])
        vs = _contract(mol, intor, comp, scripts, dms, (0, mol.nbas) * 4, vhfopt)
        for (c, O, first), v in zip(outs, vs):
            v = v.reshape(3, nao, nao)
            if first:
                g += c * numpy.einsum('xpq,pq->xp', v, O)
            else:
                g += c * numpy.einsum('xqp,qp->xp', v, O)
    de = numpy.zeros((mol.natm, 3))
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        de[ia] = -g[:, p0:p1].sum(axis=1)       # d/dR of an AO = -nabla
    return de


def veff_deriv1_energy(mol, terms):
    '''d/dR_X Re sum_t Tr[(cj_t J - ck_t K)[A_t] B_t] at fixed GHF matrices
    A_t, B_t (any symmetry), (natm, 3): the contraction of veff_deriv1 with a
    density, without the per-atom derivative matrices.'''
    nao = mol.nao_nr()
    out = []
    for A, B, cj, ck in terms:
        ra = A[:nao, :nao] + A[nao:, nao:]
        rb = B[:nao, :nao] + B[nao:, nao:]
        # Re (x y) = Re x Re y - Im x Im y
        out.append((ra.real, 'dc', rb.real, 'ba', cj))
        out.append((ra.imag, 'dc', rb.imag, 'ba', -cj))
        if ck != 0:
            ba, bb = _spin_blocks(A, nao), _spin_blocks(B, nao)
            for s, t in ('aa', 'bb', 'ab', 'ba'):
                out.append((ba[s + t].real, 'bc', bb[t + s].real, 'da', -ck))
                out.append((ba[s + t].imag, 'bc', bb[t + s].imag, 'da', ck))
    return deriv1_energies(mol, out)


def _symmetry_group():
    '''The eight label permutations of g(ab|cd) = g(ba|cd) = g(ab|dc) = g(cd|ab)
    (real AOs), as dicts letter -> letter.'''
    gens = [dict(zip('abcd', 'bacd')), dict(zip('abcd', 'abdc')), dict(zip('abcd', 'cdab'))]
    group = [dict(zip('abcd', 'abcd'))]
    while True:
        new = []
        for s in group:
            for t in gens:
                st = {x: t[s[x]] for x in 'abcd'}
                if st not in group and st not in new:
                    new.append(st)
        if not new:
            return group
        group += new


_GROUP = _symmetry_group()
# derivative position pairs (p, q): one representative per orbit of the group
_PAIR_REPS = (('a', 'a'), ('a', 'b'), ('a', 'c'))


def _fold_pairs():
    '''{rep: [sigma, ...]}: for every ordered pair (p, q) of derivative
    positions, one group element that maps it onto its representative.'''
    out = {rep: [] for rep in _PAIR_REPS}
    for p in 'abcd':
        for q in 'abcd':
            for rep in _PAIR_REPS:
                s = next((s for s in _GROUP if s[p] == rep[0] and s[q] == rep[1]), None)
                if s is not None:
                    out[rep].append(s)
                    break
    assert sum(len(v) for v in out.values()) == 16
    return out


_FOLD = _fold_pairs()


def deriv2_energies(mol, terms, atmlst=None):
    '''Second nuclear derivatives of E = sum_t coef_t sum g(ab|cd) A_t[A_pair]
    B_t[B_pair] for a list of terms (A, A_pair, B, B_pair, coef), real A, B.
    The 16 ordered pairs of derivative positions are folded by the
    permutational symmetry of g onto three (same AO, same electron, different
    electrons), so all terms share three integral passes.  Returns
    (natm, natm, 3, 3).'''
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = list(atmlst)
    natm = len(atmlst)
    if atmlst == list(range(mol.natm)):
        # unique shell quartets, all derivative integrals at once (lib/vhf/hess2e.c)
        hess = deriv2_energies_direct(mol, terms)
        if hess is not None:
            return hess
    terms = [(numpy.asarray(A, dtype=numpy.double, order='C'), ap,
              numpy.asarray(B, dtype=numpy.double, order='C'), bp, c)
             for A, ap, B, bp, c in terms]
    terms = [t for t in terms if t[4] != 0 and _nonzero(t[0]) and _nonzero(t[2])]
    hess = numpy.zeros((natm, natm, 3, 3))
    if not terms:
        return hess
    aoslices = mol.aoslice_by_atom()
    symmetric = {}
    for A, ap, B, bp, c in terms:
        for M in (A, B):
            if id(M) not in symmetric:
                symmetric[id(M)] = M.shape[0] == M.shape[1] and abs(M - M.T).max() < 1e-14
    for (p, q), sigmas in _FOLD.items():
        # relabelled terms: the derivatives sit at positions p and q; identical
        # ones (symmetric matrices, the same matrix on both sides) are merged
        merged = {}
        for A, ap, B, bp, c in terms:
            for s in sigmas:
                a1, b1 = s[ap[0]] + s[ap[1]], s[bp[0]] + s[bp[1]]
                if symmetric[id(A)]:
                    a1 = ''.join(sorted(a1))
                if symmetric[id(B)]:
                    b1 = ''.join(sorted(b1))
                if A is B and b1 < a1:
                    a1, b1 = b1, a1
                key = (id(A), a1, id(B), b1)
                c0 = merged[key][4] if key in merged else 0.
                merged[key] = (A, a1, B, b1, c0 + c)
        rterms = list(merged.values())
        intor, comp, pi = _labels_map(p, q)
        qletter = pi[q]                       # 'i' as well for the same-AO pass
        vhfopt = _deriv_vhfopt(mol, intor, [t[0] for t in terms] + [t[2] for t in terms])
        # one integral pass per atom of the first derivative index (like
        # pyscf's RHF Hessian): the output pair always carries the second
        # derivative index, which is resolved by atoms afterwards
        for i0, ia in enumerate(atmlst):
            shls_slice, rng = _slices(mol, {'i': ia})
            scripts, dms, outs = [], [], []
            for A, ap, B, bp, c in rterms:
                la, lb = pi[ap[0]], pi[ap[1]]
                lo1, lo2 = pi[bp[0]], pi[bp[1]]
                if qletter not in (lo1, lo2):
                    A, B, la, lb, lo1, lo2 = B, A, lo1, lo2, la, lb
                scripts.append('%s%s->s1%s%s' % (la, lb, lo1, lo2))
                dms.append(numpy.ascontiguousarray(A[rng[la], rng[lb]]))
                outs.append((c, B[rng[lo1], rng[lo2]], 3 if lo1 == qletter else 2))
            vs = _contract(mol, intor, comp, scripts, dms, shls_slice, vhfopt)
            for (c, Bblk, other), v in zip(outs, vs):
                # two AO derivatives: (-nabla)(-nabla) = + nabla nabla
                v = v.reshape(3, 3, *v.shape[-2:]) * Bblk
                if p == q:
                    hess[i0, i0] += c * v.sum(axis=(2, 3))
                    continue
                s = v.sum(axis=other)                          # (3, 3, n of the second index)
                for j0, ja in enumerate(atmlst):
                    q0, q1 = aoslices[ja][2:]
                    hess[i0, j0] += c * s[:, :, q0:q1].sum(axis=2)
    return hess


_libsocvhf = None
HESS2E_TOL = 1e-13


def _load_socvhf():
    global _libsocvhf
    if _libsocvhf is None:
        from socutils.lib import load_library
        try:
            lib_ = load_library('libsocvhf', 'SOCUTILS_SOCVHF_LIBRARY')
            _libsocvhf = lib_ if hasattr(lib_, 'SOCVHFhess2e_contract') else False
        except OSError:
            _libsocvhf = False
    return _libsocvhf


def deriv2_energies_direct(mol, terms):
    '''deriv2_energies over the unique shell quartets with all second-derivative
    integrals of a quartet at once (lib/vhf/hess2e.c), contracted on the fly
    with the symmetrized density weight; screened by Schwarz-type estimates
    times the densities.  None if lib/libsocvhf.so has no hess2e (or the basis
    is Cartesian).'''
    import ctypes
    from pyscf.gto import moleintor
    libv = _load_socvhf()
    if not libv or mol.cart:
        return None
    natm = mol.natm
    nao = mol.nao_nr()
    terms = [(numpy.asarray(A, dtype=numpy.double, order='C'), ap,
              numpy.asarray(B, dtype=numpy.double, order='C'), bp, c)
             for A, ap, B, bp, c in terms]
    terms = [t for t in terms if t[4] != 0 and _nonzero(t[0]) and _nonzero(t[2])]
    hess = numpy.zeros((natm, natm, 3, 3))
    if not terms:
        return hess
    pos = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    mats_a = [t[0] for t in terms]
    mats_b = [t[2] for t in terms]
    pair_a = numpy.asarray([pos[x] for t in terms for x in t[1]], dtype=numpy.int32)
    pair_b = numpy.asarray([pos[x] for t in terms for x in t[3]], dtype=numpy.int32)
    coef = numpy.asarray([t[4] for t in terms])
    pa = (ctypes.c_void_p * len(terms))(*[m.ctypes.data for m in mats_a])
    pb = (ctypes.c_void_p * len(terms))(*[m.ctypes.data for m in mats_b])
    # integral estimate per shell pair, bounding every derivative type
    q = numpy.max([_q_cond(mol, intor)[k] for intor in ('int2e_ipip1', 'int2e_ipvip1', 'int2e_ip1ip2')
                   for k in (0, 1)], axis=0)
    q = numpy.ascontiguousarray(numpy.maximum(q, q.T))
    dcond = _dm_cond(mol, mats_a + mats_b)
    atm, bas, env = mol._atm, mol._bas, mol._env
    opts = [moleintor.make_cintopt(atm, bas, env, name)
            for name in ('int2e_ipip1_sph', 'int2e_ipvip1_sph', 'int2e_ip1ip2_sph')]
    ao_loc = mol.ao_loc_nr().astype(numpy.int32)
    libv.SOCVHFhess2e_contract(
        hess.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
        ctypes.c_int(len(terms)), pa, pair_a.ctypes.data_as(ctypes.c_void_p),
        pb, pair_b.ctypes.data_as(ctypes.c_void_p), coef.ctypes.data_as(ctypes.c_void_p),
        q.ctypes.data_as(ctypes.c_void_p), dcond.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(HESS2E_TOL), opts[0], opts[1], opts[2],
        ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(natm), bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
        env.ctypes.data_as(ctypes.c_void_p))
    assert nao == ao_loc[-1]
    return hess


def deriv2_energy(mol, A, A_pair, B, B_pair, atmlst=None):
    return deriv2_energies(mol, [(A, A_pair, B, B_pair, 1.)], atmlst)


def _spin_blocks(dm, nao):
    return {'aa': dm[:nao, :nao], 'ab': dm[:nao, nao:],
            'ba': dm[nao:, :nao], 'bb': dm[nao:, nao:]}


def partial_hess_jk(mol, dm, atmlst=None, with_k=True, k_scale=1.):
    '''Second derivatives of E_J - k_scale * E_K at fixed GHF density dm.'''
    nao = mol.nao_nr()
    rho = (dm[:nao, :nao] + dm[nao:, nao:]).real
    terms = [(rho, 'ba', rho, 'dc', .5)]
    if with_k:
        blk = _spin_blocks(dm, nao)
        # time-reversal-even dm (Kramers closed shell): the bb and ba terms
        # equal the aa and ab ones
        pairs, w = (('aa', 'ab'), 2.) if _kramers_even(dm, nao) else (('aa', 'bb', 'ab', 'ba'), 1.)
        for s, t in pairs:
            dst, dts = blk[s + t], blk[t + s]
            terms.append((dst.real, 'bc', dts.real, 'da', -.5 * k_scale * w))
            terms.append((dst.imag, 'bc', dts.imag, 'da', .5 * k_scale * w))
    return deriv2_energies(mol, terms, atmlst)


def _kramers_even(dm, nao):
    from socutils.scf.ghf import _is_kramers_even
    dm = numpy.asarray(dm)
    return dm.ndim == 2 and _is_kramers_even(dm[None], nao)


def veff_deriv1(mol, dm, atmlst=None, with_k=True, k_scale=1., hermi=1):
    '''d/dR_X (J - k_scale K)[dm] at fixed dm, GHF matrices {atom: (3, 2n, 2n)}.

    hermi=1: dm Hermitian (the imaginary part of the total density drops out
    of J).  hermi=0: general complex dm, e.g. a transition density.'''
    nao = mol.nao_nr()
    if atmlst is None:
        atmlst = range(mol.natm)
    rho = dm[:nao, :nao] + dm[nao:, nao:]
    if hermi:
        terms = [(rho.real, 'dc', 'ab')]
        jphase = [1.]
    else:
        terms = [(rho.real, 'dc', 'ab'), (rho.imag, 'dc', 'ab')]
        jphase = [1., 1j]
    keys = [None] * len(terms)
    # time-reversal-even Hermitian dm: K'_bb = (K'_aa)^*, K'_ba = (K'_ab)^H
    kramers = with_k and hermi and _kramers_even(dm, nao)
    if with_k:
        blk = _spin_blocks(dm, nao)
        for st, d in blk.items():
            if kramers and st in ('bb', 'ba'):
                continue
            terms += [(d.real, 'bc', 'ad'), (d.imag, 'bc', 'ad')]
            keys += [(st, 1.), (st, 1j)]
    out = deriv1_matrices(mol, terms, atmlst)
    off = {'a': slice(0, nao), 'b': slice(nao, 2 * nao)}
    nj = len(jphase)
    res = {}
    for ia in atmlst:
        v = numpy.zeros((3, 2 * nao, 2 * nao), dtype=numpy.complex128)
        vj = sum(ph * m[ia] for ph, m in zip(jphase, out[:nj]))
        v[:, :nao, :nao] = vj
        v[:, nao:, nao:] = vj
        for key, m in zip(keys[nj:], out[nj:]):
            st, phase = key
            v[:, off[st[0]], off[st[1]]] -= k_scale * phase * m[ia]
        if kramers:
            # vj sits on aa and bb: subtract it to transform the K parts only
            kaa = v[:, :nao, :nao] - vj
            v[:, nao:, nao:] = vj + kaa.conj()
            v[:, nao:, :nao] = v[:, :nao, nao:].conj().transpose(0, 2, 1)
        res[ia] = v
    return res


def _block_diag(m, nao):
    out = numpy.zeros(m.shape[:-2] + (2 * nao, 2 * nao), dtype=m.dtype)
    out[..., :nao, :nao] = m
    out[..., nao:, nao:] = m
    return out


def x2c_hcore_deriv2_generator(mf, mol=None, step=2e-3):
    '''d2 h_X2C / dR_I dR_J: four-point central difference of the analytic
    first derivative.  Returns f(ia, ja) -> (3, 3, 2n, 2n) complex.'''
    from socutils.somf import x2c_grad
    if mol is None:
        mol = mf.mol
    x2cobj = mf.with_x2c
    natm = mol.natm
    coords = mol.atom_coords()                      # Bohr
    # d1[ja][y][ia] = d/dR_{ja,y} of (dh/dR_ia)  (3, 2n, 2n)
    d1 = {}
    for ja in range(natm):
        for y in range(3):
            vals = []
            for disp in (2 * step, step, -step, -2 * step):
                c = coords.copy()
                c[ja, y] += disp
                mold = mol.set_geom_(c, unit='Bohr', inplace=False)
                gen = x2c_grad.hcore_grad_generator_spin_orbital(x2cobj, mold)
                vals.append([gen(ia) for ia in range(natm)])
            for ia in range(natm):
                d1[ja, y, ia] = (-vals[0][ia] + 8 * vals[1][ia] - 8 * vals[2][ia] + vals[3][ia]) / (12 * step)

    def hcore_deriv2(ia, ja):
        # (x on ia, y on ja)
        return numpy.stack([d1[ja, y, ia] for y in range(3)], axis=1)
    return hcore_deriv2


def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    log = logger.new_logger(hessobj, verbose)
    time0 = (logger.process_clock(), logger.perf_counter())
    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None: mo_occ = mf.mo_occ
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)
    atmlst = list(atmlst)
    nao = mol.nao_nr()

    mocc = mo_coeff[:, mo_occ > 0]
    dm0 = mocc.dot(mocc.conj().T)
    dme0 = (mocc * mo_energy[mo_occ > 0]).dot(mocc.conj().T)
    rho = (dm0[:nao, :nao] + dm0[nao:, nao:]).real
    wrho = (dme0[:nao, :nao] + dme0[nao:, nao:]).real

    de2 = hessobj.partial_hess_veff(dm0, atmlst)
    t1 = log.timer_debug1('two-electron partial hessian', *time0)

    # overlap and hcore second derivatives
    s1aa, s1ab, s1a = rhf_hess.get_ovlp(mol)
    aoslices = mol.aoslice_by_atom()
    hcore_deriv2 = hessobj.hcore_generator(mol)
    for i0, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia][2:]
        de2[i0, i0] -= numpy.einsum('xypq,pq->xy', s1aa[:, :, p0:p1], wrho[p0:p1]) * 2
        for j0, ja in enumerate(atmlst):
            q0, q1 = aoslices[ja][2:]
            de2[i0, j0] -= numpy.einsum('xypq,pq->xy', s1ab[:, :, p0:p1, q0:q1], wrho[p0:p1, q0:q1]) * 2
            h2 = hcore_deriv2(ia, ja)
            if h2.shape[-1] != nao:
                # the finite-difference X2C second derivative is symmetric
                # only to its truncation error; use the symmetric part
                h2 = .5 * (h2 + hcore_deriv2(ja, ia).transpose(1, 0, 2, 3))
            if h2.shape[-1] == nao:
                de2[i0, j0] += numpy.einsum('xypq,qp->xy', h2, rho)
            else:
                de2[i0, j0] += numpy.einsum('xypq,qp->xy', h2, dm0).real
    log.timer('GHF partial hessian', *time0)
    return de2


def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    '''First-order Fock matrices F^X (3, 2n, 2n) at fixed density.'''
    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    mocc = mo_coeff[:, mo_occ > 0]
    dm0 = mocc.dot(mocc.conj().T)
    nao = mol.nao_nr()
    # the socutils GHF gradient class (also for a density-fitted object, whose
    # .Gradients would be pyscf's): X2C / SO-ECP hcore derivatives
    from socutils.grad import ghf_grad
    hcore_deriv = ghf_grad.Gradients(hessobj.base).hcore_generator(mol)
    v1 = hessobj.veff_deriv1(dm0, atmlst)
    h1ao = [None] * mol.natm
    for ia in atmlst:
        h1 = hcore_deriv(ia)
        if h1.shape[-1] == nao:
            h1 = _block_diag(h1, nao)
        h1ao[ia] = h1 + v1[ia]
    return h1ao


def _s1ao(mol, ia, s1a, nao):
    p0, p1 = mol.aoslice_by_atom()[ia][2:]
    s1ao = numpy.zeros((3, nao, nao))
    s1ao[:, p0:p1] += s1a[:, p0:p1]
    s1ao[:, :, p0:p1] += s1a[:, p0:p1].transpose(0, 2, 1)
    return _block_diag(s1ao, nao)


def gen_vind(mf, mo_coeff, mo_occ):
    mocc = mo_coeff[:, mo_occ > 0]
    nmo = mo_coeff.shape[1]
    nocc = mocc.shape[1]
    vresp = mf.gen_response(mo_coeff, mo_occ, hermi=1)

    def fx(mo1):
        mo1 = mo1.reshape(-1, nmo, nocc)
        dm1 = numpy.asarray([mo_coeff.dot(x).dot(mocc.conj().T) for x in mo1])
        dm1 = dm1 + dm1.conj().transpose(0, 2, 1)
        v1 = vresp(dm1)
        return numpy.asarray([mo_coeff.conj().T.dot(v).dot(mocc) for v in v1])
    return fx


def cphf_solve(fvind, mo_energy, mo_occ, h1, s1=None, max_cycle=50, tol=1e-9,
               verbose=logger.WARN, level_shift=0):
    '''Coupled-perturbed GHF with complex orbitals.

    Same equations and return values as pyscf.scf.cphf.solve, but the
    response v[U] of a complex GHF reference depends on U and U* (the density
    change is C U C_occ^H + h.c.): it is real-linear, not complex-linear, and
    the Krylov solve is carried out on the stacked real vector [Re U, Im U].
    '''
    log = logger.new_logger(verbose=verbose)
    occidx = mo_occ > 0
    viridx = ~occidx
    e_a = mo_energy[viridx]
    e_i = mo_energy[occidx]
    e_ai = 1 / (e_a[:, None] + level_shift - e_i)
    nvir, nocc = e_ai.shape
    nmo = nocc + nvir
    h1 = numpy.asarray(h1).reshape(-1, nmo, nocc)
    nset = h1.shape[0]
    if s1 is None:
        s1 = numpy.zeros_like(h1)
    s1 = numpy.asarray(s1).reshape(-1, nmo, nocc)

    hs = h1 - s1 * e_i
    mo1base = hs.astype(numpy.complex128)
    mo1base[:, viridx] *= -e_ai
    mo1base[:, occidx] = -s1[:, occidx] * .5

    def to_real(x):
        # krylov drops converged vectors: the number of sets varies
        n = x.shape[0]
        return numpy.hstack([x.real.reshape(n, -1), x.imag.reshape(n, -1)])

    def to_cplx(r):
        r = r.reshape(-1, 2, nvir * nocc)
        return (r[:, 0] + 1j * r[:, 1]).reshape(-1, nvir, nocc)

    def vind_vo(r):
        x = to_cplx(r)
        mo1 = numpy.zeros((x.shape[0], nmo, nocc), dtype=numpy.complex128)
        mo1[:, viridx] = x
        v = fvind(mo1).reshape(-1, nmo, nocc)[:, viridx]
        if level_shift != 0:
            v = v - x * level_shift
        v = v * e_ai
        return to_real(v)

    # occupied-occupied part is fixed (-S1/2); its response enters the rhs
    mo1_oo = numpy.zeros_like(mo1base)
    mo1_oo[:, occidx] = mo1base[:, occidx]
    rhs = to_real(mo1base[:, viridx] - (fvind(mo1_oo).reshape(-1, nmo, nocc)[:, viridx] * e_ai))

    # lib.krylov stops once a new Krylov vector has norm^2 < lindep (1e-13, an
    # absolute threshold), i.e. at ~3e-7 residuals whatever tol is.  Solve on
    # normalized right-hand sides and refine on the residual.
    def ksolve(b):
        scale = numpy.linalg.norm(b, axis=1)
        scale[scale == 0] = 1
        y = lib.krylov(vind_vo, b / scale[:, None], tol=tol, max_cycle=max_cycle, verbose=log)
        return y.reshape(b.shape) * scale[:, None]

    x = ksolve(rhs)
    for it in range(4):
        r = rhs - x - vind_vo(x)
        rnorm = abs(r).max()
        log.debug('cphf_solve refinement %d  |r| = %.3g', it, rnorm)
        if rnorm < tol:
            break
        x = x + ksolve(r)
    mo1 = mo1_oo.copy()
    mo1[:, viridx] = to_cplx(x)

    hs = hs + fvind(mo1).reshape(-1, nmo, nocc)
    mo1[:, viridx] = hs[:, viridx] / (e_i - e_a[:, None])
    mo_e1 = hs[:, occidx, :]
    mo_e1 += mo1[:, occidx] * (e_i[:, None] - e_i)
    return mo1, mo_e1


def solve_mo1(mf, mo_energy, mo_coeff, mo_occ, h1ao, fx=None, atmlst=None,
              max_memory=4000, verbose=None, max_cycle=50, level_shift=0):
    mol = mf.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    nao = mol.nao_nr()
    mocc = mo_coeff[:, mo_occ > 0]
    nocc = mocc.shape[1]
    nmo = mo_coeff.shape[1]
    if fx is None:
        fx = gen_vind(mf, mo_coeff, mo_occ)
    s1a = -mol.intor('int1e_ipovlp', comp=3)
    C = mo_coeff

    def ao2mo(mat):
        return numpy.asarray([C.conj().T.dot(x).dot(mocc) for x in mat])

    h1vo = numpy.vstack([ao2mo(h1ao[ia]) for ia in atmlst])
    s1vo = numpy.vstack([ao2mo(_s1ao(mol, ia, s1a, nao)) for ia in atmlst])
    tol = mf.conv_tol_cpscf * len(atmlst)
    mo1, e1 = cphf_solve(fx, mo_energy, mo_occ, h1vo, s1vo,
                         max_cycle=max_cycle, level_shift=level_shift, tol=tol,
                         verbose=verbose)
    mo1 = numpy.einsum('pq,xqi->xpi', C, mo1).reshape(-1, 3, C.shape[0], nocc)
    e1 = e1.reshape(-1, 3, nocc, nocc)
    mo1s = [None] * mol.natm
    e1s = [None] * mol.natm
    for k, ia in enumerate(atmlst):
        mo1s[ia] = mo1[k]
        e1s[ia] = e1[k]
    return mo1s, e1s


def hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
              mo1=None, mo_e1=None, h1ao=None, atmlst=None, max_memory=4000,
              verbose=None):
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (logger.process_clock(), logger.perf_counter())
    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None: mo_occ = mf.mo_occ
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)
    atmlst = list(atmlst)
    nao = mol.nao_nr()

    de2 = hessobj.partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst, max_memory, log)
    if h1ao is None:
        h1ao = hessobj.make_h1(mo_coeff, mo_occ, None, atmlst, log)
        t1 = log.timer_debug1('making H1', *time0)
    if mo1 is None or mo_e1 is None:
        mo1, mo_e1 = hessobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao, None,
                                       atmlst, max_memory, log)
        t1 = log.timer_debug1('solving MO1', *t1)

    mocc = mo_coeff[:, mo_occ > 0]
    e_occ = mo_energy[mo_occ > 0]
    nocc = mocc.shape[1]
    natm = len(atmlst)
    s1a = -mol.intor('int1e_ipovlp', comp=3)
    # Tr[h1_A dm1_B] = sum_i (C_o^H h1_A mo1_B)_ii etc.: project h1 and s1 onto
    # the occupied orbitals once per atom, then all atom pairs in one product
    A = numpy.empty((natm, 3, nocc, 2 * nao), dtype=numpy.complex128)
    B = numpy.empty((natm, 3, nocc, 2 * nao), dtype=numpy.complex128)
    Soo = numpy.empty((natm, 3, nocc, nocc), dtype=numpy.complex128)
    for i0, ia in enumerate(atmlst):
        s1ao = _s1ao(mol, ia, s1a, nao)
        A[i0] = numpy.einsum('pi,xpq->xiq', mocc.conj(), h1ao[ia])
        B[i0] = numpy.einsum('pi,xpq->xiq', mocc.conj(), s1ao)
        Soo[i0] = numpy.einsum('xiq,qj->xij', B[i0], mocc)
    M1 = numpy.asarray([mo1[ja] for ja in atmlst]).transpose(0, 1, 3, 2)          # (natm, 3, nocc, 2n)
    E1 = numpy.asarray([mo_e1[ja] for ja in atmlst])                              # (natm, 3, nocc, nocc)
    M1f = M1.reshape(3 * natm, -1)
    de = 2 * A.reshape(3 * natm, -1).dot(M1f.T).real
    de -= 2 * (B * e_occ[None, None, :, None]).reshape(3 * natm, -1).dot(M1f.T).real
    de -= Soo.reshape(3 * natm, -1).dot(E1.transpose(0, 1, 3, 2).reshape(3 * natm, -1).T).real
    de2 += de.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)
    log.timer('GHF hessian', *time0)
    return de2


class Hessian(rhf_hess.HessianBase):
    '''Non-relativistic / X2C GHF Hessian'''

    # finite-difference step (Bohr) for the X2C hcore second derivative
    x2c_fd_step = 2e-3

    def partial_hess_veff(self, dm0, atmlst):
        return partial_hess_jk(self.mol, dm0, atmlst)

    def veff_deriv1(self, dm0, atmlst):
        return veff_deriv1(self.mol, dm0, atmlst)

    def vxc_deriv1(self, dm0, atmlst):
        '''XC part of veff_deriv1 (none for HF).'''
        return None

    def hcore_generator(self, mol=None):
        if mol is None:
            mol = self.mol
        if getattr(self.base, 'with_x2c', None) is not None:
            return x2c_hcore_deriv2_generator(self.base, mol, self.x2c_fd_step)
        hcore_deriv2 = rhf_hess.HessianBase.hcore_generator(self, mol)
        if not (getattr(self.base, 'with_soc', False) and mol.has_ecp_soc()):
            return hcore_deriv2
        # scalar part (pyscf, incl. the scalar ECP) + spin-orbit ECP
        from socutils.grad import ecp_so_deriv
        gen = ecp_so_deriv.ecpso_deriv2_generator(mol)
        nao = mol.nao_nr()

        def hcore_deriv2_so(ia, ja):
            h = _block_diag(numpy.asarray(hcore_deriv2(ia, ja), dtype=numpy.complex128), nao)
            return h + ecp_so_deriv.ghf_hcore_deriv2(mol, ia, ja, gen)
        return hcore_deriv2_so

    partial_hess_elec = partial_hess_elec
    hess_elec = hess_elec
    make_h1 = make_h1

    def solve_mo1(self, mo_energy, mo_coeff, mo_occ, h1ao, fx=None, atmlst=None,
                  max_memory=4000, verbose=None):
        return solve_mo1(self.base, mo_energy, mo_coeff, mo_occ, h1ao, fx, atmlst,
                         max_memory, verbose, max_cycle=self.max_cycle,
                         level_shift=self.level_shift)


Hess = Hessian

from pyscf import scf  # noqa: E402
scf.ghf.GHF.Hessian = lib.class_as_method(Hessian)
