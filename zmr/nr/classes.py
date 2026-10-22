'''
The perturber classes for spin-restricted orbitals, in spin-free form.

With spatial indices and E_pq = sum_s p^+_s q_s, the part of V|Psi> in a class
is, for every spatial external label e (e.g. e = (a; i<j) for ccav),

    sum_P x[e, P] F_P |Phi_c Psi>,

where x[e, P] are spatial integrals and the F_P are products of spin-free
excitation operators, F_(1,t) = E_ai E_tj and F_(2,t) = E_aj E_ti for ccav.
The functions F_P |Psi> are the perturber functions of spin-free PC-NEVPT2
(Angeli, Cimiraglia, Malrieu 2002).

Written out in spin-orbitals, F_P is a sum over the spin patterns p of the
external string,

    F_P = sum_p E_(e,p) A_(P,p),

with spin-orbital external strings E_(e,p), which are orthonormal and
H_D-orthogonal for different p, and active operators A_(P,p) that lead to one
sector (n_alpha + da_p, n_beta + db_p).  A class therefore supplies

    patterns(nact)   for every p: the sector change and A_(P,p) for all P, as
                     sums of operator strings in spin-orbital indices
    blocks(ctx)      (delta[e], x[e, P]) for blocks of external labels

and the function F_P|Psi> is the tuple (A_(P,p)|Psi>)_p of CI vectors:

    UC   every pattern is an independent problem in its whole sector
    PC   the tuples are stacked, H_act = direct sum of the sector
         Hamiltonians; the span of the stacked functions is the spin-free
         first-order space, which makes the result independent of the spin
         projection of the model states.  (For a singlet this equals the
         span of the single spin-orbital strings.)

Labels with coinciding external indices (i = j, a = b) have fewer patterns
and are separate classes below (suffix "diag" in the code, same name).

Classes, parameters x and operators (s: spin of the pattern, s': the other):

    ccvv        (a<b; i<j)   O = 1 (coefficients as in the spinor code)
    ccav        (a; i<j)     (ai|tj): E_ai E_tj,  (aj|ti): E_aj E_ti
                (a; i=j)     (ai|ti): E_ai E_ti
    cavv        (a<b; i)     (ai|bt): E_ai E_bt,  (at|bi): E_at E_bi
                (a=b; i)     (at|ai): E_at E_ai
    ccaa        (i<j)        (ti|uj): E_ti E_uj,  all (t, u)
                (i=j)        (ti|ui): E_ti E_ui,  t <= u
    aavv        (a<b)        (at|bu): E_at E_bu,  all (t, u)
                (a=b)        (at|au): E_at E_au,  t <= u
    cava        (a; i)       fc_ai: E_ai,  (ai|tu): E_ai E_tu,  (au|ti): E_ti E_au
    caaa        (i)          fc_ti: E_ti,  (ti|uv): E_ti E_uv
    aaav        (a)          fc_at: E_at,  (at|uv): E_at E_uv

fc is the closed-shell core Fock matrix.
'''

import numpy

from socutils.zmr.lib import linalg as la
from socutils.zmr.nr.active import ALPHA, BETA, so
from socutils.zmr.uc.classes import ExternalBlocks, _all, _as3, _factors, _kept, _pairs, _slice

_SQRT2 = numpy.sqrt(2.0)
_SPINS = (ALPHA, BETA)


def _other(spin):
    return BETA if spin == ALPHA else ALPHA


def _dn(spin, k):
    '''(d n_alpha, d n_beta) for k electrons of the given spin.'''
    return (k, 0) if spin == ALPHA else (0, k)


def _pair_sum(x):
    '''x[p] + x[q] for all ordered pairs (p, q), flattened.'''
    return (x[:, None] + x[None, :]).ravel()


def _external(pclass, ctx, nlabel, delta_min, kernel):
    '''External sums of a doubly external class straight from the real
    three-index integrals (zmr.uc.classes.ExternalBlocks), or None if the
    integrals are not factorized.'''
    if _factors(ctx.eris) is None or nlabel <= 0:
        return None
    ext = ExternalBlocks(lambda: pclass.blocks(ctx), nlabel, delta_min,
                         lambda d: kernel(numpy.asarray(d, dtype=float)))
    ext.dtype = numpy.float64
    return ext


def _ops(nact, spins, *columns):
    '''Operator strings (rows) from columns of spatial indices and their spins.'''
    return numpy.stack([so(c, s, nact) for c, s in zip(columns, spins)], axis=1)


class Pattern:
    '''One spin pattern of the external string.

    dn: change (d n_alpha, d n_beta) of the active sector
    terms: for every parameter block, a list of (coef, daggers, ops); the
        function of parameter P is sum_terms coef[P] * string(ops[P]) |Psi>.
        An empty list means the pattern does not contain these parameters.
    '''

    def __init__(self, dn, terms):
        self.dn = dn
        self.terms = terms

    def functions(self, act, c, nelec, nparams, cache):
        '''A_(P,p) c for CI vectors c (..., dim) -> (..., nP, dim')'''
        target = act.target(nelec, self.dn)
        out = []
        for size, terms in zip(nparams, self.terms):
            block = numpy.zeros(c.shape[:-1] + (size, act.dim(target)))
            for coef, daggers, ops in terms:
                key = (daggers, ops.tobytes())
                if key not in cache:
                    cache[key] = act.apply_strings(c, nelec, [(daggers, ops)])
                block = block + numpy.asarray(coef)[..., None] * cache[key]
            out.append(block)
        return numpy.concatenate(out, axis=-2)


class SpinFreeClass:
    name = None

    def nparams(self, nact):
        '''Sizes of the parameter blocks.'''
        raise NotImplementedError

    def patterns(self, nact):
        raise NotImplementedError

    def blocks(self, ctx):
        raise NotImplementedError

    def functions(self, act, c, nelec):
        '''[(target sector, A_(P,p) c)] over the patterns p.'''
        nparams = self.nparams(act.norb)
        cache = {}
        return [(act.target(nelec, p.dn), p.functions(act, c, nelec, nparams, cache))
                for p in self.patterns(act.norb)]


class CCVV(SpinFreeClass):
    '''O = 1.  All patterns carry the same function, so they are merged:
    a_s b_s <- i_s j_s (both spins, a<b, i<j): (ai|bj) - (aj|bi);
    a_A b_B <- i_A j_B (all a, b, i, j): (ai|bj).'''
    name = 'ccvv'

    def nparams(self, nact):
        return [1]

    def patterns(self, nact):
        return [Pattern((0, 0), [[(1.0, (), _all(nact, 0))]])]

    def blocks(self, ctx):
        nc, nv = ctx.sp.ncore, ctx.sp.nvirt
        if nc < 1 or nv < 1:
            return
        cd = _factors(ctx.eris)
        if cd is None:
            g = ctx.eris.eri_block('vcvc')          # [a, i, b, j] = (ai|bj)

            def pair(i, j):
                return g[:, i, :, j]
        else:
            # (ai|bj) for one pair of core orbitals at a time
            bvc = cd('vc')

            def pair(i, j):
                return la.gemm(_slice(bvc, i), _slice(bvc, j), 'T', 'N')
        aa, bb = _pairs(nv)
        if len(aa):
            for i, j in zip(*_pairs(nc)):
                m = pair(i, j)
                v = _SQRT2 * (m[aa, bb] - m[bb, aa])[:, None]
                yield ctx.eps_v[aa] + ctx.eps_v[bb] - ctx.eps_c[i] - ctx.eps_c[j], v
        vv = _pair_sum(ctx.eps_v)
        for i in range(nc):
            for j in range(nc):
                yield vv - ctx.eps_c[i] - ctx.eps_c[j], pair(i, j).reshape(nv * nv, 1)

    def external(self, ctx):
        nc, nv = ctx.sp.ncore, ctx.sp.nvirt
        if nc < 1 or nv < 1:
            return None
        from socutils.zmr.lib import kernels
        nlabel = (nc * (nc - 1) // 2) * (nv * (nv - 1) // 2) + nc * nc * nv * nv
        return _external(
            self, ctx, nlabel, 2 * ctx.eps_v.min() - 2 * ctx.eps_c.max(),
            lambda d: _as3(kernels.wsf_ccvv(ctx.eris.cd_block('vc'), ctx.eps_v, ctx.eps_c, d)))


class CCAV(SpinFreeClass):
    '''(a; i<j): E_ai E_tj, E_aj E_ti'''
    name = 'ccav'

    def nparams(self, nact):
        return [nact, nact]

    def patterns(self, nact):
        out = []
        for s in _SPINS:
            t = ((True,), _ops(nact, (s,), numpy.arange(nact)))
            dn = _dn(s, 1)
            out += [Pattern(dn, [[(1.0,) + t], [(-1.0,) + t]]),     # a_s  <- i_s j_s
                    Pattern(dn, [[], [(-1.0,) + t]]),               # a_s' <- i_s j_s'
                    Pattern(dn, [[(-1.0,) + t], []])]               # a_s' <- j_s i_s'
        return out

    def blocks(self, ctx):
        nc, nv = ctx.sp.ncore, ctx.sp.nvirt
        if nc < 2 or nv < 1:
            return
        g = ctx.eris.eri_block('vcac')              # [a, i, t, j] = (ai|tj)
        ii, jj = _pairs(nc)
        keep = _kept(ctx, 'ccav', ii, jj)
        for i, j in zip(ii[keep], jj[keep]):
            yield (ctx.eps_v - ctx.eps_c[i] - ctx.eps_c[j],
                   numpy.concatenate([g[:, i, :, j], g[:, j, :, i]], axis=1))


class CCAVdiag(SpinFreeClass):
    '''(a; i=j): E_ai E_ti'''
    name = 'ccav'

    def nparams(self, nact):
        return [nact]

    def patterns(self, nact):
        return [Pattern(_dn(s, 1), [[(-1.0, (True,), _ops(nact, (s,), numpy.arange(nact)))]])
                for s in _SPINS]                                    # a_s' <- i_s i_s'

    def blocks(self, ctx):
        nc, nv = ctx.sp.ncore, ctx.sp.nvirt
        if nc < 1 or nv < 1:
            return
        g = ctx.eris.eri_block('vcac')
        ii = numpy.arange(nc)
        for i in ii[_kept(ctx, 'ccav', ii, ii)]:
            yield ctx.eps_v - 2 * ctx.eps_c[i], g[:, i, :, i]


class CAVV(SpinFreeClass):
    '''(a<b; i): E_ai E_bt, E_at E_bi'''
    name = 'cavv'

    def nparams(self, nact):
        return [nact, nact]

    def patterns(self, nact):
        out = []
        for s in _SPINS:
            t = ((False,), _ops(nact, (s,), numpy.arange(nact)))
            dn = _dn(s, -1)
            out += [Pattern(dn, [[(-1.0,) + t], [(1.0,) + t]]),     # a_s b_s  <- i_s
                    Pattern(dn, [[], [(1.0,) + t]]),                # a_s b_s' <- i_s'
                    Pattern(dn, [[(1.0,) + t], []])]                # b_s a_s' <- i_s'
        return out

    def blocks(self, ctx):
        nc, nv = ctx.sp.ncore, ctx.sp.nvirt
        aa, bb = _pairs(nv)
        if nc < 1 or len(aa) == 0:
            return
        cd = _factors(ctx.eris)
        if cd is None:
            g = ctx.eris.eri_block('vcva')          # [a, i, b, t] = (ai|bt)
        else:
            bvc = cd('vc')
            bva = numpy.ascontiguousarray(cd('va'))
            bva = bva.reshape(bva.shape[0], -1)       # (L, b t)
        for i in range(nc):
            if cd is None:
                m = g[:, i]                           # [a, b, t]
            else:
                m = la.gemm(_slice(bvc, i), bva, 'T', 'N').reshape(nv, nv, -1)
            yield (ctx.eps_v[aa] + ctx.eps_v[bb] - ctx.eps_c[i],
                   numpy.concatenate([m[aa, bb], m[bb, aa]], axis=1))

    def external(self, ctx):
        nc, nv, na = ctx.sp.ncore, ctx.sp.nvirt, ctx.sp.nact
        if nc < 1 or nv < 2 or na < 1:
            return None
        from socutils.zmr.lib import kernels
        return _external(
            self, ctx, nc * (nv * (nv - 1) // 2),
            numpy.sort(ctx.eps_v)[:2].sum() - ctx.eps_c.max(),
            lambda d: kernels.wsf_cavv(ctx.eris.cd_block('vc'), ctx.eris.cd_block('va'),
                                       ctx.eps_v, ctx.eps_c, d))


class CAVVdiag(SpinFreeClass):
    '''(a=b; i): E_at E_ai'''
    name = 'cavv'

    def nparams(self, nact):
        return [nact]

    def patterns(self, nact):
        return [Pattern(_dn(s, -1), [[(1.0, (False,), _ops(nact, (s,), numpy.arange(nact)))]])
                for s in _SPINS]                                    # a_s a_s' <- i_s'

    def blocks(self, ctx):
        nc, nv = ctx.sp.ncore, ctx.sp.nvirt
        if nc < 1 or nv < 1:
            return
        cd = _factors(ctx.eris)
        a = numpy.arange(nv)
        if cd is not None:
            # (ai|at) needs no four-index block
            g = numpy.einsum('Lai,Lat->ait', cd('vc'), cd('va'), optimize=True)
            for i in range(nc):
                yield 2 * ctx.eps_v - ctx.eps_c[i], g[:, i, :]
            return
        g = ctx.eris.eri_block('vcva')
        for i in range(nc):
            yield 2 * ctx.eps_v - ctx.eps_c[i], g[a, i, a, :]


class CCAA(SpinFreeClass):
    '''(i<j): E_ti E_uj for all (t, u)'''
    name = 'ccaa'

    def nparams(self, nact):
        return [nact * nact]

    def patterns(self, nact):
        t, u = _all(nact, 2).T
        d = (True, True)
        return [Pattern((2, 0), [[(1.0, d, _ops(nact, (ALPHA, ALPHA), t, u))]]),   # i_A j_A
                Pattern((0, 2), [[(1.0, d, _ops(nact, (BETA, BETA), t, u))]]),     # i_B j_B
                Pattern((1, 1), [[(1.0, d, _ops(nact, (ALPHA, BETA), t, u))]]),    # i_A j_B
                Pattern((1, 1), [[(1.0, d, _ops(nact, (ALPHA, BETA), u, t))]])]    # j_A i_B

    def blocks(self, ctx):
        nc, na = ctx.sp.ncore, ctx.sp.nact
        ii, jj = _pairs(nc)
        if len(ii) == 0 or na < 1:
            return
        keep = _kept(ctx, 'ccaa', ii, jj)
        if not keep.any():
            return
        ii, jj = ii[keep], jj[keep]
        g = ctx.eris.eri_block('acac')              # [t, i, u, j] = (ti|uj)
        x = g.transpose(1, 3, 0, 2)[ii, jj].reshape(len(ii), na * na)
        yield -ctx.eps_c[ii] - ctx.eps_c[jj], x


def _triangle(nact):
    '''(t, u) with t <= u and the weight of the second ordering (0 for t = u).'''
    t, u = numpy.triu_indices(nact)
    return t, u, (t < u).astype(float)


class CCAAdiag(SpinFreeClass):
    '''(i=j): E_ti E_ui, t <= u'''
    name = 'ccaa'

    def nparams(self, nact):
        return [nact * (nact + 1) // 2]

    def patterns(self, nact):
        t, u, w = _triangle(nact)
        d = (True, True)
        return [Pattern((1, 1), [[(1.0, d, _ops(nact, (ALPHA, BETA), t, u)),
                                  (w, d, _ops(nact, (ALPHA, BETA), u, t))]])]      # i_A i_B

    def blocks(self, ctx):
        nc, na = ctx.sp.ncore, ctx.sp.nact
        if nc < 1 or na < 1:
            return
        g = ctx.eris.eri_block('acac')
        t, u, _ = _triangle(na)
        i = numpy.arange(nc)
        i = i[_kept(ctx, 'ccaa', i, i)]
        if len(i) == 0:
            return
        k = i[:, None]
        yield -2 * ctx.eps_c[i], g[t[None, :], k, u[None, :], k]    # [i, (t<=u)] = (ti|ui)


class AAVV(SpinFreeClass):
    '''(a<b): E_at E_bu for all (t, u)'''
    name = 'aavv'

    def nparams(self, nact):
        return [nact * nact]

    def patterns(self, nact):
        t, u = _all(nact, 2).T
        d = (False, False)
        return [Pattern((-2, 0), [[(1.0, d, _ops(nact, (ALPHA, ALPHA), u, t))]]),  # a_A b_A
                Pattern((0, -2), [[(1.0, d, _ops(nact, (BETA, BETA), u, t))]]),    # a_B b_B
                Pattern((-1, -1), [[(1.0, d, _ops(nact, (BETA, ALPHA), u, t))]]),  # a_A b_B
                Pattern((-1, -1), [[(1.0, d, _ops(nact, (BETA, ALPHA), t, u))]])]  # b_A a_B

    def blocks(self, ctx):
        nv, na = ctx.sp.nvirt, ctx.sp.nact
        aa, bb = _pairs(nv)
        if len(aa) == 0 or na < 1:
            return
        g = ctx.eris.eri_block('vava')              # [a, t, b, u] = (at|bu)
        x = g.transpose(0, 2, 1, 3)[aa, bb].reshape(len(aa), na * na)
        yield ctx.eps_v[aa] + ctx.eps_v[bb], x

    def external(self, ctx):
        nv, na = ctx.sp.nvirt, ctx.sp.nact
        if nv < 2 or na < 1:
            return None
        from socutils.zmr.lib import kernels
        return _external(
            self, ctx, nv * (nv - 1) // 2, numpy.sort(ctx.eps_v)[:2].sum(),
            lambda d: kernels.wsf_aavv(ctx.eris.cd_block('va'), ctx.eps_v, d))


class AAVVdiag(SpinFreeClass):
    '''(a=b): E_at E_au, t <= u'''
    name = 'aavv'

    def nparams(self, nact):
        return [nact * (nact + 1) // 2]

    def patterns(self, nact):
        t, u, w = _triangle(nact)
        d = (False, False)
        return [Pattern((-1, -1), [[(1.0, d, _ops(nact, (BETA, ALPHA), u, t)),
                                    (w, d, _ops(nact, (BETA, ALPHA), t, u))]])]    # a_A a_B

    def blocks(self, ctx):
        nv, na = ctx.sp.nvirt, ctx.sp.nact
        if nv < 1 or na < 1:
            return
        t, u, _ = _triangle(na)
        cd = _factors(ctx.eris)
        if cd is not None:
            # (at|au) needs no four-index block
            bva = cd('va')
            yield 2 * ctx.eps_v, numpy.einsum('Lap,Lap->ap', bva[:, :, t], bva[:, :, u])
            return
        g = ctx.eris.eri_block('vava')
        a = numpy.arange(nv)[:, None]
        yield 2 * ctx.eps_v, g[a, t[None, :], a, u[None, :]]        # [a, (t<=u)] = (at|au)


class CAVA(SpinFreeClass):
    '''(a; i): E_ai, E_ai E_tu, E_ti E_au'''
    name = 'cava'

    def nparams(self, nact):
        return [1, nact * nact, nact * nact]

    def patterns(self, nact):
        t, u = _all(nact, 2).T
        d = (True, False)
        one = [(1.0, (), _all(nact, 0))]
        e_tu = [(1.0, d, _ops(nact, (s, s), t, u)) for s in _SPINS]
        out = [Pattern((0, 0), [one, e_tu, [(-1.0, d, _ops(nact, (s, s), t, u))]])
               for s in _SPINS]                                     # a_s <- i_s
        for s in _SPINS:
            o = _other(s)                                           # a_s' <- i_s
            out.append(Pattern((1, -1) if s == ALPHA else (-1, 1),
                               [[], [], [(-1.0, d, _ops(nact, (s, o), t, u))]]))
        return out

    def blocks(self, ctx):
        sp = ctx.sp
        nc, nv, na = sp.ncore, sp.nvirt, sp.nact
        if nc < 1 or nv < 1:
            return
        coul = ctx.eris.eri_block('vcaa')           # [a, i, t, u] = (ai|tu)
        exch = ctx.eris.eri_block('vaac')           # [a, u, t, i] = (au|ti)
        f_vc = ctx.fcore[sp.slice('v'), sp.slice('c')]
        for i in range(nc):
            x = numpy.concatenate([f_vc[:, i, None], coul[:, i].reshape(nv, na * na),
                                   exch[:, :, :, i].transpose(0, 2, 1).reshape(nv, na * na)],
                                  axis=1)
            yield ctx.eps_v - ctx.eps_c[i], x


class CAAA(SpinFreeClass):
    '''(i): E_ti, E_ti E_uv'''
    name = 'caaa'

    def nparams(self, nact):
        return [nact, nact**3]

    def patterns(self, nact):
        t, u, v = _all(nact, 3).T
        out = []
        for s in _SPINS:                                            # i_s
            o = _other(s)
            out.append(Pattern(_dn(s, 1), [
                [(-1.0, (True,), _ops(nact, (s,), numpy.arange(nact)))],
                [(-1.0, (True, True, False), _ops(nact, (s, s, s), t, u, v)),
                 (-1.0, (True, True, False), _ops(nact, (s, o, o), t, u, v))]]))
        return out

    def blocks(self, ctx):
        sp = ctx.sp
        nc, na = sp.ncore, sp.nact
        if nc < 1 or na < 1:
            return
        g = ctx.eris.eri_block('acaa')              # [t, i, u, v] = (ti|uv)
        f_ac = ctx.fcore[sp.slice('a'), sp.slice('c')]
        keep = _kept(ctx, 'caaa', numpy.arange(nc))
        if keep.any():
            v = numpy.concatenate([f_ac.T, g.transpose(1, 0, 2, 3).reshape(nc, na**3)], axis=1)
            yield -ctx.eps_c[keep], v[keep]


class AAAV(SpinFreeClass):
    '''(a): E_at, E_at E_uv'''
    name = 'aaav'

    def nparams(self, nact):
        return [nact, nact**3]

    def patterns(self, nact):
        t, u, v = _all(nact, 3).T
        out = []
        for s in _SPINS:                                            # a_s
            o = _other(s)
            out.append(Pattern(_dn(s, -1), [
                [(1.0, (False,), _ops(nact, (s,), numpy.arange(nact)))],
                [(1.0, (True, False, False), _ops(nact, (s, s, s), u, v, t)),
                 (1.0, (True, False, False), _ops(nact, (o, o, s), u, v, t))]]))
        return out

    def blocks(self, ctx):
        sp = ctx.sp
        nv, na = sp.nvirt, sp.nact
        if nv < 1 or na < 1:
            return
        g = ctx.eris.eri_block('vaaa')              # [a, t, u, v] = (at|uv)
        f_va = ctx.fcore[sp.slice('v'), sp.slice('a')]
        yield ctx.eps_v.copy(), numpy.concatenate([f_va, g.reshape(nv, na**3)], axis=1)


ALL_CLASSES = (CCVV(), CCAV(), CCAVdiag(), CAVV(), CAVVdiag(), CCAA(), CCAAdiag(),
               AAVV(), AAVVdiag(), CAVA(), CAAA(), AAAV())
CLASS_NAMES = ('ccvv', 'ccav', 'cavv', 'ccaa', 'aavv', 'cava', 'caaa', 'aaav')
