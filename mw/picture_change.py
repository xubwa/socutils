'''
Complete-basis-set coefficients of the sf-X2C-1e picture-change error of a
harmonic multipole operator O (lap O = 0) for a nonrelativistic density.

For a far point charge Z_B the spurious long-range energy of sf-X2C-1e is
-Z_B Tr[D (PC(O) - O)] / R^(l+1) (l: lowest nonvanishing multipole, O the
corresponding solid harmonic), and

    dM(c) = Tr[D (PC(O) - O)] = a2 / c^2 + a4 / c^4 + O(c^-6)

In a complete basis (modified spin-free Dirac, T = p^2/2, W_V = p.V p, V the
nuclear potential of the fragment; X~ = 1 + X2/c^2, X2 = T^-1 W_V/4 - (V+T)/2)

    a2 = (1/8) <lap O> = 0
    a4 = (5/32) <[T,[T,O]]> + (1/8) Re <W_V T^-1 [O,T]>
       = a4_H1 + a4_H2kin + a4_W,
    a4_H1    = (1/32) <[T,[T,O]]>       (CBS limit of H1 = 1/2 [[R,V],R^-1])
    a4_H2kin = (4/32) <[T,[T,O]]>       (-T/2 part of X2 in H2)
    a4_W     = (1/8) <W_V T^-1 [O,T]>   (T^-1 W_V/4 part of X2 in H2)

with, for harmonic O and real orbitals,
    [O,T] phi = grad O . grad phi
    <phi|[T,[T,O]]|phi> = - sum_jk <d_j phi| O_jk |d_k phi>     (O_jk = d_j d_k O)
    <phi|W_V T^-1 [O,T]|phi> = sum_a <V d_a phi | d_a u>,  u = T^-1 grad O . grad phi

The c^-4 coefficient of the relaxation of the density, Tr[D2 (lap O)/8], is
zero for the same reason as a2, so only the nonrelativistic density enters.

approx='1e' (X built from the nuclear potential of A + B) adds the response
of the decoupling to V_B.  To first order in O, with X = 1 + X2/c^2,
dX2 = T^-1 W_O/4 - O/2, the change of L = V + T - X2^T T X2/c^4 + X^T W X/4c^2
and of R = (1 + X^T T X/2c^2)^-1/2 combine to the commutator

    Q = (1/4c^4) [H0, B - B^T],  B = T dX2,  B - B^T = -[T,O]/2
      = -(1/8c^4) [T + V, [T,O]]

so a4_X = -(1/8) Tr D [T+V, [T,O]].  It vanishes for an eigenstate of T + V
(one-electron systems); for Hartree-Fock orbitals (Tr D [F, .] = 0)

    a4_X = (1/8) Tr D [G, [T,O]] = -(1/2) sum_i <G phi_i | grad O . grad phi_i>
         = (1/2) sum_i <(T + V) phi_i | grad O . grad phi_i>     (check)

(sums over doubly occupied orbitals), G = J - K.  This is the CBS limit of
the 'X-response' of examples/qdhf/14-h1h4_decomposition.py.
'''

import numpy
from vampyr import vampyr3d as vp
from socutils.mw.hf import tsum


class Multipole:
    '''A harmonic polynomial O, given as dict {(i,j,k): coef} for x^i y^j z^k
    about the point origin.'''

    def __init__(self, terms, origin=(0., 0., 0.), name='O'):
        self.terms = dict(terms)
        self.origin = numpy.asarray(origin, dtype=float)
        self.name = name
        assert self.is_harmonic()

    @classmethod
    def dipole_z(cls, origin=(0., 0., 0.)):
        return cls({(0, 0, 1): 1.}, origin, 'z')

    @classmethod
    def quadrupole_zz(cls, origin=(0., 0., 0.)):
        '''r^2 P2(cos theta) = z^2 - (x^2 + y^2)/2'''
        return cls({(0, 0, 2): 1., (2, 0, 0): -.5, (0, 2, 0): -.5}, origin, 'r^2 P2')

    @classmethod
    def octupole_zzz(cls, origin=(0., 0., 0.)):
        '''r^3 P3(cos theta) = z^3 - 3/2 z (x^2 + y^2)'''
        return cls({(0, 0, 3): 1., (2, 0, 1): -1.5, (0, 2, 1): -1.5}, origin, 'r^3 P3')

    @staticmethod
    def _diff(terms, a):
        out = {}
        for p, c in terms.items():
            if p[a] > 0:
                q = list(p)
                q[a] -= 1
                q = tuple(q)
                out[q] = out.get(q, 0.) + c * p[a]
        return out

    def derivative(self, *axes):
        t = self.terms
        for a in axes:
            t = self._diff(t, a)
        return {p: c for p, c in t.items() if c != 0}

    def is_harmonic(self):
        lap = {}
        for a in range(3):
            for p, c in self.derivative(a, a).items():
                lap[p] = lap.get(p, 0.) + c
        return all(abs(c) < 1e-14 for c in lap.values())

    def tree(self, ctx, terms=None):
        '''Project the polynomial (or one of its derivatives) onto ctx.'''
        terms = self.terms if terms is None else terms
        if not terms:
            return None
        o = self.origin

        def f(r):
            x, y, z = r[0] - o[0], r[1] - o[1], r[2] - o[2]
            return sum(c * x**p[0] * y**p[1] * z**p[2] for p, c in terms.items())
        return ctx.P(f)


def pc_coefficients(ctx, phis, op, occ=2., gphis=None):
    '''a2 check and the c^-4 coefficient (split into H1, H2kin, W parts) of
    Tr[D (PC(O) - O)] for orbitals phis with occupation occ.  With gphis
    ((J - K) phi_i) also the decoupling response a4_X of approx='1e', in two
    equivalent forms (a4_X, a4_X_check).'''
    grad_o = [op.tree(ctx, op.derivative(a)) for a in range(3)]
    hess = {}
    for j in range(3):
        for k in range(j, 3):
            terms = op.derivative(j, k)
            if terms:
                hess[j, k] = op.tree(ctx, terms)
    o_tree = op.tree(ctx)
    # products with the (unbounded) polynomials refine far out in the box;
    # cropping right away keeps the trees at the size of the orbitals
    mul = ctx.mul

    out = dict(moment=0., a2=0., kin=0., w=0., xg=0., xh=0.)
    per_orb = []
    for i, p in enumerate(phis):
        g = ctx.grad(p)
        gop = tsum(mul(grad_o[a], g[a]) for a in range(3) if grad_o[a] is not None)
        u = 2 * ctx.poisson(gop)                     # T^-1 [O,T] phi
        du = ctx.grad(u)
        w = sum(vp.dot(mul(ctx.V, g[a]), du[a]) for a in range(3))
        kin = 0.
        for (j, k), h in hess.items():
            val = vp.dot(g[j], mul(h, g[k]))
            kin -= val if j == k else 2 * val
        # (1/8) <lap O> by parts: -(1/4) <d_a phi| d_a O phi>
        a2 = -0.25 * sum(vp.dot(g[a], mul(grad_o[a], p)) for a in range(3) if grad_o[a] is not None)
        mom = vp.dot(p, mul(o_tree, p))
        xg = xh = 0.
        if gphis is not None:
            xg = vp.dot(gphis[i], gop)
            dg = ctx.grad(gop)
            xh = 0.5 * sum(vp.dot(g[a], dg[a]) for a in range(3)) + vp.dot(mul(ctx.V, p), gop)
        per_orb.append(dict(moment=mom, a2=a2, kin=kin, w=w, xg=xg, xh=xh))
        for key, val in (('moment', mom), ('a2', a2), ('kin', kin), ('w', w), ('xg', xg), ('xh', xh)):
            out[key] += occ * val
    out['a4_H1'] = out['kin'] / 32
    out['a4_H2kin'] = out['kin'] * 4 / 32
    out['a4_W'] = out['w'] / 8
    out['a4'] = out['a4_H1'] + out['a4_H2kin'] + out['a4_W']
    out['a4_X'] = -0.25 * out['xg']
    out['a4_X_check'] = 0.25 * out['xh']
    out['orbitals'] = per_orb
    return out
