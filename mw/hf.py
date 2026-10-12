'''
Closed-shell nonrelativistic Hartree-Fock with multiwavelets (VAMPyR).

Nuclei are Gaussian charge distributions Z (zeta/pi)^3/2 exp(-zeta r^2) so the
results can be matched one-to-one with a Gaussian-basis calculation using the
same nuclear model.  The orbitals are solved with the fixed-shift integral form

    phi_i = G_mu [ (mu^2 + 2 F_ii) phi_i - 2 (V + J - K) phi_i + 2 sum_{j != i} phi_j F_ji ],
    G_mu = (-lap + mu^2)^-1,   mu ~ sqrt(-2 F_ii)

with DIIS on the orbital residuals and Loewdin orthonormalization.  The
starting orbitals are the Gaussian-basis RHF orbitals of pyscf projected onto
the multiwavelet grid.

Units: atomic units, mol coordinates in bohr (mol.atom_coords()).
'''

import time
import numpy
from vampyr import vampyr3d as vp


def tsum(trees):
    '''Sum of an iterable of FunctionTrees.'''
    out = None
    for t in trees:
        out = t if out is None else out + t
    return out


class MWContext:
    '''Multiresolution analysis, operators and the nuclear potential at one precision.'''

    def __init__(self, mol, prec=1e-5, order=None, box=32, zeta=1e4, verbose=1):
        if order is None:
            order = 9
        self.mol = mol
        self.prec = prec
        self.order = order
        self.box = box
        self.zeta = zeta
        self.verbose = verbose
        self.mra = vp.MultiResolutionAnalysis(box=[-box, box], order=order)
        self.P = vp.ScalingProjector(self.mra, prec)
        self.D = vp.ABGVDerivative(self.mra, 0.5, 0.5)
        self.poisson = vp.PoissonOperator(self.mra, prec)
        self._helmholtz = {}
        self.nuclei = [(float(mol.atom_charge(i)), [float(x) for x in mol.atom_coord(i)])
                       for i in range(mol.natm)]
        rho = None
        for z, pos in self.nuclei:
            g = self.P(vp.GaussFunc(beta=zeta, alpha=z * (zeta / numpy.pi)**1.5, position=pos))
            rho = g if rho is None else rho + g
        self.rho_nuc = rho
        self.V = -4 * numpy.pi * self.poisson(rho)

    def coulomb(self, f):
        '''int f(r')/|r-r'| dr' '''
        return 4 * numpy.pi * self.poisson(f)

    def mul(self, a, b):
        '''Product cropped to the working precision.'''
        t = a * b
        t.crop(self.prec)
        return t

    def grad(self, f):
        return [self.D(f, a) for a in range(3)]

    def helmholtz(self, mu, rtol=0.15):
        '''A Helmholtz operator (-lap + m^2)^-1 with m within rtol of mu, and m.
        Any shift m is exact in the fixed-shift iteration; it only affects the
        convergence rate, so operators are reused across iterations.'''
        for m, op in self._helmholtz.items():
            if abs(m - mu) < rtol * mu:
                return op, m
        op = vp.HelmholtzOperator(self.mra, exp=float(mu), prec=self.prec)
        self._helmholtz[mu] = op
        return op, mu

    def project(self, func):
        return self.P(func)


def _cart_ao_trees(ctx, mol):
    '''Cartesian AOs of a cart=True mol as FunctionTrees, scaled to pyscf's
    normalization (checked against int1e_ovlp).'''
    from pyscf import gto
    aos = []
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        pos = [float(x) for x in mol.bas_coord(ib)]
        exps = mol.bas_exp(ib)
        coef = mol._libcint_ctr_coeff(ib)
        for ic in range(coef.shape[1]):
            for lx in range(l, -1, -1):
                for ly in range(l - lx, -1, -1):
                    lz = l - lx - ly
                    f = None
                    for e, cc in zip(exps, coef[:, ic]):
                        g = ctx.P(vp.GaussFunc(beta=float(e), alpha=float(cc), position=pos,
                                               poly_exponent=[lx, ly, lz]))
                        f = g if f is None else f + g
                    aos.append(f)
    s_ref = mol.intor('int1e_ovlp')
    s_mw = numpy.array([a.squaredNorm() for a in aos])
    scale = numpy.sqrt(numpy.diag(s_ref) / s_mw)
    aos = [float(k) * a for k, a in zip(scale, aos)]
    return aos, s_ref


def project_mos(ctx, mol, nocc=None, basis=None):
    '''Run a Gaussian-basis RHF (cartesian copy of mol, optionally another basis)
    and project its occupied orbitals onto the multiwavelet grid.'''
    from pyscf import scf
    molc = mol.copy()
    molc.cart = True
    if basis is not None:
        molc.basis = basis
    molc.verbose = 0
    molc.build()
    mf = scf.RHF(molc).run(conv_tol=1e-10)
    if nocc is None:
        nocc = mol.nelectron // 2
    aos, s_ref = _cart_ao_trees(ctx, molc)
    n = len(aos)
    for i0 in range(0, n, max(1, n // 4)):
        for j0 in range(n):
            err = abs(vp.dot(aos[i0], aos[j0]) - s_ref[i0, j0])
            if err > 1e-4:
                raise RuntimeError('AO projection does not reproduce the overlap '
                                   '(AO %d,%d: %g)' % (i0, j0, err))
    phis = []
    for i in range(nocc):
        c = mf.mo_coeff[:, i]
        phis.append(tsum(float(c[mu]) * aos[mu] for mu in range(n) if abs(c[mu]) > 1e-12))
    return lowdin(phis), mf


def lowdin(phis):
    n = len(phis)
    s = numpy.array([[vp.dot(phis[i], phis[j]) for j in range(n)] for i in range(n)])
    w, v = numpy.linalg.eigh(s)
    x = v @ numpy.diag(w**-0.5) @ v.T
    return [tsum(float(x[j, i]) * phis[j] for j in range(n) if abs(x[j, i]) > 1e-14)
            for i in range(n)]


class RHF:
    '''Closed-shell multiwavelet RHF.  nocc doubly occupied orbitals; with
    one_electron=True a single singly occupied orbital without J/K (H2+-like).
    Orbitals can be carried over between contexts of the same box and order.'''

    def __init__(self, ctx, phis, one_electron=False):
        self.ctx = ctx
        self.phis = phis
        self.one_electron = one_electron
        self.max_cycle = 60
        self.ndiis = 6
        self.conv_tol = None          # default: ctx.prec
        self.e_elec = None
        self.fock = None
        self.vphi = None
        self.t0 = time.time()

    def log(self, msg):
        if self.ctx.verbose:
            print(msg + '  (%.0f s)' % (time.time() - self.t0), flush=True)

    def potential(self, phis):
        '''(V + J - K) phi_i for all i, and the Fock and core matrices.'''
        ctx = self.ctx
        n = len(phis)
        if self.one_electron:
            vphi = [ctx.mul(ctx.V, p) for p in phis]
        else:
            rho = 2.0 * tsum(ctx.mul(p, p) for p in phis)
            J = ctx.coulomb(rho)
            kpairs = {}
            for i in range(n):
                for j in range(i, n):
                    kpairs[i, j] = kpairs[j, i] = ctx.coulomb(ctx.mul(phis[i], phis[j]))
            vj = ctx.V + J
            vphi = [ctx.mul(vj, phis[i]) - tsum(ctx.mul(phis[j], kpairs[i, j]) for j in range(n))
                    for i in range(n)]
        grads = [ctx.grad(p) for p in phis]
        t = numpy.array([[0.5 * sum(vp.dot(grads[i][a], grads[j][a]) for a in range(3))
                          for j in range(n)] for i in range(n)])
        v = numpy.array([[vp.dot(phis[i], vphi[j]) for j in range(n)] for i in range(n)])
        vphis = [ctx.mul(ctx.V, p) for p in phis]
        vne = numpy.array([[vp.dot(phis[i], vphis[j]) for j in range(n)] for i in range(n)])
        fock = t + (v + v.T) * 0.5
        hcore = t + (vne + vne.T) * 0.5
        return vphi, fock, hcore

    def gphis(self):
        '''(J - K) phi_i of the converged orbitals.'''
        return [v - self.ctx.mul(self.ctx.V, p) for v, p in zip(self.vphi, self.phis)]

    def save(self, prefix):
        for i, p in enumerate(self.phis):
            p.saveTree('%s_%d' % (prefix, i))

    def energy_elec(self, fock, hcore):
        if self.one_electron:
            return float(numpy.trace(hcore))
        return float(numpy.trace(hcore) + numpy.trace(fock))

    def kernel(self):
        ctx = self.ctx
        tol = self.conv_tol or ctx.prec
        phis = self.phis
        n = len(phis)
        hist = []
        e_old = 0.
        for it in range(self.max_cycle):
            vphi, fock, hcore = self.potential(phis)
            e = self.energy_elec(fock, hcore)
            new = []
            for i in range(n):
                g, mu = ctx.helmholtz(numpy.sqrt(-2 * fock[i, i]))
                arg = float(mu**2 + 2 * fock[i, i]) * phis[i] - 2.0 * vphi[i]
                off = [2.0 * float(fock[j, i]) * phis[j] for j in range(n) if j != i]
                if off:
                    arg = arg + tsum(off)
                new.append(g(arg))
            res = [new[i] - phis[i] for i in range(n)]
            rnorm = numpy.sqrt(sum(r.squaredNorm() for r in res))
            hist = (hist + [(phis, res)])[-self.ndiis:]
            m = len(hist)
            b = -numpy.ones((m + 1, m + 1))
            b[m, m] = 0
            for p in range(m):
                for q in range(p, m):
                    b[p, q] = b[q, p] = sum(vp.dot(hist[p][1][i], hist[q][1][i]) for i in range(n))
            rhs = numpy.zeros(m + 1)
            rhs[m] = -1
            try:
                cvec = numpy.linalg.solve(b, rhs)[:m]
            except numpy.linalg.LinAlgError:
                cvec = numpy.zeros(m)
                cvec[-1] = 1
            phis = [tsum(float(cvec[k]) * (hist[k][0][i] + hist[k][1][i]) for k in range(m)).crop(ctx.prec)
                    for i in range(n)]
            phis = lowdin(phis)
            self.log('  prec %.0e it %2d  E_elec % .12f  dE % .2e  |res| %.2e' % (
                ctx.prec, it, e, e - e_old, rnorm))
            if rnorm < tol and abs(e - e_old) < 0.1 * tol:
                break
            e_old = e
        vphi, fock, hcore = self.potential(phis)
        self.phis = phis
        self.vphi = vphi
        self.fock = fock
        self.e_elec = self.energy_elec(fock, hcore)
        self.log('  E_elec = % .12f  eps = %s' % (self.e_elec, numpy.array2string(
            numpy.linalg.eigvalsh(fock), precision=8)))
        return self.e_elec

