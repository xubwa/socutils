'''Minimization of the energy of an engine (energy and Cartesian gradient of a
geometry) in geomeTRIC's internal coordinates, with our own step logic:

  coordinates  TRIC (default; DLC, HDLC, Cartesian) from geometric.internal:
               B matrix, iterative back-transformation, the model Hessian;
               rebuilt when the bonding changes, the rotations grow large or
               the back-transformation fails (Hessian carried over through the
               Cartesian one)
  Hessian      the model Hessian, or the exact one of the engine (hessian =
               'exact'); BFGS updates with Powell damping
  step         RFO, restricted to the trust radius by the shifted Newton step;
               GEDIIS (Li, Frisch, JCTC 2, 835 (2006)) in the small-force
               region: the energy-weighted interpolation of the last points,
               then a Newton step from it
  trust        ratio of the actual to the predicted energy change; a step
               that raises the energy is rejected and replaced by the minimum
               of the cubic through the two end points along it
  convergence  geomeTRIC / Gaussian criteria (energy change, RMS / max
               gradient, RMS / max displacement), or forces under 1/100 of the
               gradient thresholds

Coordinates are flat arrays in Bohr, gradients in Hartree / Bohr.'''
import itertools
import numpy
from pyscf import lib
from pyscf.lib import logger
from geometric.molecule import Molecule
from geometric.internal import CartesianCoordinates, DelocalizedInternalCoordinates
from geometric.step import calc_drms_dmax

BOHR = lib.param.BOHR

# geomeTRIC defaults (Gaussian's force and displacement thresholds)
CONV_PARAMS = {
    'convergence_energy': 1e-6,  # Eh
    'convergence_grms': 3e-4,    # Eh / Bohr
    'convergence_gmax': 4.5e-4,  # Eh / Bohr
    'convergence_drms': 1.2e-3,  # Angstrom
    'convergence_dmax': 1.8e-3,  # Angstrom
}


class PySCFEngine:
    '''Energy and Cartesian gradient from a pyscf gradient scanner (whose SCF
    starts from the density of the previous geometry), and the Cartesian
    Hessian of the last geometry by hessian(mf) -> (natm, natm, 3, 3) (default
    mf.Hessian().kernel()).'''

    def __init__(self, g_scanner, hessian=None):
        self.scanner = g_scanner
        self.mol = g_scanner.mol.copy()
        self.hessian_method = hessian
        self.ncalc = 0
        self.nhess = 0

    @property
    def elements(self):
        return [self.mol.atom_pure_symbol(i) for i in range(self.mol.natm)]

    def coords(self):
        return self.mol.atom_coords().ravel()

    def __call__(self, x):
        self.mol.set_geom_(x.reshape(-1, 3), unit='Bohr')
        e, g = self.scanner(self.mol)
        self.ncalc += 1
        if not self.scanner.converged:
            logger.warn(self.scanner, 'SCF / gradients not converged at geometry %d', self.ncalc)
        return e, numpy.asarray(g).ravel()

    def hessian(self, x):
        mf = self.scanner.base
        if mf.mo_coeff is None or abs(mf.mol.atom_coords().ravel() - x).max() > 1e-10:
            self(x)
        h = self.hessian_method(mf) if self.hessian_method else mf.Hessian().kernel()
        self.nhess += 1
        n = x.size
        return h.transpose(0, 2, 1, 3).reshape(n, n)


def fast_engine(mf):
    '''PySCFEngine on socutils' fast RHF / RKS paths: MD J and jk_multi J + K
    builds (scf.md_j), the blocked XC potential (dft.vxc_fast), the fast
    gradients (grad.rhf_fast) and Hessians (hessian.rhf_fast).  Modifies mf.'''
    from pyscf.scf import hf
    from socutils.scf import md_j
    from socutils.dft import vxc_fast
    from socutils.grad import rhf_fast as grad_fast
    from socutils.hessian import rhf_fast as hess_fast
    dft = isinstance(mf, hf.KohnShamDFT)
    md_j.apply(mf)
    if dft:
        vxc_fast.apply(mf)
    hcls = hess_fast.RKSHessian if dft else hess_fast.RHFHessian
    return PySCFEngine(grad_fast.Gradients(mf).as_scanner(), hessian=lambda mf: hcls(mf).kernel())


def make_coords(elem, x, coordsys='tric'):
    '''geomeTRIC coordinate system of the geometry x (Bohr)'''
    M = Molecule()
    M.elem = list(elem)
    M.xyzs = [x.reshape(-1, 3) * BOHR]
    M.build_topology()
    if coordsys == 'cart':
        return CartesianCoordinates(M)
    connect, addcart = {'tric': (False, False), 'dlc': (True, False), 'hdlc': (False, True)}[coordsys]
    return DelocalizedInternalCoordinates(M, build=True, connect=connect, addcart=addcart)


def _ratio(a, d):
    '''a / d, zero where d vanishes (where a does too: the gradient has no
    component along a mode whose eigenvalue is the shift)'''
    out = numpy.zeros_like(a)
    m = abs(d) > 1e-12
    out[m] = a[m] / d[m]
    return out


def rfo_step(H, g, trust):
    '''Minimizing RFO step, or, beyond the trust radius, the shifted Newton step
    -(H - lam)^-1 g of length trust (lam below the lowest eigenvalue).
    Returns the step and lam.'''
    h, v = numpy.linalg.eigh(H)
    gt = v.T.dot(g)
    n = g.size
    aug = numpy.zeros((n + 1, n + 1))
    aug[:n, :n] = numpy.diag(h)
    aug[:n, n] = aug[n, :n] = gt
    lam = numpy.linalg.eigvalsh(aug)[0]

    def step(lam):
        return -v.dot(_ratio(gt, h - lam))
    dq = step(lam)
    if numpy.linalg.norm(dq) <= trust:
        return dq, lam
    # |step(lam)| falls monotonically as lam -> -inf below h[0]; at lo it is <= trust
    hi = lam
    lo = min(h[0], lam) - numpy.linalg.norm(g) / trust
    for _ in range(200):
        mid = .5 * (lo + hi)
        if numpy.linalg.norm(step(mid)) > trust:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-14 * max(1., abs(lo)):
            break
    return step(lo), lo


def bfgs_update(H, s, y):
    '''BFGS update of H with step s and gradient change y; Powell damping keeps
    H positive definite when s.y is small or negative'''
    Hs = H.dot(s)
    sHs = s.dot(Hs)
    if sHs < 1e-14:
        return H
    sy = s.dot(y)
    if sy < .2 * sHs:
        theta = .8 * sHs / (sHs - sy)
        y = theta * y + (1 - theta) * Hs
        sy = s.dot(y)
    return H + numpy.outer(y, y) / sy - numpy.outer(Hs, Hs) / sHs


def prfo_step(H, g, trust, mode):
    '''P-RFO step (Banerjee et al., JPC 89, 52 (1985)) uphill along eigenvector
    mode of H and downhill in the others, scaled to the trust radius.
    Returns the step and the eigenvectors of H.'''
    h, v = numpy.linalg.eigh(H)
    gt = v.T.dot(g)
    k = mode
    lp = .5 * h[k] + .5 * numpy.sqrt(h[k]**2 + 4 * gt[k]**2)
    rest = numpy.arange(g.size) != k
    hr, gr = h[rest], gt[rest]
    n = hr.size
    aug = numpy.zeros((n + 1, n + 1))
    aug[:n, :n] = numpy.diag(hr)
    aug[:n, n] = aug[n, :n] = gr
    ln = numpy.linalg.eigvalsh(aug)[0]
    st = numpy.empty_like(gt)
    st[k] = -gt[k] / (h[k] - lp) if abs(h[k] - lp) > 1e-14 else 0.
    st[rest] = -_ratio(gr, hr - ln)
    dq = v.dot(st)
    norm = numpy.linalg.norm(dq)
    if norm > trust:
        dq *= trust / norm
    return dq, v


def bofill_update(H, s, y):
    '''Bofill's update (SR1 / PSB mixture), for transition states'''
    ss = s.dot(s)
    if ss < 1e-20:
        return H
    r = y - H.dot(s)
    sr = s.dot(r)
    rr = r.dot(r)
    if rr < 1e-24:
        return H
    phi = sr * sr / (ss * rr)
    psb = (numpy.outer(r, s) + numpy.outer(s, r)) / ss - sr * numpy.outer(s, s) / ss**2
    sr1 = numpy.outer(r, r) / sr if abs(sr) > 1e-14 else 0.
    return H + phi * sr1 + (1 - phi) * psb


def gediis_coeffs(E, q, g):
    '''GEDIIS coefficients: minimize sum_i c_i E_i - 1/2 sum_i<j c_i c_j (g_i - g_j).(q_i - q_j)
    (exact for a quadratic surface) over c_i >= 0, sum c_i = 1: the lowest
    feasible stationary point of all faces of the simplex.  Returns c and the
    interpolated energy.'''
    m = len(E)
    A = numpy.empty((m, m))
    for i in range(m):
        for j in range(m):
            A[i, j] = .5 * (g[i] - g[j]).dot(q[i] - q[j])
    i0 = int(numpy.argmin(E))
    best = numpy.zeros(m)
    best[i0] = 1
    fbest = E[i0]
    for k in range(2, m + 1):
        for S in itertools.combinations(range(m), k):
            S = list(S)
            kkt = numpy.zeros((k + 1, k + 1))
            kkt[:k, :k] = A[numpy.ix_(S, S)]
            kkt[:k, k] = kkt[k, :k] = 1
            rhs = numpy.append(E[S], 1)
            try:
                sol = numpy.linalg.solve(kkt, rhs)
            except numpy.linalg.LinAlgError:
                continue
            c = sol[:k]
            if c.min() < -1e-12:
                continue
            f = c.dot(E[S]) - .5 * c.dot(A[numpy.ix_(S, S)]).dot(c)
            if f < fbest:
                fbest = f
                best = numpy.zeros(m)
                best[S] = c
    return best, fbest


def cubic_min(e0, d0, e1, d1):
    '''minimum in (0, 1] of the cubic with values e0, e1 and slopes d0, d1 at 0, 1
    (None if it has none there)'''
    # p(a) = e0 + d0 a + b a^2 + c a^3
    c = d0 + d1 - 2 * (e1 - e0)
    b = 3 * (e1 - e0) - 2 * d0 - d1
    if abs(c) < 1e-14:
        return -d0 / (2 * b) if b > 0 and 0 < -d0 / (2 * b) <= 1 else None
    disc = b * b - 3 * c * d0
    if disc < 0:
        return None
    a = (-b + numpy.sqrt(disc)) / (3 * c)     # the root with p'' = 2b + 6ca > 0
    return a if 0 < a <= 1 else None


def _grad_norms(gx):
    atomgrad = numpy.linalg.norm(gx.reshape(-1, 3), axis=1)
    return numpy.sqrt(numpy.mean(atomgrad**2)), atomgrad.max()


class Optimizer(lib.StreamObject):
    '''Minimizer of engine(x) -> (E, gradient); x0 and the elements from
    engine.coords(), engine.elements unless given to kernel.

    Attributes:
        coordsys : 'tric' (default), 'dlc', 'hdlc' or 'cart'
        hessian : 'model' or 'exact' (the engine's Hessian at the start); default
            model for minima, exact for transition states
        transition : transition-state search: P-RFO following the mode of the
            lowest eigenvalue at the start (by overlap afterwards), Bofill
            updates; the exact Hessian again every hess_recalc steps, and
            (hess_adaptive) when the updated one has had a wrong number of
            negative eigenvalues for two steps
        trust, tmax, tmin : trust radius in the internal coordinates (Bohr, rad)
        gediis : GEDIIS steps when the RMS internal gradient is below
            gediis_grms, from the last nhist points
        max_cycle : maximum number of energy + gradient evaluations
        conv_params : CONV_PARAMS keys
        criteria : 'geometric' (all five on the Cartesian gradient and the
            last step) or 'gaussian' (Gaussian's: max / RMS internal force and
            max / RMS predicted next internal step, the gmax, grms, dmax, drms
            thresholds read in Hartree, Bohr, rad; no energy criterion)
    '''
    coordsys = 'tric'
    hessian = None
    transition = False
    hess_recalc = 0
    hess_adaptive = True
    neg_eig = 1e-3
    criteria = 'geometric'
    trust = 0.3
    tmax = 0.5
    tmin = 1e-3
    trust_ts = 0.1
    tmax_ts = 0.3
    gediis = True
    gediis_grms = 1e-2
    nhist = 5
    max_cycle = 200
    reject_de = 1e-6

    def __init__(self, engine, **kwargs):
        self.engine = engine
        self.verbose = getattr(getattr(engine, 'mol', None), 'verbose', logger.NOTE)
        self.stdout = getattr(getattr(engine, 'mol', None), 'stdout', None) or __import__('sys').stdout
        self.conv_params = dict(CONV_PARAMS)
        for k, v in kwargs.items():
            if k in CONV_PARAMS:
                self.conv_params[k] = v
            else:
                setattr(self, k, v)
        self.converged = False
        self.e_tot = None
        self.ncalc = 0
        self.nhess = 0
        self.x = None

    def _new_coords(self, x, gx, ic, H, vmode=None):
        '''coordinate system rebuilt at x; H and the followed mode carried over
        through Cartesian coordinates'''
        ic1 = make_coords(self.elem, x, self.coordsys)
        gq = ic.calcGrad(x, gx)
        Hx = ic.calcHessCart(x, gq, H)
        if vmode is not None:
            vx = ic.wilsonB(x).T.dot(ic.GInverse(x)).dot(vmode)
            vmode = ic1.wilsonB(x).dot(vx)
            vmode /= numpy.linalg.norm(vmode)
        return ic1, ic1.calcHess(x, gx, Hx), vmode

    def _coords_changed(self, x, ic):
        M = Molecule()
        M.elem = list(self.elem)
        M.xyzs = [x.reshape(-1, 3) * BOHR]
        M.build_topology()
        if self.coordsys == 'cart':
            return False
        connect, addcart = {'tric': (False, False), 'dlc': (True, False), 'hdlc': (False, True)}[self.coordsys]
        ic1 = DelocalizedInternalCoordinates(M, build=False, connect=connect, addcart=addcart)
        return ic1 != ic or (hasattr(ic, 'largeRots') and ic.largeRots())

    def _step(self, ic, H, x, gq, hist, trust):
        '''internal-coordinate step and its kind'''
        dq, _ = rfo_step(H, gq, trust)
        if not self.gediis or len(hist) < 3 or numpy.sqrt(numpy.mean(gq**2)) > self.gediis_grms:
            return dq, 'rfo'
        pts = hist[-self.nhist:]
        E = numpy.array([p[1] for p in pts])
        q = [ic.calcDiff(p[0], x) for p in pts]
        g = [ic.calcGrad(p[0], p[2]) for p in pts]
        c, _ = gediis_coeffs(E, q, g)
        if c.max() > 1 - 1e-12:
            return dq, 'rfo'
        qi = sum(ci * qq for ci, qq in zip(c, q))
        gi = sum(ci * gg for ci, gg in zip(c, g))
        try:
            dqd = qi - numpy.linalg.solve(H, gi)
        except numpy.linalg.LinAlgError:
            return dq, 'rfo'
        if numpy.linalg.norm(dqd) > trust or gq.dot(dqd) + .5 * dqd.dot(H).dot(dqd) >= 0:
            return dq, 'rfo'
        return dqd, 'gediis'

    def _exact_hessian(self, ic, x, gx):
        self.nhess += 1
        return ic.calcHess(x, gx, self.engine.hessian(x))

    def kernel(self, x0=None, elem=None):
        log = logger.new_logger(self)
        eng = self.engine
        ts = self.transition
        self.elem = elem if elem is not None else eng.elements
        x = numpy.array(x0 if x0 is not None else eng.coords(), dtype=float).ravel()
        cp = self.conv_params
        ic = make_coords(self.elem, x, self.coordsys)
        E, gx = eng(x)
        ncalc = 1
        hist = [(x.copy(), E, gx.copy())]
        hess = self.hessian if self.hessian is not None else ('exact' if ts else 'model')
        H = self._exact_hessian(ic, x, gx) if hess == 'exact' else ic.guess_hessian(x)
        gq = ic.calcGrad(x, gx)
        trust = self.trust_ts if ts else self.trust
        tmax = self.tmax_ts if ts else self.tmax
        forced = None          # the line-search point after a rejected step
        vmode = None           # the followed mode (transition states)
        nwrong = 0             # consecutive steps with a wrong Hessian index
        nstep = 0
        last_exact = 0
        conv = False
        grms, gmax = _grad_norms(gx)
        log.info('geomopt start  E = %.10f  grms %.2e gmax %.2e  (%d internal coordinates)',
                 E, grms, gmax, gq.size)
        if gmax < cp['convergence_gmax'] * .01 and grms < cp['convergence_grms'] * .01:
            conv = True

        def step(ic, H, gq):
            nonlocal vmode
            if not ts:
                return self._step(ic, H, x, gq, hist, trust)
            h, v = numpy.linalg.eigh(H)
            k = 0 if vmode is None or vmode.size != gq.size else int(numpy.argmax(abs(v.T.dot(vmode))))
            dq, v = prfo_step(H, gq, trust, k)
            vmode = v[:, k]
            return dq, 'prfo%d' % k

        while not conv and ncalc < self.max_cycle:
            if forced is not None:
                dq, kind = ic.calcDiff(forced, x), 'line'
                xnew = forced
                forced = None
            else:
                dq, kind = step(ic, H, gq)
                xnew = ic.newCartesian(x, dq, verbose=0)
                if ic.bork:
                    log.info('back-transformation failed; new coordinate system')
                    ic, H, vmode = self._new_coords(x, gx, ic, H, vmode)
                    gq = ic.calcGrad(x, gx)
                    dq, kind = step(ic, H, gq)
                    xnew = ic.newCartesian(x, dq, verbose=0)
                    if ic.bork:
                        kind += '/linear'      # the first-order back-transformation
            dq = ic.calcDiff(xnew, x)
            pred = gq.dot(dq) + .5 * dq.dot(H).dot(dq)
            Enew, gxnew = eng(xnew)
            ncalc += 1
            nstep += 1
            hist.append((xnew.copy(), Enew, gxnew.copy()))
            gqnew = ic.calcGrad(xnew, gxnew)
            H = (bofill_update if ts else bfgs_update)(H, dq, gqnew - gq)
            dE = Enew - E
            dqn = numpy.linalg.norm(dq)
            rho = dE / pred if abs(pred) > 1e-12 else 1.
            grms, gmax = _grad_norms(gxnew)
            drms, dmax = calc_drms_dmax(xnew, x)
            log.info('geomopt %3d %-7s E = %.10f dE %9.2e rho %6.2f |dq| %.3e trust %.3e '
                     'grms %.2e gmax %.2e drms %.2e dmax %.2e', ncalc, kind, Enew, dE, rho, dqn, trust,
                     grms, gmax, drms, dmax)
            if not ts and dE > self.reject_de and dqn > self.tmin:
                s = xnew - x
                a = cubic_min(E, gx.dot(s), Enew, gxnew.dot(s))
                a = .5 if a is None else min(max(a, .1), .5)
                forced = x + a * s
                trust = max(a * dqn, self.tmin)
                continue
            if abs(pred) > 1e-8:
                good = rho > .75 and (not ts or rho < 1.25)
                bad = rho < .25 or (ts and rho > 1.75)
                if bad:
                    trust = max(.25 * dqn, self.tmin)
                elif good and dqn > .8 * trust:
                    trust = min(2 * trust, tmax)
            x, E, gx, gq = xnew, Enew, gxnew, gqnew
            if self.criteria == 'gaussian':
                # Gaussian's: internal forces and the predicted next step
                dqn_next = step(ic, H, gq)[0]
                conv = ((abs(gq).max() < cp['convergence_gmax'] and numpy.sqrt(numpy.mean(gq**2)) < cp['convergence_grms']
                         and abs(dqn_next).max() < cp['convergence_dmax']
                         and numpy.sqrt(numpy.mean(dqn_next**2)) < cp['convergence_drms'])
                        or (abs(gq).max() < cp['convergence_gmax'] * .01
                            and numpy.sqrt(numpy.mean(gq**2)) < cp['convergence_grms'] * .01))
            else:
                conv = ((abs(dE) < cp['convergence_energy'] and grms < cp['convergence_grms']
                         and gmax < cp['convergence_gmax'] and drms < cp['convergence_drms']
                         and dmax < cp['convergence_dmax'])
                        or (gmax < cp['convergence_gmax'] * .01 and grms < cp['convergence_grms'] * .01))
            if conv:
                break
            if self._coords_changed(x, ic):
                log.info('coordinate system rebuilt')
                ic, H, vmode = self._new_coords(x, gx, ic, H, vmode)
                gq = ic.calcGrad(x, gx)
            if ts:
                # the overall rotations / translations of TRIC carry ~zero curvature
                nneg = int((numpy.linalg.eigvalsh(H) < -self.neg_eig).sum())
                nwrong = nwrong + 1 if nneg != 1 else 0
                if ((self.hess_recalc and nstep % self.hess_recalc == 0)
                        or (self.hess_adaptive and nwrong >= 2 and nstep - last_exact >= 3)):
                    log.info('exact Hessian recomputed (%d negative eigenvalues)', nneg)
                    H = self._exact_hessian(ic, x, gx)
                    nwrong = 0
                    last_exact = nstep
        self.converged = conv
        self.e_tot = E
        self.x = x
        self.ncalc = ncalc
        if conv:
            log.note('geomopt converged in %d energy + gradient evaluations (%d Hessians), E = %.10f',
                     ncalc, self.nhess, E)
        else:
            log.note('geomopt not converged in %d evaluations, E = %.10f', ncalc, E)
        return conv, x


def optimize(method, fast=False, **kwargs):
    '''Optimize the geometry of an SCF object (or a gradient scanner, or an
    engine).  fast: the socutils fast RHF / RKS paths (fast_engine).
    kwargs: Optimizer attributes and CONV_PARAMS.  Returns (converged, mol).'''
    if isinstance(method, lib.GradScanner):
        eng = PySCFEngine(method)
    elif hasattr(method, 'nuc_grad_method'):
        eng = fast_engine(method) if fast else PySCFEngine(method.nuc_grad_method().as_scanner())
    else:
        eng = method
    opt = Optimizer(eng, **kwargs)
    conv, x = opt.kernel()
    mol = eng.mol.copy()
    mol.set_geom_(x.reshape(-1, 3), unit='Bohr')
    return conv, mol
