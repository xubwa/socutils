'''GFN-xTB through the xtb program (subprocess): an engine for Optimizer (energy
and gradient), and Cartesian Hessians, e.g. as the low-level starting Hessian
of a DFT optimization (Gaussian's ReadFC from a cheaper method).

The xtb binary is found on PATH or by $XTBBIN.  Coordinates in Bohr.'''
import os
import shutil
import subprocess
import tempfile
import numpy
from pyscf import lib

BOHR = lib.param.BOHR


def _xtb():
    exe = os.environ.get('XTBBIN') or shutil.which('xtb')
    if exe is None:
        raise RuntimeError('xtb program not found (PATH or $XTBBIN)')
    return exe


def _run(elem, x, task, charge=0, spin=0, gfn=2, nthreads=None, workdir=None):
    '''runs xtb on the geometry x (Bohr) in a scratch directory; task --grad or --hess'''
    d = tempfile.mkdtemp(prefix='xtb_', dir=workdir)
    try:
        with open(os.path.join(d, 'x.xyz'), 'w') as f:
            f.write('%d\n\n' % len(elem))
            for e, c in zip(elem, x.reshape(-1, 3) * BOHR):
                f.write('%s %.10f %.10f %.10f\n' % (e, *c))
        env = dict(os.environ)
        if nthreads:
            env['OMP_NUM_THREADS'] = str(nthreads)
        env.setdefault('OMP_STACKSIZE', '1G')
        cmd = [_xtb(), 'x.xyz', '--gfn', str(gfn), '--chrg', str(charge), '--uhf', str(spin), task]
        p = subprocess.run(cmd, cwd=d, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # xtb 6.5 may abort in its thermochemistry after writing the Hessian (linear molecules)
        out = {'--grad': 'gradient', '--hess': 'hessian'}[task]
        if p.returncode != 0 and not os.path.exists(os.path.join(d, out)):
            raise RuntimeError('xtb failed:\n' + p.stdout[-2000:])
        return d, p.stdout
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise


def _read_gradient(d, natm):
    lines = open(os.path.join(d, 'gradient')).read().split('\n')
    i = [k for k, l in enumerate(lines) if 'cycle' in l][-1]
    e = float(lines[i].split('energy =')[1].split()[0])
    g = [[float(v.replace('D', 'E')) for v in l.split()] for l in lines[i + 1 + natm:i + 1 + 2 * natm]]
    return e, numpy.array(g).ravel()


def _read_hessian(d, n):
    toks = open(os.path.join(d, 'hessian')).read().split()
    vals = [float(t) for t in toks if t[0] not in '$']
    return numpy.array(vals[:n * n]).reshape(n, n)


class XTBEngine:
    '''energy and gradient of GFN-xTB for Optimizer; hessian(x) the xtb
    Cartesian Hessian (3N, 3N)'''

    def __init__(self, elements, x0, charge=0, spin=0, gfn=2, nthreads=None):
        self.elements = list(elements)
        self.x0 = numpy.asarray(x0, dtype=float).ravel()
        self.charge, self.spin, self.gfn, self.nthreads = charge, spin, gfn, nthreads
        self.ncalc = 0
        self.nhess = 0
        self.e_last = None

    def coords(self):
        return self.x0.copy()

    def __call__(self, x):
        d, _ = _run(self.elements, x, '--grad', self.charge, self.spin, self.gfn, self.nthreads)
        try:
            e, g = _read_gradient(d, len(self.elements))
        finally:
            shutil.rmtree(d, ignore_errors=True)
        self.ncalc += 1
        self.e_last = e
        return e, g

    def hessian(self, x):
        self.nhess += 1
        return xtb_hessian(self.elements, x, self.charge, self.spin, self.gfn, self.nthreads)


def xtb_hessian(elements, x, charge=0, spin=0, gfn=2, nthreads=None):
    '''GFN-xTB Cartesian Hessian (3N, 3N) at x (Bohr), Hartree / Bohr^2'''
    d, _ = _run(elements, x, '--hess', charge, spin, gfn, nthreads)
    try:
        return _read_hessian(d, 3 * len(elements))
    finally:
        shutil.rmtree(d, ignore_errors=True)
