"""Atomic SCF engine: primitives by angular momentum -> energy, spinor energies, splittings.

The work is done by ``x2camf_atom_scf`` of the bundled pure-C X2CAMF library
(lib/libx2camf_c.so: spherical atomic four-component / X2C-1e Hartree-Fock).  Everything
here is in atomic units.
"""
import contextlib
import importlib.util
import math
import os
import tempfile

import numpy as np

_backend = None


def backend():
    """The ctypes binding of the bundled C library.  It is loaded by path: the top-level
    ``x2camf`` package may be an external C++ build without the atom_scf entry point."""
    global _backend
    if _backend is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
                            'x2camf_c', 'x2camf', '_c_backend.py')
        spec = importlib.util.spec_from_file_location('socutils_x2camf_c_backend',
                                                      os.path.abspath(path))
        _backend = importlib.util.module_from_spec(spec)
        with _quiet():
            spec.loader.exec_module(_backend)
    return _backend

LIGHT_SPEED = 137.03599967994   # the value of pyscf.lib.param.LIGHT_SPEED

ELEMENTS = ('X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn '
            'Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce '
            'Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn '
            'Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl '
            'Mc Lv Ts Og').split()

# name -> (scf_type, spin_free, gaunt, gauge); see x2camf_atom_scf in lib/x2camf/c_api
HAMILTONIANS = {
    'dc': (0, False, False, False),      # 4c Dirac-Coulomb
    'dcg': (0, False, True, False),      # 4c Dirac-Coulomb-Gaunt
    'dcb': (0, False, True, True),       # 4c Dirac-Coulomb-Breit
    'sfdc': (0, True, False, False),     # 4c spin-free Dirac-Coulomb
    'sfx2c1e': (1, True, False, False),  # spin-free X2C-1e HF
    'x2c1e': (1, False, False, False),   # X2C-1e HF, bare (one-electron) spin-orbit
    'x2camf': (2, False, False, False),  # X2C-1e HF + AMF spin-orbit of 4c DC
    'x2camf-g': (2, False, True, False),
    'x2camf-b': (2, False, True, True),
}


def charge(element):
    if isinstance(element, (int, np.integer)):
        return int(element)
    return ELEMENTS.index(element.capitalize())


def _libc_flush():
    import ctypes
    try:
        ctypes.CDLL(None).fflush(None)
    except Exception:
        pass


@contextlib.contextmanager
def _quiet(enabled=True):
    """Silence the C++ stdout of the library."""
    if not enabled:
        yield
        return
    import sys
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        _libc_flush()            # the C library prints through buffered stdio
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


class AtomResult:
    """Result of one atomic SCF.

    blocks : list of dicts, one per kappa (order s1/2, p1/2, p3/2, d3/2, ...), with keys
        l, two_j, exp (primitive exponents), e (spinor energies, ascending), c (coefficients;
        rows: large then small primitives for 4c, large only for X2C), occ (occupation per
        m_j of each spinor), ovlp, fock, h1e, den.
    """

    def __init__(self, element, prims, hamiltonian, raw):
        self.element = element
        self.prims = prims
        self.hamiltonian = hamiltonian
        self.e_scf = raw['e_scf']
        self.converged = raw['converged']
        self.blocks = raw['blocks']

    def block(self, l, two_j):
        for b in self.blocks:
            if b['l'] == l and b['two_j'] == two_j:
                return b
        raise KeyError((l, two_j))

    def eps(self, n, l, two_j):
        """Spinor energy of principal quantum number n (n = l+1 is the lowest of its l)."""
        return self.block(l, two_j)['e'][n - l - 1]

    def so_splitting(self, n, l):
        """eps(n l j=l+1/2) - eps(n l j=l-1/2)."""
        return self.eps(n, l, 2 * l + 1) - self.eps(n, l, 2 * l - 1)

    def occupied(self):
        """[(n, l, two_j, occupation per m_j, eps)] of the occupied spinors."""
        out = []
        for b in self.blocks:
            for i, f in enumerate(b['occ']):
                if abs(f) > 1e-10:
                    out.append((i + b['l'] + 1, b['l'], b['two_j'], f, b['e'][i]))
        return out

    def r_power(self, n, l, two_j, k):
        """<r^k> over the large component (renormalised) of a spinor; the X2C orbital itself
        for the two-component Hamiltonians.  Diverges for k <= -(2l+3)."""
        b = self.block(l, two_j)
        nb = len(b['exp'])
        c = b['c'][:nb, n - l - 1]
        m = radial_power_matrix(l, b['exp'], k)
        return c @ m @ c / (c @ radial_power_matrix(l, b['exp'], 0) @ c)


def radial_power_matrix(l, exps, k):
    """<g_a| r^k |g_b> for normalised radial Gaussians r^l exp(-a r^2)."""
    a = np.asarray(exps, dtype=float)

    def integral(p, s):     # int_0^inf r^p exp(-s r^2) dr
        return 0.5 * math.gamma((p + 1) / 2) / s ** ((p + 1) / 2)
    norm = np.array([1 / math.sqrt(integral(2 * l + 2, 2 * x)) for x in a])
    s = a[:, None] + a[None, :]
    return norm[:, None] * norm[None, :] * integral(2 * l + 2 + k, s)


def write_occ_file(path, element, occ):
    """occ: electrons per kappa block in the order s, p1/2, p3/2, d3/2, d5/2, f5/2, f7/2."""
    with open(path, 'w') as f:
        f.write('%%occAMFI_%s\n%d\n' % (ELEMENTS[charge(element)].upper(), len(occ)))
        f.write('\n'.join('%.15g' % x for x in occ) + '\n')


def flatten(prims):
    """{l: exponents} -> (shell, exp) arrays grouped by l as the library requires, lmax+1."""
    lmax = max(l for l in prims if len(prims[l]))
    shell, exps = [], []
    for l in range(lmax + 1):
        e = sorted((float(x) for x in prims.get(l, ())), reverse=True)
        if not e:
            raise ValueError('no primitives for l = %d: every l up to lmax needs some' % l)
        shell += [l] * len(e)
        exps += e
    return np.array(shell, dtype=np.int32), np.array(exps), lmax + 1


NUCLEAR_MODELS = {'fermi': 1, 'gaussian': 2, 'sphere': 3}
FM = 1.0 / 52917.721067      # bohr per fm (CODATA 2014, as GRASP2018)


def fermi_from_isodata(path):
    """('fermi', c, a) in bohr from a GRASP isodata file (None for a point nucleus)."""
    lines = open(path).read().splitlines()
    val = {lines[i].strip(): float(lines[i + 1]) for i in range(0, len(lines) - 1, 2)}
    a, c = val['Fermi distribution parameter a:'], val['Fermi distribution parameter c:']
    return ('fermi', c * FM, a * FM) if c > 0 else None


def atom(element, prims, hamiltonian='dc', occ=None, aoc=False, gau_nuc=False,
         c=LIGHT_SPEED, conv=1e-9, max_iter=100, verbose=0, nucleus=None):
    """Atomic SCF of `element` in the uncontracted basis `prims` = {l: [exponents]}.

    hamiltonian : key of HAMILTONIANS.
    occ : electrons per kappa block (s, p1/2, p3/2, d3/2, ...); None for the built-in
        ground-state table (open shells: electrons spread evenly over the l shell).
    aoc : average-of-configuration SCF for the open shell instead of the plain
        fractional-occupation SCF.
    gau_nuc : the built-in Gaussian nucleus (analytic integrals) instead of a point charge.
    nucleus : ('fermi', c, a), ('gaussian', exponent) or ('sphere', radius), lengths in
        bohr: nuclear models integrated numerically (exclusive with gau_nuc).
    """
    scf_type, spin_free, gaunt, gauge = HAMILTONIANS[hamiltonian]
    shell, exps, nshell = flatten(prims)
    with tempfile.TemporaryDirectory() as tmp:
        # the library looks for "%occAMFI_<EL>" in this file and falls back to its table
        occ_file = os.path.join(tmp, 'occ')
        if occ is not None:
            write_occ_file(occ_file, element, occ)
        with _quiet(verbose == 0):
            raw = backend().atom_scf(charge(element), nshell, len(shell), verbose, shell, exps,
                                     speed_of_light=c, scf_type=scf_type, spin_free=spin_free,
                                     gaunt=gaunt, gauge=gauge, gau_nuc=gau_nuc, aoc=aoc,
                                     occ_file=occ_file, conv=conv, max_iter=max_iter,
                                     nuc_model=NUCLEAR_MODELS[nucleus[0]] if nucleus else 0,
                                     nuc_p1=nucleus[1] if nucleus else 0.0,
                                     nuc_p2=nucleus[2] if nucleus and len(nucleus) > 2 else 0.0)
    return AtomResult(element, {l: list(exps[shell == l]) for l in range(nshell)},
                      hamiltonian, raw)


def exponent_gradient(element, prims, spin_free=False, gau_nuc=False, occ=None,
                      c=LIGHT_SPEED, conv=1e-9, max_iter=100, nucleus=None, step=1e-3,
                      verbose=0):
    """4c Dirac-Coulomb atomic SCF (fractional occupations) and dE/d ln(alpha) of every
    primitive: returns (e_scf, {l: [gradient per exponent, as sorted in flatten()]}).
    One SCF plus six integral evaluations, whatever the number of primitives."""
    shell, exps, nshell = flatten(prims)
    with tempfile.TemporaryDirectory() as tmp:
        occ_file = os.path.join(tmp, 'occ')
        if occ is not None:
            write_occ_file(occ_file, element, occ)
        with _quiet(verbose == 0):
            e, g = backend().atom_exp_grad(
                charge(element), nshell, len(shell), verbose, shell, exps, speed_of_light=c,
                spin_free=spin_free, gau_nuc=gau_nuc, occ_file=occ_file, conv=conv,
                max_iter=max_iter, nuc_model=NUCLEAR_MODELS[nucleus[0]] if nucleus else 0,
                nuc_p1=nucleus[1] if nucleus else 0.0,
                nuc_p2=nucleus[2] if nucleus and len(nucleus) > 2 else 0.0, step=step)
    return e, {l: list(g[shell == l]) for l in range(nshell)}


def prims_from_pyscf(element, basis):
    """Primitive exponents {l: [...]} of a PySCF basis-set name (duplicates removed)."""
    from pyscf import gto
    prims = {}
    for sh in gto.uncontract(gto.basis.load(basis, element)):
        prims.setdefault(sh[0], set()).add(sh[1][0])
    return {l: sorted(v, reverse=True) for l, v in prims.items()}

