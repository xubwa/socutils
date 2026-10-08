"""j-adapted (spinor) basis sets as PySCF basis names.

PySCF's basis files are NWChem format, which cannot carry kappa.  The sets live here as Python
modules in the form PySCF uses for its own Dyall sets: one variable per element holding the
internal-format list with kappa, `Th = [[l, kappa, [exp, c, ...], ...], ...]`.  register()
makes every such module of this directory (and of the directories in SOCUTILS_BASIS_PATH)
known to PySCF under its file name, so that

    gto.M(atom=..., basis={'Th': 'dyvtz-sog', 'O': 'cc-pvtz'})

works; `import socutils` calls it.  Nothing in the PySCF installation is modified.

A j-adapted set is meant for spinor calculations only.  In a scalar (real spherical or
Cartesian AO) calculation both contractions of every shell enter as separate functions: the
basis is larger than intended and nearly linearly dependent.  register() therefore also makes
every SCF object warn when it is built in a scalar AO basis for a molecule with such a set.
"""
import os
import sys
import glob
import warnings
import importlib.util

import numpy
from pyscf import gto
from pyscf.gto import basis as _pyscf_basis
from pyscf.scf import hf as _hf

_PACKAGE = _pyscf_basis.__name__            # 'pyscf.gto.basis'
registered = {}                             # PySCF alias key -> path of the module


class ScalarWithSpinorBasisWarning(UserWarning):
    pass


def register(*paths):
    """Register the basis modules found in `paths` (files or directories; default: this
    directory and those in the environment variable SOCUTILS_BASIS_PATH).  Returns the names."""
    if not paths:
        paths = [os.path.dirname(os.path.abspath(__file__))]
        paths += [p for p in os.environ.get('SOCUTILS_BASIS_PATH', '').split(os.pathsep) if p]
    names = []
    for p in paths:
        p = os.path.expanduser(p)
        for f in sorted(glob.glob(os.path.join(p, '*.py'))) if os.path.isdir(p) else [p]:
            stem = os.path.splitext(os.path.basename(f))[0]
            if stem.startswith('_'):
                continue
            key = _pyscf_basis._format_basis_name(stem)
            modname = 'socutils_' + stem.replace('-', '_')
            spec = importlib.util.spec_from_file_location('%s.%s' % (_PACKAGE, modname), f)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules[spec.name] = mod
            _pyscf_basis.ALIAS[key] = modname
            registered[key] = f
            names.append(stem)
    _install_scalar_warning()
    return names


def has_kappa(mol):
    """True if the molecule carries functions restricted to one kappa (a j-adapted set)."""
    return bool(numpy.any(mol._bas[:, gto.KAPPA_OF] != 0))


def _scalar_ao(mf):
    """True if the SCF object works in the scalar AO basis of its molecule (or two copies of
    it, GHF) rather than in the spinor basis."""
    mol = mf.mol
    try:
        s = numpy.asarray(mf.get_ovlp())
    except Exception:
        return False
    if s.ndim != 2:
        return False
    n, nao = s.shape[0], mol.nao_nr()
    if n not in (nao, 2 * nao):
        return False
    return bool(numpy.allclose(s[:nao, :nao], mol.intor_symmetric('int1e_ovlp'), atol=1e-8))


def _install_scalar_warning():
    if getattr(_hf.SCF.build, '_socutils_spinor_basis_check', False):
        return
    build = _hf.SCF.build

    def checked_build(self, mol=None):
        out = build(self, mol)
        m = self.mol if mol is None else mol
        if isinstance(m, gto.Mole) and has_kappa(m) and _scalar_ao(self):
            warnings.warn(
                '%s is a scalar (non-spinor) calculation, but the basis of this molecule is '
                'j-adapted (contractions per kappa).  Both contractions of every shell enter as '
                'separate functions here: %d AOs, nearly linearly dependent.  Use a spinor method, '
                'or the scalar counterpart of the basis set.'
                % (self.__class__.__name__, m.nao_nr()), ScalarWithSpinorBasisWarning, stacklevel=2)
        return out

    checked_build._socutils_spinor_basis_check = True
    _hf.SCF.build = checked_build
