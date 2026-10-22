'''PC-NEVPT2 vs ORCA 6.0.1 FIC-NEVPT2 (= PC-NEVPT2) with ORCA's CASSCF
orbitals (stored in data/orca; ORCA is not needed to run the tests).

ORCA settings: NoFrozenCore, TrafoStep Exact, D3TPre = D4TPre = 1e-30 (the
default CI truncation of the 3-/4-RDM intermediates changes caaa/aaav by up
to 4e-9).  ORCA removes spin-free PC functions whose metric eigenvalue is
below 1e-6 (fixed; NThresh/DThresh do not change it), which removes a few
physical functions of small norm in some classes; zmr keeps them (round-off
null space only).  Those classes are reproduced with the spin-adapted
reference at threshold 1e-6 (slow tests).

The non-relativistic orbitals are also expanded to spin-orbitals and rotated
by a random complex unitary transformation inside the core, active and
virtual spaces (alpha-beta mixing, complex phases); every class must be
unchanged.'''
import functools
import json
import os

import numpy
import pytest
from pyscf import gto, mcscf, scf
from pyscf.tools import molden
from scipy.stats import unitary_group

from socutils.zmr import fock
from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.interface import pyscf_nr
from socutils.zmr.pc.nevpt2 import PCNEVPT2
from socutils.zmr.reference import spin_adapted_pc
from socutils.zmr.uc.nevpt2 import UCNEVPT2

DATA = os.path.join(os.path.dirname(__file__), 'data', 'orca')
REF = json.load(open(os.path.join(DATA, 'reference.json')))
CLASSES = ('ccvv', 'ccav', 'cavv', 'ccaa', 'aavv', 'cava', 'caaa', 'aaav')
# classes where ORCA's fixed 1e-6 metric threshold removes physical functions
TRUNCATED = {'h2o_casscf44': ('aavv', 'cava'), 'n2_casscf66': ('caaa', 'aaav')}


def orca_case(name):
    ref = REF[name]
    mol = gto.M(atom=ref['atom'], basis=ref['basis'], verbose=0)
    _, _, mo, _, _, _ = molden.load(os.path.join(DATA, name + '.molden'))
    s = mol.intor('int1e_ovlp')
    # the molden basis section of ORCA is not read correctly by PySCF, but the
    # coefficients are those of PySCF's own cc-pVDZ (checked here)
    assert abs(mo.T @ s @ mo - numpy.eye(mo.shape[1])).max() < 1e-6
    w, v = numpy.linalg.eigh(mo.T @ s @ mo)
    mo = mo @ (v / numpy.sqrt(w)) @ v.T
    return mol, mo, ref


def _eris(mol, mo, ref):
    return pyscf_nr.mo_eris(mol, scf.hf.get_hcore(mol), mo, 0, ref['ncore'], ref['ncas'])


def _complex_eris(mol, mo, ref, seed=11):
    c = pyscf_nr._spin_orbital_coeff(mo)
    sizes = (2 * ref['ncore'], 2 * ref['ncas'], c.shape[1] - 2 * (ref['ncore'] + ref['ncas']))
    u = numpy.zeros((c.shape[1],) * 2, dtype=complex)
    off = 0
    for k, n in enumerate(sizes):
        u[off:off + n, off:off + n] = unitary_group.rvs(n, random_state=seed + k)
        off += n
    return pyscf_nr.ghf_eris(mol, scf.hf.get_hcore(mol), c @ u, 0, *sizes[:2])


@pytest.fixture(scope='module', params=['h2o_casscf44',
                                       pytest.param('n2_casscf66', marks=pytest.mark.slow)])
def single_state(request):
    mol, mo, ref = orca_case(request.param)
    eris = _eris(mol, mo, ref)
    pc = PCNEVPT2(eris, ref['nelecas'])
    pc.kernel()
    return request.param, mol, mo, ref, pc


def test_casscf_energy_reproduced(single_state):
    name, mol, mo, ref, pc = single_state
    assert abs(pc.hmodel[0, 0].real - ref['casscf_energy']) < 2e-8


def test_pc_matches_orca_fic(single_state):
    name, mol, mo, ref, pc = single_state
    for cls in CLASSES:
        e, e_orca = pc.h2_by_class[cls][0, 0].real, ref['fic_nevpt2'][cls]
        if cls in TRUNCATED[name]:
            # ORCA drops physical functions here: its energy lies above
            assert e < e_orca, cls
        else:
            assert abs(e - e_orca) < 1e-9, (cls, e - e_orca)


def test_complex_rotation_invariance(single_state):
    name, mol, mo, ref, pc = single_state
    eris_c = _complex_eris(mol, mo, ref)
    assert abs(eris_c.h1.imag).max() > 1.0
    pc_c = PCNEVPT2(eris_c, ref['nelecas'])
    pc_c.kernel()
    uc = UCNEVPT2(_eris(mol, mo, ref), ref['nelecas'])
    uc_c = UCNEVPT2(eris_c, ref['nelecas'])
    uc.kernel()
    uc_c.kernel()
    for cls in CLASSES:
        assert abs(pc_c.h2_by_class[cls][0, 0] - pc.h2_by_class[cls][0, 0]) < 1e-12, cls
        assert abs(uc_c.h2_by_class[cls][0, 0] - uc.h2_by_class[cls][0, 0]) < 1e-12, cls


def test_state_averaged_roots():
    '''ORCA's per-root FIC-NEVPT2 after SA-CASSCF uses each root's own Fock
    matrix: zmr single-state runs on the SA orbitals reproduce it (roots 1, 2
    in every class; root 0 except where the 1e-6 metric threshold acts).'''
    mol, mo, ref = orca_case('h2o_sa3_casscf44')
    eris = _eris(mol, mo, ref)
    sec = ActiveSpace.from_eris(eris).sector(ref['nelecas'])
    e_core = ActiveSpace.from_eris(eris).e_core
    for root, (e_cas, orca) in enumerate(zip(ref['casscf_states'], ref['fic_nevpt2_roots'])):
        k = int(numpy.argmin(abs(sec.lam + e_core - e_cas)))
        assert abs(sec.lam[k] + e_core - e_cas) < 1e-8
        pc = PCNEVPT2(eris, ref['nelecas'], civecs=sec.vecs[:, k])
        pc.kernel()
        for cls in CLASSES:
            e = pc.h2_by_class[cls][0, 0].real
            if root == 0 and cls in ('ccaa', 'aavv', 'cava'):
                assert e < orca[cls], (root, cls)
            else:
                assert abs(e - orca[cls]) < 1e-9, (root, cls, e - orca[cls])


@functools.lru_cache(maxsize=None)
def _spin_adapted(name, cls, civec_index=0):
    mol, mo, ref = orca_case(name)
    mc = mcscf.CASCI(scf.RHF(mol), ref['ncas'], ref['nelecas'])
    mc.verbose = 0
    mc.kernel(mo)
    # canonicalize spatial orbitals so that no alpha-beta mixing occurs later
    mo_can, _, _ = mc.canonicalize(mo, cas_natorb=False)
    eris, nelec = pyscf_nr.from_casci(mc, mo_coeff=mo_can, lazy=False)
    act = ActiveSpace.from_eris(eris)
    sec = act.sector(nelec)
    civec = sec.vecs[:, civec_index]
    f = fock.generalized_fock(eris, act.averaged_rdm1(civec, nelec))
    c, v = eris.spaces.slice('c'), eris.spaces.slice('v')
    res = spin_adapted_pc.class_energy(eris, civec, nelec, numpy.diag(f[c, c]).real,
                                       numpy.diag(f[v, v]).real, cls, (1e-12, 1e-6))
    pc = PCNEVPT2(eris, nelec, civecs=civec)
    pc.kernel()
    return res, pc.h2_by_class[cls][0, 0].real


@pytest.mark.slow
@pytest.mark.parametrize('cls, skipped', [('cava', 612), ('aavv', None)])
def test_orca_threshold_reproduced(cls, skipped):
    '''Spin-adapted functions: removing only the null space gives zmr's
    energy, removing metric eigenvalues below 1e-6 gives ORCA's.'''
    ((t0, e0, _, _), (t1, e1, n1, _)), zmr = _spin_adapted('h2o_casscf44', cls)
    assert abs(e0 - zmr) < 1e-10
    assert abs(e1 - REF['h2o_casscf44']['fic_nevpt2'][cls]) < 1e-9
    if skipped is not None:
        assert n1 == skipped
