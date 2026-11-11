'''zmr.uc.density: the unrelaxed ccvv density of NEVPT2 against finite fields (frozen
orbitals and model states), the native kernel against the Python one, and the Kramers
pairs against all core orbitals.'''
import copy
import os
import sys

import numpy
import pytest
from pyscf import gto

sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture(scope='module')
def hbr():
    from pyscf.x2c import x2c
    from socutils.scf import double_group_symm as dgs
    from socutils.symm import spinor_basis as sb
    from socutils.zmr.interface import socutils_x2c
    from socutils.zmr.pc.nevpt2 import PCNEVPT2
    from test_cholesky import factorize
    mol = sb.to_symm_frame(gto.M(atom='H 0 0 0; Br 0 0 1.4144', basis='6-31g',
                                 symmetry=True, verbose=0))
    mf = dgs.SymAdaptedRHF(mol)
    mf.with_x2c = x2c.SpinorX2CHelper(mol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.kernel()
    eris = socutils_x2c.from_scf(mf, ncore=mol.nelectron - 2, nact=4, cderi=factorize(mol))
    pt = PCNEVPT2(eris, 2, nstate=1)
    pt.verbose = 0
    pt.max_memory = 8000
    pt.kernel()
    return mol, pt


def _ctx(pt):
    from socutils.zmr.uc import classes
    qc = pt.quasi_canonical
    return classes.Context(qc.eris, qc.eps_core, qc.eps_virt, None)


def test_ccvv_kernels_and_kramers(hbr):
    from socutils.zmr.uc import density
    _, pt = hbr
    ctx = _ctx(pt)
    assert ctx.eris.time_reversal is not None
    ref = density.ccvv_reference(ctx)
    for kr in (False, True):
        e, dcc, dvv = density.ccvv(ctx, kramers=kr)
        assert abs(e - ref[0]) < 1e-11, kr
        assert abs(dcc - ref[1]).max() < 1e-11, kr
        assert abs(dvv - ref[2]).max() < 1e-11, kr
    assert abs(ref[0] - pt.h2_by_class['ccvv'][0, 0].real) < 1e-11
    for d in ref[1:]:
        assert abs(d - d.conj().T).max() < 1e-12


def test_ccvv_finite_field(hbr):
    '''dE_ccvv / dlambda for h -> h + lambda O (O in the core-core or virtual-virtual
    block of the quasi-canonical orbitals) against sum O * D.'''
    from socutils.zmr.pc.nevpt2 import PCNEVPT2
    from socutils.zmr.uc import density
    mol, pt = hbr
    qc = pt.quasi_canonical
    sp = qc.eris.spaces
    _, dcc, dvv = density.ccvv(_ctx(pt))
    c = qc.eris.mo_coeff
    x = mol.intor('int1e_ovlp_spinor') @ c
    rng = numpy.random.default_rng(7)

    def energy(o, lam):
        e2 = copy.copy(qc.eris)
        e2.hcore_ao = qc.eris.hcore_ao + lam * (x @ o @ x.conj().T)
        e2.h1 = c.conj().T @ e2.hcore_ao @ c
        p = PCNEVPT2(e2, 2, civecs=pt.civecs)
        p.verbose = 0
        p.max_memory = 8000
        p.kernel()
        return p.h2_by_class['ccvv'][0, 0].real

    for label, d in (('v', dvv), ('c', dcc)):
        sl = sp.slice(label)
        n = sl.stop - sl.start
        y = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        o = numpy.zeros((sp.nmo, sp.nmo), dtype=complex)
        o[sl, sl] = 0.5 * (y + y.conj().T)
        fd = [(energy(o, h) - energy(o, -h)) / (2 * h) for h in (2e-4, 1e-4)]
        # Richardson: the central difference has an error of order h^2
        fd = (4 * fd[1] - fd[0]) / 3
        analytic = float(numpy.sum(o[sl, sl] * d).real)
        assert abs(fd - analytic) < 1e-7 * max(1.0, abs(analytic)) + 1e-10, label


@pytest.mark.parametrize('name', ['cavv', 'ccav'])
def test_active_classes_finite_field(hbr, name):
    '''cavv and ccav (Python reference): energy against NEVPT2, and the core-core,
    active-active and virtual-virtual blocks of the density against finite fields.'''
    from socutils.zmr.pc.nevpt2 import PCNEVPT2
    from socutils.zmr.uc import density
    mol, pt = hbr
    qc = pt.quasi_canonical
    sp = qc.eris.spaces
    e, dcc, daa, dvv = getattr(density, name + '_reference')(pt, _ctx(pt))
    assert abs(e - pt.h2_by_class[name][0, 0].real) < 1e-10
    c = qc.eris.mo_coeff
    x = mol.intor('int1e_ovlp_spinor') @ c
    rng = numpy.random.default_rng(11)

    def energy(o, lam):
        e2 = copy.copy(qc.eris)
        e2.hcore_ao = qc.eris.hcore_ao + lam * (x @ o @ x.conj().T)
        e2.h1 = c.conj().T @ e2.hcore_ao @ c
        p = PCNEVPT2(e2, 2, civecs=pt.civecs)
        p.verbose = 0
        p.max_memory = 8000
        p.kernel()
        return p.h2_by_class[name][0, 0].real

    for label, d in (('c', dcc), ('a', daa), ('v', dvv)):
        sl = sp.slice(label)
        n = sl.stop - sl.start
        y = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        o = numpy.zeros((sp.nmo, sp.nmo), dtype=complex)
        o[sl, sl] = 0.5 * (y + y.conj().T)
        fd = [(energy(o, h) - energy(o, -h)) / (2 * h) for h in (2e-4, 1e-4)]
        fd = (4 * fd[1] - fd[0]) / 3
        analytic = float(numpy.sum(o[sl, sl] * d).real)
        assert abs(fd - analytic) < 1e-6 * max(1.0, abs(analytic)) + 1e-9, (label, fd, analytic)
