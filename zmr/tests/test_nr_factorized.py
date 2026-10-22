'''Spin-restricted code with three-index integrals: the real spin-free
kernels of the doubly external classes, the blocks without four-index
integrals, and the core-valence separation, against the dense spin-restricted
code and the spin-orbital code.'''
import numpy
import pytest
from pyscf import gto, mcscf, scf

from socutils.zmr.interface import pyscf_nr
from socutils.zmr.lib import kernels
from socutils.zmr.nr import PCNEVPT2, UCNEVPT2
from socutils.zmr.nr import classes as nr_classes
from socutils.zmr.nr.active import ActiveSpace
from socutils.zmr.pc.nevpt2 import PCNEVPT2 as SOPCNEVPT2
from socutils.zmr.tests.test_cholesky import factorize
from socutils.zmr.uc import classes, engine


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASSCF(mf, 4, 4)
    mc.conv_tol = 1e-11
    mc.kernel()
    return mc, factorize(mol)


@pytest.mark.parametrize('mode', [-1, 0, 1])
def test_spin_free_kernels(water, mode):
    '''W of every doubly external spin-free class from the three-index
    integrals against the sum over its listed labels (exact integrals).'''
    mc, cderi = water
    dense, _ = pyscf_nr.nr_from_casci(mc)
    eris, _ = pyscf_nr.nr_from_casci(mc, cderi=cderi)
    rng = numpy.random.default_rng(3)
    sp = eris.spaces
    eps_c = numpy.sort(rng.normal(size=sp.ncore)) - 3
    eps_v = numpy.sort(rng.normal(size=sp.nvirt)) + 3
    fcore = dense.generalized_fock(numpy.zeros((sp.nact, sp.nact)))
    ctx = classes.Context(eris, eps_c, eps_v, fcore)
    ref_ctx = classes.Context(dense, eps_c, eps_v, fcore)
    d = numpy.array([-0.3, 0.0, 0.7, 2.5])
    bvc, bva = eris.cd_block('vc'), eris.cd_block('va')
    assert bvc.dtype == numpy.float64 and bva.dtype == numpy.float64
    if kernels.wsf_ccvv(bvc, eps_v, eps_c, d, mode) is None:
        pytest.skip('libzmr not built')
    got = {nr_classes.CCVV: kernels.wsf_ccvv(bvc, eps_v, eps_c, d, mode),
           nr_classes.CAVV: kernels.wsf_cavv(bvc, bva, eps_v, eps_c, d, mode),
           nr_classes.AAVV: kernels.wsf_aavv(bva, eps_v, d)}
    for cls, w in got.items():
        pclass = cls()
        ext = engine.class_blocks(pclass, ctx)
        assert isinstance(ext, classes.ExternalBlocks) and ext._list is None
        assert ext.dtype == numpy.float64
        blocks = list(pclass.blocks(ref_ctx))
        delta = numpy.concatenate([x for x, _ in blocks])
        v = numpy.concatenate([x for _, x in blocks], axis=0)
        assert ext.nlabel == len(delta)
        assert abs(ext.delta_min - delta.min()) < 1e-12
        ref = numpy.array([v.T @ (v / (x + delta)[:, None]) for x in d])
        assert abs(w.reshape(ref.shape) - ref).max() < 1e-10 * abs(ref).max(), cls.__name__
        assert abs(ext.w_nodes(d).reshape(ref.shape) - ref).max() < 1e-10 * abs(ref).max()


def test_blocks_without_four_index_integrals(water):
    '''Every class lists the same labels from three-index integrals.'''
    mc, cderi = water
    dense, _ = pyscf_nr.nr_from_casci(mc, nfrozen=1)
    eris, _ = pyscf_nr.nr_from_casci(mc, nfrozen=1, cderi=cderi)
    sp = eris.spaces
    rng = numpy.random.default_rng(4)
    eps_c = numpy.sort(rng.normal(size=sp.ncore)) - 3
    eps_v = numpy.sort(rng.normal(size=sp.nvirt)) + 3
    fcore = dense.generalized_fock(numpy.zeros((sp.nact, sp.nact)))
    for pclass in nr_classes.ALL_CLASSES:
        a = list(pclass.blocks(classes.Context(dense, eps_c, eps_v, fcore)))
        b = list(pclass.blocks(classes.Context(eris, eps_c, eps_v, fcore)))
        assert len(a) == len(b)
        for (d1, v1), (d2, v2) in zip(a, b):
            assert abs(d1 - d2).max() < 1e-12
            assert abs(v1 - v2).max() < 1e-9, type(pclass).__name__


@pytest.mark.parametrize('method, options', [
    (PCNEVPT2, dict(contraction='union')), (PCNEVPT2, dict(contraction='state')), (UCNEVPT2, {})])
@pytest.mark.parametrize('nstate', [1, 3])
def test_energies_match_the_dense_code(water, method, options, nstate, monkeypatch):
    mc, cderi = water
    dense, nelec = pyscf_nr.nr_from_casci(mc, nfrozen=1)
    eris, _ = pyscf_nr.nr_from_casci(mc, nfrozen=1, cderi=cderi)
    civecs = ActiveSpace.from_eris(dense).solve(nelec, nstate)[1]
    used = []
    direct = engine.direct_contract_t
    monkeypatch.setattr(engine, 'direct_contract_t',
                        lambda *args: used.append(direct(*args)) or used[-1])
    out = []
    for h in (dense, eris):
        pt = method(h, nelec, civecs=civecs, **options)
        pt.verbose = 0
        pt.kernel()
        out.append(pt)
    assert out[1].heff.dtype == numpy.float64
    assert abs(out[1].heff - out[0].heff).max() < 1e-9
    for name, a in out[0].h2_by_class.items():
        assert abs(out[1].h2_by_class[name] - a).max() < 1e-9, name
    if options.get('contraction') != 'state' and kernels.wsf_aavv(
            eris.cd_block('va'), numpy.ones(eris.spaces.nvirt), numpy.zeros(1)) is not None:
        # the external sums came from the kernels
        assert sum(x is not None for x in used) >= 3


def _core_hole(water, active, cderi=None):
    '''Spatial orbital `active` empty (two holes), the other occupied ones core.'''
    mc, _ = water
    mo = mc._scf.mo_coeff
    occupied = [i for i in range(5) if i != active]
    order = occupied + [active] + list(range(5, mo.shape[1]))
    args = (mc.mol, mc.get_hcore(), mo[:, order], 0, len(occupied), 1)
    return pyscf_nr.nr_eris(*args, cderi=cderi), pyscf_nr.mo_eris(*args)


@pytest.mark.parametrize('factorized', [False, True])
def test_core_valence_separation_as_in_the_spin_orbital_code(water, factorized):
    '''Two holes in O 2s (a closed-shell state): the spin-free and the
    spin-orbital contracted spaces coincide, so CVS energies, decay channels
    and the inner core agree.'''
    nr, so = _core_hole(water, 1, water[1] if factorized else None)
    ref = SOPCNEVPT2(so, 0, nstate=1)
    ref.verbose = 0
    ref.cvs = True
    ref.kernel()
    pt = PCNEVPT2(nr, (0, 0), nstate=1)
    pt.verbose = 0
    pt.cvs = True
    pt.kernel()
    assert list(pt.cvs_core) == [0]                     # O 1s
    assert abs(pt.e_tot - ref.e_tot).max() < 1e-8
    assert abs(pt.e_tot_decay - ref.e_tot_decay).max() < 1e-8
    for name in ('ccaa', 'ccav', 'caaa'):
        assert abs(pt.h2_by_class[name] - ref.h2_by_class[name]).max() < 1e-8, name
        assert abs(pt.h2_decay_by_class[name] - ref.h2_decay_by_class[name]).max() < 1e-8
    full = PCNEVPT2(nr, (0, 0), nstate=1)
    full.verbose = 0
    with pytest.warns(UserWarning):
        full.kernel()
    assert abs(pt.heff_decay - full.heff).max() < 1e-9 or pt.decay_window[1] > 0
