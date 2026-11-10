#
# Fast J / K for two-component (spinor / GHF) mean fields through the real
# scalar-AO builds of socutils.scf.md_j / jk_multi.
#
'''J and K of two-component densities from real scalar-AO builds.

The molecular two-electron integrals of X2C / X2CAMF are spin free, so with
the GHF blocks D_st (s, t in a, b) of a spinor density

    J_aa = J_bb = J[Re (D_aa + D_bb)],    K_st = K[Re D_st] + i K[Im D_st],

and for a Hermitian density K_ba = K_ab^H.  Re D_aa, Re D_bb are symmetric
(one hermi=1 build that also gives J), Im D_aa, Im D_bb anti-symmetric and
Re / Im D_ab general (one hermi=0 K-only build); for a time-reversal-even
(Kramers closed-shell) density D_bb = D_aa^*, so only Re D_aa, Im D_aa, Re /
Im D_ab are built.  The real builds go through a real RHF proxy of the same
molecule with socutils.scf.md_j applied (MD J engine / generated J/K kernels
/ in-core ERIs when they fit), so improvements there carry over.

    from socutils.scf import spinor_jk
    vj, vk = spinor_jk.get_jk_ghf(mf, mol, dm_ghf, hermi=1)       # (.., 2nao, 2nao)
'''
import numpy
from pyscf import lib, scf
from pyscf.lib import logger

ENABLE = True       # spinor_hf.SCF / socutils GHF route their J / K here (else pyscf's GHF get_jk)
KRAMERS_TOL = 1e-10


def available(mol):
    from socutils.scf import jk_multi
    return not mol.cart and jk_multi.available(mol)


def _proxy(mf, mol):
    '''a real RHF of mol with md_j applied, cached on mf (one per molecule)'''
    p = getattr(mf, '_spinor_jk_proxy', None)
    if p is None or p.mol is not mol:
        from socutils.scf import md_j
        p = scf.RHF(mol)
        p.verbose = 0
        p.stdout = mf.stdout
        md_j.apply(p)
        mf._spinor_jk_proxy = p
    p.max_memory = mf.max_memory
    p.conv_tol = getattr(mf, 'conv_tol', 1e-9)
    return p


def _is_kramers_even(aa, ab, ba, bb, tol=KRAMERS_TOL):
    scale = max(abs(aa).max(), abs(ab).max(), 1e-300)
    return (abs(bb - aa.conj()).max() < tol * scale and abs(ba + ab.conj()).max() < tol * scale)


def get_jk_ghf(mf, mol, dm, hermi=1, with_j=True, with_k=True):
    '''J, K of GHF-form densities (..., 2 nao, 2 nao) (complex or real) in the
    scalar-AO spin-block basis; hermi=1 asserts Hermitian densities.'''
    t0 = (logger.process_clock(), logger.perf_counter())
    dm = numpy.asarray(dm)
    nao = mol.nao_nr()
    shape = dm.shape
    dms = dm.reshape(-1, 2 * nao, 2 * nao)
    n = dms.shape[0]
    aa, ab = dms[:, :nao, :nao], dms[:, :nao, nao:]
    ba, bb = dms[:, nao:, :nao], dms[:, nao:, nao:]
    p = _proxy(mf, mol)
    cplx = numpy.iscomplexobj(dms)
    kramers = hermi == 1 and _is_kramers_even(aa, ab, ba, bb)

    # symmetric real parts: J and K in one hermi=1 build (hermi=0 densities: general builds)
    sym = [aa.real] if kramers else [aa.real, bb.real]
    gen = []
    if with_k:
        if cplx:
            gen += [aa.imag] if kramers else [aa.imag, bb.imag]
            gen += [ab.real, ab.imag]
            if hermi != 1:
                gen += [ba.real, ba.imag]
        else:
            gen += [ab] if hermi == 1 else [ab, ba]
    vj = vk = None
    nsym = len(sym)
    S = numpy.ascontiguousarray(numpy.concatenate(sym))
    js, ks = p.get_jk(mol, S, 1 if hermi == 1 else 0, with_j, with_k)
    if with_j:
        js = numpy.asarray(js).reshape(nsym, n, nao, nao)
        J = 2 * js[0] if kramers else js[0] + js[1]
        if cplx and hermi != 1:
            # a non-Hermitian density: Im (D_aa + D_bb) need not be anti-symmetric, J of it is not zero
            ji = p.get_jk(mol, numpy.ascontiguousarray(aa.imag + bb.imag), 0, True, False)[0]
            J = J + 1j * numpy.asarray(ji).reshape(n, nao, nao)
        vj = numpy.zeros((n, 2 * nao, 2 * nao), dtype=dms.dtype)
        vj[:, :nao, :nao] = vj[:, nao:, nao:] = J
    if with_k:
        ks = numpy.asarray(ks).reshape(nsym, n, nao, nao)
        G = numpy.ascontiguousarray(numpy.concatenate(gen))
        kg = numpy.asarray(p.get_jk(mol, G, 0, False, True)[1]).reshape(len(gen), n, nao, nao)
        vk = numpy.zeros((n, 2 * nao, 2 * nao), dtype=dms.dtype)
        if cplx:
            if kramers:
                kaa = ks[0] + 1j * kg[0]
                vk[:, :nao, :nao] = kaa
                vk[:, nao:, nao:] = kaa.conj()
                i = 1
            else:
                vk[:, :nao, :nao] = ks[0] + 1j * kg[0]
                vk[:, nao:, nao:] = ks[1] + 1j * kg[1]
                i = 2
            kab = kg[i] + 1j * kg[i + 1]
            vk[:, :nao, nao:] = kab
            if hermi == 1:
                vk[:, nao:, :nao] = kab.conj().transpose(0, 2, 1)
            else:
                vk[:, nao:, :nao] = kg[i + 2] + 1j * kg[i + 3]
        else:
            vk[:, :nao, :nao] = ks[0]
            vk[:, nao:, nao:] = ks[0] if kramers else ks[1]
            vk[:, :nao, nao:] = kg[0]
            vk[:, nao:, :nao] = kg[0].T if hermi == 1 else kg[1]
    logger.timer_debug1(mf, 'two-component J/K by real scalar builds (%d sym, %d general)' % (
        nsym * n, len(gen) * n), *t0)
    if vj is not None:
        vj = vj.reshape(shape)
    if vk is not None:
        vk = vk.reshape(shape)
    return vj, vk
