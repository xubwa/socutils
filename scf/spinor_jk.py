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

RIJ / COSX (RIJCOSX, or the per-object attribute mf.rijcosx): J by density
fitting (a real RHF proxy density_fit(only_dfj=True) through md_j: the
resident L^-1 (P|ij), or the direct RI-J engine rij_direct when that does not
fit), K by COSX (socutils.scf.cosx) of the real pieces of each spinor density
in one multi-density call per spinor build, incrementally (K[D] = K[D_prev] +
K[D - D_prev] of the spinor density, a full build every cosx.REBUILD_NSTEPS),
on GRIDX2 (GRIDX1 -> 2 with cosx.SCF_GRID_SWITCH), the exchange energy
recomputed on GRIDX3 after the SCF (finalize, the spinor SCF's scf()):

    mf = spinor_hf.RHF(mol).x2camf()
    mf.rijcosx = True                   # or 'rij' (RIJ, exact K), 'cosx' (exact J, COSX K)
    mf.rijcosx_auxbasis = None          # None: AutoAux of mol (AUXBASIS); any pyscf basis spec,
                                        # or a callable mol -> basis (e.g. an acCD generator)
    mf.kernel()

COSX needs cosx.available(mol) (bra l <= the kernel's CX_LMAX); otherwise K is
the exact build (logged).  The direct RI-J engine needs rij_direct.available
(bra l <= 3, aux l <= 6); md_j otherwise keeps the resident tensor or pyscf's
DF J, so RIJ itself always works.
'''
import numpy
from pyscf import lib, scf
from pyscf.lib import logger

ENABLE = True       # spinor_hf.SCF / socutils GHF route their J / K here (else pyscf's GHF get_jk)
KRAMERS_TOL = 1e-10
RIJCOSX = False     # False | True / 'rijcosx' | 'rij' | 'cosx': default of mf.rijcosx (None -> this)
AUXBASIS = None     # the RIJ auxiliary basis: None -> pyscf AutoAux of the molecule, a basis spec, or mol -> basis
FINAL_GRID = 3      # COSX level of the final exchange energy (None: not recomputed)
# The COSX grids' radial rule (Krack-Koster, n = 15 intacc - 40 + 5 ROW: 50 shells for Au at GRIDX2) does not
# reach the tight core of the all-electron bases of the sixth-row atoms: AuH scalar X2C RHF, exchange energy
# error on GRIDX2 / GRIDX3 (Eh): x2c-SVPall -1.1e-3 / +3.5e-4, ano-rcc-vdzp +7.8e-4 / -4.8e-4, dyall-v2z +1.3e3
# (an s primitive of exponent 5.9e7 the grid does not see, S_g singular, the fit explodes); with Au at intacc 8
# (110 shells, +70 % points on AuH): +2.6e-6 / +5.2e-6, +9.1e-5 / +5.9e-5, +2.1e-6 / +2.9e-6.  HgCl2 x2c-SVPall
# -1.7e-3 -> -2.0e-4 (the rest angular: -0.6e-4 with Lebedev table 4).  Fifth row: HI x2c-SVPall 1.9e-5 as is,
# but I2 ano-rcc-vdzp -3.7e-4 / +3.6e-4 on GRIDX2 / GRIDX3 -> +5.9e-5 on both (intacc >= 6 converged).
COSX_HEAVY_Z = 36           # atoms beyond this get the radial grid of COSX_HEAVY_INTACC on every COSX level
COSX_HEAVY_INTACC = 8.0
COSX_FIT_RANGE = (.5, 2.)   # fit_quality bounds of a usable grid; else +2 intacc on the atoms beyond Z = 18,
COSX_GRID_RETRIES = 2       # ... at most this many times, then exact K


def mode(mf):
    '''the fast-path mode of mf: '' (exact), 'rij', 'cosx' or 'rijcosx' '''
    m = getattr(mf, 'rijcosx', None)
    if m is None:
        m = RIJCOSX
    if m is True:
        return 'rijcosx'
    if not m:
        return ''
    m = str(m).lower()
    if m not in ('rij', 'cosx', 'rijcosx'):
        raise ValueError('rijcosx mode %r' % m)
    return m


def auxbasis(mf, mol):
    '''the RIJ auxiliary basis of mf for mol (mf.rijcosx_auxbasis, else AUXBASIS; None: AutoAux)'''
    aux = getattr(mf, 'rijcosx_auxbasis', None)
    if aux is None:
        aux = AUXBASIS
    if aux is None:
        from pyscf.df.autoaux import autoaux
        return autoaux(mol)
    if callable(aux):
        return aux(mol)
    return aux


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


def _rij_proxy(mf, mol):
    '''a real RHF of mol, density_fit(only_dfj=True) with the auxiliary basis of auxbasis(), md_j applied
    (resident L^-1 (P|ij) or the direct RI-J engine), cached on mf'''
    p = getattr(mf, '_spinor_rij_proxy', None)
    aux = getattr(mf, 'rijcosx_auxbasis', None)
    if aux is None:
        aux = AUXBASIS
    if p is None or p.mol is not mol or p._aux_spec is not aux:
        from socutils.scf import md_j
        p = scf.RHF(mol).density_fit(auxbasis=auxbasis(mf, mol), only_dfj=True)
        p.verbose = mf.verbose
        p.stdout = mf.stdout
        p.max_memory = mf.max_memory
        p.with_df.verbose = 0
        p.with_df.build()
        p._aux_spec = aux
        logger.info(mf, 'spinor RIJ: auxiliary basis %d functions, lmax %d',
                    p.with_df.auxmol.nao, p.with_df.auxmol._bas[:, 1].max())
        p.verbose = 0
        md_j.apply(p)
        mf._spinor_rij_proxy = p
    p.max_memory = mf.max_memory
    p.conv_tol = getattr(mf, 'conv_tol', 1e-9)
    return p


# ---------------------------------------------------------------------------------------------------------------
# COSX K of the real pieces of a spinor density

def _pieces(aa, ab, ba, bb, cplx, kramers, hermi):
    '''the real matrices whose K make up K of the GHF density: [(block, part, kind, (n, nao, nao))], kind
    'S' symmetric, 'A' anti-symmetric, 'G' general.  Hermitian: Re D_aa, Re D_bb symmetric, Im antisymmetric;
    time-reversal even (D_bb = D_aa^*, D_ba = -D_ab^*, Hermitian) also D_ab^T = -D_ab.'''
    out = []
    g = hermi != 1
    out.append(('aa', 're', 'G' if g else 'S', aa.real))
    if not kramers:
        out.append(('bb', 're', 'G' if g else 'S', bb.real))
    if cplx:
        out.append(('aa', 'im', 'G' if g else 'A', aa.imag))
        if not kramers:
            out.append(('bb', 'im', 'G' if g else 'A', bb.imag))
    kab = 'A' if kramers else ('G')
    out.append(('ab', 're', kab, ab.real))
    if cplx:
        out.append(('ab', 'im', kab, ab.imag))
    if g:
        out.append(('ba', 're', 'G', ba.real))
        if cplx:
            out.append(('ba', 'im', 'G', ba.imag))
    return out


def _assemble(pieces, ks, n, nao, dtype, kramers, hermi):
    '''K (n, 2nao, 2nao) of the GHF density from the K of its real pieces'''
    blk = {}
    for (b, part, kind, _), k in zip(pieces, ks):
        if part == 're':
            blk[b] = k + blk.get(b, 0)
        else:
            blk[b] = 1j * k + blk.get(b, 0)
    if kramers:
        blk['bb'] = blk['aa'].conj()
    if 'ba' not in blk:
        blk['ba'] = blk['ab'].conj().transpose(0, 2, 1)
    vk = numpy.zeros((n, 2 * nao, 2 * nao), dtype=dtype)
    vk[:, :nao, :nao] = blk['aa']
    vk[:, :nao, nao:] = blk['ab']
    vk[:, nao:, :nao] = blk['ba']
    vk[:, nao:, nao:] = blk['bb']
    return vk


def _project(pieces, ks):
    '''K of symmetric (anti-symmetric) pieces symmetrised (anti-symmetrised), as the exact K'''
    for i, (_, _, kind, _) in enumerate(pieces):
        if kind == 'S':
            ks[i] = (ks[i] + ks[i].transpose(0, 2, 1)) * .5
        elif kind == 'A':
            ks[i] = (ks[i] - ks[i].transpose(0, 2, 1)) * .5
    return ks


def _stack(pieces):
    return numpy.ascontiguousarray(numpy.concatenate([x for *_, x in pieces]))


def _state(mf, mol):
    st = getattr(mf, '_spinor_cosx', None)
    if st is None or st['mol'] is not mol:
        st = mf._spinor_cosx = {'mol': mol, 'k': {}, 'last': None}
    return st


def grid_args(mol, level, extra=0.):
    '''the COSX grid of level for the all-electron heavy-element bases of the two-component SCF:
    cosx.gridx_args, with the radial grid of the atoms beyond Z = COSX_HEAVY_Z at intacc
    COSX_HEAVY_INTACC (+ extra; also for the atoms beyond Z = 18 when extra > 0)'''
    from pyscf import gto
    from socutils.scf import cosx
    args = cosx.gridx_args(mol, level)
    ei = {}
    for i in range(mol.natm):
        s = mol.atom_pure_symbol(i)
        z = gto.charge(s)
        if z > COSX_HEAVY_Z or (extra > 0 and z > 18):
            ei[s] = max(args.get('intacc', 0), COSX_HEAVY_INTACC) + extra
    if ei:
        args['element_intacc'] = ei
    return args


def fit_quality(k):
    '''the eigenvalue range of X^T S_g X (X^T S X = 1): 1 where the grid integrates the basis exactly;
    near 0 for a function the grid does not see (a tight core primitive between the inner radial shells)'''
    s = k.mol.intor_symmetric('int1e_ovlp')
    e, v = numpy.linalg.eigh(s)
    keep = e > 1e-10 * e.max()
    x = v[:, keep] / numpy.sqrt(e[keep])
    mu = numpy.linalg.eigvalsh(x.T @ k._sg @ x)
    return mu.min(), mu.max()


def _kobj(mf, st, level):
    '''the COSX object of grid level (cached), None if no grid up to COSX_GRID_RETRIES refinements
    represents the basis (fit_quality outside COSX_FIT_RANGE)'''
    from socutils.scf import cosx
    if level not in st['k']:
        mol = st['mol']
        k = None
        for it in range(COSX_GRID_RETRIES + 1):
            g = cosx.make_grids(mol, **grid_args(mol, level, 2. * it))
            k = cosx.COSX(mol, g, fit=True, tol=min(cosx.TOL, .1 * getattr(mf, 'conv_tol', 1e-9)))
            k.verbose = mf.verbose
            k.stdout = mf.stdout
            k.build()
            lo, hi = fit_quality(k)
            logger.info(mf, 'spinor COSX: GRIDX%d %d points, fit eigenvalues [%.3g, %.3g]',
                        level, k._weights.size, lo, hi)
            if COSX_FIT_RANGE[0] < lo and hi < COSX_FIT_RANGE[1]:
                break
            k = None
        if k is None:
            logger.warn(mf, 'spinor COSX: GRIDX%d does not represent the basis (tight primitives): exact K', level)
        st['k'][level] = k
    return st['k'][level]


def _cosx_k(k, pieces, n, nao, dtype, kramers, hermi, tol=None):
    '''K of the GHF density by one (multi-density) COSX call on the real pieces'''
    tol0 = k.tol
    if tol is not None:
        k.tol = tol
    try:
        ks = k.get_k(_stack(pieces), hermi=0)
    finally:
        k.tol = tol0
    ks = _project(pieces, list(ks.reshape(len(pieces), n, nao, nao)))
    return _assemble(pieces, ks, n, nao, dtype, kramers, hermi)


def _exact_k(p, mol, pieces, n, nao, dtype, kramers, hermi):
    '''K of the GHF density from the exact builds of the proxy p (symmetric pieces hermi=1)'''
    ks = [None] * len(pieces)
    for herm in (1, 0):
        idx = [i for i, pc in enumerate(pieces) if (pc[2] == 'S') == (herm == 1)]
        if idx:
            kk = p.get_jk(mol, _stack([pieces[i] for i in idx]), herm, False, True)[1]
            kk = numpy.asarray(kk).reshape(len(idx), n, nao, nao)
            for j, i in enumerate(idx):
                ks[i] = kk[j]
    return _assemble(pieces, _project(pieces, ks), n, nao, dtype, kramers, hermi)


def _cosx_wanted(mf, mol):
    from socutils.scf import cosx
    ok = cosx.available(mol)
    if not ok and not getattr(mf, '_spinor_cosx_warned', False):
        logger.warn(mf, 'spinor COSX: not available for this basis (l > CX_LMAX or no kernel): exact K')
        mf._spinor_cosx_warned = True
    return ok


def get_k_cosx(mf, mol, dm, hermi, aa, ab, ba, bb, cplx, kramers):
    '''COSX K of GHF densities (n, 2nao, 2nao); a single Hermitian density incrementally (state on mf)'''
    from socutils.scf import cosx
    n, nao = dm.shape[0], mol.nao_nr()
    st = _state(mf, mol)
    pieces = _pieces(aa, ab, ba, bb, cplx, kramers, hermi)
    if n != 1 or hermi != 1:
        k = _kobj(mf, st, 2)
        if k is None:
            return _exact_k(_proxy(mf, mol), mol, pieces, n, nao, dm.dtype, kramers, hermi)
        return _cosx_k(k, pieces, n, nao, dm.dtype, kramers, hermi)
    last = st['last']
    if last is None:
        lev = 1 if cosx.SCF_GRID_SWITCH > 0 else 2
    else:
        lev = last['lev']
        if lev == 1 and abs(dm - last['dm']).max() < cosx.SCF_GRID_SWITCH:
            lev = 2
    k = _kobj(mf, st, lev)
    if k is None:
        st['last'] = None
        return _exact_k(_proxy(mf, mol), mol, pieces, n, nao, dm.dtype, kramers, hermi)
    if (last is not None and last['lev'] == lev and last['n'] < cosx.REBUILD_NSTEPS
            and last['dm'].shape == dm.shape):
        ddm = dm - last['dm']
        if numpy.iscomplexobj(ddm) != cplx:
            ddm = ddm.astype(dm.dtype)
        d_aa, d_ab = ddm[:, :nao, :nao], ddm[:, :nao, nao:]
        d_ba, d_bb = ddm[:, nao:, :nao], ddm[:, nao:, nao:]
        dkr = _is_kramers_even(d_aa, d_ab, d_ba, d_bb)
        dp = _pieces(d_aa, d_ab, d_ba, d_bb, cplx, dkr, hermi)
        vk = last['vk'] + _cosx_k(k, dp, n, nao, numpy.result_type(dm.dtype, last['vk'].dtype), dkr, hermi)
        nb = last['n'] + 1
        logger.debug(mf, 'spinor COSX: incremental K (%d since the full build, %d real densities), grid %d',
                     nb, len(dp), lev)
    else:
        vk = _cosx_k(k, pieces, n, nao, dm.dtype, kramers, hermi, tol=max(k.tol, cosx.TOL_FULL))
        nb = 0
        logger.debug(mf, 'spinor COSX: full K (%d real densities), grid %d', len(pieces), lev)
    st['last'] = {'lev': lev, 'dm': dm.copy(), 'vk': vk.copy(), 'n': nb}
    return vk


def reset(mf):
    '''start of an SCF: drop the incremental COSX state; the fast path owns the incremental builds'''
    st = getattr(mf, '_spinor_cosx', None)
    if st is not None:
        st['last'] = None
    if mode(mf):
        mf.direct_scf = False


def finalize(mf):
    '''after the SCF: the exchange energy of the last Fock build's density recomputed on FINAL_GRID, mf.e_tot
    corrected (mf._cosx_de); returns mf.e_tot'''
    st = getattr(mf, '_spinor_cosx', None)
    mf._cosx_de = 0.
    if FINAL_GRID is None or st is None or st['last'] is None or st['last']['lev'] == FINAL_GRID:
        return mf.e_tot
    t0 = (logger.process_clock(), logger.perf_counter())
    mol = st['mol']
    nao = mol.nao_nr()
    dm, k_scf = st['last']['dm'], st['last']['vk']
    e_fin = exchange_energy(mf, mol, dm[0], FINAL_GRID)
    if e_fin is None:
        logger.warn(mf, 'spinor COSX: no usable GRIDX%d, the SCF grid energy kept', FINAL_GRID)
        return mf.e_tot
    hyb = 1.
    if hasattr(mf, 'xc'):
        hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)[2]
    e_scf = numpy.einsum('ij,ji', dm[0], k_scf[0]).real
    de = -.5 * hyb * (e_fin - e_scf)
    mf._cosx_de = de
    logger.info(mf, 'spinor COSX: exchange energy recomputed on GRIDX%d: change %.9f, E = %.12f',
                FINAL_GRID, de, mf.e_tot + de)
    logger.timer(mf, 'spinor COSX final grid', *t0)
    mf.e_tot = mf.e_tot + de
    return mf.e_tot


def exchange_energy(mf, mol, dm, level):
    '''Re tr(D K[D]) of a Hermitian GHF density (2nao, 2nao) on COSX grid level, energy only
    (= sum_st <Re D_st, K[Re D_st]> + <Im D_st, K[Im D_st]>, Frobenius)'''
    from socutils.scf import cosx
    nao = mol.nao_nr()
    aa, ab = dm[:nao, :nao], dm[:nao, nao:]
    ba, bb = dm[nao:, :nao], dm[nao:, nao:]
    cplx = numpy.iscomplexobj(dm)
    kramers = _is_kramers_even(aa, ab, ba, bb)
    pieces = _pieces(aa[None], ab[None], ba[None], bb[None], cplx, kramers, 1)
    # weights: D_bb's pieces (Kramers: equal to D_aa's) and D_ba = D_ab^H (same Frobenius forms)
    # COSX.energy(X) is tr(X K[X]) = sum X^T o K: +<X, K> for symmetric X, -<X, K> for anti-symmetric X; a
    # general piece by its symmetric and anti-symmetric parts (the cross terms vanish for the exact K)
    w, xs = [], []
    for b, _, kind, x in pieces:
        wb = (2. if kramers else 1.) if b in ('aa', 'bb') else 2.
        if kind in 'SG':
            xs.append(x if kind == 'S' else (x + x.transpose(0, 2, 1)) * .5)
            w.append(wb)
        if kind in 'AG':
            xs.append(x if kind == 'A' else (x - x.transpose(0, 2, 1)) * .5)
            w.append(-wb)
    k = _kobj(mf, _state(mf, mol), level)
    if k is None:
        return None
    e = k.energy(numpy.ascontiguousarray(numpy.concatenate(xs)), tol=cosx.TOL_ENERGY)
    return float(numpy.dot(w, e))


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
    cplx = numpy.iscomplexobj(dms)
    kramers = hermi == 1 and _is_kramers_even(aa, ab, ba, bb)
    m = mode(mf)
    if m:
        out = _get_jk_fast(mf, mol, dms, hermi, with_j, with_k, m, aa, ab, ba, bb, cplx, kramers)
        logger.timer_debug1(mf, 'two-component J/K (%s)' % m, *t0)
        return tuple(None if v is None else v.reshape(shape) for v in out)
    p = _proxy(mf, mol)

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


def _get_jk_fast(mf, mol, dms, hermi, with_j, with_k, m, aa, ab, ba, bb, cplx, kramers):
    '''J (RIJ for 'rij*', else exact) and K (COSX for '*cosx' when available, else exact) of GHF densities'''
    if mf.direct_scf:
        # the incremental builds are this module's (on the spinor density), pyscf's Fock increments off
        mf.direct_scf = False
    n, nao = dms.shape[0], mol.nao_nr()
    vj = vk = None
    if with_j:
        p = _rij_proxy(mf, mol) if m.startswith('rij') else _proxy(mf, mol)
        dj = numpy.ascontiguousarray(aa.real + bb.real)
        J = numpy.asarray(p.get_jk(mol, dj[0] if n == 1 else dj, 1 if hermi == 1 else 0, True, False)[0])
        J = J.reshape(n, nao, nao)
        if cplx and hermi != 1:
            ji = p.get_jk(mol, numpy.ascontiguousarray(aa.imag + bb.imag), 0, True, False)[0]
            J = J + 1j * numpy.asarray(ji).reshape(n, nao, nao)
        vj = numpy.zeros((n, 2 * nao, 2 * nao), dtype=dms.dtype)
        vj[:, :nao, :nao] = vj[:, nao:, nao:] = J
    if with_k:
        if m.endswith('cosx') and _cosx_wanted(mf, mol):
            vk = get_k_cosx(mf, mol, dms, hermi, aa, ab, ba, bb, cplx, kramers)
        else:
            pieces = _pieces(aa, ab, ba, bb, cplx, kramers, hermi)
            vk = _exact_k(_proxy(mf, mol), mol, pieces, n, nao, dms.dtype, kramers, hermi)
    return vj, vk
