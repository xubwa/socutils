import ctypes, os, numpy, time
from pyscf import gto
from pyscf.lib import param
from pyscf.gto.mole import (ATM_SLOTS, BAS_SLOTS, PTR_COORD, ATOM_OF, ANG_OF, NPRIM_OF, NCTR_OF, PTR_EXP, PTR_COEFF,
                            PTR_ENV_START, gaussian_int, conc_env)
_here = os.path.dirname(os.path.abspath(__file__))
# the libcint built with spsp3c1e.cl (see README) and the kernel library
_cint = ctypes.CDLL(os.environ.get('OCJ_LIBCINT', os.path.join(_here, 'libcint.so.6')), mode=os.RTLD_LOCAL)
_ocj = ctypes.CDLL(os.environ.get('OCJ_LIB', os.path.join(_here, 'libocj.so')))
_LOC = {0: 'int3c1e', 1: 'int3c1e_r2_origk', 2: 'int3c1e_r4_origk', 3: 'int3c1e_r6_origk'}
ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)


def kshells(coords, prims, nctr):
    '''env/bas/atm of contracted s shells, shell i at coords[i], prims[i] = (exponents,
    coefficient matrix nprim x nctr); coefficients multiply exp(-z r^2) exactly'''
    nb = len(prims)
    atm = numpy.zeros((nb, ATM_SLOTS), dtype=numpy.int32)
    bas = numpy.zeros((nb, BAS_SLOTS), dtype=numpy.int32)
    env = [numpy.zeros(PTR_ENV_START)]
    p = PTR_ENV_START
    for i, (xyz, (ez, cf)) in enumerate(zip(coords, prims)):
        atm[i, PTR_COORD] = p; env.append(numpy.asarray(xyz, float)); p += 3
        bas[i, ATOM_OF] = i; bas[i, NPRIM_OF] = len(ez); bas[i, NCTR_OF] = nctr
        bas[i, PTR_EXP] = p; env.append(numpy.asarray(ez, float)); p += len(ez)
        bas[i, PTR_COEFF] = p; env.append((numpy.asarray(cf, float) * 2 * numpy.sqrt(numpy.pi)).T.ravel()); p += cf.size
    return atm, bas, numpy.hstack(env)


_LOC_SS = {0: 'int3c1e_ipvip1', 1: 'int3c1e_ipvip1_r2_origk', 2: 'int3c1e_ipvip1_r4_origk',
           3: 'int3c1e_ipvip1_r6_origk'}
_TMAP1 = numpy.zeros(1, numpy.int32)
_TMAP9 = numpy.array([3 * b + a for a in range(3) for b in range(3)], numpy.int32)


def to_spinor(xmol, ll, a9):
    '''LL spinor matrix from the scalar <phi|V|phi>; SS (sigma.p V sigma.p) from the nine
    <d_a phi|V|d_b phi> (index 3a+b): sum_ab A_ab s_a s_b = W0 + i sigma.W.  The sph->spinor
    maps are block sparse (<= 2l+1 entries per column): sparse products, O(N^2 l).'''
    import scipy.sparse
    ca, cb = [scipy.sparse.csc_matrix(c) for c in xmol.sph2spinor_coeff()]
    cah, cbh = ca.conj().T.tocsr(), cb.conj().T.tocsr()

    def sand(left, mm, right):          # left^H-type sparse (n2c x nao) @ dense mm @ sparse right
        return numpy.asarray(left @ numpy.asarray(right.T @ mm.T).T)
    out_ll = None
    if ll is not None:
        out_ll = sand(cah, ll, ca) + sand(cbh, ll, cb)
    if a9 is None:
        return out_ll, None
    a = a9.reshape(3, 3, *a9.shape[1:])
    w0 = a[0, 0] + a[1, 1] + a[2, 2]
    wx, wy, wz = a[1, 2] - a[2, 1], a[2, 0] - a[0, 2], a[0, 1] - a[1, 0]
    ss = (sand(cah, w0 + 1j * wz, ca) + sand(cah, 1j * wx + wy, cb)
          + sand(cbh, 1j * wx - wy, ca) + sand(cbh, w0 - 1j * wz, cb))
    return out_ll, ss


def _groups(zs, width):
    '''split exponents into groups of log10-width `width`'''
    zs = numpy.asarray(sorted(zs))
    lab = numpy.floor(numpy.log10(zs) / width).astype(int)
    return [zs[lab == g] for g in numpy.unique(lab)]


def j_kernel(xmol, terms, eps=1e-13, timing=None, width=1.):
    '''neutral-atom potentials V_na (4c: LL and SS = sigma.p V sigma.p, not yet / 4c^2;
    2c: LL) as spinor matrices from per-atom potential terms
    terms[K] = ((erf4, loc4), (erf2, loc2)): erf {z: E}, loc {m: {z: L}}.
    Every atom's exponents are split into groups of width `width` decades; each erf group
    with its share of the nuclear charge, -sum_g E_z erfc(sqrt(z) r)/r, and each local group
    is screened with its own smallest exponent.'''
    t0 = time.perf_counter()
    natm = xmol.natm
    norm = lambda z: 1 / (4 * numpy.pi * gaussian_int(2, z))
    coords = numpy.array([xmol.atom_coord(ia) for ia in range(natm)])
    prims, kinds, kz = [], [], []
    pts = []                                   # point-charge entries: (atom, zeta, q4, q2)
    for ia, ((erf4, loc4), (erf2, loc2)) in enumerate(terms):
        for z in _groups(set(erf4) | set(erf2), width):
            e4 = numpy.array([erf4.get(x, 0.) for x in z])
            e2 = numpy.array([erf2.get(x, 0.) for x in z])
            prims.append((z, numpy.c_[e4 * norm(z), e2 * norm(z)]))
            kinds.append(('erf', ia)); kz.append(z.min())
            pts.append((ia, z.min(), -e4.sum(), -e2.sum()))
        for m in sorted(set(loc4) | set(loc2)):
            d4, d2 = loc4.get(m, {}), loc2.get(m, {})
            for z in _groups(set(d4) | set(d2), width):
                prims.append((z, numpy.array([[d4.get(x, 0.), d2.get(x, 0.)] for x in z])))
                kinds.append(('loc', ia, m)); kz.append(z.min())
    kz = numpy.array(kz)
    katm, kbas, kenv = kshells([coords[k[1]] for k in kinds], prims, 2)
    atm, bas, env = conc_env(xmol._atm, xmol._bas, xmol._env, katm, kbas, kenv)
    atm = numpy.asarray(atm, numpy.int32); bas = numpy.asarray(bas, numpy.int32); env = numpy.asarray(env)
    nbas_ao = xmol.nbas
    ao_loc = numpy.asarray(xmol.ao_loc_nr(), dtype=numpy.int32)
    nao = ao_loc[-1]
    amin = numpy.array([xmol.bas_exp(i).min() for i in range(nbas_ao)])
    cut = -numpy.log(eps)
    ll = numpy.zeros((1, 2, nao, nao))          # [comp][4c/2c]
    a9 = numpy.zeros((9, 2, nao, nao))
    cnt = ctypes.c_long(0)
    kept = [0]

    def run3c(name, out, comp, tmap, ksel):
        ksh = numpy.asarray(nbas_ao + ksel, numpy.int32)
        kc = numpy.ascontiguousarray(coords[[kinds[i][1] for i in ksel]])
        kzs = numpy.ascontiguousarray(kz[ksel])
        _ocj.OCJ_3c_drv(getattr(_cint, name + '_sph'), ptr(out), ctypes.c_int(2), ctypes.c_int(comp), ptr(tmap),
                        ctypes.c_int(nao), ptr(ao_loc), ctypes.c_int(nbas_ao), ptr(amin), ctypes.c_int(len(ksel)),
                        ptr(ksh), ptr(kc), ptr(kzs), ctypes.c_double(cut), ptr(atm), ctypes.c_int(len(atm)),
                        ptr(bas), ctypes.c_int(len(bas)), ptr(env), ctypes.byref(cnt))
        kept[0] += cnt.value
    erf_sel = numpy.array([i for i, k in enumerate(kinds) if k[0] == 'erf'])
    run3c('int3c2e', ll, 1, _TMAP1, erf_sel)
    run3c('int3c2e_ipvip1', a9, 9, _TMAP9, erf_sel)
    for m in sorted(set(k[2] for k in kinds if k[0] == 'loc')):
        sel = numpy.array([i for i, k in enumerate(kinds) if k[0] == 'loc' and k[2] == m])
        run3c(_LOC[m], ll, 1, _TMAP1, sel)
        run3c(_LOC_SS[m], a9, 9, _TMAP9, sel)
    t1 = time.perf_counter()
    pc = numpy.ascontiguousarray(coords[[p[0] for p in pts]])
    pz = numpy.array([p[1] for p in pts])
    vn = numpy.zeros((2, 1, nao, nao))
    vn9 = numpy.zeros((9, nao, nao))
    for name, out, comp, tmap, q in (('int1e_rinv', vn[0], 1, _TMAP1, [p[2] for p in pts]),
                                     ('int1e_rinv', vn[1], 1, _TMAP1, [p[3] for p in pts]),
                                     ('int1e_iprinvip', vn9, 9, _TMAP9, [p[2] for p in pts])):
        q = numpy.array(q)
        _ocj.OCJ_rinv_drv(getattr(_cint, name + '_sph'), ptr(out), ctypes.c_int(comp), ptr(tmap), ctypes.c_int(nao),
                          ptr(ao_loc), ctypes.c_int(nbas_ao), ptr(amin), ctypes.c_int(len(pts)), ptr(pc), ptr(pz),
                          ptr(q), ctypes.c_double(cut), ptr(atm), ctypes.c_int(len(atm)), ptr(bas), ctypes.c_int(len(bas)),
                          ptr(env), ctypes.c_int(len(env)))
    t2 = time.perf_counter()
    l4, s4 = to_spinor(xmol, ll[0, 0] + vn[0, 0], a9[:, 0] + vn9)
    l2 = to_spinor(xmol, ll[0, 1] + vn[1, 0], None)[0]
    t3 = time.perf_counter()
    if timing is not None:
        timing.extend([t1 - t0, t2 - t1, t3 - t2, kept[0]])
    return l4, s4, l2
