import numpy, scipy
import time
from numpy.linalg import norm
from functools import reduce
from pyscf import gto, scf, lib
from pyscf.gto import moleintor
from pyscf.lib import logger
from pyscf.lib.parameters import LIGHT_SPEED
from pyscf.scf import dhf, jk, _vhf
#from pyscf.shciscf import socutils
from pyscf.x2c import sfx2c1e
from pyscf.x2c import x2c
from spinor2sph import spinor2sph_soc
x2camf  = None
try:
    import x2camf
except ImportError:
    pass

def get_hxr(mol, uncontract=True):
    if (uncontract):
        xmol, contr_coeff = x2c.X2C(mol).get_xmol()
    else:
        xmol, contr_coeff = mol, None

    c = LIGHT_SPEED
    t = xmol.intor_symmetric('int1e_kin')
    v = xmol.intor_symmetric('int1e_nuc')
    s = xmol.intor_symmetric('int1e_ovlp')
    w = xmol.intor_symmetric('int1e_pnucp')

    h1, x, r = _x2c1e_hxrmat(t, v, w, s, c)
    if (uncontract):
        h1 = reduce(numpy.dot, (contr_coeff.T, h1, contr_coeff))

    return h1, x, r


def _x2c1e_hxrmat(t, v, w, s, c):
    nao = s.shape[0]
    n2 = nao * 2
    h = numpy.zeros((n2, n2), dtype=v.dtype)
    m = numpy.zeros((n2, n2), dtype=v.dtype)
    h[:nao, :nao] = v
    h[:nao, nao:] = t
    h[nao:, :nao] = t
    h[nao:, nao:] = w * (.25 / c**2) - t
    m[:nao, :nao] = s
    m[nao:, nao:] = t * (.5 / c**2)

    e, a = scipy.linalg.eigh(h, m)
    cl = a[:nao, nao:]
    cs = a[nao:, nao:]

    b = numpy.dot(cl, cl.T.conj())
    x = reduce(numpy.dot, (cs, cl.T.conj(), numpy.linalg.inv(b)))

    s1 = s + reduce(numpy.dot, (x.T.conj(), t, x)) * (.5 / c**2)
    tx = reduce(numpy.dot, (t, x))
    h1 = (h[:nao, :nao] + h[:nao, nao:].dot(x) + x.T.conj().dot(h[nao:, :nao]) +
          reduce(numpy.dot, (x.T.conj(), h[nao:, nao:], x)))

    sa = x2c._invsqrt(s)
    sb = x2c._invsqrt(reduce(numpy.dot, (sa, s1, sa)))
    r = reduce(numpy.dot, (sa, sb, sa, s))
    h1out = reduce(numpy.dot, (r.T.conj(), h1, r))
    return h1out, x, r

def get_wso(mol):
    nb = mol.nao_nr()
    wso = numpy.zeros((3, nb, nb))
    for iatom in range(mol.natm):
        zA = mol.atom_charge(iatom)
        xyz = mol.atom_coord(iatom)
        mol.set_rinv_orig(xyz)
        # sign due to integration by part
        wso += zA * mol.intor('cint1e_prinvxp_sph', 3)
    return wso


def get_hso1e_bp(mol):
    return get_wso(mol)


def get_fso2e_bp(mol, dm):

    fso_0, fso_1, fso_2 = jk.get_jk(
        mol, [dm, dm, dm],
        scripts=['ijkl,kl->ij', 'ijkl,jk->il', 'ijkl,li->kj'],
        intor='cint2e_p1vxp1_sph')

    return fso_0 - 1.5 * fso_1 - 1.5 * fso_2


def get_p(dm, x, rp):
    p_ll = rp.dot(dm.dot(rp.T))
    p_ls = p_ll.dot(x.T)
    p_ss = x.dot(p_ll.dot(x.T))
    return p_ll, p_ls, p_ss


def get_hso1e_x2c1(mol, unc=True):
    h1, x, rp = get_hxr(mol, uncontract=unc)
    nb = x.shape[0]
    hso1e = numpy.zeros((3, nb, nb))
    wso = get_wso(mol)
    for ic in range(3):
        hso1e[ic] = reduce(numpy.dot, (rp.T, x.T, wso[ic], x, rp))
    return hso1e


def get_fso2e_x2c(mol, dm, unc=True):
    ''' Two-electron x2c operator with memory saving strategy '''

    nb = mol.nao_nr()
    h1, x, rp = get_hxr(mol, uncontract=unc)
    p_ll, p_ls, p_ss = get_p(0.5 * dm, x, rp)

    fso2e = numpy.zeros((3, nb, nb))
    gso_ll = numpy.zeros((3, nb, nb))
    gso_ls = numpy.zeros((3, nb, nb))
    gso_ss = numpy.zeros((3, nb, nb))
    from pyscf.gto import moleintor
    nbas = mol.nbas
    max_double = mol.max_memory / 8.0 * 1.0e6
    max_basis = pow(max_double / 9., 1. / 4.)
    ao_loc_orig = moleintor.make_loc(mol._bas, 'int2e_ip1_ip2_sph')
    shl_size = []
    shl_slice = [0]
    ao_loc = [0]
    if nb > max_basis:
        for i in range(0, nbas - 1):
            if (ao_loc_orig[i + 1] - ao_loc[-1] > max_basis
                    and ao_loc_orig[i] - ao_loc[-1]):
                ao_loc.append(ao_loc_orig[i])
                shl_size.append(ao_loc[-1] - ao_loc[-2])
                shl_slice.append(i)
    if ao_loc[-1] is not ao_loc_orig[-1]:
        ao_loc.append(ao_loc_orig[-1])
        shl_size.append(ao_loc[-1] - ao_loc[-2])
        shl_slice.append(nbas)
    nbas = len(shl_size)
    logger.info(
        mol,
        "Cutting basis functions into %d batches, need to calculate %d integrals batches.",
        nbas, nbas**4)

    start = time.process_time()
    for i in range(0, nbas):
        for j in range(0, nbas):
            for k in range(0, nbas):
                for l in range(0, nbas):
                    start_this = time.process_time()
                    ddint = mol.intor('int2e_ip1ip2_sph',
                                      comp=9,
                                      shls_slice=[
                                          shl_slice[i], shl_slice[i + 1],
                                          shl_slice[j], shl_slice[j + 1],
                                          shl_slice[k], shl_slice[k + 1],
                                          shl_slice[l], shl_slice[l + 1]
                                      ]).reshape(3, 3, -1)
                    kint = numpy.zeros(3 * shl_size[i] * shl_size[j] *
                                       shl_size[k] * shl_size[l]).reshape(
                                           3, shl_size[i], shl_size[j],
                                           shl_size[k], shl_size[l])
                    kint[0] = (ddint[1, 2] - ddint[2, 1]).reshape(
                        shl_size[i], shl_size[j], shl_size[k], shl_size[l])
                    kint[1] = (ddint[2, 0] - ddint[0, 2]).reshape(
                        shl_size[i], shl_size[j], shl_size[k], shl_size[l])
                    kint[2] = (ddint[0, 1] - ddint[1, 0]).reshape(
                        shl_size[i], shl_size[j], shl_size[k], shl_size[l])

                    gso_ll[:, ao_loc[j]:ao_loc[j+1], ao_loc[l]:ao_loc[l+1]] \
                      +=-2.0*lib.einsum('ilmkn,lk->imn', kint, \
                        p_ss[ao_loc[i]:ao_loc[i+1], ao_loc[k]:ao_loc[k+1]])
                    gso_ls[:, ao_loc[i]:ao_loc[i+1], ao_loc[l]:ao_loc[l+1]] \
                      +=-1.0*lib.einsum('imlkn,lk->imn', kint, \
                        p_ls[ao_loc[j]:ao_loc[j+1], ao_loc[k]:ao_loc[k+1]])
                    gso_ls[:, ao_loc[j]:ao_loc[j+1], ao_loc[l]:ao_loc[l+1]] \
                      +=-1.0*lib.einsum('ilmkn,lk->imn', kint, \
                        p_ls[ao_loc[i]:ao_loc[i+1], ao_loc[k]:ao_loc[k+1]])
                    gso_ss[:, ao_loc[i]:ao_loc[i+1], ao_loc[j]:ao_loc[j+1]] \
                      +=-2.0*lib.einsum('imnkl,lk->imn', kint, \
                        p_ll[ao_loc[l]:ao_loc[l+1], ao_loc[k]:ao_loc[k+1]])\
                        -2.0*lib.einsum('imnlk,lk->imn', kint, \
                        p_ll[ao_loc[k]:ao_loc[k+1], ao_loc[l]:ao_loc[l+1]])
                    gso_ss[:, ao_loc[i]:ao_loc[i+1], ao_loc[k]:ao_loc[k+1]] \
                      += 2.0*lib.einsum('imlnk,lk->imn', kint, \
                        p_ll[ao_loc[j]:ao_loc[j+1], ao_loc[l]:ao_loc[l+1]])

                    logger.info(mol, "Time elapsed for %dth batch in %d batches is %g, cumulates time is %g.", \
                    i*nbas**3+j*nbas**2+k*nbas+l+1, nbas*4, time.process_time()-start_this, time.process_time()-start)

    for comp in range(0, 3):
        fso2e[comp] = gso_ll[comp] + gso_ls[comp].dot(x) + x.T.dot(
            -gso_ls[comp].T) + x.T.dot(gso_ss[comp].dot(x))
        fso2e[comp] = reduce(numpy.dot, (rp.T, fso2e[comp], rp))

    logger.info(mol, 'Two electron part of SOC integral is done')

    return fso2e


def print_int1e(h1, name):
    xyz = ["X", "Y", "Z"]
    for k in range(3):
        with open('%s.' % (name) + xyz[k], 'w') as fout:
            fout.write('%d\n' % h1[k].shape[0])
            for i in range(h1[k].shape[0]):
                for j in range(h1[k].shape[0]):
                    if (abs(h1[k, i, j]) > 1.e-8):
                        fout.write('%16.10g %4d %4d\n' %
                                   (h1[k, i, j].real, i + 1, j + 1))


def get_soc_integrals(method, dm=None, pc1e=None, pc2e=None, unc=None, atomic=True):
    factor = 0.5 / (LIGHT_SPEED)**2

    # treat x2c and bp in seperate workflows.
    has_ecp = method.mol.has_ecp()
    mol = method.mol

    if pc1e is None:
        pc1e = 'None'
    if pc2e is None:
        pc2e = 'None'
    if (pc1e is 'None' and pc2e is 'None'):
        if has_ecp:
            hso1e = mol.intor('ECPso')
            print('''
            WARNING: no picture change effects provided, only soecp term included. 
            Make sure you used a ecp with spin-orbit terms.
            ''')
        else:
            AssertionError(
                'No picture change effects provided, no soc effects included')

    if (has_ecp and ('x2c' in pc1e or 'x2c' in pc2e)):
        raise AssertionError('X2C should not be used together with ECP at any time.')


    if unc is None:
        # when there is ecp, or both pc1e and pc2e are bp, use contracted basis by default.
        unc = 2 * has_ecp + ('bp' in pc1e) + ('bp' in pc2e) < 2

    if dm is None:
        dm = method.make_rdm1()

    if unc:
        xmol, contr_coeff = sfx2c1e.SpinFreeX2C(mol).get_xmol()
        dm = reduce(numpy.dot, (contr_coeff, dm, contr_coeff.T))
    else:
        xmol, contr_coeff = method.mol, numpy.eye(method.mol.nao_nr())

    nb = xmol.nao_nr()
    hso = numpy.zeros((3, nb, nb), dtype=complex)

    if (has_ecp):
        hso += mol.intor('ECPso')

    if (pc1e == 'bp'):
        hso += factor * get_wso(xmol)
    elif (pc1e == 'x2c1'):
        hso += factor * get_hso1e_x2c1(xmol, unc=unc)
    elif (pc1e == None):
        hso += 0.0
    else:
        AssertionError('pc1e=%s is not a valid option.' % pc1e)

    if (atomic != True):
        if (pc2e == 'bp'):
            hso -= factor * get_fso2e_bp(xmol, dm)
        elif (pc2e == 'x2c'):
            hso -= factor * get_fso2e_x2c(xmol, dm, unc=unc)
        elif (pc2e == None):
            hso += 0.0
        else:
            AssertionError('pc2e=%s is not a valid option.' % pc2e)
    else:
        if (pc2e == 'bp'):
            NotImplementedError('Atomic mean-field for BP is not implemented yet, set atomic = False instead.')
        elif (pc2e == 'x2c'):
            if x2camf:
                x2cobj = x2c.X2C(xmol)
                spinor = x2camf.amfi(x2cobj, spin_free=False, two_c=False, with_gaunt=True, with_gauge=True)
                print(xmol.nao_2c(), spinor.shape)
                hso -= 2. * spinor2sph_soc(xmol, spinor)[1:]
            else:
                AssertionError('AMF calculation requires x2camf package. Install x2camf with pip install git+https://github.com/warlocat/x2camf')

    # convert back to contracted basis
    result = numpy.zeros((3, mol.nao_nr(), mol.nao_nr()))
    for ic in range(3):
        result[ic] = reduce(numpy.dot, (contr_coeff.T, hso[ic], contr_coeff))
    
    return result


def write_gtensor_integrals(mc, atomlist=None):
    mol = mc.mol
    ncore, ncas = mc.ncore, mc.ncas
    nb = mol.nao_nr()
    if atomlist is None:
        h1ao = mol.intor('cint1e_cg_irxp_sph', comp=3)
    else:
        h1ao = numpy.zeros((3, nb, nb))
        aoslice = mc.mol.aoslice_by_atom()
        for iatom in atomlist:
            for jatom in atomlist:
                iao_start = aoslice[iatom, 2]
                jao_start = aoslice[jatom, 2]
                iao_end = aoslice[iatom, 3]
                jao_end = aoslice[jatom, 3]
                ishl_start = aoslice[iatom, 0]
                jshl_start = aoslice[jatom, 0]
                ishl_end = aoslice[iatom, 1]
                jshl_end = aoslice[jatom, 1]
                h1ao[:, iao_start: iao_end, jao_start: jao_end]\
                    += mol.intor('cint1e_cg_irxp_sph', 3, shls_slice=[ishl_start, ishl_end, jshl_start, jshl_end]).reshape(3, iao_end-iao_start, jao_end-jao_start)

    h1 = lib.einsum('xpq,pi,qj->xij', h1ao, mc.mo_coeff, mc.mo_coeff)
    print_int1e(h1[:, ncore:ncore + ncas, ncore:ncore + ncas], 'GTensor')


def write_soc_integrals(mc, dm=None, pc1e='bp', pc2e='bp', unc=None, atomic=True):
    if dm is None:
        try:
            dm = mc.make_rdm1()
        except:
            dm = mc._scf.make_rdm1(mc.mo_coeff)

    hso = get_soc_integrals(mc, dm=dm, pc1e=pc1e, pc2e=pc2e, unc=unc, atomic=atomic)
    ncore, ncas = mc.ncore, mc.ncas
    h1 = lib.einsum('xpq,pi,qj->xij', hso, mc.mo_coeff,
                    mc.mo_coeff)[:, ncore:ncore + ncas, ncore:ncore + ncas]
    print_int1e(h1, 'SOC')
    print(h1.imag)



def get_jk_sf_coulomb(mol, dm, hermi=1, coulomb_allow='SSSS',
                   opt_llll=None, opt_ssll=None, opt_ssss=None, omega=None, verbose=None):
    log = logger.new_logger(mol, verbose)
    with mol.with_range_coulomb(omega):
        if coulomb_allow.upper() == 'LLLL':
            log.debug('Coulomb integral: (LL|LL)')
            j1, k1 = dhf._call_veff_llll(mol, dm, hermi, opt_llll)
            n2c = j1.shape[1]
            vj = numpy.zeros_like(dm)
            vk = numpy.zeros_like(dm)
            vj[...,:n2c,:n2c] = j1
            vk[...,:n2c,:n2c] = k1
        elif coulomb_allow.upper() == 'SSLL' \
          or coulomb_allow.upper() == 'LLSS':
            log.debug('Coulomb integral: (LL|LL) + (SS|LL)')
            vj, vk = _call_veff_sf_ssll(mol, dm, hermi, opt_ssll)
            j1, k1 = dhf._call_veff_llll(mol, dm, hermi, opt_llll)
            n2c = j1.shape[1]
            vj[...,:n2c,:n2c] += j1
            vk[...,:n2c,:n2c] += k1
        else: # coulomb_allow == 'SSSS'
            log.debug('Coulomb integral: (LL|LL) + (SS|LL) + (SS|SS)')
            vj, vk = _call_veff_sf_ssll(mol, dm, hermi, opt_ssll)
            j1, k1 = dhf._call_veff_llll(mol, dm, hermi, opt_llll)
            n2c = j1.shape[1]
            vj[...,:n2c,:n2c] += j1
            vk[...,:n2c,:n2c] += k1
            j1, k1 = _call_veff_sf_ssss(mol, dm, hermi, opt_ssss)
            vj[...,n2c:,n2c:] += j1
            vk[...,n2c:,n2c:] += k1

    return vj, vk


def _call_veff_sf_ssll(mol, dm, hermi=1, mf_opt=None):
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        n_dm = 1
        n2c = dm.shape[0] // 2
        dmll = dm[:n2c,:n2c].copy()
        dmsl = dm[n2c:,:n2c].copy()
        dmss = dm[n2c:,n2c:].copy()
        dms = (dmll, dmss, dmsl)
    else:
        n_dm = len(dm)
        n2c = dm[0].shape[0] // 2
        dms = [dmi[:n2c,:n2c].copy() for dmi in dm] \
            + [dmi[n2c:,n2c:].copy() for dmi in dm] \
            + [dmi[n2c:,:n2c].copy() for dmi in dm]
    jks = ('lk->s2ij',) * n_dm \
        + ('ji->s2kl',) * n_dm \
        + ('jk->s1il',) * n_dm
    c1 = .5 / lib.param.LIGHT_SPEED
    vx = _vhf.rdirect_bindm('int2e_pp1_spinor', 's4', jks, dms, 1,
                            mol._atm, mol._bas, mol._env, mf_opt) * c1**2
    vj = numpy.zeros((n_dm,n2c*2,n2c*2), dtype=numpy.complex)
    vk = numpy.zeros((n_dm,n2c*2,n2c*2), dtype=numpy.complex)
    vj[:,n2c:,n2c:] = vx[      :n_dm  ,:,:]
    vj[:,:n2c,:n2c] = vx[n_dm  :n_dm*2,:,:]
    vk[:,n2c:,:n2c] = vx[n_dm*2:      ,:,:]
    if n_dm == 1:
        vj = vj.reshape(vj.shape[1:])
        vk = vk.reshape(vk.shape[1:])
    return dhf._jk_triu_(mol, vj, vk, hermi)


def _call_veff_sf_ssss(mol, dm, hermi=1, mf_opt=None):
    c1 = .5 / lib.param.LIGHT_SPEED
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        n2c = dm.shape[0] // 2
        dms = dm[n2c:,n2c:].copy()
    else:
        n2c = dm[0].shape[0] // 2
        dms = []
        for dmi in dm:
            dms.append(dmi[n2c:,n2c:].copy())
    vj, vk = _vhf.rdirect_mapdm('int2e_pp1pp2_spinor', 's8',
                                ('ji->s2kl', 'jk->s1il'), dms, 1,
                                mol._atm, mol._bas, mol._env, mf_opt) * c1**4
    return dhf._jk_triu_(mol, vj, vk, hermi)


if __name__ == '__main__':
    mol = gto.M(verbose=4,
                atom=[["O", (0., 0., 0.)], [1, (0., -0.757, 0.587)],
                      [1, (0., 0.757, 0.587)]],
                basis='sto-3g')

    mf = scf.RHF(mol).run()
    dm = mf.make_rdm1()
    hso_jk = get_fso2e_bp(mol, dm)
    hso_ref = socutils.get_fso2e_bp(mol, dm)
    print(hso_jk[0])
    print(hso_ref[0])
    print(norm(hso_jk - hso_ref))
