import numpy, scipy
from functools import reduce
from pyscf.lib.parameters import LIGHT_SPEED
from pyscf import lib
from socutils.somf import x2c_grad
from socutils.tools.spinor2sph import pauli_decompose

x2camf  = None
try:
    import x2camf
except ImportError:
    pass


def get_psoc_pbc_x2camf(cell, kpts=None, gaunt=True, gauge=True, atm_pt=True):
    '''
    Perturbative treatment of spin-orbit coupling within X2CAMF scheme.
    The SOC contributions are evaluated in a consistent way to include only first-order terms.

    Ref.
    Mol. Phys. 118, e1768313 (2020); DOI:10.1080/00268976.2020.1768313
    Attention: The formula in the reference paper correspond to the case of atm_pt = False.

    Args:
        cell: cell object
        gaunt: include Gaunt integrals
        gauge: include gauge integrals
        atm_pt: use atomic perturbation
        atm_pt = False:
            The molecular one-electron 4c SO integral (Wso) and AMF two-electron 4c SO integrals are
            treated as the perturbation together.
        atm_pt = True:
            The molecular one-electron 4c SO integral (Wso) is treated as the perturbation solely and
            then augmented with the AMF pt integrals.
        These two choices are supposed to give very similar results.

    Returns:
        Perturbative SOC integrals in pauli representation
    '''
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    
    from pyscf.pbc.x2c import sfx2c1e as pbc_sfx2c1e
    from pyscf.pbc.x2c import x2c1e as pbc_x2c1e
    xcell, contr_coeff = pbc_x2c1e.SpinOrbitalX2C1EHelper(cell).get_xmol()
    from pyscf.pbc.df import df
    with_df = df.DF(xcell)

    c = LIGHT_SPEED
    t_aa = xcell.pbc_intor('int1e_kin', 1, lib.HERMITIAN, kpts_lst)
    t    = numpy.array([pbc_x2c1e._block_diag(t_aa[k]) for k in range(len(kpts_lst))])
    s_aa = xcell.pbc_intor('int1e_ovlp', 1, lib.HERMITIAN, kpts_lst)
    s = numpy.array([pbc_x2c1e._block_diag(s_aa[k]) for k in range(len(kpts_lst))])
    if cell.pseudo:
        raise NotImplementedError
    else:
        v_aa = lib.asarray(with_df.get_nuc(kpts_lst))
        v = numpy.array([pbc_x2c1e._block_diag(v_aa[k]) for k in range(len(kpts_lst))])
    
    wsf_aa = pbc_sfx2c1e.get_pnucp(with_df, kpts_lst)
    wso_blocks = pbc_x2c1e.get_pbc_pvxp(with_df, kpts_lst)
    wsf_aa_zero = numpy.zeros_like(wsf_aa)
    wso_blocks_zero = numpy.zeros_like(wso_blocks)
    wsf = []
    wso = []
    for k in range(len(kpts_lst)):
        w_spblk = numpy.vstack([wso_blocks[k], wsf_aa_zero[k,None]])
        wso.append(pbc_x2c1e._sigma_dot(w_spblk))
        w_spblk = numpy.vstack([wso_blocks_zero[k], wsf_aa[k,None]])
        wsf.append(pbc_x2c1e._sigma_dot(w_spblk))
    wso = lib.asarray(wso)
    wsf = lib.asarray(wsf)

    if(atm_pt):
        amfi = x2camf.amfi(pbc_x2c1e.SpinOrbitalX2C1EHelper(cell), 
                           printLevel = cell.verbose,
                           with_gaunt=gaunt, with_gauge=gauge, pt=True, int4c=False)
    else:
        amfi = x2camf.amfi(pbc_x2c1e.SpinOrbitalX2C1EHelper(cell), 
                           printLevel = cell.verbose,
                           with_gaunt=gaunt, with_gauge=gauge, pt=True, int4c=True)
    
    hfw1_kpts = []
    for k in range(len(kpts_lst)):
        n2c = wsf[k].shape[0]
        n4c = n2c * 2

        h4c1 = numpy.zeros((n4c, n4c), dtype=wso[k].dtype)
        h4c1[n2c:, n2c:] = wso[k] * (.25 / c**2)

        if(atm_pt):
            hfw1k = x2c_grad.x2c1e_hfw1(xcell,h4c1,t=t[k],v=v[k],s=s[k],w=wsf)
            hfw1k += amfi
        else:
            h4c1 += amfi
            hfw1k = x2c_grad.x2c1e_hfw1(xcell,h4c1,t=t[k],v=v[k],s=s[k],w=wsf)
    
        hfw1_sphk = pauli_decompose(hfw1k)
        # convert back to contracted basis
        result = numpy.zeros((4, cell.nao_nr(), cell.nao_nr()), dtype=hfw1_sphk.dtype)
        for ic in range(4):
            result[ic] = reduce(numpy.dot, (contr_coeff.T, hfw1_sphk[ic], contr_coeff))
        hfw1_kpts.append(result)
    if kpts is None or numpy.shape(kpts) == (3,):
        hfw1_kpts = hfw1_kpts[0]
    return hfw1_kpts
