# Author: Xubo Wang <xubo.wang@outlook.com>
# Date: 2024/2/21
import numpy
import numpy as np
import time
import h5py
from pyscf import lib, scf, gto, mcscf
from pyscf.lib import logger, einsum
from functools import reduce
from scipy.linalg import expm as expmat
from numpy.linalg import norm
from .hf_superci import GMRES
from . import zcahf, zcasci, zmcscf, zmc_ao2mo, zmc_superci, davidson

def gen_g_hop_slow(casscf, mo, casdm1, casdm2, eris):
    # first attempt, using a fully mo driven algorithm.
    if casscf.mo_coeff is None:
        casscf.mo_coeff = mo
    ncas = casscf.ncas
    nelecas = casscf.nelecas
    ncore = casscf.ncore
    nocc = ncas + ncore
    nmo = mo.shape[1]

    ################# gradient #################
    dm_core = np.zeros((nmo, nmo), dtype=complex)
    dm_active = np.zeros((nmo, nmo), dtype=complex)
    idx = np.arange(ncore)
    dm_core[idx, idx] = 1
    dm_active[ncore:nocc, ncore:nocc] = casdm1
    dm1 = dm_core + dm_active
    h1e_mo = reduce(np.dot, (mo.T.conj(), casscf.get_hcore(), mo))
    vj_c, vk_c = casscf.get_jk(casscf.mol, reduce(np.dot, (mo, dm_core, mo.T.conj())))
    vj_a, vk_a = casscf.get_jk(casscf.mol, reduce(np.dot, (mo, dm_active, mo.T.conj())))
    vhf_c = reduce(np.dot, (mo.T.conj(), vj_c - vk_c, mo))
    vhf_a = reduce(np.dot, (mo.T.conj(), vj_a - vk_a, mo))
    vhf_ca = vhf_c + vhf_a
    A = np.zeros((nmo, nmo), dtype=complex)
    A[:, :ncore] = h1e_mo[:, :ncore] + vhf_ca[:, :ncore]
    A[:, ncore:nocc] = np.dot(h1e_mo[:, ncore:nocc] + vhf_c[:, ncore:nocc], casdm1)

    paaa = eris.paaa
    g_dm2 = einsum('puvw,tuvw->pt', paaa, casdm2)
    A[:, ncore:nocc] += g_dm2
    g_orb = casscf.pack_uniq_var(A - A.T.conj())

    # define effective fock for evaluation of hessian
    fock_eff = h1e_mo + vhf_ca
    fock_core = h1e_mo + vhf_c

    def hop(x):

        x1 = casscf.unpack_uniq_var(x)

        # copying blocks to put them in contiguous memory
        x1_ai = x1[nocc:,:ncore].copy()
        x1_ti = x1[ncore:nocc,:ncore].copy()
        x1_at = x1[nocc:,ncore:nocc].copy()


        x2 = np.zeros_like(x1, dtype=complex)
        x2_ai = np.zeros_like(x1_ai, dtype=complex)
        x2_ti = np.zeros_like(x1_ti, dtype=complex)
        x2_at = np.zeros_like(x1_at, dtype=complex)

        # h needs to be contracted with x and x^*.
        # the return value is hx + hx^*.

        # aibj 1, aibu 2, aiuj 3, atbj 4, atbu 5, atuj 6, tibj 7, tibu 8, tiuj 9
        # enumerating terms

        # aibj term (1)
        # x^* term, h_{ai,bj}^{\kappa^{*}\kappa^{*}}=g_{ijab}^{*}

        ccvv = eris.pppp[:ncore,:ncore,nocc:,nocc:]
        vccv = eris.pppp[nocc:,:ncore,:ncore,nocc:]
        x2_ai += einsum('ijab,bj->ai',ccvv, x1_ai.conj()) # cd contraction, exchange style

        # x term,
        # h_{ai,bj}^{\kappa^{*}\kappa}=
        #   \dfrac{1}{2}\delta_{ab}(A_{ij}+A_{ji}^{\dagger})
        #   -F_{ab}\delta_{ij}
        #   -g_{ajib}

        x2_ai += 0.5 * einsum('ij,aj->ai', A[:ncore,:ncore], x1_ai)
        x2_ai -= einsum('ab,bi->ai',fock_eff[nocc:,nocc:], x1_ai)
        x2_ai -= einsum('ajib,bj->ai', vccv, x1_ai) # cd contractoin, exchange style

        # c_intermediate
        # C_pirt=C_ript=g_iupr D_ut = -g_uipr D_ut
        # aibu term (2)
        # x^* term,
        # h_{ai,bu}^{\kappa^{*}\kappa^{*}}
        #   =C_{aibu}^{\dagger}
        # C_{aibu}=(g_ivab*D_vu)

        cavv = eris.pppp[:ncore,ncore:nocc,nocc:,nocc:]
        c_cavv = einsum('ivab,vu->iuab', cavv, casdm1) # c_aibu reordered
        x2_ai += einsum('iuab,bu->ai', c_cavv, x1_at).conj()
        
        # buai term (atbj) (4)
        # x^* term,
        # h_{bu,ai}^{\kappa^{*}\kappa^{*}}
        #   =C_{buai}^{\dagger}

        x2_at += einsum('iuab,ai->bu', c_cavv, x1_ai).conj()

        # b intermediate
        # B_{pi,ts} = g_{sipu}D_{tu}
        # B_{pt,is}&=g_{usip}D_{ut}
        # B_{st,ip}&=g_{upis}D_{ut}=g_{sipu}^*D_{tu}^*=B_{pi,ts}^*
        # aibu term (2)
        # x term
        # h_{ai,bu}^{\kappa^{*}\kappa} =
        #   \dfrac{1}{2}\delta_{ab}(A_{iu}+A_{ui}^{\dagger})
        #   +g_{vaib}D_{vu} （B_{bu,ia}) K_vi^ab

        avcv = eris.pppp[ncore:nocc,nocc:,:ncore,nocc:]
        b_vacv = einsum('vaib,vu->buia', avcv, casdm1)
        x2_ai += einsum('buia,bu->ai', b_vacv, x1_at)
        x2_ai += 0.5*einsum('au,iu->ai',x1_at, A[:ncore,ncore:nocc]+A[ncore:nocc,:ncore].conj().T)

        # buai term (4)
        # x term
        # h_{bu,ai}^{\kappa^{*}\kappa}
        #   =\dfrac{1}{2}\delta_{ab}(A_{ui}+A_{iu}^{\dagger})
        #   +B_{bu,ia}^*)

        x2_at += einsum('buia,ai->bu', b_vacv, x1_ai.conj()).conj()
        x2_at += 0.5*einsum('ai,ti->at', x1_ai, A[ncore:nocc,:ncore]+A[:ncore,ncore:nocc].conj().T)

        # aiuj term x^* (3)
        # h_{ai,uj}^{\kappa^{*}\kappa^{*}}
        #  = B_{ju,ia}+\dfrac{1}{2}(C_{ai,uj}^{\dagger}+C_{uj,ai}^{\dagger})
        #  =g_vaij*D_uv+g_{ijau}^{*}

        avcc = eris.pppp[ncore:nocc,nocc:,:ncore,:ncore]
        b_cacv = einsum('vaij,uv->juia', avcc, casdm1)

        x2_ai += einsum('juia,uj->ai', b_cacv, x1_ti.conj())
        x2_ai += einsum('uaji,uj->ai', avcc, x1_ti.conj()) # g_ijau^* = g_uaji

        # ujai term x^* (7)
        # h_{uj,ai}^{\kappa^{*}\kappa^{*}}
        #   =g_vaij^* * D_vu + g_ijau
        # g_ijau,x_uj^*=g_uaji^*,x_uj^*

        x2_ti += einsum('juia,ai->uj', b_cacv, x1_ai).conj()
        x2_ti += einsum('uaji,ai->uj', avcc, x1_ai).conj() # g_ijau = g_uaji^*

        # aiuj term x (3)
        # h_{ai,uj}^{\kappa^{*}\kappa}
        #   =\dfrac{1}{2}\delta_{ij}(A_{ua})-\delta_{ij}F_{au}-g_{ajiu}+g_{ijau}^{*}

        vcca = eris.pppp[nocc:, :ncore, :ncore, ncore:nocc]
        x2_ai += 0.5*einsum('ua,ui->ai', A[ncore:nocc,nocc:], x1_ti)
        x2_ai += -1.0*einsum('au,ui->ai', fock_eff[nocc:,ncore:nocc], x1_ti)
        x2_ai += einsum('ajiu,uj->ai', vcca, x1_ti)

        # ujai term x (7)
        # h_{uj,ai}^{\kappa^{*}\kappa}
        #   = \dfrac{1}{2}\delta_{ij}(A_{ua}^{\dagger}-F_{ua})-g_{uija}+g_{ijau}
        
        x2_ti += 0.5*einsum('ua,ai->ui', A[ncore:nocc,nocc:].conj(), x1_ai)
        x2_ti += -1.0*einsum('ua,ai->ui', fock_eff[ncore:nocc,nocc:], x1_ai)
        x2_ti += -1.0*einsum('ajiu,ai->uj', vcca, x1_ai.conj()).conj()
        x2_ti += einsum('uaji,ai->uj', avcc, x1_ai.conj()).conj() #ccva=avcc^*

        # atbu term x^* (5)
        # h_{at,bu}^{\kappa^{*}\kappa^{*}}
        #  =g_{vwab}d_{vtwu} # J

        aavv = eris.pppp[ncore:nocc,ncore:nocc,nocc:,nocc:]
        h_atbu_ss = einsum('vwab,vtwu->atbu', aavv, casdm2)
        x2_at += einsum('atbu,bu->at', h_atbu_ss, x1_at.conj())

        # atbu term x (5)
        # h_{at,bu}^{\kappa^{*}\kappa}
        #   = \dfrac{1}{2}\left\{ \delta_{ab}(A_{tu}+A_{ut}^{\dagger})\right\}
        #    -F_{ba}^{core}D_{tu}+
        #    (g_{avwb}d_{tuvw})

        vaav = eris.pppp[nocc:,ncore:nocc,ncore:nocc,nocc:]
        h_atbu__s = einsum('avwb,tuvw->atbu', vaav,casdm2)
        x2_at += einsum('atbu,bu->at', h_atbu__s, x1_at)
        # not sure whether act-act block of a matrix is hermitian
        x2_at += 0.5*einsum('tu,at->at', A[ncore:nocc,ncore:nocc]+A[ncore:nocc,ncore:nocc].T.conj(), x1_at)
        x2_at -= einsum('ab,tu,bu->at', fock_core[nocc:,nocc:], casdm1, x1_at)

        # atuj term x^* (6)
        # h_{at,uj}^{\kappa^{*}\kappa^{*}}
        #   =\dfrac{1}{2}\delta_{tu}A_{ja}
        #   +B_{ju,ta} (g_avwj*d_utvw-F^{core}_{ja}D_{ut})
        #   +C_{at,uj}^{\dagger} (C_uj,at=g_jvua*D_vt)C_{pirt}=g_{iupr}D_{ut}
        vaac = eris.pppp[nocc:,ncore:nocc,ncore:nocc,:ncore]
        b_caav = einsum('avwj,utvw->juta', vaac, casdm2)
        c_acva_conj = einsum('auvj,vt->ujat', vaac, casdm1.conj())
        x2_at += 0.5*einsum('ja,uj->au', A[:ncore,nocc:], x1_ti.conj())
        x2_at += einsum('juta,uj->at', b_caav, x1_ti.conj())
        x2_at += einsum('ujat,uj->at', c_acva_conj, x1_ti.conj())

        # ujat term x^* (8)
        # h_{uj,at}^{\kappa^{*}\kappa^{*}}
        #  =B_{ju,ta} = g_avwj*d_tuvw - F^{core}_{aj}D_{tu}
        #  +C_{uj,at}^{\dagger}
        x2_ti += einsum('juta,at->uj', b_caav, x1_at.conj())
        x2_ti += einsum('ujat,at->uj', c_acva_conj, x1_at.conj())

        # atuj term x (6)
        # h_{at,uj}^{\kappa^{*}\kappa}
        #  = B_{uj,ta} g_{uvaj}D_tv
        #   +C_{at,ju}^{\dagger} g_vwaj d_vtwu
        aavc = eris.pppp[ncore:nocc,ncore:nocc,nocc:,:ncore]
        b_ujta = einsum('uvaj,tv->ujta', aavc, casdm1)
        c_vaca = einsum('vwaj,vtwu->atju', aavc, casdm2)
        x2_at += einsum('ujta,uj->at', b_ujta, x1_ti)
        x2_at += einsum('atju,uj->at', c_vaca, x1_ti.conj()).conj()

        # ujat term x (8)
        # h_{uj,at}^{\kappa^{*}\kappa}
        # = 0
        
        x2_ti += 0. # placeholder

        # tiuj term x^* (9)
        # h_{ti,uj}^{\kappa^{*}\kappa^{*}}
        # = B_{it,ju}+B_{ju,it} g_vuji D_vt + g_vtij D_vu
        #   +C_{ju,it}+C_{ti,uj}^{\dagger} g_vwji*d_vuwt + g_ijtu^*

        aacc = eris.pppp[ncore:nocc,ncore:nocc,:ncore,:ncore]
        b_caca = einsum('vuji,vt->itju', aacc, casdm1)
        c_acac = einsum('vwji,vuwt->tiuj', aacc, casdm2)
        x2_ti += einsum('itju,uj->ti', b_caca, x1_ti.conj())
        x2_ti += einsum('juit,uj->ti', b_caca, x1_ti.conj())
        x2_ti += einsum('tiuj,uj->ti', c_acac, x1_ti.conj())

        # tiuj term x (9)
        # h_{ti,uj}^{\kappa^{*}\kappa}
        #   =\delta_{ij}(A_{ut}+A_{tu}^{\dagger}-2F_{tu})
        #   +\delta_{tu}(A_{ij}+A_{ji}^{\dagger})
        #   +B_{it,uj}-g_{tjiu}+C_{it,uj}+C_{ti,ju}^{\dagger}
        acca = eris.pppp[ncore:nocc,:ncore,:ncore,ncore:nocc]
        A_aa = A[ncore:nocc,ncore:nocc]
        A_core = A[:ncore,:ncore]
        x2_ti += einsum('tu,ui->ti', A_aa + A_aa.T.conj(), x1_ti)
        x2_ti += einsum('ij,tj->ti', A_core + A_core.T.conj(), x1_ti)
        caac = eris.pppp[:ncore,ncore:nocc,ncore:nocc,:ncore]
        b_caac = einsum('jvwi,utvw->ituj', caac, casdm2)# b_ptus=g_sxyp d_utxy
        x2_ti += einsum('ituj,uj->ti', b_caac, x1_ti)
        # x2_ti -= einsum('ituj,uj->ti', acca, x1_ti) # clumsy, I don't see where this is from
        c_caac = einsum('jvui,vt->ituj', caac, casdm1) #c_ituj = c_ujit = g_jvui D_vt c_pirt=g_{iupr}D_{ut}
        x2_ti += einsum('ituj,uj->ti', c_caac, x1_ti)
        x2_ti += einsum('juti,uj->ti', c_caac, x1_ti.conj()).conj()
        x2[nocc:,:ncore] = x2_ai
        x2[nocc:,ncore:nocc] = x2_at
        x2[ncore:nocc,:ncore] = x2_ti
        return casscf.pack_uniq_var(x2)
    print(g_orb.shape)
    from scipy.sparse.linalg import LinearOperator
    n_uniq_var = g_orb.shape[0]
    print(n_uniq_var)
    h_op = LinearOperator((n_uniq_var,n_uniq_var), matvec=hop)

    return g_orb, h_op


def gen_g_hop(casscf, mo, casdm1, casdm2, eris):
    # first attempt, using a fully mo driven algorithm.
    # adopt blocks from gen_g_hop_slow one by one
    if casscf.mo_coeff is None:
        casscf.mo_coeff = mo
    ncas = casscf.ncas
    nelecas = casscf.nelecas
    ncore = casscf.ncore
    nocc = ncas + ncore
    nmo = mo.shape[1]

    ################# gradient #################
    dm_core = np.zeros((nmo, nmo), dtype=complex)
    dm_active = np.zeros((nmo, nmo), dtype=complex)
    idx = np.arange(ncore)
    dm_core[idx, idx] = 1
    dm_active[ncore:nocc, ncore:nocc] = casdm1
    dm1 = dm_core + dm_active
    h1e_mo = reduce(np.dot, (mo.T.conj(), casscf.get_hcore(), mo))
    vj_c, vk_c = casscf.get_jk(casscf.mol, reduce(np.dot, (mo, dm_core, mo.T.conj())))
    vj_a, vk_a = casscf.get_jk(casscf.mol, reduce(np.dot, (mo, dm_active, mo.T.conj())))
    vhf_c = reduce(np.dot, (mo.T.conj(), vj_c - vk_c, mo))
    vhf_a = reduce(np.dot, (mo.T.conj(), vj_a - vk_a, mo))
    vhf_ca = vhf_c + vhf_a
    A = np.zeros((nmo, nmo), dtype=complex)
    A[:, :ncore] = h1e_mo[:, :ncore] + vhf_ca[:, :ncore]
    A[:, ncore:nocc] = np.dot(h1e_mo[:, ncore:nocc] + vhf_c[:, ncore:nocc], casdm1)

    paaa = eris.paaa
    g_dm2 = einsum('puvw,tuvw->pt', paaa, casdm2)
    A[:, ncore:nocc] += g_dm2
    g_orb = casscf.pack_uniq_var(A - A.T.conj())

    # define effective fock for evaluation of hessian
    fock_eff = h1e_mo + vhf_ca
    fock_core = h1e_mo + vhf_c

    # hdiag
    y = lib.einsum('pu,qu->pq', (h1e_mo + vhf_c)[ncore:nocc, ncore:nocc], casdm1)
    h_diag = np.ones((nmo, nmo), dtype=complex)
    for v_idx in range(nocc,nmo):
        for i_idx in range(ncore):
            h_diag[v_idx, i_idx]=fock_eff[v_idx,v_idx]-fock_eff[i_idx, i_idx]

    for a_idx in range(ncore, nocc):
        for i_idx in range(ncore):
            d_tt = dm1[a_idx, a_idx]
            h_diag[a_idx, i_idx] = fock_eff[a_idx, a_idx] - (1.-d_tt)*fock_eff[i_idx, i_idx]\
                - y[a_idx-ncore, a_idx-ncore] - g_dm2[a_idx, a_idx-ncore]
    
    for v_idx in range(nocc, nmo):
        for a_idx in range(ncore, nocc):
            h_diag[v_idx, a_idx] = fock_eff[v_idx, v_idx] * dm1[a_idx, a_idx] - y[a_idx-ncore, a_idx-ncore] - g_dm2[a_idx, a_idx-ncore]
    
    #h_{diag,ai}=A_{ii}-F_{aa}-g_{aiia}
    #h_{diag,ti}=2A_{tt}-2F_{tt}+2A_{ii}+B_{itti}-g_{tiit}+C_{itti}+C_{tiit}^{*}
    #h_{diag,at}=A_{tt}-F_{aa}^{core}D_{tt}+g_{avwa}d_{ttvw}

    def hop(x):

        x1 = casscf.unpack_uniq_var(x)

        # copying blocks to put them in contiguous memory
        x1_ai = x1[nocc:,:ncore].copy()
        x1_ti = x1[ncore:nocc,:ncore].copy()
        x1_at = x1[nocc:,ncore:nocc].copy()


        x2 = np.zeros_like(x1, dtype=complex)
        x2_ai = np.zeros_like(x1_ai, dtype=complex)
        x2_ti = np.zeros_like(x1_ti, dtype=complex)
        x2_at = np.zeros_like(x1_at, dtype=complex)

        # h needs to be contracted with x and x^*.
        # the return value is hx + hx^*.

        # aibj 1, aibu 2, aiuj 3, atbj 4, atbu 5, atuj 6, tibj 7, tibu 8, tiuj 9
        # enumerating terms

        # aibj term (1)
        # x^* term, h_{ai,bj}^{\kappa^{*}\kappa^{*}}=g_{ijab}^{*}

        #ccvv = eris.pppp[:ncore,:ncore,nocc:,nocc:]
        #vccv = eris.pppp[nocc:,:ncore,:ncore,nocc:]
        #x2_ai += einsum('ijab,bj->ai',ccvv, x1_ai.conj()) # cd contraction, exchange style
        # switch to cd optimized implementation
        # g_ijab^* x_bj^* = (g_ijab x_bj)^*
        x2_ai += einsum('Lij,Lab,bj->ai', eris.cd_cc, eris.cd_vv, x1_ai).conj()

        # aibj x term,
        # h_{ai,bj}^{\kappa^{*}\kappa}=
        #   \dfrac{1}{2}\delta_{ab}(A_{ij}+A_{ji}^{\dagger})
        #   -F_{ab}\delta_{ij}
        #   -g_{ajib}

        x2_ai += 0.5 * einsum('ij,aj->ai', A[:ncore,:ncore], x1_ai)
        x2_ai -= einsum('ab,bi->ai',fock_eff[nocc:,nocc:], x1_ai)
        x2_ai -= einsum('Laj,Lib,bj->ai', eris.cd_vc, eris.cd_cv, x1_ai)

        # c_intermediate
        # C_pirt=C_ript=g_iupr D_ut = -g_uipr D_ut
        # aibu term (2)
        # x^* term,
        # h_{ai,bu}^{\kappa^{*}\kappa^{*}}
        #   =C_{aibu}^{\dagger}
        # C_{aibu}=(g_ivab*D_vu)

        #cavv = eris.pppp[:ncore,ncore:nocc,nocc:,nocc:]
        #c_cavv = einsum('ivab,vu->iuab', cavv, casdm1) # c_aibu reordered
        # einsum('Liv,Lab,vu,bu->ai')
        cd_ca_1transform = einsum('Liv,vu->Liu', eris.cd_ca, casdm1)
        cd_vv_1contract = einsum('Lab,bu->Lau', eris.cd_vv, x1_at)
        x2_ai += einsum('Liu,Lau->ai', cd_ca_1transform, cd_vv_1contract).conj()
        
        # buai term (atbj) (4)
        # x^* term,
        # h_{bu,ai}^{\kappa^{*}\kappa^{*}}
        #   =C_{buai}^{\dagger}

        # x2_at += einsum('iuab,ai->bu', c_cavv, x1_ai).conj()
        cd_vv_1contract = einsum('Lab,ai->Lib', eris.cd_vv, x1_ai)
        x2_at += einsum('Liu,Lib->bu', cd_ca_1transform, cd_vv_1contract).conj()

        # b intermediate
        # B_{pi,ts} = g_{sipu}D_{tu}
        # B_{pt,is}&=g_{usip}D_{ut}
        # B_{st,ip}&=g_{upis}D_{ut}=g_{sipu}^*D_{tu}^*=B_{pi,ts}^*
        # aibu term (2)
        # x term
        # h_{ai,bu}^{\kappa^{*}\kappa} =
        #   \dfrac{1}{2}\delta_{ab}(A_{iu}+A_{ui}^{\dagger})
        #   +g_{vaib}D_{vu} （B_{bu,ia}) K_vi^ab

        # avcv = eris.pppp[ncore:nocc,nocc:,:ncore,nocc:]
        # b_vacv = einsum('vaib,vu->buia', avcv, casdm1)
        # einsum('Lva,Lib,vu,bu->ai')
        cd_va_1transform = einsum('Lva,vu->Lua', eris.cd_av, casdm1)
        cd_cv_1contract = einsum('Lib,bu->Liu', eris.cd_cv, x1_at)
        x2_ai += einsum('Lua,Liu->ai', cd_va_1transform, cd_cv_1contract)
        x2_ai += 0.5*einsum('au,iu->ai',x1_at, A[:ncore,ncore:nocc]+A[ncore:nocc,:ncore].conj().T)

        # buai term (4)
        # x term
        # h_{bu,ai}^{\kappa^{*}\kappa}
        #   =\dfrac{1}{2}\delta_{ab}(A_{ui}+A_{iu}^{\dagger})
        #   +B_{bu,ia}^*)

        # einsum('Lva,Lib,vu,ai->bu')
        tmp = einsum('Lua,ai->Lui', cd_va_1transform, x1_ai.conj())
        # x2_at += einsum('buia,ai->bu', b_vacv, x1_ai.conj()).conj()
        x2_at += einsum('Lui,Lib->bu', tmp, eris.cd_cv).conj()
        x2_at += 0.5*einsum('ai,ti->at', x1_ai, A[ncore:nocc,:ncore]+A[:ncore,ncore:nocc].conj().T)

        # aiuj term x^* (3)
        # h_{ai,uj}^{\kappa^{*}\kappa^{*}}
        #  = B_{ju,ia}+\dfrac{1}{2}(C_{ai,uj}^{\dagger}+C_{uj,ai}^{\dagger})
        #  =g_vaij*D_uv+g_{ijau}^{*}

        # avcc = eris.pppp[ncore:nocc,nocc:,:ncore,:ncore]
        # b_cacv = einsum('vaij,uv->juia', avcc, casdm1)
        # x2_ai += einsum('juia,uj->ai', b_cacv, x1_ti.conj())

        # einsum('Lva,Lij,uv,uj->ai')
        cd_av_imd = einsum('Lva,vj->Laj', eris.cd_av, einsum('uv,uj->vj', casdm1, x1_ti))
        x2_ai += einsum('Laj,Lij->ai', cd_av_imd, eris.cd_cc)
        
        # einsum('Lua,uj,Lji->ai)
        # x2_ai += einsum('uaji,uj->ai', avcc, x1_ti.conj()) # g_ijau^* = g_uaji
        cd_av_1contract = einsum('Lua,uj->Laj', eris.cd_av, x1_ti.conj())
        x2_ai += einsum('Laj,Lij->ai', cd_av_1contract, eris.cd_cc)

        # ujai term x^* (7)
        # h_{uj,ai}^{\kappa^{*}\kappa^{*}}
        #   =g_vaij^* * D_vu + g_ijau
        # g_ijau,x_uj^*=g_uaji^*,x_uj^*

        # x2_ti += einsum('juia,ai->uj', b_cacv, x1_ai).conj()
        # x2_ti += einsum('uaji,ai->uj', avcc, x1_ai).conj() # g_ijau = g_uaji^*
        # einsum('Lva,Lij,uv,ai->uj)
        cd_av_imd = einsum('Lva,uv,ai->Lui', eris.cd_av, casdm1, x1_ai)
        x2_ti += einsum('Lui,Lij->uj', cd_av_imd, eris.cd_cc).conj()

        # aiuj term x (3)
        # h_{ai,uj}^{\kappa^{*}\kappa}
        #   =\dfrac{1}{2}\delta_{ij}(A_{ua})-\delta_{ij}F_{au}-g_{ajiu}+g_{ijau}^{*}

        x2_ai += 0.5*einsum('ua,ui->ai', A[ncore:nocc,nocc:], x1_ti)
        x2_ai += -1.0*einsum('au,ui->ai', fock_eff[nocc:,ncore:nocc], x1_ti)
        # einsum('Laj,Liu,uj->ai')
        # x2_ai += einsum('ajiu,uj->ai', vcca, x1_ti)
        cd_ca_1contract = einsum('Liu,uj->Lij', eris.cd_ca, x1_ti)
        x2_ai += einsum('Laj,Lij->ai', eris.cd_vc, cd_ca_1contract)
        

        # ujai term x (7)
        # h_{uj,ai}^{\kappa^{*}\kappa}
        #   = \dfrac{1}{2}\delta_{ij}(A_{ua}^{\dagger}-F_{ua})-g_{uija}+g_{ijau}
        
        x2_ti += 0.5*einsum('ua,ai->ui', A[ncore:nocc,nocc:].conj(), x1_ai)
        x2_ti += -1.0*einsum('ua,ai->ui', fock_eff[ncore:nocc,nocc:], x1_ai)
    
        # x2_ti += -1.0*einsum('ajiu,ai->uj', vcca, x1_ai.conj()).conj()
        # x2_ti += einsum('uaji,ai->uj', avcc, x1_ai.conj()).conj() #ccva=avcc^*
        cd_vc_1contract = einsum('Laj,ai->Lij', eris.cd_vc, x1_ai.conj())
        x2_ti += -1.0*einsum('Lij,Liu->uj', cd_vc_1contract, eris.cd_ca).conj()
        cd_av_1contract = einsum('Lua,ai->Lui', eris.cd_av, x1_ai.conj())
        x2_ti += einsum('Lui,Lji->uj', cd_av_1contract, eris.cd_cc).conj()

        # atbu term x^* (5)
        # h_{at,bu}^{\kappa^{*}\kappa^{*}}
        #  =g_{vwab}d_{vtwu} # J

        #aavv = eris.pppp[ncore:nocc,ncore:nocc,nocc:,nocc:]
        #h_atbu_ss = einsum('vwab,vtwu->atbu', aavv, casdm2)
        #x2_at += einsum('atbu,bu->at', h_atbu_ss, x1_at.conj())

        cd_aa_2transform = einsum('Lvw, vtwu->Ltu', eris.cd_aa, casdm2)
        tmp = einsum('Ltu,bu->Ltb', cd_aa_2transform, x1_at.conj())
        x2_at += einsum('Ltb,Lab->at', tmp, eris.cd_vv)
        # atbu term x (5)
        # h_{at,bu}^{\kappa^{*}\kappa}
        #   = \dfrac{1}{2}\left\{ \delta_{ab}(A_{tu}+A_{ut}^{\dagger})\right\}
        #    -F_{ba}^{core}D_{tu}+
        #    (g_{avwb}d_{tuvw})

        vaav = einsum('Lav,Lwb->avwb', eris.cd_va, eris.cd_av)
        h_atbu__s = einsum('avwb,tuvw->atbu', vaav,casdm2)
        x2_at += einsum('atbu,bu->at', h_atbu__s, x1_at)
        # not sure whether act-act block of a matrix is hermitian
        x2_at += 0.5*einsum('tu,at->at', A[ncore:nocc,ncore:nocc]+A[ncore:nocc,ncore:nocc].T.conj(), x1_at)
        x2_at -= einsum('ab,tu,bu->at', fock_core[nocc:,nocc:], casdm1, x1_at)

        # atuj term x^* (6)
        # h_{at,uj}^{\kappa^{*}\kappa^{*}}
        #   =\dfrac{1}{2}\delta_{tu}A_{ja}
        #   +B_{ju,ta} (g_avwj*d_utvw-F^{core}_{ja}D_{ut})
        #   +C_{at,uj}^{\dagger} (C_uj,at=g_jvua*D_vt)C_{pirt}=g_{iupr}D_{ut}
        vaac = einsum('Lav,Lwj->avwj', eris.cd_va, eris.cd_ac)
        b_caav = einsum('avwj,utvw->juta', vaac, casdm2)
        c_acva_conj = einsum('auvj,vt->ujat', vaac, casdm1.conj())
        x2_at += 0.5*einsum('ja,uj->au', A[:ncore,nocc:], x1_ti.conj())
        x2_at += einsum('juta,uj->at', b_caav, x1_ti.conj())
        x2_at += einsum('ujat,uj->at', c_acva_conj, x1_ti.conj())

        # ujat term x^* (8)
        # h_{uj,at}^{\kappa^{*}\kappa^{*}}
        #  =B_{ju,ta} = g_avwj*d_tuvw - F^{core}_{aj}D_{tu}
        #  +C_{uj,at}^{\dagger}
        x2_ti += einsum('juta,at->uj', b_caav, x1_at.conj())
        x2_ti += einsum('ujat,at->uj', c_acva_conj, x1_at.conj())

        # atuj term x (6)
        # h_{at,uj}^{\kappa^{*}\kappa}
        #  = B_{uj,ta} g_{uvaj}D_tv
        #   +C_{at,ju}^{\dagger} g_vwaj d_vtwu
        aavc = einsum('Ltu,Lai->tuai', eris.cd_aa, eris.cd_vc)
        b_ujta = einsum('uvaj,tv->ujta', aavc, casdm1)
        c_vaca = einsum('vwaj,vtwu->atju', aavc, casdm2)
        del aavc
        x2_at += einsum('ujta,uj->at', b_ujta, x1_ti)
        x2_at += einsum('atju,uj->at', c_vaca, x1_ti.conj()).conj()

        # ujat term x (8)
        # h_{uj,at}^{\kappa^{*}\kappa}
        # = 0
        
        x2_ti += 0. # placeholder

        # tiuj term x^* (9)
        # h_{ti,uj}^{\kappa^{*}\kappa^{*}}
        # = B_{it,ju}+B_{ju,it} g_vuji D_vt + g_vtij D_vu
        #   +C_{ju,it}+C_{ti,uj}^{\dagger} g_vwji*d_vuwt + g_ijtu^*

        #aacc = eris.pppp[ncore:nocc,ncore:nocc,:ncore,:ncore]
        cd_aa_2transform = einsum('Lvw,vuwt->Ltu', eris.cd_aa, casdm2)
        #c_acac = einsum('vwji,vuwt->tiuj', aacc, casdm2)
        #x2_ti += einsum('tiuj,uj->ti', c_acac, x1_ti.conj())
        tmp = einsum('Ltu,uj->Ltj', cd_aa_2transform, x1_ti.conj())
        x2_ti += einsum('Ltj,Lji->ti', tmp, eris.cd_cc)
        #b_caca = einsum('vuji,vt->itju', aacc, casdm1)
        cd_aa_1transform = einsum('Lvu,vt->Ltu', eris.cd_aa, casdm1)
        tmp = einsum('Ltu,uj->Ltj', cd_aa_1transform, x1_ti.conj())
        tmp = einsum('Lvu,vt,uj->Ltj', eris.cd_aa, casdm1, x1_ti.conj())
        #x2_ti += einsum('itju,uj->ti', b_caca, x1_ti.conj())
        x2_ti += einsum('Ltj,Lji->ti', tmp, eris.cd_cc)
        #x2_ti += einsum('ituj,ti->uj', b_caca, x1_ti.conj())
        tmp = einsum('vt,ti->vi', casdm1, x1_ti.conj())
        tmp = einsum('Lvu,vi->Liu', eris.cd_aa, tmp)
        x2_ti += einsum('Liu,Lij->uj', tmp, eris.cd_cc)

        # tiuj term x (9)
        # h_{ti,uj}^{\kappa^{*}\kappa}
        #   =\delta_{ij}(A_{ut}+A_{tu}^{\dagger}-2F_{tu})
        #   +\delta_{tu}(A_{ij}+A_{ji}^{\dagger})
        #   +B_{it,uj}-g_{tjiu}+C_{it,uj}+C_{ti,ju}^{\dagger}
        A_aa = A[ncore:nocc,ncore:nocc]
        A_core = A[:ncore,:ncore]
        x2_ti += einsum('tu,ui->ti', A_aa + A_aa.T.conj(), x1_ti)
        x2_ti += einsum('ij,tj->ti', A_core + A_core.T.conj(), x1_ti)
        tmp = einsum('Ljv,uj->Luv', eris.cd_ca, x1_ti)
        tmp = einsum('Luv,utvw->Ltw', tmp, casdm2)
        x2_ti += einsum('Ltw,Lwi->ti', tmp, eris.cd_ac)
        # x2_ti -= einsum('ituj,uj->ti', acca, x1_ti) # clumsy, I don't see where this is from
        #c_caac = einsum('jvui,vt->ituj', caac, casdm1) #c_ituj = c_ujit = g_jvui D_vt c_pirt=g_{iupr}D_{ut}
        
        #x2_ti += einsum('ituj,uj->ti', c_caac, x1_ti)
        #x2_ti += einsum('ituj,ju->ti', c_caac, x1_ti.T.conj()).conj()
        tmp = einsum('Ljv,vt->Ltj', eris.cd_ca, casdm1)
        tmp = einsum('Lui,uj->Lij', eris.cd_ac, x1_ti)
        x2[nocc:,:ncore] = x2_ai
        x2[nocc:,ncore:nocc] = x2_at
        x2[ncore:nocc,:ncore] = x2_ti
        #print(norm(x2_at), norm(x2_ai), norm(x2_ti))
        return casscf.pack_uniq_var(x2)
    
    from scipy.sparse.linalg import LinearOperator
    n_uniq_var = g_orb.shape[0]
    h_op = LinearOperator((n_uniq_var,n_uniq_var), matvec=hop)

    def h_diag_inv(x):
        return x/(casscf.pack_uniq_var(h_diag+h_diag.T))        
    precond = LinearOperator((n_uniq_var, n_uniq_var), h_diag_inv) 
    return g_orb, casscf.pack_uniq_var(h_diag+h_diag.T), hop, precond

def mcscf_superci(mc, mo_coeff, max_stepsize=0.2, conv_tol=1e-9, conv_tol_grad=None,
                  verbose=logger.INFO, cderi=None):
    log = logger.new_logger(mc, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Start Super-CI Spinor MCSCF')
    mol = mc.mol
    #mo = form_kramers(mo_coeff)
    mo = mo_coeff
    nmo = mo_coeff.shape[1]
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    
    mci = mc.view(zcasci.CASCI)
    if cderi is None:
        cderi = zmc_ao2mo.chunked_cholesky(mol)
    #eris = zmc_ao2mo._ERIS(mc, mo, level=4)
    eris = zmc_ao2mo._CDERIS(mc, mo, cderi=cderi, level=4)
    mci = zmcscf._fake_h_for_fast_casci(mc, mo, eris)
    e_tot, e_cas, fcivec = mci.kernel(mo, verbose=verbose)
    mc.e_tot, mc.e_cas = e_tot, e_cas
    mc._finalize()
    #e_tot, e_cas, fcivec = mc.casci(mo, ci0=None, eris=eris)
    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(conv_tol)
        logger.info(mc, 'Set conv_tol_grad to %g', conv_tol_grad)

    conv = False
    norm_gorb = norm_gci = -1
    de, elast = e_tot, e_tot

    t1m = log.timer('Initializing 1-step CASSCF', *cput0)
    casdm1, casdm2 = mc.fcisolver.make_rdm12(fcivec, ncas, mc.nelecas)
    norm_ddm = 1e2
    casdm1_prev = casdm1_last = casdm1
    t3m = t2m = log.timer('CAS DM', *t1m)

    imacro = 0
    xs = []
    ys = []
    rhos = []
    g_prev = None
    x_prev = None
    rejected = False
    trust_radii = 0.7
    e_last = e_tot
    dr = None
    while not conv and imacro < mc.max_cycle_macro:
        g, h_diag, hop, precond = gen_g_hop(mc, mo, casdm1, casdm2, eris)
        g, h_diag, hop, precond = zmc_superci.gen_g_hop(mc, mo, casdm1, casdm2, eris)
        norm_gorb = np.linalg.norm(g)
        print(
            f'Iter {imacro:3d}: E = {e_tot:20.15f}  dE = {de:12.10f}' +
            f'  norm(grad) = {norm_gorb:8.6f} '
        )

        if abs(de) < conv_tol and norm_gorb < conv_tol_grad:
            conv = True
        if conv:
            break

        gbar = g
        x0 = -gbar/h_diag
        from scipy.sparse.linalg import gmres, lgmres
        class gmres_counter(object):
            def __init__(self, disp=True):
                self._disp = disp
                self.niter = 0
                self.callbacks = []
            def __call__(self, rk=None):
                self.callbacks.append(str(rk))
                self.niter += 1
                if self._disp:
                    print('%s' %(str(rk)))
        #counter = gmres_counter()
        #x, _ = gmres(hop, -gbar, maxiter=20, callback=counter, M=precond)
        #x, _ = gmres(hop, -trust_radii*gbar, maxiter=20, M=precond)
        if imacro > 0:
            if not rejected:
                ys.append(g - g_prev)
                xs.append(x_prev)
                rhos.append(np.dot(ys[-1].conj(), xs[-1]))
            if len(ys) > 10:
                ys.pop(0)
                xs.pop(0)
                rhos.pop(0)
            print(rejected)
            print(lib.fp(ys[-1]), lib.fp(xs[-1]))
            gbar, a = zmc_superci.precondition_grad(gbar, xs, ys, rhos)
            x, _ = gmres(hop, -trust_radii*gbar, M=precond, maxiter=20)
            #x, _ = davidson(hop, gbar, h_diag, max_iter=100)
            x = zmc_superci.postprocess_x(x, xs, ys, rhos, a)
        else:
            x, _ = gmres(hop, -trust_radii*gbar, M=precond, maxiter=20)
        #x, _ = GMRES(hop, -gbar, x0=x0)#h_diag)
        #x, _ = davidson(hop, gbar, h_diag)

        dr = mc.unpack_uniq_var(x)
        step_control=max_stepsize
        if norm(dr) > step_control:
            print('step rescaled')
            dr = dr*(step_control/norm(dr))
        rotation = expmat(dr)
        #rotation = ensure_kramers(rotation)
        #print(np.where(abs(rotation-1.0)>0.01), rotation[abs(rotation-1.0)>0.01])
        mo_new = np.dot(mo, rotation)
        # e_tot, e_cas, fcivec, _, _ = mci.kernel(mo)
        #eris = zmc_ao2mo._ERIS(mc, mo_new, level=2)
        eris = zmc_ao2mo._CDERIS(mc, mo_new, cderi=cderi, level=4)
        mci = zmcscf._fake_h_for_fast_casci(mc, mo_new, eris)
        e_tot, e_cas, fcivec = mci.kernel(mo_new, ci0=None, verbose=verbose)

        # trus radius control
        de = e_tot - e_last
        e2 = 0.5 * np.dot(x.T.conj(), g)
        r = de/e2
        print(de, e2)
        if de > 1e-3:
            trust_radii *= 0.7
            print(f'step rejected, shrink trust radius to {trust_radii:.4f}')
            rejected=True
            continue
        if r < 0.25:
            trust_radii *= 0.8
            print(f'r value too small, shrink the trust radius')
        elif r > 0.75:
            if trust_radii < 1.0:
                trust_radii *= 1.25
            print(f'r value pretty large, we can uplift the trust radius')
        else:
            print(f'normal step')
        #if trust_radii < 1e-1:
        #    trust_radii = 1e-1
        #print(trust_radii)
        rotation = expmat(dr)
        nvar = rotation.shape[0]
        for i in range(nvar):
            if abs(rotation[i,i])>1.01 or abs(rotation[i,i]) < 0.99:
                print(rotation[i,i]>1.01, rotation[i,i]<0.99,i,j,rotation[i,i])
            for j in range(i):
                if abs(rotation[i,j])>0.01:
                    print(rotation[i,j], i,j)
        rejected=False
        mo = mo_new
        e_last = e_tot        
        x_prev = x
        g_prev = g

        casdm1, casdm2 = mc.fcisolver.make_rdm12(fcivec, ncas, mc.nelecas)
        '''
        step_control = 0.1#min(0.1, np.linalg.norm(g))
        if np.linalg.norm(dr) > step_control:
            #scale_step_size = 0.1 / np.linalg.norm(dr)
            #dr[abs(dr)>0.05] = 0.005
            print(f'Step size rescaled from {np.linalg.norm(dr)}')
            #dr[abs(dr)>1e-3]=1e-3# *= 0.1/np.linalg.norm(dr)
            dr *= step_control/np.linalg.norm(dr)
        #dr[abs(dr)<1e-9]=0.0
        mo = np.dot(mo, expmat(dr))
        e_last = e_tot
        # e_tot, e_cas, fcivec, _, _ = mci.kernel(mo)
        #eris = zmc_ao2mo._ERIS(mc, mo, level=2)
        eris = zmc_ao2mo._CDERIS(mc, mo, cderi=cderi)
        mci = _fake_h_for_fast_casci(mc, mo, eris)
        e_tot, e_cas, fcivec = mci.kernel(mo, ci0=None)
        de = e_tot - e_last
        '''
        #casdm1, casdm2 = mc.fcisolver.make_rdm12(fcivec, ncas, mc.nelecas, verbose=verbose)
        imacro += 1
        if verbose >= logger.INFO:
            mc.e_tot = e_tot
            mc.e_cas = e_cas
            mc._finalize()
    return conv, e_tot, e_cas, fcivec, mo, None

def mc2step(casscf, mo_coeff, cderi=None, tol=1e-8, conv_tol_grad=None, ci0=None, callback=None,
            verbose=None, dump_chk=True):
    if verbose is None:
        verbose = casscf.verbose
    if callback is None:
        callback = casscf.callback
    if cderi is None:
        cderi = zmc_ao2mo.chunked_cholesky(casscf.mol)

    mo = mo_coeff
    nmo = mo.shape[1]
    ncore = casscf.ncore
    ncas = casscf.ncas
    nocc = ncore + ncas
    eris = zmc_ao2mo._CDERIS(casscf, mo, cderi=cderi, level=2)
    mci = zmcscf._fake_h_for_fast_casci(casscf, mo, eris)
    e_tot, e_cas, fcivec = mci.kernel(mo, verbose=verbose)
    print(e_tot, e_cas, e_tot-e_cas)

    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(tol)
        logger.info(casscf, 'Set conv_tol_grad to %g', conv_tol_grad)
    ah_conv_tol = conv_tol_grad
    conv_tol_ddm = conv_tol_grad * 3
    conv = False
    de, elast = e_tot, e_tot
    casdm1 = 0
    r0 = None

    imacro = 0
    while not conv and imacro < casscf.max_cycle_macro:
        imacro += 1
        casdm1_old = casdm1
        casdm1, casdm2 = casscf.fcisolver.make_rdm12(fcivec, ncas, casscf.nelecas)
        norm_ddm = np.linalg.norm(casdm1 - casdm1_old)
        
        gorb, hdiag, hop, _ = gen_g_hop(casscf, mo, casdm1, casdm2, eris)
        # Dump hdiag and gorb into a hdf5 file
        with h5py.File('hdiag_r.h5', 'w') as f:
            f['hdiag']=hdiag
            f['gorb']=casscf.unpack_uniq_var(gorb)
        exit()
        x, eig, eigs, ihop = davidson.davidson(hop, gorb, hdiag, mmax=5)
        x = casscf.unpack_uniq_var(x)
        u = expmat(x)
        #mo = casscf.rotate_mo(mo)
        mo = np.dot(mo, u)
        eris = zmc_ao2mo._CDERIS(casscf, mo, cderi=cderi, level=2)
        mci = zmcscf._fake_h_for_fast_casci(casscf, mo, eris)
        e_tot, e_cas, fcivec = mci.kernel(mo, verbose=verbose)
        if (abs(de) < tol and norm_gorb < conv_tol_grad and
            norm_ddm < conv_tol_ddm):
            conv = True

    return conv, e_tot, e_cas, fcivec, mo
