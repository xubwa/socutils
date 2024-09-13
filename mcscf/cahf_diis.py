import numpy as np

from functools import reduce

from socutils.mcscf import zcahf, zmcscf, zmc_ao2mo
from pyscf import scf, gto, lib
from pyscf.mcscf import addons
from pyscf.lib import logger


def mc_diis(mc, mo_coeff, mo_cap=None, conv_tol=None, conv_tol_grad=None, active_index=None, verbose=logger.INFO, subspace=None, cderi=None, diis_start=10, damp_factor=0.75, trace_orbital=True):
    t1m = t2m = t3m = (logger.process_clock(), logger.perf_counter())
    log= logger.new_logger(mc, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    if conv_tol is None:
        conv_tol = mc.conv_tol
    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(conv_tol)
    mol = mc.mol
    nmo = mo_coeff.shape[1]
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    s1e = mc._scf.get_ovlp()

    mc_diis = scf.diis.CDIIS()
    h1e = mc._scf.get_hcore()
    #_, mc_diis.Corth = mc._scf.eig(h1e, s1e)
    mc_diis.Corth=mo_coeff


    mo_initial = mo_coeff.copy()
    mo_initial_active = mo_initial[:,ncore:nocc]
    
    if cderi is None:
        cderi = zmcscf.chunked_cholesky(mol)

    imacro = 0
    conv = False
    e_last = 0.0
    t2m = t3m = log.timer('Initialize MCSCF', *t2m)
    mo = mo_coeff
    fock = None
    print(f'max_cycle:{mc.max_cycle_macro}')
    while not conv and imacro < mc.max_cycle_macro:
        eris = zmc_ao2mo._CDERIS(mc, mo, cderi=cderi, level=2)
        t3m = log.timer('Integral transformation', *t3m)

        mci = zmcscf._fake_h_for_fast_casci(mc, mo, eris)
        e_tot, e_cas, fcivec = mci.kernel(mo, verbose=verbose)
        print(e_tot, e_cas)
        mc.e_tot, mc.e_cas = e_tot, e_cas

        casdm1, casdm2 = mc.fcisolver.make_rdm12(fcivec, ncas, mc.nelecas)
        dm_core = np.zeros((nmo, nmo), dtype=complex)
        dm_active = np.zeros((nmo, nmo), dtype=complex)
        idx = np.arange(ncore)
        dm_core[idx, idx] = 1 
        dm_active[ncore:nocc, ncore:nocc] = casdm1
        dm1 = dm_core + dm_active
        dm1_ao = reduce(np.dot, (mo, dm1, mo.T.conj()))
        h1e_mo = reduce(np.dot, (mo.T.conj(), h1e, mo))
        vj_c, vk_c = mc.get_jk(mc.mol,
                                       reduce(np.dot, (mo, dm_core, mo.T.conj())))
        vj_a, vk_a = mc.get_jk(mc.mol,
                                       reduce(np.dot, (mo, dm_active, mo.T.conj())))
        vhf_c = reduce(np.dot, (mo.T.conj(), vj_c - vk_c, mo))
        vhf_a = reduce(np.dot, (mo.T.conj(), vj_a - vk_a, mo))
        vhf_ca = vhf_c + vhf_a

        g = np.zeros((nmo, nmo), dtype=complex)
        #g[:, :ncore] = h1e_mo[:, :ncore] + vhf_ca[:, :ncore]
        #g[:, ncore:nocc] = np.dot(
        #h1e_mo[:, ncore:nocc] + vhf_c[:, ncore:nocc], casdm1)
        g[ncore:, ncore:nocc] = np.dot(
            h1e_mo[:, ncore:nocc] + vhf_c[:, ncore:nocc], casdm1)[ncore:,:]
        g[ncore:, :ncore] = h1e_mo[ncore:, :ncore] + vhf_ca[ncore:, :ncore]
        g[nocc:,ncore:nocc] = np.dot(h1e_mo[nocc:,ncore:nocc] + vhf_c[nocc:,ncore:nocc], casdm1)
        g[ncore:nocc,:ncore] = np.dot(
        casdm1, h1e_mo[ncore:nocc, :ncore] + vhf_c[ncore:nocc,:ncore])
        cd_pa = eris.cd_pa
        cd_aa = eris.cd_aa
        
        tmp = lib.einsum('Lvw,tuvw->Ltu', cd_aa, casdm2)
        g_dm2 = lib.einsum('Lpu,Ltu->pt', cd_pa, tmp)

        g[ncore:nocc,:ncore] += g_dm2.T.conj()[:,:ncore]
        g[nocc:,ncore:nocc] += g_dm2[nocc:,:]
        mo_active = np.dot(mo[:,ncore:nocc], casdm1)
        mo_core = mo[:,:ncore]
        mo_virt = mo[:,nocc:]
        
        fock_dm2_ca = reduce(np.dot, (mo_core, g_dm2[:ncore], mo_active.T.conj()))
        fock_dm2_aa = reduce(np.dot, (mo_active, g_dm2[ncore:nocc,:], mo_active.T.conj()))
        fock_dm2_ai = reduce(np.dot, (mo_virt, g_dm2[nocc:,:], mo_active.T.conj()))
        fock_dm2 = fock_dm2_ca+fock_dm2_ca.T.conj()+fock_dm2_aa+fock_dm2_ai+fock_dm2_ai.T.conj()
        #print(np.linalg.norm(fock_dm2), np.linalg.norm(fock_dm2-fock_dm2.T.conj()))
        fock_prev = fock
        nelecas = mc.nelecas
        #f = nelecas / ncas
        #a = (ncas-1) * nelecas / (nelecas-1) / ncas
        #alpha = (1.-a)/(1.-f)
        fock = vhf_c + h1e_mo

        fock[:ncore,ncore:nocc] = np.dot((h1e_mo+vhf_c)[:ncore,ncore:nocc], casdm1)
        fock[:ncore,ncore:nocc] += g_dm2[:ncore,:]
        fock[ncore:nocc,:ncore] = fock[:ncore,ncore:nocc].T.conj().copy()

        fock[nocc:,ncore:nocc] = np.dot((h1e_mo+vhf_c)[nocc:,ncore:nocc], casdm1)
        fock[nocc:,ncore:nocc] += g_dm2[nocc:,:]
        fock[ncore:nocc,nocc:] = fock[nocc:,ncore:nocc].T.conj()

        fock[ncore:nocc,ncore:nocc] = np.dot(casdm1, (h1e_mo+vhf_c)[ncore:nocc,ncore:nocc])

        fock[:ncore,nocc:] += vhf_a[:ncore,nocc:]
        fock[nocc:,:ncore] += vhf_a[nocc:,:ncore]

        fock[:ncore,:ncore] += vhf_a[:ncore,:ncore]
        fock[nocc:,nocc:] += vhf_a[nocc:,nocc:]
        fock[ncore:nocc,ncore:nocc] += g_dm2[ncore:nocc,:]

        grad = mc.pack_uniq_var(g-g.T.conj())
        grad2 = mc.pack_uniq_var(fock)
        grad = g - g.T.conj()
        grad[:ncore,:ncore]=0.0
        grad[ncore:nocc,ncore:nocc]=0.0
        grad[nocc:,nocc:] = 0.0
        fock = reduce(np.dot, (s1e, mo, fock, mo.T.conj(), s1e.T.conj()))

        ################################################################################################
        # evaluate open shell fock contribution avoid the usage of eri under mo basis.
        occ_active = mc.nelecas / mc.ncas
        occ_core = 1.0
        coupling_active = (mc.nelecas - 1) / (mc.ncas - 1)

        vhf_active = vhf_a / occ_active
        fock_mo = np.zeros((nmo, nmo), dtype=complex)
        fock_active = (h1e_mo+vhf_c) + vhf_active * coupling_active
        fock_core = h1e_mo + vhf_c + vhf_active * occ_active
        fock_mo = vhf_c + h1e_mo # fock_c
        fock_mo[:ncore,:ncore] = fock_core[:ncore,:ncore] # fock_c
        fock_mo[nocc:,nocc:] = fock_core[nocc:,nocc:] # fock_c
        fock_mo[nocc:,:ncore] = fock_core[nocc:,:ncore] # fock_c
        fock_mo[:ncore,nocc:] = fock_core[:ncore:,nocc:] # fock_c

        fock_mo[:ncore,ncore:nocc] = fock_core[:ncore,ncore:nocc]*1.0 - fock_active[:ncore,ncore:nocc]*occ_active
        fock_mo[ncore:nocc,:ncore] = fock_mo[:ncore,ncore:nocc].T.conj().copy()

        fock_mo[nocc:,ncore:nocc] = fock_active[nocc:,ncore:nocc]*occ_active
        fock_mo[ncore:nocc,nocc:] = fock_mo[nocc:,ncore:nocc].T.conj()

        fock_mo[ncore:nocc,ncore:nocc] = fock_active[ncore:nocc,ncore:nocc]*occ_active
        
        fock_ao = reduce(np.dot, (s1e, mo, fock_mo, mo.T.conj(), s1e.T.conj()))

        #print('difference', np.linalg.norm(fock-fock_ao))
        #new block ends.
        ################################################################################################
        #fock_diis = reduce(np.dot, (s1e, mo, vhf_c + h1e_mo + vhf_a, mo.T.conj(), s1e.T.conj()))
        def damping(f, f_prev, factor):
            return f*(1-factor) + f_prev*factor
        if imacro > diis_start:# and abs(e_tot - e_last) < 1e-2:
            print('diis step')
            #fock = mc_diis.update(fock, grad_ao)
            fock = mc_diis.update(s1e, dm1_ao, fock)
        elif imacro > 0:
            print(f'damping step with factor {damp_factor}')
            fock = damping(fock, fock_prev, damp_factor)
        log.info(f'cycle {imacro}: E = {e_tot}  dE = {e_tot - e_last}  |g|={np.linalg.norm(grad):.4g}, {np.linalg.norm(grad2):.4g}')
        if (abs(e_tot - e_last) < conv_tol) and (np.linalg.norm(grad) < conv_tol_grad):
            conv = True
        e_last = e_tot
        mo_energy, mo = mc._scf.eig(fock, s1e)
        mo_unsrt = mo.copy()
        #print(mo_energy.irrep_tag)
        if active_index is not None:
            mo = addons.sort_mo(mc, mo, active_index, base=1)
            print(mo_energy[np.array(active_index)-1])
        # pick mo with max overlap with original active space
        elif trace_orbital is True:
            picked_index = np.zeros(ncas, dtype=int)
            '''
            for i in range(mc.ncas):
                mo_initial = mo_initial_active[:,i]
                ovlp = lib.einsum('i,j,jp->p', moi.T, s1e, mo)
                #print(f'ovlp:{ovlp}, picked index:{np.argmax(np.abs(ovlp))}')
                picked_index[i] = np.argmax(np.abs(ovlp))
            '''
            if subspace is None:
                subspace = [ncas]

            for ispace, space in enumerate(subspace):
                if mo_cap is None:
                    mo_cap = mo.shape[1]
                start = sum(subspace[:ispace])
                end = start + space
                ovlp = lib.einsum('ia,ij,jp->ap', mo_initial_active[:,start:end].conj(), s1e, mo)
                ovlp = lib.einsum('ap,ap->p', ovlp, ovlp.conj())
                print(start, end, subspace)
                print(ovlp.shape)
                picked_index[start:end] = np.argsort(np.abs(ovlp))[-space:]
                # try implement a eneryg augmented scheme to pick orbital
                idx_list = []
                energy_list = []
                ovlp_list = []
                for idx, ovlp_i in enumerate(ovlp):
                    if idx < mo_cap: # a arbitrary threshold
                        idx_list.append(idx)
                        energy_list.append(mo_energy[idx])
                        ovlp_list.append(ovlp_i)
                idx_list = np.array(idx_list)
                energy_list = np.array(energy_list)
                ovlp_list = np.array(ovlp_list)
                #picked_index[start:end] = idx_list[np.argsort(energy_list)[:space]]
                picked_index[start:end] = idx_list[np.argsort(ovlp_list)[-space:]]
                print(np.argsort(np.abs(ovlp_list))[-space-4:])
                print(np.sort(np.abs(ovlp))[-space-4:])
                print(picked_index)
            print(picked_index)
            print(mo_energy[picked_index])
            mo = addons.sort_mo(mc, mo, picked_index, base=0)
        imacro += 1
        mc.mo_coeff = mo
    return mc, mo_unsrt, mo_energy
 
