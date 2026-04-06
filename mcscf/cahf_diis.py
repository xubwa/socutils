import numpy as np

from functools import reduce

from socutils.mcscf import zcahf, zmcscf, zmc_ao2mo
from socutils.tools import analyze
from pyscf import scf, gto, lib
from pyscf.mcscf import addons
from pyscf.lib import logger


def mc_diis(mc, mo_coeff, target_orbital=None, mo_cap=None, conv_tol=None,
    conv_tol_grad=None, active_index=None, verbose=logger.INFO,
    subspace=None, cderi=None, diis_start=10, damp_factor=0.75, level_shift=0.0,
    scaling_factor=None,
    trace_orbital=True, frozen=None, initial_ovlp=True, drive='mo'):
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
    mc.mo_coeff = mo_coeff

    mc_diis = scf.diis.CDIIS()
    h1e = mc._scf.get_hcore()
    _, mc_diis.Corth = mc._scf.eig(h1e, s1e)
    #mc_diis.Corth=mo_coeff


    mo_initial = mo_coeff.copy()
    mo_initial_active = mo_initial[:,ncore:nocc]
    if target_orbital is not None:
        mo_initial_active = target_orbital
    

    imacro = 0
    conv = False
    e_last = 0.0
    t2m = t3m = log.timer('Initialize MCSCF', *t2m)
    mo = mo_coeff
    fock = None
    vj_c, vk_c = None, None  # cache core JK to avoid redundant get_jk call in CASCI
    print(f'max_cycle:{mc.max_cycle_macro}')
    while not conv and imacro < mc.max_cycle_macro:
        eris = zmc_ao2mo._CDERIS(mc, mo, cderi=cderi, level=2)
        t3m = log.timer('Integral transformation', *t3m)

        mci = zmcscf._fake_h_for_fast_casci(mc, mo, eris)
        # if cached vj_c, vk_c available, precompute h1eff and energy_core
        # to avoid redundant get_jk call inside CASCI kernel
        if vj_c is not None:
            mo_core = mo[:, :ncore]
            mo_cas = mo[:, ncore:nocc]
            dm_core_ao = np.dot(mo_core, mo_core.T.conj())
            corevhf = vj_c - vk_c
            h1eff = reduce(np.dot, (mo_cas.T.conj(), h1e + corevhf, mo_cas))
            energy_core = mc.energy_nuc()
            energy_core += np.einsum('ij,ji', dm_core_ao, h1e)
            energy_core += np.einsum('ij,ji', dm_core_ao, corevhf) * 0.5
            mci.get_h1eff = lambda *args, h1=h1eff, ec=energy_core: (h1, ec)
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
        dm_core_ao = reduce(np.dot, (mo, dm_core, mo.T.conj()))
        dm_act_ao = reduce(np.dot, (mo, dm_active, mo.T.conj()))
        core_occ = dm_core.diagonal()
        act_occ = dm_active.diagonal()
        #vj_c, vk_c = mc.get_jk(mc.mol, reduce(np.dot, (mo, dm_core, mo.T.conj())))
        #vj_a, vk_a = mc.get_jk(mc.mol, reduce(np.dot, (mo, dm_active, mo.T.conj())))
        print('core fock')
        vj_c, vk_c = eris.get_jk(dm_core_ao, mo_coeff=mo, mo_occ=core_occ)
        vhf_c = reduce(np.dot, (mo.T.conj(), vj_c - vk_c, mo))
        print('active fock')
        vj_a_mo, vk_a_mo = eris.get_jk_active_mo(casdm1)
        vhf_a = vj_a_mo - vk_a_mo
        vhf_ca = vhf_c + vhf_a

        g = np.zeros((nmo, nmo), dtype=complex)
        g[:, :ncore] = h1e_mo[:, :ncore] + vhf_ca[:, :ncore]
        g[:, ncore:nocc] = np.dot(h1e_mo[:, ncore:nocc] + vhf_c[:, ncore:nocc], casdm1)

        cd_pa = eris.cd_pa
        cd_aa = eris.cd_aa
        
        tmp = lib.einsum('Lvw,tuvw->Ltu', cd_aa, casdm2)
        g_dm2 = lib.einsum('Lpu,Ltu->pt', cd_pa, tmp)

        g[:, ncore:nocc] += g_dm2

        mo_active = np.dot(mo[:,ncore:nocc], casdm1)
        mo_core = mo[:,:ncore]
        mo_virt = mo[:,nocc:]
        
        fock_prev = fock
        nelecas = mc.nelecas
        #f = nelecas / ncas
        #a = (ncas-1) * nelecas / (nelecas-1) / ncas
        #alpha = (1.-a)/(1.-f)
        fock = vhf_c + h1e_mo
        fock[:ncore,:ncore] += vhf_a[:ncore,:ncore]
        fock[nocc:,nocc:] += vhf_a[nocc:,nocc:]
        fock[nocc:,:ncore] += vhf_a[nocc:,:ncore]
        fock[:ncore,nocc:] += vhf_a[:ncore,nocc:]
        #fock[ncore:nocc,ncore:nocc] += vhf_a[ncore:nocc,ncore:nocc]
        fock[:ncore,ncore:nocc] += vhf_a[:ncore,ncore:nocc] - g[:ncore, ncore:nocc]
        fock[ncore:nocc, :ncore] = fock[:ncore,ncore:nocc].T.conj().copy()
        fock[nocc:, ncore:nocc] = g[nocc:,ncore:nocc] 
        fock[ncore:nocc,nocc:] = fock[nocc:,ncore:nocc].T.conj()
        fock[ncore:nocc, ncore:nocc] = g[ncore:nocc,ncore:nocc]
        grad = g - g.T.conj()
        if frozen is not None:
            fock_keep = fock[frozen, frozen]
            grad_keep = grad[frozen, frozen]
            fock[frozen, :] = 0.0
            fock[:, frozen] = 0.0
            grad[frozen, :] = 0.0
            grad[:, frozen] = 0.0
            fock[frozen, frozen] = fock_keep
            grad[frozen, frozen] = grad_keep
   
        grad = mc.pack_uniq_var(grad)
        grad1 = mc.pack_uniq_var(fock)

        #casdm1_inv = casdm1.copy()
        #casdm1_inv = np.diag(1.0/np.diagonal(casdm1))
        #for i in range(ncore,nocc):
        #    fock[i,i] /= casdm1[i-ncore,i-ncore]
        #    print(i, fock[i,i])
        #fock[ncore:nocc,ncore:nocc] = np.dot(fock[ncore:nocc,ncore:nocc], casdm1_inv)
        print(f'applying level shift {level_shift}')
        fock[ncore:nocc,ncore:nocc] -= np.diag(np.full(nocc-ncore,level_shift))
        if scaling_factor is not None:
            fock[ncore:nocc,ncore:nocc] /= scaling_factor
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
        print(f'applying level shift {level_shift}')
        fock_mo[ncore:nocc,ncore:nocc] -= np.diag(np.full(nocc-ncore,level_shift))
        if scaling_factor is not None:
            fock_mo[ncore:nocc,ncore:nocc] /= scaling_factor
        #casdm1_inv = casdm1.copy()
        #casdm1_inv = np.diag(1.0/np.diagonal(casdm1))
        #fock_mo[ncore:nocc,ncore:nocc] = np.dot(fock_mo[ncore:nocc,ncore:nocc], casdm1_inv)
        grad2 = mc.pack_uniq_var(fock_mo)
        fock_ao = reduce(np.dot, (s1e, mo, fock_mo, mo.T.conj(), s1e.T.conj()))


        print('difference', np.linalg.norm(fock-fock_ao))
        print('difference', np.linalg.norm(mc.unpack_uniq_var(grad1)-mc.unpack_uniq_var(grad2)))
        if drive == 'ao':
            fock = fock_ao
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
        log.info(f'cycle {imacro}: E = {e_tot}  dE = {e_tot - e_last}  |g|={np.linalg.norm(grad):.4g}, {np.linalg.norm(grad1):.4g}')
        #if (abs(e_tot - e_last) < conv_tol): # and (np.linalg.norm(grad2) < conv_tol_grad):
        if (abs(e_tot - e_last) < conv_tol) and (np.linalg.norm(grad1) < conv_tol_grad):
            conv = True
        e_last = e_tot
        mo_old = mo
        mo_energy, mo = mc._scf.eig(fock, s1e)
        print(mo_energy)
        mo_unsrt = mo.copy()
        if hasattr(mc._scf, "irrep_mo") and mc._scf.irrep_mo is not None:
            mo_unsrt = lib.tag_array(mo.copy(), orbsym=mc._scf.irrep_mo)
            print(mo_energy.irrep_tag[:nocc])
        if active_index is not None:
            mo = addons.sort_mo(mc, mo_unsrt, active_index, base=1)
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
                if initial_ovlp:
                    ovlp = lib.einsum('ia,ij,jp->ap', mo_initial_active[:,start:end].conj(), s1e, mo)
                else:
                    print('ovlp with previous step mo')
                    print(start, end)
                    ovlp = lib.einsum('ia,ij,jp->ap', mc.mo_coeff[:,ncore:nocc].conj(), s1e, mo)
                ovlp = lib.einsum('ap,ap->p', ovlp, ovlp.conj())
                print(start, end, subspace)
                print(ovlp.shape)
                picked_index[start:end] = np.argsort(np.abs(ovlp))[-space:]
                # try implement a energy augmented scheme to pick orbital
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
            if hasattr(mc._scf, "irrep_mo") and mc._scf.irrep_mo is not None:
                mo = addons.sort_mo(mc, lib.tag_array(mo, orbsym=mc._scf.irrep_mo), picked_index, base=0)
            else:
                mo = addons.sort_mo(mc, mo_unsrt, picked_index, base=0)
            analyze.analyze(mol, mo[:,ncore:nocc], mo_energy[picked_index])
        else:
            print('active orbital selected by energy ordering')
            analyze.analyze(mol, mo[:,ncore:nocc], mo_energy[ncore:nocc])
        imacro += 1
        mc.mo_coeff = mo.copy()
        #mc.irrep = mo.orbsym
    return mc, mo_unsrt, mo_energy
 
