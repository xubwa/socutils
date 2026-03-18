import numpy as np

from functools import reduce

from socutils.mcscf import zcahf, zmcscf, zmc_ao2mo
from pyscf import scf, gto, lib
from pyscf.mcscf import addons
from pyscf.lib import logger


def mc_diis(mc, mo_coeff, target_orbital=None, mo_cap=None, conv_tol=None,
    conv_tol_grad=None, active_index=None, verbose=logger.INFO,
    subspace=None, diis_start=10, damp_factor=0.75, level_shift=0.0,
    trace_orbital=True, frozen=None, initial_ovlp=True, drive='mo'):
    t1m = t2m = t3m = (logger.process_clock(), logger.perf_counter())
    log= logger.new_logger(mc, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    if conv_tol is None:
        conv_tol = mc.conv_tol
    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(conv_tol)
    mol = mc.mol
    nmo_4c = mo_coeff.shape[1]
    nNeg = nmo_4c // 2 # number of negative energy states
    ncore = mc.ncore + nNeg
    ncas = mc.ncas
    nocc = ncore + ncas
    s1e = mc._scf.get_ovlp()

    mc_diis = scf.diis.CDIIS()
    h1e = mc._scf.get_hcore()
    #_, mc_diis.Corth = mc._scf.eig(h1e, s1e)
    mc_diis.Corth=mo_coeff


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
    print(f'max_cycle:{mc.max_cycle_macro}')
    while not conv and imacro < mc.max_cycle_macro:
        casdm1, casdm2 = mc.fcisolver.make_rdm12(None, ncas, mc.nelecas)
        dm_core = np.zeros((nmo_4c, nmo_4c), dtype=complex)
        dm_active = np.zeros((nmo_4c, nmo_4c), dtype=complex)
        idx = np.arange(nNeg, ncore)
        dm_core[idx, idx] = 1 
        dm_active[ncore:nocc, ncore:nocc] = casdm1
        dm1 = dm_core + dm_active
        dm1_ao = reduce(np.dot, (mo, dm1, mo.T.conj()))
        h1e_mo = reduce(np.dot, (mo.T.conj(), h1e, mo))
        vj_c, vk_c = mc.get_jk(mc.mol, reduce(np.dot, (mo, dm_core, mo.T.conj())))
        vj_a, vk_a = mc.get_jk(mc.mol, reduce(np.dot, (mo, dm_active, mo.T.conj())))
        vhf_c = reduce(np.dot, (mo.T.conj(), vj_c - vk_c, mo))
        vhf_a = reduce(np.dot, (mo.T.conj(), vj_a - vk_a, mo))
        vhf_ca = vhf_c + vhf_a

        # this is not the actual energy expression for CAHF.
        # I am putting it here as a placeholder.
        e_tot = np.einsum('ij,ji->', dm1, h1e_mo).real + mol.energy_nuc()
        e_coul = 0.5*np.einsum('ij,ji->', vhf_ca, dm1).real
        print(dm1, e_tot, e_coul)
        e_tot += e_coul

        fock_prev = fock
        ################################################################################################
        # evaluate open shell fock contribution avoid the usage of eri under mo basis.
        occ_active = mc.nelecas / mc.ncas
        occ_core = 1.0
        coupling_active = (mc.nelecas - 1) / (mc.ncas - 1)

        vhf_active = vhf_a / occ_active
        fock_mo = np.zeros((nmo_4c, nmo_4c), dtype=complex)
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
        #casdm1_inv = casdm1.copy()
        #casdm1_inv = np.diag(1.0/np.diagonal(casdm1))
        #fock_mo[ncore:nocc,ncore:nocc] = np.dot(fock_mo[ncore:nocc,ncore:nocc], casdm1_inv)
        #grad2 = mc.pack_uniq_var(fock_mo)
        mask = np.zeros((nmo_4c, nmo_4c), dtype=bool)
        mask[ncore:nocc, :ncore] = True
        mask[nocc:,:nocc] = True
        grad2 = fock_mo[mask]
        print(grad2.shape)
        fock_ao = reduce(np.dot, (s1e, mo, fock_mo, mo.T.conj(), s1e.T.conj()))


        if drive == 'ao':
            fock = fock_ao
        else:
            raise NotImplementedError
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
        log.info(f'cycle {imacro}: E = {e_tot}  dE = {e_tot - e_last}  |g|={np.linalg.norm(grad2):.4g}')
        #if (abs(e_tot - e_last) < conv_tol): # and (np.linalg.norm(grad2) < conv_tol_grad):
        if (abs(e_tot-e_last) < conv_tol) and (np.linalg.norm(grad2) < conv_tol_grad):
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
            #mo = addons.sort_mo(mc, lib.tag_array(mo, orbsym=mc._scf.irrep_mo), picked_index, base=0)
            mo = addons.sort_mo(mc, mo_unsrt, picked_index, base=0)
            from socutils.tools import analyze
            #analyze.analyze(mol, mo[nNeg:,ncore:nocc], mo_energy[picked_index])
        imacro += 1
        mc.mo_coeff = mo.copy()
        #mc.irrep = mo.orbsym
    return mc, mo_unsrt, mo_energy
 
