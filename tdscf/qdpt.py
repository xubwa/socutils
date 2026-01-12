import numpy, scipy
from pyscf import lib

def transition_spin_density_t_0(xy_t, sz_t):
    assert sz_t in [0, 1, -1]
    norm = numpy.sum(xy_t**2)
    xy_t_normalized = xy_t / numpy.sqrt(norm)
    no, nv = xy_t.shape
    nmo = nv + no
    tdm = numpy.zeros((nmo, nmo))
    tdm[no:, :no] = xy_t_normalized.T
    if sz_t == 0:
        tdm *= numpy.sqrt(2)

    return tdm

def transition_spin_density_t_s(xy_t, xy_s, sz_t):
    assert sz_t in [0, 1, -1]
    norm = numpy.sum(xy_t**2)
    xy_t_normalized = xy_t / numpy.sqrt(norm)
    norm = numpy.sum(xy_s**2)
    xy_s_normalized = xy_s / numpy.sqrt(norm)
    no, nv = xy_t.shape
    nmo = nv + no
    tdm = numpy.zeros((nmo, nmo))
    
    sum_i = lib.einsum("ib,ia->ba", xy_t_normalized, xy_s_normalized)
    sum_a = lib.einsum("ja,ia->ij", xy_t_normalized, xy_s_normalized)
    tdm[no:, no:] = sum_i / numpy.sqrt(2)
    tdm[:no, :no] = -sum_a / numpy.sqrt(2)
    if sz_t == 0:
        tdm *= numpy.sqrt(2)

    return tdm

def transition_spin_density_t_t(xy_t1, xy_t2, sz_t1, sz_t2):
    assert sz_t1 in [0, 1, -1]
    assert sz_t2 in [0, 1, -1]
    assert abs(sz_t1 - sz_t2) <= 1
    if sz_t1 < sz_t2:
        return transition_spin_density_t_t(xy_t2, xy_t1, sz_t2, sz_t1)
    norm = numpy.sum(xy_t1**2)
    xy_t1_normalized = xy_t1 / numpy.sqrt(norm)
    norm = numpy.sum(xy_t2**2)
    xy_t2_normalized = xy_t2 / numpy.sqrt(norm)
    no, nv = xy_t1.shape
    nmo = nv + no
    tdm = numpy.zeros((nmo, nmo))

    sum_i = lib.einsum("ib,ia->ba", xy_t1_normalized, xy_t2_normalized)
    sum_a = lib.einsum("ja,ia->ij", xy_t1_normalized, xy_t2_normalized)
    if sz_t1 == 1 and sz_t2 == 1:
        tdm[no:, no:] = sum_i
        tdm[:no, :no] = sum_a
    elif sz_t1 == -1 and sz_t2 == -1:
        tdm[no:, no:] = -sum_i
        tdm[:no, :no] = -sum_a
    elif sz_t1 == 1 and sz_t2 == 0:
        tdm[no:, no:] = -sum_i / numpy.sqrt(2)
        tdm[:no, :no] = -sum_a / numpy.sqrt(2)
    elif sz_t1 == 0 and sz_t2 == -1:
        tdm[no:, no:] = sum_i / numpy.sqrt(2)
        tdm[:no, :no] = sum_a / numpy.sqrt(2)
        
    return tdm

def contract_soc_tdm(socints_mo, tdm, sz_diff):
    assert sz_diff in [0, 1, -1]
    if sz_diff == 0:
        res = numpy.trace(tdm @ socints_mo[3])
    elif sz_diff == 1:
        res = numpy.trace(tdm @ socints_mo[1] - 1j * tdm @ socints_mo[2])
    elif sz_diff == -1:
        res = numpy.trace(tdm @ socints_mo[1] + 1j * tdm @ socints_mo[2])
    return 1.0j * res

def qdpt(tda_singlet, tda_triplet, socints_mo):
    nroots_s = tda_singlet.nroots
    nroots_t = tda_triplet.nroots
    nroots_total = 1 + nroots_t*3 + nroots_s
    mat_qdpt = numpy.zeros((nroots_total, nroots_total), dtype=numpy.complex128)
    mat_qdpt[0,0] = 0.0  # reference energy
    for i in range(nroots_s):
        mat_qdpt[1+i, 1+i] = tda_singlet.e[i]
    for i in range(nroots_t):
        for sz in range(-1, 2):
            pos_t = 1 + nroots_s + i*3 + (sz + 1)
            mat_qdpt[pos_t, pos_t] = tda_triplet.e[i]
            tdm = transition_spin_density_t_0(tda_triplet.xy[i][0], sz)
            soc_me = contract_soc_tdm(socints_mo, tdm, sz)
            mat_qdpt[pos_t, 0] = soc_me
            mat_qdpt[0, pos_t] = numpy.conj(soc_me)
            for j in range(nroots_s):
                pos_s = 1 + j
                tdm_ts = transition_spin_density_t_s(tda_triplet.xy[i][0], tda_singlet.xy[j][0], sz)
                soc_me_ts = contract_soc_tdm(socints_mo, tdm_ts, sz)
                mat_qdpt[pos_t, pos_s] = soc_me_ts
                mat_qdpt[pos_s, pos_t] = numpy.conj(soc_me_ts)

            for j in range(i):
                for sz2 in range(-1, 2):
                    if abs(sz - sz2) > 1:
                        continue
                    pos_t2 = 1 + nroots_s + j*3 + (sz2 + 1)
                    tdm_tt = transition_spin_density_t_t(tda_triplet.xy[i][0], tda_triplet.xy[j][0], sz, sz2)
                    soc_me_tt = contract_soc_tdm(socints_mo, tdm_tt, sz - sz2)
                    mat_qdpt[pos_t, pos_t2] = soc_me_tt
                    mat_qdpt[pos_t2, pos_t] = numpy.conj(soc_me_tt)

    assert numpy.allclose(mat_qdpt, mat_qdpt.conj().T)
    return scipy.linalg.eigh(mat_qdpt)
        
    


if __name__ == "__main__":
    from pyscf import gto, scf, dft
    from pyscf.tdscf.rks import TDA
    mol = gto.Mole()
    mol.atom = """
O      0.000000    0.000000    0.601105
C     -0.000000    0.000000   -0.598757
H      0.000000   -0.944973   -1.202781
H      0.000000    0.944973   -1.202781
    """
    mol.basis = "def2svp"
    mol.verbose = 4
    mol.build()

    from socutils.somf import somf_pt
    x2cints = somf_pt.get_psoc_x2camf(mol)

    mf = dft.RKS(mol,xc="b3lyp")
    mf.kernel()
    socints_mo = numpy.array([mf.mo_coeff.T @ x2cints[i] @ mf.mo_coeff for i in range(4)])

    s_tda = TDA(mf)
    s_tda.singlet = True
    s_tda.nroots = 5
    s_tda.kernel()
    
    t_tda = TDA(mf)
    t_tda.singlet = False
    t_tda.nroots = 5
    t_tda.kernel()

    # t_state = 3 - 1
    # print(numpy.sum(t_tda.xy[t_state][0]**2))  # norm
    # tdm_0 = transition_spin_density_t_0(t_tda.xy[t_state][0], 0)
    # print(tdm_0.shape)
    # print(numpy.trace(tdm_0 @ socints_mo[3]) * 219474.63, "*j")  # Lz

    # tdm_1 = transition_spin_density_t_0(t_tda.xy[t_state][0], 1)
    # print(numpy.trace(tdm_1 @ socints_mo[1]) * 219474.63)
    # print(numpy.trace(tdm_1 @ socints_mo[2]) * 219474.63)

    # t_state = 2 - 1
    # s_state = 3 - 1
    # tdm = transition_spin_density_t_s(t_tda.xy[t_state][0], s_tda.xy[s_state][0], 0)
    # print(numpy.trace(tdm @ socints_mo[3]) * 219474.63, "Sz")
    # print(numpy.trace(tdm @ socints_mo[1]) * 219474.63, "Sx")
    # print(numpy.trace(tdm @ socints_mo[2]) * 219474.63, "Sy")

    e, c = qdpt(s_tda, t_tda, socints_mo)
    print("QDPT energies (cm-1):")
    print((e - e[0]) * 219474.63)
    