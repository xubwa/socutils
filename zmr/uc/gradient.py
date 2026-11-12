'''
Orbital gradient of the PC-NEVPT2 second-order energy at fixed CI vectors for large
external spaces (real orbitals; the derivation and the reference with all amplitudes in
memory are in zmr.reference.orbital_gradient).

For a derivative G[P, a, i] = dE/dB[P, a, i] of the core-virtual block of the three-index
integrals, its part of the gradient sum_P B[P] (G[P] + G[P]^T) is

    T1[r, i] = sum_{P a} B[P, r, a] G[P, a, i]    (any r: through the AO vectors, the
                                                    virtual index of G back-transformed,
                                                    so no B[P, r, a] block is formed)
    T2[r, a] = sum_{P i} B[P, r, i] G[P, a, i]    (B[P, r, i]: cd_pc of the source)

at the cost of one exchange-type build (N_aux N_ao^2 n_core).
'''

import numpy
from pyscf import lib as pyscf_lib


def ao_back(eris, gam, space='v', out_space='c'):
    '''T1[r, i] = sum_{P x} B[P, r, x] G[P, x, i] for all orbitals r, G[P, x, i] with x in
    space (back-transformed to the AO basis) and i in out_space, from the AO vectors of
    the factorized source, block by block.'''
    fn = eris.eri_fn
    c_all = fn.components(eris.mo_coeff)
    c_x = fn.components(eris.coeff(space))
    nao = fn.nao
    ni = gam.shape[2]
    y = [numpy.zeros((nao, ni), dtype=numpy.result_type(gam, *c_x)) for _ in c_x]
    for p0, p1, packed in fn.loop(nao * ni):
        ao = pyscf_lib.unpack_tril(numpy.asarray(packed))         # (np, nao, nao)
        g = gam[p0:p1]
        for k, cx in enumerate(c_x):
            x = numpy.einsum('mx,Pxi->Pmi', cx, g, optimize=True)  # (np, nao, ni)
            y[k] += numpy.einsum('Pnm,Pmi->ni', ao, x, optimize=True)
    return sum(c.conj().T @ yk for c, yk in zip(c_all, y))


def explicit_vc(eris, gam):
    '''sum_P B[P] (G[P] + G[P]^T) for G nonzero only in the virtual-core block, G[P, a, i]:
    an (nmo, nmo) matrix with T1 in the core columns and T2 in the virtual columns.'''
    sp = eris.spaces
    n = sp.nmo
    out = numpy.zeros((n, n), dtype=numpy.result_type(gam, float))
    out[:, sp.slice('c')] += ao_back(eris, gam, 'v')
    out[:, sp.slice('v')] += numpy.einsum('Pri,Pai->ra', eris.cd_pc, gam, optimize=True)
    return out


def ccvv(ctx):
    '''(E, D_cc, D_vv, explicit part of the orbital gradient) of ccvv from the native
    kernel: the explicit part is sum_P B[P] (G[P] + G[P]^T) with
    G[P, a, i] = -2 sum_jb t[ij, ab] B[P, b, j].'''
    from socutils.zmr.lib import kernels
    res = kernels.d_ccvv(ctx.eris.cd_block('vc'), ctx.eps_v, ctx.eps_c, gamma=True)
    if res is None:
        raise NotImplementedError('needs libzmr')
    e, dcc, dvv, gam = res
    return e, dcc, dvv, explicit_vc(ctx.eris, gam)
