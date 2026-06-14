#
# Direct element-wise check of the connected first-T3 (R3_from_t2) in
# zccsdt._t3_residual against a full-tensor assembly of gccsd_t's get_wv_ijk
# connected 'w'.  No convergence, no energy reconstruction ambiguity.
#
import numpy as np
from pyscf import gto, lib
from socutils.scf import spinor_hf
from socutils.cc import gccsd_t
from socutils.cc import zccsdt as zt

einsum = lib.einsum

mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
            basis='sto-3g', verbose=0)
mf = spinor_hf.SCF(mol); mf.kernel()
cc = zt.ZCCSDT(mf, frozen=2)
eris = cc.ao2mo(cc.mo_coeff); cc.eris = eris
nocc = cc.nocc; nvir = cc.nmo - nocc
mo_e = eris.mo_energy
eia = mo_e[:nocc, None] - mo_e[None, nocc:]
eijab = lib.direct_sum('ia,jb->ijab', eia, eia)

t1 = np.zeros((nocc, nvir), dtype=complex)
t2 = np.asarray(eris.oovv).conj() / eijab
t3 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=complex)

# ---- Reference connected w as a full antisymmetric tensor, from gccsd_t ----
ovvv = np.asarray(eris.ovvv).transpose(0, 2, 3, 1).conj()
ooov = np.asarray(eris.ooov).transpose(0, 1, 3, 2)
t2T = t2.transpose(0, 2, 3, 1)

def get_w_ijk(i, j, k):
    w = einsum('ae,bce->abc', t2[j, k], ovvv[i])
    w += einsum('bcm,am->abc', t2T[i], ooov[j, k])
    w = w + w.transpose(2, 0, 1) + w.transpose(1, 2, 0)
    return w

# Assemble W_ref[i,j,k,a,b,c] over ALL i,j,k using the same i<j<k combination
# that gccsd_t uses to build the antisymmetric triple block.
W_ref = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=complex)
for i in range(nocc):
    for j in range(nocc):
        for k in range(nocc):
            W_ref[i, j, k] = get_w_ijk(i, j, k)
# This per-(ijk) w is already abc-antisymmetrized (cyclic sum). The full
# occ-antisymmetrization is implicit in gccsd_t via the i<j<k loop + wijk+wkij-wjik.
# To get a fully antisymmetric reference tensor, antisymmetrize W_ref over ijk:
def antisym_ijk(t):
    return (t
            - t.transpose(1, 0, 2, 3, 4, 5)
            - t.transpose(2, 1, 0, 3, 4, 5)
            - t.transpose(0, 2, 1, 3, 4, 5)
            + t.transpose(1, 2, 0, 3, 4, 5)
            + t.transpose(2, 0, 1, 3, 4, 5))

# ---- Code's connected residual ----
Foo_o = np.zeros((nocc, nocc), dtype=complex)
Fvv_o = np.zeros((nvir, nvir), dtype=complex)
R3 = zt._t3_residual(cc, t1, t2, t3, eris,
                     Foo_o, Fvv_o,
                     np.asarray(eris.oovv).conj(),
                     np.asarray(eris.ooov).conj(),
                     np.asarray(eris.ovvv).conj(),
                     np.asarray(eris.oooo),
                     np.asarray(eris.ovov),
                     np.asarray(eris.vvvv))

# The gccsd_t (T) energy, computed two ways using the SAME w.
# Way A: gccsd_t.kernel
etA = gccsd_t.kernel(cc, eris, t1, t2, verbose=0).real

# Way B: energy from W_ref and disconnected v, exactly mirroring gccsd_t occ_loop
oovv = np.asarray(eris.oovv).conj()
eabc = lib.direct_sum('a+b+c->abc', mo_e[nocc:], mo_e[nocc:], mo_e[nocc:])
eijk = lib.direct_sum('i+j+k->ijk', mo_e[:nocc], mo_e[:nocc], mo_e[:nocc])
def get_wv_ijk(i, j, k):
    w = einsum('ae,bce->abc', t2[j, k], ovvv[i])
    w += einsum('bcm,am->abc', t2T[i], ooov[j, k])
    v = -einsum('a,bc->abc', t1[i], oovv[j, k])
    v += w
    w = w + w.transpose(2, 0, 1) + w.transpose(1, 2, 0)
    return w, v
etB = 0
for i in range(nocc):
    for j in range(i):
        for k in range(j):
            wijk, vijk = get_wv_ijk(i, j, k)
            wkij, vkij = get_wv_ijk(k, i, j)
            wjik, vjik = get_wv_ijk(j, i, k)
            w = wijk + wkij - wjik
            v = vijk + vkij - vjik
            w = w / (eijk[i, j, k] - eabc)
            etB += einsum('abc,abc', w, v.conj())
etB = (etB / 2).real
print('gccsd_t kernel (T)      =', etA)
print('manual occ_loop (T)     =', etB, ' (sanity, should equal kernel)')

# Now the crux: does R3 (code connected residual) equal the antisymmetrized w?
W_full = antisym_ijk(W_ref)   # fully antisym over ijk; abc already antisym
print()
print('shape R3', R3.shape)
print('max|R3|              =', np.abs(R3).max())
print('max|W_full|          =', np.abs(W_full).max())
# Try direct, and a few scalar prefactors
for f in [1.0, 0.5, 1/3., 2.0, 3.0, 6.0, 1/6.]:
    d = np.abs(R3 - f * W_full).max()
    print('  max|R3 - %6.4f*W_full| = %.3e' % (f, d))
