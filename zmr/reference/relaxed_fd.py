'''
Relaxed one-electron properties of states of a (state-averaged) PySCF CASSCF through the
Lagrangian, with the derivatives of the target energy taken by finite differences: the
reference for the analytic relaxed densities (small molecules only).

Parametrization of PySCF's second-order CASSCF (pyscf.mcscf.newton_casscf):

    orbitals  mo' = mo expm(unpack(x_orb)),   CI vectors  c_k' = (c_k + x_k) / |c_k + x_k|

g(x) the gradient of the averaged energy (gen_g_hop), J = dg/dx (its Hessian product).  For
a target energy E(x) (a state of the SA-CASSCF, or that plus a second-order correction):

    J^T z = -dE/dx                         (pseudo-inverse: the norms of the CI vectors and
                                            rotations among degenerate averaged states)
    dE/dlambda = dE/dlambda|_x + z . dg/dlambda
               = sum O D_target + sum O D_z

for h -> h + lambda O, with D_z the derivative of the averaged density along z
(lagrangian_density): one solution z gives the relaxed density for every one-electron
operator.

    mc = polish(casscf)                    tight convergence (Newton on the exact Hessian)
    z = zvector(mc, rhs)                   rhs = fd_gradient(energy_fn, mc)
    d = lagrangian_density(mc, z)          (nao, nao), the part of the relaxed density
'''

import numpy
from pyscf.fci import direct_spin1
from pyscf.mcscf import newton_casscf


def _ci_list(mc):
    return list(mc.ci) if isinstance(mc.ci, (list, tuple)) else [mc.ci]


def _weights(mc):
    return list(getattr(mc, 'weights', [1.0]))


def ngorb(mc, mo=None):
    mo = mc.mo_coeff if mo is None else mo
    return numpy.count_nonzero(mc.uniq_var_indices(mo.shape[1], mc.ncore, mc.ncas, mc.frozen))


def move(mc, mo, ci, x):
    '''(mo', [c_k']) for the parameters x.'''
    n = ngorb(mc, mo)
    mo1 = mo @ mc.update_rotate_matrix(x[:n])
    out, p = [], n
    for c in ci:
        d = c.ravel() + x[p:p + c.size]
        out.append((d / numpy.linalg.norm(d)).reshape(c.shape))
        p += c.size
    return mo1, out


def state_energies(mc, mo, ci):
    '''<c_k|H|c_k> in the orbitals mo (CI vectors not re-solved).'''
    h1, ecore = mc.get_h1eff(mo)
    h2 = mc.get_h2eff(mo)
    return numpy.array([ecore + direct_spin1.energy(h1, h2, c, mc.ncas, mc.nelecas) for c in ci])


def gradient_hessian(mc, mo=None, ci=None):
    '''(g, J) of the averaged energy, J as a dense matrix.'''
    mo = mc.mo_coeff if mo is None else mo
    ci = _ci_list(mc) if ci is None else ci
    arg = ci if len(ci) > 1 else ci[0]
    g, _, hop, _ = newton_casscf.gen_g_hop(mc, mo, arg, mc.ao2mo(mo))[:4]
    g = numpy.asarray(g)
    jac = numpy.array([numpy.asarray(hop(e)) for e in numpy.eye(len(g))]).T
    return g, jac


def polish(mc, tol=1e-11, max_cycle=12):
    '''Newton steps on the positive part of the exact Hessian after the PySCF solution
    (step length bounded, averaged energy decreasing); sets mc.mo_coeff, mc.ci and
    mc.gnorm.'''
    w = _weights(mc)
    mo, ci = mc.mo_coeff, _ci_list(mc)
    e_old = numpy.dot(w, state_energies(mc, mo, ci))
    for _ in range(max_cycle):
        g, jac = gradient_hessian(mc, mo, ci)
        if numpy.linalg.norm(g) < tol:
            break
        ev, vec = numpy.linalg.eigh(0.5 * (jac + jac.T))
        keep = ev > 1e-6
        x = -vec[:, keep] @ ((vec[:, keep].T @ g) / ev[keep])
        if numpy.linalg.norm(x) > 0.05:
            x *= 0.05 / numpy.linalg.norm(x)
        for _ in range(10):
            mo1, ci1 = move(mc, mo, ci, x)
            e1 = numpy.dot(w, state_energies(mc, mo1, ci1))
            if e1 <= e_old + 1e-12:
                break
            x *= 0.5
        mo, ci, e_old = mo1, ci1, e1
    mc.mo_coeff = mo
    mc.ci = ci if len(ci) > 1 else ci[0]
    mc.gnorm = float(numpy.linalg.norm(g))
    return mc


def fd_gradient(energy_fn, mc, step=1e-4):
    '''dE/dx by central differences, energy_fn(mo, ci) the target energy.'''
    mo, ci = mc.mo_coeff, _ci_list(mc)
    n = ngorb(mc, mo) + sum(c.size for c in ci)
    out = numpy.zeros(n)
    for k in range(n):
        x = numpy.zeros(n)
        x[k] = step
        out[k] = (energy_fn(*move(mc, mo, ci, x)) - energy_fn(*move(mc, mo, ci, -x))) / (2 * step)
    return out


def zvector(mc, rhs):
    '''z with J^T z = -rhs (least squares: the zero modes of the averaged energy).'''
    _, jac = gradient_hessian(mc)
    return -numpy.linalg.lstsq(jac.T, rhs, rcond=1e-10)[0]


def lagrangian_density(mc, z):
    '''D_z (AO basis): the derivative of the averaged density sum_k w_k D_k along the
    parameters z, so that z . dg/dlambda = sum O D_z for h -> h + lambda O.'''
    mo, ci, w = mc.mo_coeff, _ci_list(mc), _weights(mc)
    nmo = mo.shape[1]
    ncore, ncas = mc.ncore, mc.ncas
    n = ngorb(mc, mo)
    kmat = mc.unpack_uniq_var(z[:n])                   # mo(t) = mo (1 + t K + ...)
    gam = numpy.zeros((nmo, nmo))                      # averaged 1-RDM, MO basis
    dgam = numpy.zeros((nmo, nmo))                     # its derivative from the CI part
    gam[:ncore, :ncore] = 2 * numpy.eye(ncore)
    act = slice(ncore, ncore + ncas)
    p = n
    for wk, c in zip(w, ci):
        zk = z[p:p + c.size].reshape(c.shape)
        p += c.size
        gam[act, act] += wk * direct_spin1.make_rdm1(c, ncas, mc.nelecas)
        t = direct_spin1.trans_rdm1(zk, c, ncas, mc.nelecas)       # <z|E|c>
        # c + t z normalized: d gamma = <z|E|c> + <c|E|z> - 2 (c.z) gamma
        dgam[act, act] += wk * (t + t.T - 2 * float(numpy.dot(c.ravel(), zk.ravel()))
                                * direct_spin1.make_rdm1(c, ncas, mc.nelecas))
    # D(t) = mo (1 + tK) gamma (1 + tK)^T mo^T + mo t dgam mo^T
    d_mo = kmat @ gam + gam @ kmat.T + dgam
    return mo @ d_mo @ mo.T
