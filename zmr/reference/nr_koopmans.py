'''
Brute-force metric and Koopmans matrices of the spin-free PC-NEVPT2
functions, for checks of zmr.nr.rdm on small active spaces: the functions
F_P |Psi_J> are built as CI vectors (the spin patterns stacked,
zmr.nr.classes) and H_act is applied to every one of them.

    S[I, P, J, Q] = <I| F_P^+ F_Q |J>,   K[I, P, J, Q] = <I| F_P^+ H_act F_Q |J>
'''

import numpy


def class_matrices(pclass, act, civecs, nelec):
    '''(S, K) of one class (zmr.nr.classes), or None if no spin pattern has
    a target sector.'''
    civecs = numpy.asarray(civecs)
    nstate = civecs.shape[1]
    s = k = None
    for target, phi in pclass.functions(act, civecs.T, tuple(nelec)):
        dim = act.dim(target)
        if dim == 0:
            continue
        nx = phi.shape[1]
        f = phi.reshape(nstate * nx, dim)
        hf = act.apply_h(target, numpy.ascontiguousarray(f.T)).T
        if s is None:
            s = numpy.zeros((nstate * nx, nstate * nx))
            k = numpy.zeros_like(s)
        s += f @ f.T
        k += f @ hf.T
    if s is None:
        return None
    shape = (nstate, nx, nstate, nx)
    return s.reshape(shape), k.reshape(shape)
