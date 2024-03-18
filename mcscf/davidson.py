import numpy as np
import numpy
import time

def compute_lambda_(mat1, mat2, x_):
    nlast = mat1.shape[0]
    assert (nlast > 1)
    lambda_test = 1.0
    lambda_lasttest = 0.0
    stepsize_lasttest = 0.0
    stepsize = 0.0
    maxstepsize = 1.0
    iok = 0
    for i in range(10):
        scr = mat1 + mat2 * (1.0 / lambda_test)
        e, c = np.linalg.eigh(scr)

        ivec = -1
        for j in range(nlast):
            if abs(c[0, j]) <= 1.1 and abs(c[0, j]) > 0.1:
                ivec = j
                break
        if ivec < 0:
            raise Exception('logical error in AugHess')
        c[:, ivec] = c[:, ivec] / (c[0, ivec])
        step = np.dot(x_[1:, :nlast], c[:nlast, ivec])
        stepsize = np.linalg.norm(step[1:]) / abs(lambda_test)
        #print(ivec, e, stepsize, lambda_test)

        if i == 0:
            if stepsize <= maxstepsize:
                break
            lambda_lasttest = lambda_test
            lambda_test = stepsize / maxstepsize
        else:
            if abs(stepsize - maxstepsize) / maxstepsize < 0.01:
                break
            if (stepsize > maxstepsize):
                lambda_lasttest = lambda_test
                lambda_test = lambda_test * (stepsize / maxstepsize)
            else:
                if (iok > 2):
                    break
                iok += 1
                d1 = maxstepsize - stepsize
                d2 = stepsize_lasttest - maxstepsize
                if (d1 == 0.0 or d1 == -d2):
                    break
                lambda_lasttest_ = lambda_lasttest
                lambda_lasttest = lambda_test
                lambda_test = d1 / (d1 + d2) * lambda_lasttest_ + d2 / (d1 + d2) * lambda_test
            if lambda_test < 1.0:
                lambda_test = 1.0
            stepsize_lasttest = stepsize
    return lambda_test, stepsize


def davidson(hop, g, hdiag, tol=1e-5, neig=1, mmax=20, max_space=10):
    # Setup the subspace trial vectors
    print('No. of start vectors:', 1)
    neig = neig
    ptype = g.dtype
    print('No. of desired Eigenvalues:', neig)
    n = g.shape[0] + 1
    x = np.zeros((n, mmax + 1), dtype=ptype)  # holder for trial vectors as iterations progress
    sigma = np.zeros((n, mmax + 1), dtype=ptype)
    ritz = np.zeros((n, n), dtype=ptype)
    #-------------------------------------------------------------------------------
    # Begin iterations
    #-------------------------------------------------------------------------------
    #initial extra (1,0) vector
    start = time.time()
    g0 = g.copy()
    x[0, 0] = 1.0
    sigma[1:, 0] = g

    # "first" guess vector

    for i in range(1, n):
        if abs(hdiag[i - 1]) > 1.0e-8:
            x[i, 1] = -g[i - 1] / (hdiag[i - 1])
        else:
            print('else')
            x[i, 1] = -g[i - 1] / (hdiag[i - 1] + 0.001)

    x[:, 1] = x[:, 1] / np.linalg.norm(x[:, 1])
    norm_grad = np.linalg.norm(g)
    for m in range(1, mmax + 1):
        sigma[1:, m] = hop(x[1:, m])
        sigma[0, m] = np.dot(g.conj(), x[1:, m]).real
        # Matrix-vector products, form the projected Hamiltonian in the subspace
        T = np.linalg.multi_dot((x[:, :m + 1].T.conj(), sigma[:, :m + 1])).real
        #print(T)
        #T=(T+T.T)*0.5
        mat1 = numpy.zeros((m + 1, m + 1))
        mat2 = numpy.zeros((m + 1, m + 1))
        mat2[1:, 1:] = T[1:, 1:]
        mat1 = T - mat2
        lambda_, stepsize = compute_lambda_(mat1, mat2, x)
        #lambda_ = 1.0
        #stepsize = np.linalg.norm(x)
        T = mat1 + mat2 * (1.0 / lambda_)
        e, vects = np.linalg.eigh(T[:m + 1, :m + 1])
        for j in range(m + 1):
            if abs(vects[0, j]) <= 1.1 and abs(vects[0, j]) > 0.1:
                ivec = j
                break
        if ivec < 0:
            raise Exception('logical error in AugHess')
        elif ivec > 0:
            print('... the vector found in AugHess was not the lowest eigenvector ...')

        ritz[:, 0] = np.dot(sigma[:, :m + 1] - lambda_ * e[ivec] * x[:, :m + 1], vects[:, ivec])
        err = np.linalg.norm(ritz[1:, 0])
        g = g0+np.dot(sigma[1:, :m + 1], vects[:, ivec])
        
        print(f'iter {m}, ivec {ivec}, c[ivec], {vects[0,ivec]:6.2f}, eps={e[ivec]:12.8e}, res={err:8.6e},  lambda={lambda_:6.2f}, step={stepsize}')
        # print(e[:min(4, m)])
        # print(vects[0,:min(4,m)])
        if err < norm_grad/10.:
            print('Davidson converged at iteration no.:', m - 1)
            end = time.time()
            print('Davidson time:', end - start)
            x_ = np.dot(x[:, :m + 1], vects[:, ivec])
            return x_[1:]/x_[0], e[ivec], e#*abs(x_[0]), e[ivec], e
        elif m is mmax:
            print('Max iteration reached')
            x_ = np.dot(x[:, :m + 1], vects[:, ivec])
            return x_[1:]/x_[0], e[ivec], e#*abs(x_[0]), e[ivec], e
        else:
            for idx in range(hdiag.shape[0]):
                denom = hdiag[idx] - e[ivec] * lambda_
                if abs(denom) > 1e-8:
                    ritz[idx + 1, 0] = -ritz[idx + 1, 0] / denom
                else:
                    print('else')
                    ritz[idx + 1, 0] = 0.0 #-ritz[idx + 1, 0] / 1e-8
            # orthonormalize ritz vector
            for idx in range(m + 1):
                ritz[:, 0] = ritz[:, 0] - (np.dot(x[:, idx].conj(), ritz[:, 0]) * x[:, idx])  #/np.linalg.norm(x[:,idx])
            ritz[:, 0] = ritz[:, 0] / (np.linalg.norm(ritz[:, 0]))
            x[:, m + 1] = ritz[:, 0]

