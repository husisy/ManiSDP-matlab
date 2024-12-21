import os
import time
import numpy as np
import scipy.sparse

from manisdp import (generate_theta_example, unittrace_cvxpy, manisdp_unittrace, manisdp_onlyunitdiag, read_maxcut_Gset)

np_rng = np.random.default_rng()

def demo_example_theta():
    # WARNING: python version "manisdp_unittrace" is slower than matlab version by about 10x
    C, A, b = generate_theta_example(n=50)

    X_cvxpy, fval_cvxpy = unittrace_cvxpy(C, A.toarray().reshape(-1, *C.shape), b)

    kwargs = dict(sigma0=1, tol=1e-8, TR_maxinner=20, TR_maxiter=4, tao=1e-3, theta=1e-2, delta=10, line_search=True, alpha=0.01)
    return_info, Y = manisdp_unittrace(C, A, b, **kwargs)

    print('cvxpy obj:', fval_cvxpy)
    print('manisdp obj:', return_info['cost'])


def demo_maxcut():
    z0 = read_maxcut_Gset(os.path.join('..', 'data', 'Gset', 'G32.txt'))
    matC = -0.25 * scipy.sparse.csc_matrix(z0)
    return_info, Y = manisdp_onlyunitdiag(matC, p0=40)
    print('cost:', return_info['cost'])


if __name__ == '__main__':
    demo_example_theta()

    demo_maxcut()
