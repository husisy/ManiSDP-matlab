import time
import numpy as np
import cvxpy
import pymanopt
import scipy.sparse

np_rng = np.random.default_rng()

def unittrace_cvxpy(matC:np.ndarray, matA:np.ndarray, vecb:np.ndarray):
    assert (matC.ndim==2) and (matC.shape[0]==matC.shape[1])
    assert isinstance(matA, np.ndarray), 'not support scipy.sparse'
    assert (matA.ndim==3) and (vecb.ndim==1) and matA.shape==(vecb.shape[0], matC.shape[0], matC.shape[1])
    assert np.abs(matA-matA.transpose(0,2,1)).max() < 1e-10, 'matA must be symmetric'
    n = matC.shape[0]
    cvxX = cvxpy.Variable((n,n), PSD=True)
    tmp0 = cvxpy.reshape(cvxX, n*n, order='C')
    obj = cvxpy.Minimize(cvxpy.sum(cvxpy.matmul(tmp0, matC.reshape(-1))))
    constraint = [
        cvxX==cvxX.T,
        cvxpy.trace(cvxX)==1,
        (matA.reshape(vecb.shape[0],-1) @ tmp0)==vecb,
    ]
    prob = cvxpy.Problem(obj, constraint)
    prob.solve()
    return cvxX.value, prob.value


def hf_cost_wrapper_np(A, At, C, b, y, p:int, sigma:float):
    # A (csr,float64,(m,n*n))
    # At (csr,float64,(n*n,m))
    assert (C.ndim==2) and (C.shape[0]==C.shape[1])
    n = C.shape[0]
    manifold = pymanopt.manifolds.Sphere(n, p)
    _global = dict(last_x=None, last_H=None, fval=None, grad=None, hess=None, eS=None, z=None)

    @pymanopt.function.numpy(manifold)
    def hf_cost(Y):
        if (_global['last_x'] is not None) and (Y.shape==_global['last_x'].shape) and np.all(Y==_global['last_x']) and (_global['fval'] is not None):
            fval = _global['fval']
        else:
            _global['last_x'] = Y # Y of shape (n,p)
            X = (Y @ Y.T).reshape(-1)
            Axb = A @ X - b - y/sigma
            fval = np.dot(C.reshape(-1), X) + sigma/2 * np.dot(Axb, Axb)
            eS = C + sigma * (At @ Axb).reshape(n,n)
            z = np.dot(X, eS.reshape(-1))
            grad = 2*eS @ (2*Y) - 2*z*Y
            _global['fval'] = fval
            _global['grad'] = grad
            _global['eS'] = eS
            _global['z'] = z
        return fval

    @pymanopt.function.numpy(manifold)
    def hf_grad(Y):
        if (_global['last_x'] is not None) and (Y.shape==_global['last_x'].shape) and np.all(Y==_global['last_x']) and (_global['grad'] is not None):
            grad = _global['grad']
        else:
            hf_cost(Y)
            grad = _global['grad']
        return grad

    @pymanopt.function.numpy(manifold)
    def hf_hess(Y, H):
        # Y (np,float64,(n,p))
        # H (np,float64,(n,p))
        tag = (_global['last_x'] is not None) and (Y.shape==_global['last_x'].shape) and np.all(Y==_global['last_x']) and (_global['grad'] is not None)
        if tag and (_global['last_H'] is not None) and np.all(H==_global['last_H']) and (_global['hess'] is not None):
            hess = _global['hess']
        else:
            if not tag: #make sure fval and grad are updated
                hf_cost(Y)
            AyU = (At @ (A @ (H @ Y.T).reshape(-1))).reshape(n, n)
            # AyU = (((H @ Y.T).reshape(-1) @ At) @ A).reshape(n,n)
            hess = 2*_global['eS'] @ H + 4*sigma*(AyU @ Y)
            tmp1 = np.dot(hess.reshape(-1), Y.reshape(-1))
            hess -= tmp1*Y + 2*_global['z']*H
            _global['last_H'] = H
            _global['hess'] = hess
        return hess
    prob = pymanopt.Problem(manifold, hf_cost, euclidean_gradient=hf_grad, euclidean_hessian=hf_hess)
    return manifold, hf_cost, prob

def _co(Y, A, C, b, y, sigma):
    tmp0 = (Y @ Y.T).reshape(-1)
    tmp1 = A @ tmp0 - b - y/sigma
    ret = np.dot(C.reshape(-1), tmp0) + sigma/2 * np.dot(tmp1, tmp1)
    return ret

def hf_line_search(Y, U, A, C, b, y, sigma):
    alpha = 0.2
    cost0 = _co(Y, A, C, b, y, sigma)
    nY = Y + alpha*U
    nY /= np.linalg.norm(nY, ord='fro')
    for _ in range(15):
        if _co(nY, A, C, b, y, sigma) - cost0 <= -1e-3:
            break
        alpha *= 0.8
        nY = Y + alpha*U
        nY /= np.linalg.norm(nY, ord='fro')
    return nY



def manisdp_unittrace(C, A, b, *, p0=1, AL_maxiter=1000, gama=2, sigma0=10, sigma_min=1e2, sigma_max=1e7,
            tol=1e-8, theta=1e-2, delta=10, alpha=0.1, tolgradnorm=1e-8, TR_maxinner=40,
            TR_maxiter=3, tao=1/6e3, line_search=True, Y0=None, **kwargs):
    print('WARNING: python version "manisdp_unittrace()" is slower than matlab version by about 10x')
    kwargs['p0'] = p0
    kwargs['AL_maxiter'] = AL_maxiter
    kwargs['gama'] = gama
    kwargs['sigma0'] = sigma0
    kwargs['sigma_min'] = sigma_min
    kwargs['sigma_max'] = sigma_max
    kwargs['tol'] = tol
    kwargs['theta'] = theta
    kwargs['delta'] = delta
    kwargs['alpha'] = alpha
    kwargs['tolgradnorm'] = tolgradnorm
    kwargs['TR_maxinner'] = TR_maxinner
    kwargs['TR_maxiter'] = TR_maxiter
    kwargs['tao'] = tao
    kwargs['line_search'] = line_search
    kwargs['Y0'] = Y0

    assert (C.ndim==2) and (C.shape[0]==C.shape[1])
    n = C.shape[0]
    assert np.abs(C - C.T).max() < 1e-10, 'C must be symmetric'
    assert (A.ndim==2) and (A.shape[1]==n*n) and (A.shape[0]==b.shape[0])
    if not scipy.sparse.issparse(A):
        A = scipy.sparse.csr_matrix(A) # A (csr,float64,(m,n*n))
    At = A.T.tocsr() #(csr,float64,(n*n,m))
    tmp0 = A.toarray().reshape(-1, n, n)
    assert np.abs(tmp0-tmp0.transpose(0,2,1)).max() < 1e-10

    sigma = kwargs['sigma0']
    y = np.zeros(b.shape)
    normb = np.linalg.norm(b) + 1
    Y = kwargs['Y0']
    U = None

    t0 = time.monotonic()
    return_info = dict(status=0)
    for ind0 in range(kwargs['AL_maxiter']):
        if U is not None:
            Y = hf_line_search(Y, U, A, C, b, y, sigma)
        tmp0 = kwargs['p0'] if (Y is None) else Y.shape[1]
        manifold, hf_cost, prob = hf_cost_wrapper_np(A, At, C, b, y, tmp0, sigma)
        optimizer = pymanopt.optimizers.TrustRegions(max_iterations=kwargs['AL_maxiter'], min_gradient_norm=kwargs['tolgradnorm'], verbosity=0)
        result = optimizer.run(prob, initial_point=Y, maxinner=kwargs['TR_maxinner'])
        gradnorm = result.gradient_norm
        Y = result.point
        X = Y @ Y.T
        obj = np.dot(X.reshape(-1), C.reshape(-1)) #warning: obj is not the same as result.cost
        tmp0 = A @ X.reshape(-1) - b
        pinf = np.linalg.norm(tmp0) / normb
        y = y - sigma*tmp0
        eS = C - (At @ y).reshape(n,n)
        z = np.dot(X.reshape(-1), eS.reshape(-1))
        S = eS - z*np.eye(n)
        EVL,EVC = np.linalg.eigh(S)
        dinf = max(0, -EVL[0])/(1+EVL[-1])
        by = np.dot(b, y) + z
        gap = abs(obj - by) / (abs(by) + abs(obj) + 1)
        svd_U,svd_S,_ = np.linalg.svd(Y, full_matrices=False)
        r = (svd_S >= kwargs['theta']*svd_S[0]).sum()
        print(f'Iter {ind0+1}, obj:{obj:0.8f}, gap:{gap:0.1e}, pinf:{pinf:0.1e}, dinf:{dinf:0.1e}, gradnorm:{gradnorm:0.1e}, r:{r}, Y.shape:{result.point.shape}, time:{(time.monotonic()-t0):0.2f}s')
        eta = max(pinf, gap, dinf)
        if eta < kwargs['tol']:
            print('Optimality is reached!')
            break
        if (ind0>0) and (ind0%20==0):
            if (ind0>50) and (gap > gap0) and (pinf > pinf0) and (dinf > dinf0):
                return_info['status'] = 2
                print('Slow progress!')
                break
            else:
                gap0 = gap
                pinf0 = pinf
                dinf0 = dinf
        if r<=Y.shape[1]-1:
            Y = svd_U[:,:r] * svd_S[:r]
        nne = min((EVL<0).sum(), kwargs['delta'])
        if kwargs['line_search']:
            U = np.concatenate([np.zeros_like(Y), EVC[:,:nne]], axis=1)
            Y = np.concatenate([Y,np.zeros((n,nne), dtype=Y.dtype)], axis=1)
        else:
            Y = np.concatenate([Y, kwargs['alpha']*EVC[:,:nne]], axis=1)
            Y /= np.linalg.norm(Y, ord='fro')
        if pinf < kwargs['tao']*gradnorm:
            sigma = max(sigma/kwargs['gama'], kwargs['sigma_min'])
        else:
            sigma = min(sigma*kwargs['gama'], kwargs['sigma_max'])
    return_info['cost'] = obj
    return return_info, Y


def generate_theta_example(n:int):
    tmp0 = np_rng.integers(1, n+1, size=(10*n,2))
    Omega = np.array(sorted({(x,y) for x,y in tmp0.tolist() if x<y}))
    m = Omega.shape[0]
    C = -np.ones((n,n))
    b = np.array([0]*m + [1])

    tmp0 = np.stack([(Omega[:,0]-1)*n + Omega[:,1]-1, (Omega[:,1]-1)*n+Omega[:,0]-1], axis=1).reshape(-1)
    tmp1 = np.arange(n)*n + np.arange(n)
    row = np.concatenate([tmp0, tmp1])
    col = np.concatenate([np.repeat(np.arange(m), [2]), np.ones(n)*m])
    val = np.ones(col.shape)
    A = scipy.sparse.csr_array((val, (col,row)), shape=(m+1, n*n))
    return C, A, b
