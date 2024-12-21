import numpy as np
import scipy.sparse
import pymanopt
import time

try:
    import torch #torch is optional
except ImportError:
    torch = None


def read_maxcut_Gset(path:str):
    with open(path, 'r') as fid:
        tmp0 = [[int(y) for y in x.strip().split()] for x in fid]
    assert len(tmp0[0])==2
    num_vertex,num_edge = tmp0[0]
    assert len(tmp0)==(num_edge+1)
    tmp1 = {(x[0],x[1]) for x in tmp0[1:]}
    assert len(tmp1)==len(set(tmp1))
    assert all(((x[1],x[0]) not in tmp1) for x in tmp0[1:])
    assert all(x[0]!=x[1] for x in tmp0[1:])
    ret = np.zeros((num_vertex,num_vertex), dtype=np.int64)
    tmp2 = np.array(tmp0[1:], dtype=np.int64)
    ret[tmp2[:,0]-1, tmp2[:,1]-1] = -tmp2[:,2]
    ret[tmp2[:,1]-1, tmp2[:,0]-1] = -tmp2[:,2]
    ret[np.arange(num_vertex), np.arange(num_vertex)] = -ret.sum(axis=1)
    return ret

def hf_cost_wrapper_torch(matC_torch, p):
    manifold = pymanopt.manifolds.Oblique(p, matC_torch.shape[0])
    @pymanopt.function.pytorch(manifold)
    def hf_cost_torch(X):
        ret = torch.dot((X @ matC_torch).reshape(-1), X.reshape(-1)).sum()
        return ret
    prob = pymanopt.Problem(manifold, hf_cost_torch)
    return manifold, hf_cost_torch, prob


def hf_cost_wrapper_np(matC, p):
    manifold = pymanopt.manifolds.Oblique(p, matC.shape[0])
    _global = dict(last_x=None, last_H=None, fval=None, grad=None, eG=None, hess=None)

    @pymanopt.function.numpy(manifold)
    def hf_cost(X):
        if (_global['last_x'] is not None) and (X.shape==_global['last_x'].shape) and np.all(X==_global['last_x']) and (_global['fval'] is not None):
            fval = _global['fval']
        else:
            _global['last_x'] = X
            YC = X @ matC
            eG = (YC*X).sum(axis=0)
            fval = eG.sum()
            grad = 2*(YC - X * eG)
            _global['fval'] = fval
            _global['grad'] = grad
            _global['eG'] = eG
        return fval

    @pymanopt.function.numpy(manifold)
    def hf_grad(X):
        if (_global['last_x'] is not None) and (X.shape==_global['last_x'].shape) and np.all(X==_global['last_x']) and (_global['grad'] is not None):
            grad = _global['grad']
        else:
            hf_cost(X)
            grad = _global['grad']
        return grad

    @pymanopt.function.numpy(manifold)
    def hf_hess(X, H):
        tag = (_global['last_x'] is not None) and (X.shape==_global['last_x'].shape) and np.all(X==_global['last_x']) and (_global['grad'] is not None)
        if tag and (_global['last_H'] is not None) and np.all(H==_global['last_H']) and (_global['hess'] is not None):
            hess = _global['hess']
        else:
            if not tag: #make sure fval and grad are updated
                hf_cost(X)
            eH = H @ matC
            hess = 2*(eH - X * ((X * eH).sum(axis=0)) - H*_global['eG'])
            _global['last_H'] = H
            _global['hess'] = hess
        return hess

    prob = pymanopt.Problem(manifold, hf_cost, euclidean_gradient=hf_grad, euclidean_hessian=hf_hess)
    return manifold, hf_cost, prob


def manisdp_onlyunitdiag(matC, *, p0=2, AL_maxiter=20, tol=1e-8, theta=0.1, delta=8, alpha=0.5,
                    tolgradnorm=1e-8, TR_maxinner=100, TR_maxiter=40, **kwargs):
    # preferred csc sparse matrix
    kwargs['p0'] = p0
    kwargs['AL_maxiter'] = AL_maxiter
    kwargs['tol'] = tol
    kwargs['theta'] = theta
    kwargs['delta'] = delta
    kwargs['alpha'] = alpha
    kwargs['tolgradnorm'] = tolgradnorm
    kwargs['TR_maxinner'] = TR_maxinner
    kwargs['TR_maxiter'] = TR_maxiter
    assert kwargs.get('line_search',0)==0, 'line_search not implemented' #TODO, see ManiSDP-matlab
    # matC_torch = torch.from_numpy(matC.toarray()).to_sparse_csc()

    t0 = time.monotonic()
    return_info = dict(status=0)
    Y = None
    for ind0 in range(kwargs['AL_maxiter']):
        # _, hf_cost_torch, prob = hf_cost_wrapper_torch(matC_torch, kwargs['p0'] if (Y is None) else Y.shape[0])
        _, hf_cost_torch, prob = hf_cost_wrapper_np(matC, kwargs['p0'] if (Y is None) else Y.shape[0])
        optimizer = pymanopt.optimizers.TrustRegions(max_iterations=kwargs['AL_maxiter'], min_gradient_norm=kwargs['tolgradnorm'], verbosity=0)
        result = optimizer.run(prob, initial_point=Y, maxinner=kwargs['TR_maxinner'])

        tmp0 = np.einsum(matC @ result.point.T, [0,1], result.point, [1,0], [0], optimize=True)
        S = matC - scipy.sparse.diags(tmp0, format=matC.format)
        EVL,EVC =  np.linalg.eigh(S.toarray())
        dinf = max(0, -EVL[0]) / (1+EVL[-1])
        _,S,V = np.linalg.svd(result.point, full_matrices=False)
        r = sum(S >= kwargs['theta']*S[0])
        print(f'Iter {ind0+1}, obj:{result.cost:0.8f}, dinf:{dinf:0.1e}, r:{r}, Y.shape:{result.point.shape}, time:{(time.monotonic()-t0):0.2f}s')
        if dinf < kwargs['tol']:
            print('Optimality is reached!')
            break
        if (ind0>0) and (ind0%20 == 0):
            if (ind0 > 50) and (dinf > dinf0):
                return_info['status'] = 2
                print('Slow progress!')
                break
            else:
                dinf0 = dinf
        if r < result.point.shape[0]:
            Y = S[:r].reshape(-1,1) * V[:r]
        nne = max(min((EVL < 0).sum(), kwargs['delta']), 1)
        Y = np.concatenate([Y, kwargs['alpha'] * EVC[:,:nne].T], axis=0)
        Y /= np.linalg.norm(Y, axis=0, keepdims=True)
        print(Y.shape, result.cost, hf_cost_torch(Y))
    return_info['cost'] = result.cost
    return return_info, Y
