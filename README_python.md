# manisdp-Python

quickstart

```bash
pip install git+https://github.com/husisy/zzz233.git@zc-dev
```

## Example: `manisdp_onlyunitdiag()`

max-cut problem

adopted from `example/example_maxcut.m` [link](https://github.com/wangjie212/ManiSDP-matlab/blob/main/example/example_maxcut.m)

save `data/Gset/G32.txt` in the repo [link](https://github.com/wangjie212/ManiSDP-matlab/blob/main/data/Gset/G32.txt) as `G32.txt` on your local machine.

```python
import scipy.sparse
from manisdp import read_maxcut_Gset, manisdp_onlyunitdiag

z0 = read_maxcut_Gset('G32.txt')
matC = -0.25 * scipy.sparse.csc_matrix(z0)
return_info, Y = manisdp_onlyunitdiag(matC, p0=40)
print('cost:', return_info['cost'])
X = Y.T @ Y
```

$$ \begin{align*}\min_{X}&\;\mathrm{Tr}[CX]\\
\mathrm{s.t.}&\begin{cases}
X\succeq0\\
X_{ii}=1
\end{cases}\end{align*} $$

## Example: `manisdp_unittrace()`

theta example?

adopted from `example/example_theta.m` [link](https://github.com/wangjie212/ManiSDP-matlab/blob/main/example/example_theta.m)

$$ \begin{align*} \min_{X}&\mathrm{Tr}[CX]\\
\mathrm{s.t.}&\begin{cases}
X\succeq0\\
\mathrm{Tr}[A_{i}X]=b_{i}
\end{cases} \end{align*} $$

```Python
from manisdp import generate_theta_example, unittrace_cvxpy, manisdp_unittrace
C, A, b = generate_theta_example(n=50)
X_cvxpy, fval_cvxpy = unittrace_cvxpy(C, A.toarray().reshape(-1, *C.shape), b)
kwargs = dict(sigma0=1, tol=1e-8, TR_maxinner=20, TR_maxiter=4, tao=1e-3, theta=1e-2, delta=10, line_search=True, alpha=0.01)
return_info, Y = manisdp_unittrace(C, A, b, **kwargs)

print('cvxpy obj:', fval_cvxpy)
print('manisdp obj:', return_info['cost'])
```

WARNING: `manisdp_unittrace` is 10x slower than that matlab version. After doing some profiling, I think the sparse matrix multiplication is the bottleneck.
