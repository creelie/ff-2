"""Exact dim der(H(x)O) and the centre of T = C(x)H(x)O (to get dim der_R(T) = 2 dim der(H(x)O))."""
import numpy as np, runpy
ns = runpy.run_path('dixon_checks_lib.py')
MH, MO, MC, tensor_algebra = ns['MH'], ns['MO'], ns['MC'], ns['tensor_algebra']
MHO = tensor_algebra(MH, MO); n = 32
G = np.zeros((n * n, n * n))
# accumulate A^T A over output blocks to limit memory: R[i,j,m] linear in D
cols = []
for a in range(n):
    for b in range(n):
        D = np.zeros((n, n)); D[a, b] = 1
        R = np.einsum('ijk,mk->ijm', MHO, D) - np.einsum('li,ljm->ijm', D, MHO) - np.einsum('lj,ilm->ijm', D, MHO)
        cols.append(R.ravel().astype(np.float32))
A = np.array(cols, dtype=np.float32)   # (1024, 32768)
G = (A.astype(np.float64) @ A.T.astype(np.float64))
w = np.linalg.eigvalsh(G)
print("dim der(H(x)O) =", int((w < 1e-7 * w.max()).sum()))
MT = tensor_algebra(MC, MHO); N = 64
E = np.eye(N)
# centre: z with zx = xz and (z,x,y)=(x,z,y)=(x,y,z)=0 for all basis x,y
rows = []
L = np.einsum('i,ijk->ikj', np.ones(1), np.zeros((1, 1, 1))) if False else None
for x in range(N):
    for y in range(N):
        pass
# linear conditions on z (coordinates): commutator and associators
comm = np.einsum('zjk->zjk', MT) - np.einsum('jzk->zjk', MT)            # (z x) - (x z), index [z,x,k]
xy = MT  # [x,y,k]
assoc1 = np.einsum('zxk,kyl->zxyl', MT, MT) - np.einsum('xyk,zkl->zxyl', MT, MT)   # (z x) y - z (x y)
assoc2 = np.einsum('xzk,kyl->zxyl', MT, MT) - np.einsum('zyk,xkl->zxyl', MT, MT)   # (x z) y - x (z y)
assoc3 = np.einsum('xyk,kzl->zxyl', MT, MT) - np.einsum('yzk,xkl->zxyl', MT, MT)   # (x y) z - x (y z)
C = np.concatenate([comm.reshape(N, -1), assoc1.reshape(N, -1), assoc2.reshape(N, -1), assoc3.reshape(N, -1)], axis=1)
s = np.linalg.svd(C, compute_uv=False)
print("dim of centre (commutative and associative) of C(x)H(x)O =", int((s < 1e-9 * s.max()).sum()))
