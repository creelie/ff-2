"""Checks of arXiv:2512.02271 (Dixon-Rosenfeld planes), Sec. 2-3:
 - Corollary cor:CxO: J_3(C x H) with biquaternionic off-diagonal and real diagonal, via the map
   gamma: z0 + z_k i_k -> conj(z0) - z_k i_k, "is a Jordan algebra";
 - eq. (Alg1): L_3(CxH, O) = su2 + f4 + (7,26), dim 237 = 3+52+7*26  (der(CxH) taken as su2);
 - eq. (Alg3): L_3(CxO, H) = g2 + sp6 + (15,14), dim 245 = 14+21+15*14 (der(CxO) taken as g2).
"""
import itertools
import numpy as np

def tensor_algebra(MA, MB):
    """structure constants of the real tensor product algebra A (x) B"""
    a, b = MA.shape[0], MB.shape[0]
    M = np.einsum('ijk,lmn->iljmkn', MA, MB).reshape(a * b, a * b, a * b)
    return M
# C: basis 1,i
MC = np.zeros((2, 2, 2)); MC[0, 0, 0] = 1; MC[0, 1, 1] = 1; MC[1, 0, 1] = 1; MC[1, 1, 0] = -1
# H: basis 1,i,j,k
MH = np.zeros((4, 4, 4))
for x in range(4): MH[0, x, x] = 1; MH[x, 0, x] = 1
for x in range(1, 4): MH[x, x, 0] = -1
for (x, y, z) in [(1, 2, 3), (2, 3, 1), (3, 1, 2)]:
    MH[x, y, z] = 1; MH[y, x, z] = -1
# O: as in f4_signature.py
MO = np.zeros((8, 8, 8))
for x in range(8): MO[0, x, x] = 1; MO[x, 0, x] = 1
for x in range(1, 8): MO[x, x, 0] = -1
for t in range(7):
    a_, b_, c_ = t % 7 + 1, (t + 1) % 7 + 1, (t + 3) % 7 + 1
    for (x, y, z) in [(a_, b_, c_), (b_, c_, a_), (c_, a_, b_)]:
        MO[x, y, z] = 1; MO[y, x, z] = -1
mul = lambda M, x, y: np.einsum('i,j,ijk->k', x, y, M)

