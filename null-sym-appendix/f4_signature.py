"""der(J_{1,2}^O) and the signature of its invariant form on the 26.

Checks arXiv:1906.10709 (PLB 808 (2020) 135674), Sec. 4: claimed
"anomalous" embedding f4(-20) in so(25,1) with 26 = 26, and
f4(-20) = der(J_{1,2}^O), 26 = (J_{1,2}^O)_0.
"""
import itertools
import numpy as np

# ---- octonions: e0=1, e_i e_{i+1} = e_{i+3} (indices mod 7 in 1..7) ----
M = np.zeros((8, 8, 8))
for i in range(8):
    M[0, i, i] = 1; M[i, 0, i] = 1
for i in range(1, 8):
    M[i, i, 0] = -1
for i in range(7):
    a, b, c = (i % 7) + 1, ((i + 1) % 7) + 1, ((i + 3) % 7) + 1
    for (x, y, z) in [(a, b, c), (b, c, a), (c, a, b)]:
        M[x, y, z] = 1; M[y, x, z] = -1
def omul(x, y): return np.einsum('i,j,ijk->k', x, y, M)
def oconj(x): return np.concatenate([[x[0]], -x[1:]])
rng = np.random.default_rng(1)
x, y = rng.standard_normal(8), rng.standard_normal(8)
assert np.allclose(omul(omul(x, x), y), omul(x, omul(x, y)))  # alternative
assert np.allclose(omul(x, y) @ omul(x, y), (x @ x) * (y @ y))  # composition
print("octonion table: alternative and normed: OK")

eta = np.array([1.0, 1.0, -1.0])
# element: dict of 3x3 octonion matrix as array (3,3,8); Hermitian H, product X.Y = 1/2(H eta K + K eta H)
def mat(v):
    H = np.zeros((3, 3, 8))
    H[0, 0, 0], H[1, 1, 0], H[2, 2, 0] = v[0], v[1], v[2]
    off = [(0, 1), (0, 2), (1, 2)]
    for t, (i, j) in enumerate(off):
        o = v[3 + 8 * t: 11 + 8 * t]
        H[i, j] = o; H[j, i] = oconj(o)
    return H
def vec(H):
    off = [(0, 1), (0, 2), (1, 2)]
    assert all(np.allclose(H[j, i], oconj(H[i, j])) for (i, j) in off)
    assert all(np.allclose(H[i, i, 1:], 0) for i in range(3))
    return np.concatenate([[H[0, 0, 0], H[1, 1, 0], H[2, 2, 0]]] + [H[i, j] for (i, j) in off])
def mmul(A, Bm):
    C = np.zeros((3, 3, 8))
    for i in range(3):
        for k in range(3):
            for j in range(3):
                C[i, k] += omul(A[i, j], Bm[j, k]) * eta[j]
    return C
def jprod(u, v):
    A, Bm = mat(u), mat(v)
    return vec(0.5 * (mmul(A, Bm) + mmul(Bm, A)))
N = 27
C = np.zeros((N, N, N))
E = np.eye(N)
for i in range(N):
    for j in range(i, N):
        C[i, j] = jprod(E[i], E[j]); C[j, i] = C[i, j]
def jp(u, v): return np.einsum('i,j,ijk->k', u, v, C)
u, v = rng.standard_normal(N), rng.standard_normal(N)
uu = jp(u, u)
print("Jordan identity residual:", np.abs(jp(jp(u, v), uu) - jp(u, jp(v, uu))).max())
unit = np.array([1, 1, -1] + [0] * 24, dtype=float)  # H = eta is the unit
print("unit check residual:", np.abs(jp(unit, v) - v).max())

# ---- derivations ----
rows = []
for a in range(N):
    for b in range(N):
        D = np.zeros((N, N)); D[a, b] = 1.0
        # R[i,j,m] = sum_k C[i,j,k] D[m,k] - sum_l D[l,i] C[l,j,m] - sum_l D[l,j] C[i,l,m]
        R = np.einsum('ijk,mk->ijm', C, D) - np.einsum('li,ljm->ijm', D, C) - np.einsum('lj,ilm->ijm', D, C)
        iu = np.triu_indices(N)
        rows.append(R[iu[0], iu[1], :].ravel())
A = np.array(rows).T  # equations x unknowns
G = A.T @ A
w, U = np.linalg.eigh(G)
null = U[:, w < 1e-9 * w.max()]
print("dim der(J_{1,2}^O) =", null.shape[1])
Ds = [null[:, t].reshape(N, N) for t in range(null.shape[1])]
# Killing-type form tr(D1 D2) on the 27: signature identifies the real form
K = np.array([[np.trace(D1 @ D2) for D2 in Ds] for D1 in Ds])
kw = np.linalg.eigvalsh(K)
print("trace form on der: (#pos, #neg, #zero) =", (int((kw > 1e-8).sum()), int((kw < -1e-8).sum()), int((abs(kw) <= 1e-8).sum())),
      " [f4(-20): 16 noncompact (+), 36 compact (-)]")
# ---- the 26: span of images of derivations ----
img = np.hstack(Ds)
uu_, s, _ = np.linalg.svd(img)
W = uu_[:, s > 1e-8]
print("dim of span of images (the 26):", W.shape[1], "; unit annihilated by all D:", max(np.abs(D @ unit).max() for D in Ds) < 1e-9)
# invariant symmetric forms on W: B with (P D P)^T B + B (P D P) = 0, D restricted to W
Dw = [W.T @ D @ W for D in Ds]
n = W.shape[1]
symb = []
for i in range(n):
    for j in range(i, n):
        S = np.zeros((n, n)); S[i, j] = S[j, i] = 1; symb.append(S)
eqs = np.array([np.concatenate([(Dk.T @ S + S @ Dk).ravel() for Dk in Dw]) for S in symb]).T
w2, U2 = np.linalg.eigh(eqs.T @ eqs)
nb = U2[:, w2 < 1e-9 * w2.max()]
print("dim of invariant symmetric bilinear forms on the 26:", nb.shape[1])
Binv = sum(c * S for c, S in zip(nb[:, 0], symb))
bw = np.linalg.eigvalsh(Binv)
sig = (int((bw > 1e-9).sum()), int((bw < -1e-9).sum()))
print("signature of the (unique up to scale) invariant form on the 26:", sig, " => f4(-20) sits in so(10,16), not so(25,1)")
# cross-check with the Jordan trace form T(X.Y), T(X) = tr(H eta)
T = lambda z: z[0] * eta[0] + z[1] * eta[1] + z[2] * eta[2]
TF = np.array([[T(jp(W[:, i], W[:, j])) for j in range(n)] for i in range(n)])
tw = np.linalg.eigvalsh(TF)
print("Jordan trace form T(X.Y) on the 26: signature", (int((tw > 1e-9).sum()), int((tw < -1e-9).sum())))
