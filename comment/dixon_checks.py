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

def der_dim(M):
    n = M.shape[0]
    rows = []
    for a in range(n):
        for b in range(n):
            D = np.zeros((n, n)); D[a, b] = 1
            R = np.einsum('ijk,mk->ijm', M, D) - np.einsum('li,ljm->ijm', D, M) - np.einsum('lj,ilm->ijm', D, M)
            rows.append(R.ravel())
    A = np.array(rows).T
    s = np.linalg.svd(A, compute_uv=False)
    return int((s < 1e-9 * s.max()).sum())
MCH = tensor_algebra(MC, MH)   # index = 4*c + h
MCO = tensor_algebra(MC, MO)   # index = 8*c + o
print("dim der(H) =", der_dim(MH), "; dim der(O) =", der_dim(MO))
print("dim der_R(C(x)H) =", der_dim(MCH), " (paper uses su2, dim 3)")
print("dim der_R(C(x)O) =", der_dim(MCO), " (paper uses g2, dim 14)")

# D_{x,y} = [Lx,Ly]+[Lx,Ry]+[Rx,Ry] spans (paper's Tits bracket, item 3)
def L(M, x): return np.einsum('i,ijk->kj', x, M)
def Rm(M, y): return np.einsum('j,ijk->ki', y, M)
def Dspan(M, n):
    Ds = []
    E = np.eye(n)
    for p, q in itertools.combinations(range(1, n), 2):  # A' = real-trace-zero part = span of non-identity basis
        x, y = E[p], E[q]
        Lx, Ly, Rx, Ry = L(M, x), L(M, y), Rm(M, x), Rm(M, y)
        Ds.append((Lx @ Ly - Ly @ Lx + Lx @ Ry - Ry @ Lx + Rx @ Ry - Ry @ Rx).ravel())
    return np.linalg.matrix_rank(np.array(Ds), tol=1e-9)
print("dim span{D_{a,b}: a,b in (CxH)'} =", Dspan(MCH, 8), "; in (CxO)' =", Dspan(MCO, 16),
      "; in H' =", Dspan(MH, 4), "; in O' =", Dspan(MO, 8))

# the map gamma on CxH: z0 + z_k i_k -> conj(z0) - z_k i_k
def gamma(x):
    x = x.reshape(2, 4).copy()   # x[c,h]: c=0 real part, c=1 imaginary part of z_h
    x[1, 0] *= -1                # conj(z0)
    x[:, 1:] *= -1               # -z_k i_k
    return x.ravel()
E8 = np.eye(8)
icx = E8[4 * 1 + 0]   # i (x) 1
i1 = E8[4 * 0 + 1]    # 1 (x) i_1
lhs = gamma(mul(MCH, icx, i1)); rhs = mul(MCH, gamma(i1), gamma(icx))
print("gamma(x y) == gamma(y) gamma(x) for x=i, y=i_1 ?", np.allclose(lhs, rhs), "| gamma(xy) =", lhs, " gamma(y)gamma(x) =", rhs)
bad = 0
rng = np.random.default_rng(3)
for _ in range(100):
    x, y = rng.standard_normal(8), rng.standard_normal(8)
    bad += not np.allclose(gamma(mul(MCH, x, y)), mul(MCH, gamma(y), gamma(x)))
print("random pairs violating anti-automorphism property:", bad, "/ 100")
# closure of gamma-Hermitian 3x3 matrices with real diagonal under X.Y = (XY+YX)/2
def herm(entries):  # entries: dict (i,j)->element for i<j ; diag real
    X = np.zeros((3, 3, 8))
    for (i, j), v in entries.items():
        X[i, j] = v; X[j, i] = gamma(v)
    return X
def mm(X, Y):
    Z = np.zeros((3, 3, 8))
    for i in range(3):
        for k in range(3):
            for j in range(3):
                Z[i, k] += mul(MCH, X[i, j], Y[j, k])
    return Z
X = herm({(0, 1): icx}); Y = herm({(0, 1): i1})
J = 0.5 * (mm(X, Y) + mm(Y, X))
print("(X.Y)_11 for X_12 = i, Y_12 = i_1:", J[0, 0], "-> real?", np.allclose(J[0, 0][1:], 0))
# minimal dimension of symmetric elements over all involutions of CxH: involution types of M2(C)
print("symmetric-element dims of the involution types of CxH ~ M_2(C): orthogonal 6, symplectic 2, unitary 4 (all > 1)")
# check the three standard representatives
def symdim(f):
    A = np.array([f(E8[t]) for t in range(8)]).T
    return 8 - np.linalg.matrix_rank(A - np.eye(8))
def is_inv(f):
    ok = all(np.allclose(f(mul(MCH, E8[a], E8[b])), mul(MCH, f(E8[b]), f(E8[a]))) for a in range(8) for b in range(8))
    return ok and all(np.allclose(f(f(E8[a])), E8[a]) for a in range(8))
def sympl(x):  # z0 - z_k i_k with z0 NOT conjugated (C-linear quaternion conjugation)
    x = x.reshape(2, 4).copy(); x[:, 1:] *= -1; return x.ravel()
def unit(x):   # conj(z0) + conj(z_k)(-i_k): complex conj (x) quaternion conj
    x = x.reshape(2, 4).copy(); x[1, :] *= -1; x[:, 1:] *= -1; return x.ravel()
for name, f in [("paper's gamma", gamma), ("C-linear quaternion conjugation", sympl), ("conj (x) quaternion conj", unit)]:
    print(f"  {name}: anti-automorphic involution? {is_inv(f)}; dim of symmetric elements = {symdim(f)}")

# ---- der of the Dixon algebra T = C (x) H (x) O (64-dim): lower bound ----
# claim (arXiv:2303.11334, eq. der(T)): der(T) = su2 + g2 (dim 17)
MHO = tensor_algebra(MH, MO)          # 32-dim, index 8*h + o
MT = tensor_algebra(MC, MHO)          # 64-dim, index 32*c + (8*h+o)
def is_der(D, M):
    R = np.einsum('ijk,mk->ijm', M, D) - np.einsum('li,ljm->ijm', D, M) - np.einsum('lj,ilm->ijm', D, M)
    return np.abs(R).max() < 1e-9
# derivations of H and O (bases from nullspaces), lifted to T
def der_basis(M):
    n = M.shape[0]; rows = []
    for a in range(n):
        for b in range(n):
            D = np.zeros((n, n)); D[a, b] = 1
            R = np.einsum('ijk,mk->ijm', M, D) - np.einsum('li,ljm->ijm', D, M) - np.einsum('lj,ilm->ijm', D, M)
            rows.append(R.ravel())
    A = np.array(rows).T
    u, s, vt = np.linalg.svd(A)
    null = vt[s.size - (s < 1e-9 * s.max()).sum():] if (s < 1e-9 * s.max()).sum() else np.zeros((0, n * n))
    # robust: take right-singular vectors with tiny singular values
    w, U = np.linalg.eigh(A.T @ A)
    return [U[:, t].reshape(n, n) for t in range(n * n) if w[t] < 1e-9 * w.max()]
dH, dO = der_basis(MH), der_basis(MO)
I2, I4, I8 = np.eye(2), np.eye(4), np.eye(8)
Jc = np.array([[0, -1], [1, 0]])  # multiplication by i on C (basis 1,i): 1->i, i->-1
lifts = []
for D in dH:
    lifts.append(np.kron(I2, np.kron(D, I8)))
for D in dO:
    lifts.append(np.kron(I2, np.kron(I4, D)))
base = list(lifts)
lifts += [np.kron(Jc, X[:32, :32]) for X in base]   # i * D
print("Dixon T: #candidate derivations", len(lifts), "; all satisfy the derivation identity:",
      all(is_der(X, MT) for X in lifts))
print("rank of their span:", np.linalg.matrix_rank(np.array([X.ravel() for X in lifts]), tol=1e-9),
      " => dim der(C(x)H(x)O) >= 34 > 17 = dim(su2+g2)")
