"""Leech lattice minimal vectors and the Co_0 permutation module R[Lambda_4].

Checks claims of arXiv:2008.06742 (Monstrous M-theory), footnote in Sec. 2.2:
  196560 = 2*95680 + 4576 + 2*276 + 3*24          (form 1)
  196560 = 2*95680 + 4576 + 299 + 276 + 2*24 + 1  (form 2)
and the Sec. 7 "Leech algebra" claim that the 196560 minimal vectors play
the role of roots.
"""
import itertools, math
import numpy as np

# ---------- extended binary Golay code via QR code mod 23 ----------
p = 23
Q = sorted({(x * x) % p for x in range(1, p)})
assert len(Q) == 11
INF = 23  # coordinate index for infinity; F_23 elements 0..22 are indices 0..22

def gf2_rank(rows):
    rows = [int(''.join(map(str, r)), 2) for r in rows]
    rank = 0
    width = 24
    for bit in reversed(range(width)):
        piv = None
        for i in range(rank, len(rows)):
            if rows[i] >> bit & 1:
                piv = i; break
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for i in range(len(rows)):
            if i != rank and rows[i] >> bit & 1:
                rows[i] ^= rows[rank]
        rank += 1
    return rank

gens = []
for s in range(p):
    v = [0] * 24
    for q in Q:
        v[(q + s) % p] = 1
    v[INF] = sum(v[:23]) % 2  # parity extension
    gens.append(v)
allones = [1] * 24
gens.append(allones)
# reduce to a basis
basis = []
for g in gens:
    if gf2_rank(basis + [g]) > len(basis):
        basis.append(g)
assert len(basis) == 12, len(basis)
B = np.array(basis, dtype=np.int64)
codewords = np.array([np.array(c) @ B % 2 for c in itertools.product([0, 1], repeat=12)], dtype=np.int8)
wt = codewords.sum(1)
wd = {int(w): int((wt == w).sum()) for w in sorted(set(wt.tolist()))}
print("Golay weight distribution:", wd)
assert wd == {0: 1, 8: 759, 12: 2576, 16: 759, 24: 1}
codeset = {tuple(c) for c in codewords.tolist()}

# PSL(2,23) element x -> -1/x (swaps 0 and infinity); check it preserves the code
def mob(x):
    if x == INF: return 0
    if x == 0: return INF
    return (-pow(x, p - 2, p)) % p
perm = [mob(i) for i in range(24)]
ok = all(tuple(c[perm[i]] for i in range(24)) in codeset for c in codewords.tolist())
print("x->-1/x preserves extended Golay code:", ok)
assert ok

# ---------- Leech minimal vectors, scaled by sqrt(8): norm 32 ----------
octads = codewords[wt == 8]
vecs = []
for o in octads:
    idx = np.nonzero(o)[0]
    for signs in itertools.product([1, -1], repeat=8):
        if signs.count(-1) % 2 == 0:
            v = np.zeros(24, dtype=np.int8); v[idx] = 2 * np.array(signs); vecs.append(v)
n1 = len(vecs)
for i, j in itertools.combinations(range(24), 2):
    for si, sj in itertools.product([4, -4], repeat=2):
        v = np.zeros(24, dtype=np.int8); v[i] = si; v[j] = sj; vecs.append(v)
n2 = len(vecs) - n1
for c in codewords:
    for j in range(24):
        v = np.ones(24, dtype=np.int8); v[j] = -3
        v = v * (1 - 2 * c.astype(np.int8)); vecs.append(v)
n3 = len(vecs) - n1 - n2
V = np.array(vecs, dtype=np.int64)
print("shape counts (2^8 0^16, 4^2 0^22, 3 1^23):", n1, n2, n3, "total", len(V))
assert (n1, n2, n3) == (97152, 1104, 98304) and len(V) == 196560
assert np.all((V * V).sum(1) == 32)
assert len({tuple(r) for r in V.tolist()}) == 196560
# Lattice check: all pairwise inner products of a sample with all vectors are in 8Z/... (norm-4 scale: integers)
G = (V[:2000] @ V.T)
assert np.all(G % 8 == 0)
ips = sorted(set((G // 8).ravel().tolist()))
print("inner products (norm-4 normalisation) occurring:", ips)
v0 = V[0]
d = {k: int(((V @ v0) // 8 == k).sum()) for k in ips}
print("inner-product distribution w.r.t. one minimal vector:", d)
assert d == {-4: 1, -2: 4600, -1: 47104, 0: 93150, 1: 47104, 2: 4600, 4: 1}

# ---------- spherical design strength ----------
rng = np.random.default_rng(0)
Vf = V.astype(np.float64) / math.sqrt(8.0)  # norm 4
print("spherical design test: sum_v (v.x)^(2k) vs isotropic average")
for k in range(1, 8):
    x = rng.standard_normal(24)
    lhs = ((Vf @ x) ** (2 * k)).sum()
    # isotropic: N * |v|^{2k}|x|^{2k} * (2k-1)!! / prod_{i=0}^{k-1}(24+2i)
    dfact = math.prod(range(1, 2 * k, 2))
    den = math.prod(24 + 2 * i for i in range(k))
    rhs = len(Vf) * 4.0 ** k * (x @ x) ** k * dfact / den
    print(f"  k={k}: rel. deviation {abs(lhs - rhs) / rhs:.3e}")

# ---------- central element -1 on R[Lambda_4] ----------
# -1 is in Co_0 (it preserves the lattice) and maps v -> -v; no minimal vector is fixed.
fixed = int(np.all(V == -V, axis=1).sum())
print("number of minimal vectors fixed by -1:", fixed, "=> trace(-1 | R[Lambda_4]) = 0")
print("=> dim(+1 eigenspace) = dim(-1 eigenspace) = 98280;  +1 eigenspace = R[Lambda_4/{+-1}]")

def subset_sums(parts):
    sums = {0}
    for d in parts:
        sums |= {s + d for s in sums}
    return sums
for name, parts in [("form 1", [95680, 95680, 4576, 276, 276, 24, 24, 24]),
                    ("form 2", [95680, 95680, 4576, 299, 276, 24, 24, 1])]:
    assert sum(parts) == 196560
    ss = subset_sums(parts)
    print(f"{name}: sum={sum(parts)}; can a sub-multiset of constituents have dim 98280? {98280 in ss}")
    near = sorted(ss, key=lambda s: abs(s - 98280))[:3]
    print(f"   closest achievable dims to 98280: {near}")
    print(f"   multiplicity of trivial rep: {parts.count(1)} (any permutation module has >= 1)")

# Their own footnote (Sec. 3): R[Lambda_4/+-1] = 98280 = 80730+17250+299+1 (Co_1-irreducibles).
print("their footnote 98280 = 80730+17250+299+1 :", 80730 + 17250 + 299 + 1 == 98280,
      "-- these constituents are absent from forms 1/2")

# true decomposition (consistent with ATLAS/11-design): harmonic pieces k=0..5 plus 80730
harm = [math.comb(24 + k - 1, k) - (math.comb(24 + k - 3, k - 2) if k >= 2 else 0) for k in range(6)]
print("dim Harm_k(R^24), k=0..5:", harm, "sum", sum(harm), "remainder", 196560 - sum(harm))

# ---------- Leech "roots": Cartan integers ----------
u = V[0]; w = V[np.nonzero((V @ u) // 8 == 1)[0][0]]
cart = 2 * (u @ w) / (w @ w)
print("pair of minimal vectors with <u,w>=1 (norm 4): 2<u,w>/<w,w> =", cart, "(not an integer)")
print("max #roots of a rank-24 reduced root system: B24/C24 have 2*24^2 =", 2 * 24 ** 2)

# ---------- second, independent obstruction: harmonic submodules ----------
# 11-design => for f in Harm_k, k<=5: sum_v f(v)^2 = N * avg_{S^23} f^2 > 0, so the
# Co_0-equivariant evaluation map Harm_k -> R[Lambda_4] is injective.
# Hence R[Lambda_4] contains Co_0-submodules of dims 24, 299, 2576, 17250, 95680.
for name, parts in [("form 1", [95680, 95680, 4576, 276, 276, 24, 24, 24]),
                    ("form 2", [95680, 95680, 4576, 299, 276, 24, 24, 1])]:
    ss = subset_sums(parts)
    print(name, "sub-multiset sums available for Harm_k dims:",
          {h: (h in ss) for h in harm[1:]})
