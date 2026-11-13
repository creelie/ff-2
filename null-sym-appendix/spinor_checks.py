"""Spinor branching and symmetry checks (weights in doubled units, i.e. 2*weight).

(I)  arXiv:2202.08459 (Universe 7 (2021) 462), eqs. (11-brane), (19-brane):
     32768 of so(28,4) under so(4,4)+so(8)+so(16) and so(4,4)+so(16)+so(8).
(H)  arXiv:1906.10709, eq. after (a-2-3): 16384 (MW of so(27,3)) = (+)_i wedge^{2i+1} 15,
     called "so_15-covariant".
(J)  arXiv:2412.20191, eq. (se8): {Q,Q} ~ gamma^{ab} M_ab on the 128 of so(12,4).
"""
import itertools
from collections import Counter
import numpy as np

def spinor(n, chir):  # D_n semispinor weights, doubled: (+-1)^n, #minus even (chir=+1) / odd (-1)
    out = []
    for s in itertools.product([1, -1], repeat=n):
        if (s.count(-1) % 2 == 0) == (chir == 1):
            out.append(s)
    return out
def vector(n):
    out = []
    for i in range(n):
        for sg in (2, -2):
            w = [0] * n; w[i] = sg; out.append(tuple(w))
    return out
D4 = {'v': vector(4), 's': spinor(4, 1), 'c': spinor(4, -1)}
D8 = {'+': spinor(8, 1), '-': spinor(8, -1)}  # '+' = 128, '-' = 128'

def tensor(*ws):
    return Counter(tuple(itertools.chain(*t)) for t in itertools.product(*ws))

print("=== (I) WDM 32768 branchings ===")
S16p = spinor(16, 1)
true_11 = Counter(S16p)  # coordinates split (4 | 4 | 8)
claim_11 = [('s', 'c', '-'), ('c', 's', '-'), ('c', 's', '+'), ('s', 'c', '+')]
claim_19 = [('s', '-', 'c'), ('c', '+', 'c'), ('c', '+', 's'), ('s', '-', 's')]  # (so44, so16, so8)

def claimed_counter(claim, order, rel):
    r1, r2, r16 = rel
    c = Counter()
    for t in claim:
        if order == '11':
            a, b, x = t
            c += tensor(D4[r1[a]], D4[r2[b]], D8[r16[x]])
        else:
            a, x, b = t
            c += tensor(D4[r1[a]], D8[r16[x]], D4[r2[b]])
    return c
perms = [dict(zip('vsc', p)) for p in itertools.permutations('vsc')]
flips = [{'+': '+', '-': '-'}, {'+': '-', '-': '+'}]
for name, claim, order in [('(11-brane)', claim_11, '11'), ('(19-brane)', claim_19, '19')]:
    for chir in (1, -1):
        truth = Counter(spinor(16, chir))
        hits = 0
        for r1 in perms:
            for r2 in perms:
                for r16 in flips:
                    if claimed_counter(claim, order, (r1, r2, r16)) == truth:
                        hits += 1
        print(f"{name}: #(triality x triality x parity) relabelings (72) matching the "
              f"{'S+' if chir == 1 else 'S-'} of D16: {hits}")
# the true branching
def true_branch(order):
    out = []
    for a in 'sc':
        for b in 'sc':
            for x in '+-':
                sa = 1 if a == 's' else -1; sb = 1 if b == 's' else -1; sx = 1 if x == '+' else -1
                if sa * sb * sx == 1:
                    out.append((a, b, x))
    return out
tb = true_branch('11')
assert claimed_counter(tb, '11', (perms[0], perms[0], flips[0])) == true_11
print("true: 32768 = " + " + ".join(f"(8{a},8{b},128{'' if x == '+' else chr(39)})" for a, b, x in tb),
      "  [labels up to triality/parity]")
# internal contradiction: their two lines give the same restriction for 128 and 128' of so(12,4)
r128 = Counter(); r128p = Counter()
for a, b, x in claim_11:
    (r128 if x == '-' else r128p)[(a, b)] += 1   # line 1: (128,128') + (128',128)
print("(11-brane) as stated: so(12,4) 128 restricts to", dict(r128), "; 128' restricts to", dict(r128p))
def lab(ws):
    out = Counter()
    for a in 'sc':
        for b in 'sc':
            if Counter(tensor(D4[a], D4[b])) <= Counter(ws):
                out[(a, b)] += 1
    return dict(out)
print("  actual restriction of D8 semispinors to so(4,4)+so(8): S+ ->", lab(D8['+']), "; S- ->", lab(D8['-']))
print("  => the two inequivalent semispinors restrict differently; the stated line 2 assigns both the same pair.")
print("\n=== (H) so(15)-covariance of the so(27,3) MW spinor ===")
B7V = [tuple(2 if k == i else 0 for k in range(7)) for i in range(7)] + \
      [tuple(-2 if k == i else 0 for k in range(7)) for i in range(7)] + [(0,) * 7]
def wedge(k):
    return Counter(tuple(map(sum, zip(*c))) if c else (0,) * 7 for c in itertools.combinations(B7V, k))
odd = Counter(); even = Counter()
for k in range(16):
    (odd if k % 2 else even).update(wedge(k))
print("sum dims odd/even forms of 15:", sum(odd.values()), sum(even.values()))
S15 = {1: spinor(15, 1), -1: spinor(15, -1)}
def restrict(ws, mp):
    return Counter(tuple(sum(w[j] for j in mp[i]) for i in range(7)) for w in ws)
mp_c = [[i, 7 + i] for i in range(7)]      # C^30 = V + V  (eps_j, eps_{7+j} -> e_j ; eps_15 -> 0)
mp_b = [[i] for i in range(7)]             # C^30 = V + 15*1 (eps_8..eps_15 -> 0)
B7S = Counter(spinor(7, 1) + spinor(7, -1))  # 128 of B7 (doubled weights (+-1)^7)
for chir in (1, -1):
    rc = restrict(S15[chir], mp_c); rb = restrict(S15[chir], mp_b)
    print(f"D15 S{'+' if chir == 1 else '-'}: (V+V) restriction == odd forms: {rc == odd}; == even forms: {rc == even}")
    print(f"           (V+15) restriction == 128 x (B7 spinor 128): "
          f"{rb == Counter({w: 128 * m for w, m in B7S.items()})}; contains weight of 15's highest weight e1: "
          f"{rb[(2, 0, 0, 0, 0, 0, 0)] > 0}")
sigs = set()
for p_ in range(16):
    q_ = 15 - p_
    sigs |= {(2 * p_, 2 * q_), (15, 15)}
print("signatures (s,t) of R^30 for real so(15)-forms acting as V+V (incl. so(15,C)_R):", sorted(sigs))
print("(27,3) or (3,27) among them:", (27, 3) in sigs or (3, 27) in sigs)

print("\n=== (J) symmetry of C Gamma^(k) on the 128 of D8 ===")
s1 = np.array([[0, 1], [1, 0]], complex); s2 = np.array([[0, -1j], [1j, 0]]); s3 = np.diag([1, -1]).astype(complex); I2 = np.eye(2)
def kron(*ms):
    out = np.array([[1]], complex)
    for m in ms: out = np.kron(out, m)
    return out
n = 8
gam = []
for k in range(n):
    for s in (s1, s2):
        gam.append(kron(*([s3] * k + [s] + [I2] * (n - k - 1))))
gam = [g for g in gam]
for a in range(16):
    for b in range(16):
        assert np.allclose(gam[a] @ gam[b] + gam[b] @ gam[a], 2 * (a == b) * np.eye(256))
Asym = [g for g in gam if np.allclose(g.T, -g)]
Cm = np.eye(256, dtype=complex)
for g in Asym: Cm = Cm @ g
signs = {complex(np.round((Cm @ g @ np.linalg.inv(Cm))[np.nonzero(g.T)][0] / g.T[np.nonzero(g.T)][0], 8)) for g in gam}
print("C gamma_a C^-1 = s gamma_a^T with s in", signs)
chir = np.eye(256, dtype=complex)
for g in gam: chir = chir @ g
chir = chir * (1j) ** (-8) if not np.allclose(chir @ chir, np.eye(256)) else chir
ev, U = np.linalg.eigh((chir + chir.conj().T) / 2)
Pp = U[:, ev > 0]  # 128-dim
print("dim S+ =", Pp.shape[1])
rng = np.random.default_rng(2)
res = {}
for k in (0, 2, 4, 6, 8):
    syms = set()
    for _ in range(6):
        idx = sorted(rng.choice(16, size=k, replace=False))
        G = np.eye(256, dtype=complex)
        for i in idx: G = G @ gam[i]
        Bf = Pp.T @ Cm @ G @ Pp
        if np.abs(Bf).max() < 1e-9:
            syms.add('zero'); continue
        if np.allclose(Bf.T, Bf): syms.add('sym')
        elif np.allclose(Bf.T, -Bf): syms.add('antisym')
        else: syms.add('mixed')
    res[k] = syms
    print(f"  k={k}: C Gamma^(k) restricted to S+ x S+ is {syms}")
print("=> S^2(128) = wedge^0 + wedge^4 + wedge^8_+ = 1+1820+6435 = 8256 =", 128 * 129 // 2,
      "; wedge^2(128) = wedge^2 + wedge^6 = 120+8008 = 8128 =", 128 * 127 // 2)
print("=> the adjoint 120 = wedge^2 occurs only in the ANTISYMMETRIC square: an so(16)- (hence so(12,4)-)"
      " equivariant symmetric map S^2(128) -> so(16) is zero.")
