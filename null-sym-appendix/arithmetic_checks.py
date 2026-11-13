"""Arithmetic / representation-count checks for arXiv:2008.06742, 1906.10709, 1811.06101."""
from math import comb
L = lambda k: comb(24, k)
S = lambda: 2048
print("== 2008.06742 ==")
# Griess assignment, Sec. 3 (after eq. fieldc')
parts = {'L5': (L(5), 1), 'L4': (L(4), 4), 'L3': (L(3), 6), 'L2': (L(2), 4), 'L1': (L(1), 1)}
tot = sum(d * m for d, m in parts.values())
tr = sum(d * m * (-1) ** int(k[1]) for k, (d, m) in parts.items())
print("98280 = L5+4L4+6L3+4L2+L1 :", tot, tot == 98280)
print("trace of -1 in SO(24) on that SO(24)-module:", tr,
      "| trace of -1 in Co_0 on R[Lambda_4/+-1]: 98280 (acts trivially)")
print("dim of (-1)-eigenspace of the assigned module:", sum(d * m for k, (d, m) in parts.items() if int(k[1]) % 2))
print("196884 = 1+299+98280+98304:", 1 + 299 + 98280 + 98304 == 196884, "; 98304 = 24*4096:", 24 * 4096 == 98304)
# fieldc'
print("fieldc': 324+2*53130+6*12650+6*2300+2*300 =", 324 + 2 * 53130 + 6 * 12650 + 6 * 2300 + 2 * 300)
print("dims of so(25) forms: L5,L4,L3,L2 of 25 =", [comb(25, k) for k in (5, 4, 3, 2)], "; Sym^2_0(25) =", 25 * 26 // 2 - 1)
# Leech algebra
print("L24: 624 + 2*(53130+3*12650+3*2300) =", 624 + 2 * (53130 + 3 * 12650 + 3 * 2300), "; minus 24 =",
      624 + 2 * (53130 + 3 * 12650 + 3 * 2300) - 24)
distinct = 600 + 2 * (comb(25, 5) + comb(25, 4) + comb(25, 3))
print("L24 as su(25)-module a24+V+Vbar, V = L5+3L4+3L3 of C^25: distinct nonzero Cartan weights =", distinct,
      "(not 196560); max weight multiplicity = 3")
print("98280 = C(28,5):", comb(28, 5) == 98280, "; 196560 = 3*240*273:", 3 * 240 * 273 == 196560)
# weak triality etc.
psi = 24 * 2048 - 2048
psi2 = 2048 * comb(24, 2) - psi - 2048
print("psi = 24*2048-2048 =", psi, "; psi^(2) =", psi2)
print("pt: 2048 = 24+2024:", 24 + 2024 == 2048)
print("ptt:", 2 * 276 + 2 * 2024 + 4 * L(4) == psi, 2 * 276 + 2 * 2024 + L(5) == psi)
print("terza:", 14 + 8 * 24 + 276 + 5 * 2024 + 3 * L(4) + 3 * L(5) + L(7) == psi2, " L7 =", L(7))
# K(q) coefficients from T_2B = eta(t)^24/eta(2t)^24 + 24 = q^-1 prod (1+q^n)^-24 + 24
N = 10
poly = [0] * (N + 2); poly[0] = 1
for n in range(1, N + 2):
    # multiply by (1+q^n)^-24 = sum_k binom(-24,k) q^{nk}
    new = [0] * (N + 2)
    for i, c in enumerate(poly):
        if c == 0: continue
        k = 0
        while i + n * k < N + 2:
            new[i + n * k] += c * comb(24 + k - 1, k) * (-1) ** k
            k += 1
    poly = new
T2B = {e - 1: c for e, c in enumerate(poly)}
T2B[0] += 24
print("T_2B coefficients q^-1..q^7:", [T2B[e] for e in range(-1, 8)])
K = [1, 276, 2048, 11202, 49152, 184024, 614400, 1881471]
print("K(q) (eq. pf) coefficients q^-1/2..q^7/2:", K, "; |T_2B| match:", [abs(T2B[e]) for e in range(-1, 8) if e != 0] == K)
print("K(q) = -T_2B(-q^(1/2)) holds; literal 'q^(1/2) Z^2B(q)' would give exponents -1/2, 3/2, 5/2,... with alternating signs")
# dec-2 lines
dec2 = {
    276: {2: 1}, 2048: {1: 1, 3: 1}, 11202: {1: 1, 2: 2, 4: 1}, 49152: {1: 1, 2: 2, 3: 3, 5: 1},
    184024: {1: 1, 2: 3, 3: 3, 5: 1, 6: 1}, 614400: {0: 16, 1: 8, 2: 7, 3: 2, 5: 3, 6: 1, 7: 1},
    1881471: {0: 23, 1: 2, 2: 4, 3: 1, 4: 1, 5: 3, 6: 2, 8: 2}}
for val, mult in dec2.items():
    s = sum(m * L(p) for p, m in mult.items())
    viol = [(p, m * L(p), L(p + 1)) for p, m in mult.items() if m * L(p) > L(p + 1)]
    print(f"dec-2 {val}: sum={s} ok={s == val}; violations of #L^p*|L^p| <= |L^(p+1)|: {viol}")
print("== 1906.10709 ==")
print("odd forms of 15:", [comb(15, k) for k in range(1, 16, 2)], sum(comb(15, k) for k in range(1, 16, 2)))
print("even forms of 15:", [comb(15, k) for k in range(0, 16, 2)], "(paper lists 25' instead of 15')")
print("== EP / e7^(2) root count ==")
def simple_max(r):
    c = [r * (r + 1)]
    if r >= 2: c.append(2 * r * r)
    if r >= 4: c.append(2 * r * (r - 1))
    c += {2: [12], 4: [48], 6: [72], 7: [126], 8: [240]}.get(r, [])
    return max(c)
def maxroots(r):  # max #roots of a semisimple Lie algebra of rank r (DP over simple components)
    f = [0] * (r + 1)
    for n in range(1, r + 1):
        f[n] = max(simple_max(k) + f[n - k] for k in range(1, n + 1))
    return f[r]
dim = 190 + 3 + 1024; r = 11
print("so20+su2+(512,2): dim", dim, "rank", r, "=> #roots if semisimple:", dim - r, "> max over rank-11 semisimple =", maxroots(r))
print("Leech: 196560 'roots' vs max over rank-24 semisimple =", maxroots(24))
for n in range(1, 4):
    d = (8 * n + 8) * (8 * n + 7) // 2 + 2 ** (4 * n + 3); rk = 4 * n + 4
    print(f"e8^({n}) = so({8*n+4},4)+2^{4*n+3}: dim {d}, rank {rk}, roots-if-Lie {d-rk} vs max {maxroots(rk)}")
