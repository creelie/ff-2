# Checks for the 27+3, GUT, Dixon–Rosenfeld and Monstrous M-theory refutations

Programs that recheck the finite arithmetic in two papers by Deep Bhattacharjee. Every proof in the papers is analytic; the programs only confirm numbers.

- `null-sym-appendix/`: the further obstructions appendix of "Residual-Dimension Obstructions for Null-Projected Supersymmetric Yang–Mills Theories" (f4(−20) signature, spinor branchings, symmetric squares, root counts).
- `comment/`: "Derivations, Jordan Structures and Leech Vectors in Dixon–Rosenfeld and Monstrous Models" (derivation algebras of C⊗H, C⊗O, C⊗H⊗O; Jordan closure; Leech minimal vectors and Co_0).

Run each script from its folder with `python3 <script>.py`; it prints the output stored in the matching `.log`. Requires Python 3 with numpy, sympy and mpmath.

Author: Deep Bhattacharjee. Licence: see `LICENSE`.
