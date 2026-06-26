"""
When does compression suffice as a keep-criterion, and when must you price utility?

A controlled abstraction-library experiment for world-model planning, designed so the
two quantities the keep-decision trades off are INDEPENDENT knobs:

  - Each candidate skill i abstracts one segment of a task. It has a frequency f_i
    (fraction of tasks that need that segment) and a hardness h_i (the blind-search
    cost of that segment, modeled as B**h_i model-rollouts if the segment is not
    abstracted).

  - All segments share a description length D, so the MDL gain of keeping skill i is
    proportional to f_i ALONE. Compression measures statistical regularity, not search
    value: it is blind to hardness by construction.

  - The planning utility of keeping skill i is f_i * B**h_i: on every task that needs
    the segment, the macro replaces a blind B**h_i search with one O(B+|L|) lookup.

Under a carrying-cost budget K (keep at most K skills; a larger library also raises the
per-step matching floor B+|L|), an MDL keep-rule keeps the K most FREQUENT skills, while
a utility keep-rule keeps the K highest-(f_i * B**h_i) skills.

The expected held-out planning cost of a library L is exact (no Monte-Carlo):

    E[cost | L] = sum_i  f_i * ( (B + |L|)          if i in L
                                 B**h_i              otherwise )

We sweep two things: rho, the correlation between frequency and hardness, and the budget
K. The prediction (Minton's utility problem vs MDL, stated as a condition): the two keep-
rules AGREE when frequency and hardness are correlated, and utility DOMINATES when hard
skills are rare (negative rho), because MDL spends its budget on frequent-but-easy skills
and drops the rare-but-critical ones.
"""
from __future__ import annotations
import numpy as np

B = 4          # primitive branching factor
N = 30         # number of candidate skills (one per segment type)
SEEDS = range(16)


def population(rho: float, seed: int):
    """N skills whose frequency and hardness have target rank-correlation rho."""
    g = np.random.default_rng(seed)
    a = g.standard_normal(N)
    b = g.standard_normal(N)
    hz = rho * a + np.sqrt(max(0.0, 1.0 - rho * rho)) * b

    def rank01(z):                       # map to evenly spaced ranks in [0, 1]
        r = np.empty(N)
        r[np.argsort(z)] = np.linspace(0.0, 1.0, N)
        return r

    freq = 0.02 + 0.48 * rank01(a)       # frequencies in [0.02, 0.50]
    hard = 1 + np.rint(6 * rank01(hz)).astype(int)   # hardness exponent in 1..7
    return freq, hard


def ecost(freq, hard, L: set[int]) -> float:
    """Exact expected held-out planning cost under library L."""
    inL = np.zeros(N, dtype=bool)
    if L:
        inL[list(L)] = True
    covered = (B + len(L))                          # cheap lookup, grows with |L|
    uncovered = np.power(float(B), hard)            # blind search of the segment
    per_segment = np.where(inL, covered, uncovered)
    return float(np.sum(freq * per_segment))


def mdl_lib(freq, hard, K):                          # MDL gain ∝ frequency
    return set(np.argsort(-freq)[:K].tolist())


def util_lib(freq, hard, K):                          # utility ∝ frequency * B**hardness
    u = freq * np.power(float(B), hard)
    return set(np.argsort(-u)[:K].tolist())


def freq_threshold_lib(freq, hard, K):                # naive: same ranking as MDL here
    return mdl_lib(freq, hard, K)


# ---------------------------------------------------------------- 1-D slice (text)
def slice_table(K):
    print(f"\nFrequency↔hardness correlation sweep at budget K={K} "
          f"(cost = expected held-out planning rollouts):")
    print(f"{'rho':>6}{'MDL cost':>14}{'utility cost':>14}{'MDL/util':>10}  regime")
    print("-" * 60)
    for rho in [-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9]:
        cm = np.mean([ecost(*population(rho, s)[:2], mdl_lib(*population(rho, s), K)) for s in SEEDS])
        cu = np.mean([ecost(*population(rho, s)[:2], util_lib(*population(rho, s), K)) for s in SEEDS])
        ratio = cm / cu
        regime = "agree" if ratio < 1.5 else ("utility wins" if ratio < 20 else "utility wins BIG")
        print(f"{rho:>6.1f}{cm:>14.1f}{cu:>14.1f}{ratio:>10.2f}  {regime}")


def baseline_table(rho, K):
    print(f"\nBaselines at rho={rho}, K={K} (mean over seeds):")
    rules = {
        "no-library":     lambda f, h: set(),
        "accumulate-all": lambda f, h: set(range(N)),
        "frequency-topK": freq_threshold_lib,
        "MDL-keep":       mdl_lib,
        "utility-keep":   util_lib,
    }
    print(f"{'rule':<16}{'|L|':>5}{'held-out cost':>16}")
    print("-" * 40)
    for name, rule in rules.items():
        costs, sizes = [], []
        for s in SEEDS:
            f, h = population(rho, s)
            L = rule(f, h) if name in ("no-library", "accumulate-all") else rule(f, h, K)
            costs.append(ecost(f, h, L)); sizes.append(len(L))
        print(f"{name:<16}{int(np.mean(sizes)):>5}{np.mean(costs):>16.1f}")


# ---------------------------------------------------------------- phase diagram (PNG)
def phase_diagram(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rhos = np.linspace(-1.0, 1.0, 41)
    Ks = np.arange(1, N + 1)
    grid = np.zeros((len(Ks), len(rhos)))
    for j, rho in enumerate(rhos):
        pops = [population(rho, s) for s in SEEDS]
        for i, K in enumerate(Ks):
            vals = [np.log10(ecost(f, h, mdl_lib(f, h, K)) / ecost(f, h, util_lib(f, h, K)))
                    for (f, h) in pops]
            grid[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="magma",
                   extent=[rhos[0], rhos[-1], Ks[0], Ks[-1]])
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("log₁₀(MDL cost / utility cost)   — bright = utility wins")
    ax.set_xlabel("frequency–hardness correlation  ρ")
    ax.set_ylabel("library budget  K")
    ax.set_title("When compression is enough, and when you must price utility")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"\nwrote {path}")
    # headline numbers
    print(f"max utility advantage on grid: {10**grid.max():.0f}x "
          f"(at ρ={rhos[grid.max(0).argmax()]:.2f}); "
          f"median over grid: {10**np.median(grid):.2f}x")


if __name__ == "__main__":
    K = N // 3
    baseline_table(rho=-0.6, K=K)
    slice_table(K)
    phase_diagram("figures/phase_diagram.png")
