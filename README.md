# mine-then-keep

Controlled experiments for the position/survey *"Mine, Then Keep: Acquiring Reusable
Abstractions for World-Model Planning."* Agents that plan with a world model learn an
abstraction library to plan in coarse moves; the hard question is the **keep-criterion** —
which proposed abstractions to retain. This repo isolates that decision.

## The question

When is **compression (MDL)** enough as a keep-rule, and when must you price **planning
utility** directly (Minton's macro-operator utility problem)? The two are usually treated
as separate literatures. We make them comparable on a controlled domain with *independent
knobs*.

## Design (`experiments/phase_diagram.py`)

Each candidate skill abstracts one task segment and has two **independent** properties:

- **frequency** `f` — fraction of tasks needing that segment;
- **hardness** `h` — blind-search cost of the segment, `B**h` model-rollouts if not abstracted.

All segments share a description length, so an **MDL** keep-rule's gain is proportional to
`f` *alone* — compression is blind to hardness by construction. A **utility** keep-rule
scores `f · B**h`. Under a carrying-cost budget `K`, MDL keeps the `K` most frequent skills;
utility keeps the `K` highest-`f·B**h`. Expected held-out planning cost is exact:

```
E[cost | L] = Σ_i  f_i · ( (B+|L|)   if i∈L     # cheap lookup, grows with library size
                            B**h_i    otherwise )  # blind search of the segment
```

We sweep `ρ`, the correlation between frequency and hardness, and the budget `K`.

## Result

Keep-pressure is the dominant effect, and the choice of keep-rule is conditional:

| rule | held-out cost (ρ=−0.6, K=10) |
|---|---|
| no-library | 10356 |
| accumulate-all (|L|=30) | 265 |
| frequency / MDL keep | 8739 |
| **utility keep** | **831** |

Utility beats MDL by **2.5×–11×**, the advantage largest when frequency and hardness are
uncorrelated and collapsing toward parity as they correlate (the agreement regime). MDL is a
good keep-rule wherever statistical regularity tracks search value; it fails precisely on the
**rare-but-critical** abstraction that compression cannot see. The phase boundary is
`figures/phase_diagram.png`.

## Run

```
uv run python experiments/phase_diagram.py
```

## Status / roadmap

- [x] controlled phase diagram (MDL vs utility, independent knobs) — the mechanism result
- [x] Blocksworld external validity (`experiments/blocksworld.py`) — macros mined from solved
  plans, planning cost = search nodes. The two keep-rules nearly coincide and the gap widens
  with difficulty: utility/MDL = 1.08× (5 blocks), 1.18× (6), 1.37× (7). Standard planning sits
  in the agreement corner and drifts toward divergence as the domain hardens.
- [ ] Logistics + larger Blocksworld — trace the full agreement-to-divergence drift
- [ ] ARC-AGI-3 within-game cross-level reuse — the live world-model agent demonstration

## License

Copyright (C) 2026 June Kim. Licensed under the [GNU Affero General Public License v3.0 or later](LICENSE) (`AGPL-3.0-or-later`), a strong copyleft license: anyone who runs a modified version, including over a network, must make the corresponding source available under the same terms.
