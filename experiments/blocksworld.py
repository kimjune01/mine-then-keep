"""
External validity for the keep-criterion result, in Blocksworld (HTN-learning's home
turf). Same ablation as the synthetic phase diagram, but the abstractions are real
learned macro-operators and "planning cost" is real search-node expansions.

Pipeline:
  1. Solve a training set of random Blocksworld tasks with greedy best-first search
     (move(block, dest) formulation), recording the optimal-ish plans.
  2. Mine LIFTED macro candidates: frequent windows of consecutive moves, with blocks
     replaced by variables in first-occurrence order ("table" kept literal). The same
     candidate stream feeds every keep-rule.
  3. Score candidates two ways:
       MDL gain   ~ support * (window_len - 1)        (compression of the plan corpus)
       utility    ~ mean search-node reduction per held-out task when the macro is added
  4. Keep top-K under each rule; replan held-out tasks with macro-augmented search;
     report mean nodes expanded.

The question is whether the synthetic phenomenon transfers: does a compression keep-rule
miss macros that cut search but do not recur often, the way it did in the toy?
"""
from __future__ import annotations
import heapq, itertools, random
from collections import Counter

T = "T"  # the table


# ---------------------------------------------------------------- Blocksworld core
def clear(state, b):
    return b != T and all(on != b for _, on in state)

def positions(state):
    return {blk: on for blk, on in state}

def primitive_moves(state, blocks):
    pos = positions(state)
    movers = [b for b in blocks if clear(state, b)]
    dests = [T] + [b for b in blocks if clear(state, b)]
    for b in movers:
        for d in dests:
            if d != b and pos[b] != d:
                yield ("move", b, d)

def apply_move(state, mv):
    _, b, d = mv
    return frozenset((blk, (d if blk == b else on)) for blk, on in state)

def h(state, goal):  # blocks not yet on their goal support
    g = dict(goal)
    return sum(1 for blk, on in state if blk in g and g[blk] != on)

def gbfs(start, goal, blocks, macros=(), node_cap=20000):
    """Greedy best-first search; returns (plan, nodes_expanded) or (None, nodes)."""
    goalset = frozenset(goal.items())
    start_h = h(start, goal)
    pq = [(start_h, 0, start, [])]
    seen = {start}
    nodes = 0
    while pq and nodes < node_cap:
        _, g, state, plan = heapq.heappop(pq)
        nodes += 1
        if goalset <= state:
            return plan, nodes
        # primitive successors
        succ = [(apply_move(state, mv), [mv]) for mv in primitive_moves(state, blocks)]
        # macro successors (one super-step each)
        for mac in macros:
            for ns, seq in apply_macro(state, mac, blocks):
                succ.append((ns, seq))
        for ns, seq in succ:
            if ns not in seen:
                seen.add(ns)
                heapq.heappush(pq, (h(ns, goal), g + len(seq), ns, plan + seq))
    return None, nodes


# ---------------------------------------------------------------- lifted macros
def lift(window):
    """Replace concrete blocks by variables in first-occurrence order; keep T."""
    m, out = {}, []
    for (_, b, d) in window:
        for x in (b, d):
            if x != T and x not in m:
                m[x] = f"v{len(m)}"
        out.append(("move", m[b], m[d] if d != T else T))
    return tuple(out)

def apply_macro(state, macro, blocks):
    """Yield (resulting_state, ground_sequence) for every valid binding of the lifted
    macro's variables to distinct blocks, executable in order from `state`."""
    vars_ = []
    for (_, b, d) in macro:
        for x in (b, d):
            if x != T and x not in vars_:
                vars_.append(x)
    for perm in itertools.permutations(blocks, len(vars_)):
        bind = dict(zip(vars_, perm))
        s, seq, ok = state, [], True
        for (_, b, d) in macro:
            gb = bind[b]
            gd = T if d == T else bind[d]
            mv = ("move", gb, gd)
            if not clear(s, gb) or (gd != T and not clear(s, gd)) or positions(s)[gb] == gd:
                ok = False; break
            s = apply_move(s, mv); seq.append(mv)
        if ok and len(seq) == len(macro):
            yield s, seq


# ---------------------------------------------------------------- tasks + corpus
def random_state(blocks, rng):
    order = blocks[:]; rng.shuffle(order)
    state, tops = [], {}
    for b in order:
        # put b on table or on a current top, forming towers
        choices = [T] + [t for t in tops]
        d = rng.choice(choices)
        state.append((b, d))
        if d != T: tops.pop(d, None)
        tops[b] = True
    return frozenset(state)

def random_task(blocks, rng):
    return random_state(blocks, rng), dict(random_state(blocks, rng))

def mine_macros(plans, lo=2, hi=3, min_support=4):
    cnt = Counter()
    for plan in plans:
        seen = set()
        for L in range(lo, hi + 1):
            for i in range(len(plan) - L + 1):
                seen.add(lift(plan[i:i + L]))
        for m in seen:
            cnt[m] += 1
    return {m: c for m, c in cnt.items() if c >= min_support}


# ---------------------------------------------------------------- keep rules
def mdl_score(macro, support):       # compression of plan corpus
    return support * (len(macro) - 1)

def keep_mdl(cands, K):
    return [m for m, _ in sorted(cands.items(), key=lambda kv: -mdl_score(kv[0], kv[1]))[:K]]

def keep_frequency(cands, K):
    return [m for m, _ in sorted(cands.items(), key=lambda kv: -kv[1])[:K]]

def keep_utility(cands, K, train, blocks):
    """Greedy: repeatedly add the macro with the largest mean node-reduction on train."""
    chosen, pool = [], dict(cands)
    base = [gbfs(s, g, blocks, macros=chosen)[1] for s, g in train]
    for _ in range(K):
        best, best_gain = None, 0
        for m in pool:
            red = []
            for (s, g), b0 in zip(train, base):
                n = gbfs(s, g, blocks, macros=chosen + [m])[1]
                red.append(b0 - n)
            gain = sum(red) / len(red)
            if gain > best_gain:
                best, best_gain, best_red = m, gain, red
        if best is None: break
        chosen.append(best); pool.pop(best)
        base = [b0 - r for b0, r in zip(base, best_red)]
    return chosen


# ---------------------------------------------------------------- experiment
def run(nblocks=5, n_train=60, n_test=40, K=6, seed=1):
    rng = random.Random(seed)
    blocks = [chr(ord("a") + i) for i in range(nblocks)]
    train_tasks, train_plans = [], []
    while len(train_tasks) < n_train:
        s, g = random_task(blocks, rng)
        plan, _ = gbfs(s, g, blocks)
        if plan:
            train_tasks.append((s, g)); train_plans.append(plan)
    test_tasks = []
    while len(test_tasks) < n_test:
        s, g = random_task(blocks, rng)
        if gbfs(s, g, blocks)[0]:
            test_tasks.append((s, g))

    cands = mine_macros(train_plans)
    libs = {
        "no-library":     [],
        "accumulate-all": list(cands),
        "frequency-topK": keep_frequency(cands, K),
        "MDL-keep":       keep_mdl(cands, K),
        "utility-keep":   keep_utility(cands, K, train_tasks, blocks),
    }
    print(f"blocks={nblocks} train={n_train} test={n_test} K={K} "
          f"candidates_mined={len(cands)}\n")
    print(f"{'keep rule':<16}{'|L|':>5}{'mean nodes (held-out)':>24}")
    print("-" * 46)
    rows = {}
    for name, L in libs.items():
        nodes = [gbfs(s, g, blocks, macros=L)[1] for s, g in test_tasks]
        rows[name] = sum(nodes) / len(nodes)
        print(f"{name:<16}{len(L):>5}{rows[name]:>24.1f}")
    print("-" * 46)
    mdl, util = rows["MDL-keep"], rows["utility-keep"]
    print(f"utility vs MDL on held-out nodes: {mdl/util:.2f}x "
          f"({'utility cheaper' if util < mdl else 'MDL cheaper'})")
    return rows


def selftest():
    blocks = ["a", "b", "c"]
    start = frozenset([("a", T), ("b", T), ("c", "a")])      # Sussman anomaly
    goal = {"a": "b", "b": "c"}
    plan, nodes = gbfs(start, goal, blocks)
    assert plan and h(apply_plan(start, plan), goal) == 0, "planner failed Sussman"
    print(f"selftest ok: Sussman solved in {len(plan)} moves, {nodes} nodes expanded")

def apply_plan(state, plan):
    for mv in plan: state = apply_move(state, mv)
    return state


if __name__ == "__main__":
    selftest()
    print()
    for nb in (5, 6, 7):
        run(nblocks=nb, n_train=50, n_test=40, K=8)
        print()
