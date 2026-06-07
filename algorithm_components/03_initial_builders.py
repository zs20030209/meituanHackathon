# Auto-generated algorithm component library.
# These files are LLM composition material for autosolver_agent.py.
# They are not the contest submission entrypoint and are intentionally decoupled from the submit file at agent runtime.

import math
import random
import time
import itertools
import heapq


TL = 9.65
EPS = 1e-9
PEN = 100.0
DENSE30 = 16.75

# ---- component: greedy ----
def greedy(gs, cs, mode, ddl, rng=None):
    def key(c):
        g = gs[c.g]
        sc = single(g, c)
        if mode == 0:
            k = (sc / g.n, c.s, -c.p)
        elif mode == 1:
            k = (c.s, -c.p, -g.n)
        elif mode == 2:
            k = (-g.n, sc / g.n, c.s)
        elif mode == 3:
            k = (-c.p, sc / g.n, c.s)
        else:
            k = (c.s / max(0.03, c.p) / g.n, c.s)
        return k if rng is None else k + (rng.random() * 0.25,)
    pool = sorted(cs[:min(len(cs), 14000)] if rng else cs, key=key)
    sel, um, ur = {}, 0, set()
    for i, c in enumerate(pool):
        if i & 511 == 0 and time.perf_counter() >= ddl:
            break
        g = gs[c.g]
        if (g.mask & um) or c.r in ur:
            continue
        sel[c.g] = [c.r]
        um |= g.mask
        ur.add(c.r)
    return sel

# ---- component: exact_cover_seed ----
def exact_cover_seed(gs, rids, ddl, max_cfg=14, maxr=3, bit_lim=70, node_lim=140000):
    full = 0
    for g in gs:
        full |= g.mask
    nt = pc(full)
    if nt == 0 or nt > 15 or time.perf_counter() >= ddl:
        return {}
    bybit = {}
    for gi, g in enumerate(gs):
        if (gi & 31) == 0 and time.perf_counter() >= ddl:
            break
        pool, seen = [], set()
        lim = 10 if nt <= 8 else 7
        lists = (
            g.cs[:lim],
            sorted(g.cs, key=lambda c: (single(g, c), c.s, -c.p))[:lim],
            sorted(g.cs, key=lambda c: (-c.p, c.s))[:max(5, lim // 2)],
        )
        for arr in lists:
            for c in arr:
                if c.r not in seen:
                    seen.add(c.r)
                    pool.append(c)
        cfgs, seen_rs = [], set()
        for k in range(1, min(maxr, len(pool)) + 1):
            for comb in itertools.combinations(pool[:lim], k):
                rs = tuple(sorted(c.r for c in comb))
                if rs in seen_rs:
                    continue
                seen_rs.add(rs)
                rb = 0
                for r in rs:
                    rb |= 1 << r
                cfgs.append((gcost_cands(g, comb), rs, rb, gi, g.mask))
        cfgs.sort(key=lambda x: (x[0] / g.n, len(x[1]), x[0]))
        for cfg in cfgs[:max_cfg]:
            mm, b = g.mask, 0
            while mm:
                if mm & 1:
                    bybit.setdefault(b, []).append(cfg)
                b += 1
                mm >>= 1
    for b in bybit:
        bybit[b].sort(key=lambda x: (x[0] / pc(x[4]), len(x[1]), x[0]))
        bybit[b] = bybit[b][:bit_lim]
    best_cov, best_cost, best_sel = -1, 1e100, None
    memo, nodes = {}, [0]

    def first(mask):
        return (mask & -mask).bit_length() - 1

    def save(chosen, cost, rem):
        nonlocal_best[0] = pc(full) - pc(rem)
        if nonlocal_best[0] > nonlocal_best[1] or (nonlocal_best[0] == nonlocal_best[1] and cost < nonlocal_best[2] - EPS):
            sel = {}
            for _, rs, _, gi, _ in chosen:
                sel[gi] = list(rs)
            nonlocal_best[1], nonlocal_best[2], nonlocal_best[3] = nonlocal_best[0], cost, sel

    nonlocal_best = [0, best_cov, best_cost, best_sel]

    def dfs(rem, used, cost, chosen):
        if time.perf_counter() >= ddl or nodes[0] >= node_lim:
            return
        nodes[0] += 1
        save(chosen, cost, rem)
        if rem == 0:
            return
        if nonlocal_best[1] == nt and cost >= nonlocal_best[2] - EPS:
            return
        key = (rem, used)
        if memo.get(key, 1e100) <= cost + EPS:
            return
        memo[key] = cost
        b = first(rem)
        tried = False
        for cfg in bybit.get(b, ()):
            cst, rs, rb, gi, gm = cfg
            if gm & ~rem or rb & used:
                continue
            tried = True
            chosen.append(cfg)
            dfs(rem ^ gm, used | rb, cost + cst, chosen)
            chosen.pop()
        if not tried or nt <= 10:
            dfs(rem ^ (1 << b), used, cost, chosen)

    dfs(full, 0, 0.0, [])
    return nonlocal_best[3] if nonlocal_best[3] is not None else {}

# ---- component: singleton_seed ----
def singleton_seed(gs, ddl):
    singles = [(i, g) for i, g in enumerate(gs) if g.n == 1]
    if not singles:
        return {}
    used, sel = set(), {}
    singles.sort(key=lambda x: (single(x[1], x[1].best), x[1].key))
    for gi, g in singles:
        for c in g.cs:
            if c.r not in used:
                sel[gi] = [c.r]
                used.add(c.r)
                break
    if len(sel) != len(singles):
        return sel
    if time.perf_counter() < ddl:
        sel = expand(gs, sel, ddl, 9, 1500, 200)
    if time.perf_counter() < ddl:
        sel = reassign(gs, sel, ddl, 4, 180)
    if time.perf_counter() < ddl:
        sel = move_extra(gs, sel, ddl, 160)
    if time.perf_counter() < ddl:
        sel = swap_riders(gs, sel, ddl, 160)
    return sel

# ---- component: mcf_singleton_seed ----
def mcf_singleton_seed(gs, rids, ddl):
    groups = [i for i, g in enumerate(gs) if g.n == 1]
    if not groups or len(groups) > len(rids) or time.perf_counter() >= ddl:
        return {}
    sel = assign_first_mcf(gs, groups, len(rids), ddl)
    if len(sel) != len(groups):
        return {}
    if time.perf_counter() < ddl:
        sel = expand(gs, sel, ddl, 9, 900, 180)
    if time.perf_counter() < ddl:
        sel = reassign(gs, sel, ddl, 3, 160)
    if time.perf_counter() < ddl:
        sel = move_extra(gs, sel, ddl, 130)
    if time.perf_counter() < ddl:
        sel = swap_riders(gs, sel, ddl, 130)
    return sel

# ---- component: pair_plan_seed ----
def pair_plan_seed(gs, rids, k, ddl, cache, mcf=False, qoff=0, jitter=0.0, seed=0):
    tasks, sg, pg = task_maps(gs)
    nt = len(tasks)
    if nt == 0 or k < 0 or k > nt // 2:
        return {}
    rows = max(1, nt - k)
    q = int(float(len(rids)) / rows + 0.5)
    q = max(1, min(10, q + qoff))
    sc = []
    for t in tasks:
        sc.append(approx_k_cost(gs, sg[t], q, cache))
    weights, edges = {}, []
    rng = random.Random(seed) if jitter > 0.0 else None
    for (a, b), gi in pg.items():
        w = sc[a] + sc[b] - approx_k_cost(gs, gi, q, cache)
        weights[(a, b)] = w
        edges.append((w + (rng.random() * jitter if rng else 0.0), a, b))
    pairs = choose_pairs(edges, weights, pg, nt, k)
    paired, groups = set(), []
    for a, b, _ in pairs:
        gi = pg.get((a, b))
        if gi is not None:
            paired.add(a)
            paired.add(b)
            groups.append(gi)
    for i, t in enumerate(tasks):
        if i not in paired:
            groups.append(sg[t])
    sel = assign_first_mcf(gs, groups, len(rids), ddl) if mcf else {}
    if len(sel) != len(groups):
        used, sel = set(), {}
        groups.sort(key=lambda gi: (single(gs[gi], gs[gi].best) / gs[gi].n, gs[gi].key))
        for gi in groups:
            for c in gs[gi].cs:
                if c.r not in used:
                    used.add(c.r)
                    sel[gi] = [c.r]
                    break
    if time.perf_counter() < ddl:
        sel = expand(gs, sel, ddl, 10, 1200, 180)
    if time.perf_counter() < ddl:
        sel = reassign(gs, sel, ddl, 3, 160)
    if time.perf_counter() < ddl:
        sel = move_extra(gs, sel, ddl, 180)
    if time.perf_counter() < ddl:
        sel = swap_riders(gs, sel, ddl, 180)
    return sel

# ---- component: stochastic_pair_plan_seed ----
def stochastic_pair_plan_seed(gs, rids, k, ddl, cache, qoff=0, tries=24, evals=4, jitter=18.0, seed=0):
    tasks, sg, pg = task_maps(gs)
    nt = len(tasks)
    if nt == 0 or k < 0 or k > nt // 2 or time.perf_counter() >= ddl:
        return {}
    rows = max(1, nt - k)
    q = int(float(len(rids)) / rows + 0.5)
    q = max(1, min(10, q + qoff))
    sc = []
    for t in tasks:
        sc.append(approx_k_cost(gs, sg[t], q, cache))
    weights, base_edges = {}, []
    for (a, b), gi in pg.items():
        w = sc[a] + sc[b] - approx_k_cost(gs, gi, q, cache)
        weights[(a, b)] = w
        base_edges.append((w, a, b))
    if not base_edges:
        return {}
    rng = random.Random(seed + nt * 97 + k * 17 + qoff * 31)
    seen, plans = set(), []
    for it in range(tries):
        if time.perf_counter() >= ddl:
            break
        if it == 0:
            edges = base_edges
        else:
            amp = jitter * (0.35 + 0.08 * (it % 7))
            edges = [(w + (rng.random() - rng.random()) * amp, a, b) for w, a, b in base_edges]
        pairs = choose_pairs(edges, weights, pg, nt, k)
        if not pairs:
            continue
        key = tuple(sorted((min(a, b), max(a, b)) for a, b, _ in pairs))
        if key in seen:
            continue
        seen.add(key)
        score = sum(weights.get((min(a, b), max(a, b)), -1e80) for a, b, _ in pairs)
        plans.append((score, key))
    if not plans:
        return {}
    plans.sort(reverse=True)
    best, bv = None, None
    for _, key in plans[:evals]:
        if time.perf_counter() >= ddl:
            break
        paired, groups = set(), []
        for a, b in key:
            gi = pg.get((a, b))
            if gi is not None:
                paired.add(a)
                paired.add(b)
                groups.append(gi)
        for i, t in enumerate(tasks):
            if i not in paired:
                groups.append(sg[t])
        if len(groups) > len(rids):
            continue
        sel = assign_first_mcf(gs, groups, len(rids), ddl)
        if len(sel) != len(groups):
            used, sel = set(), {}
            groups.sort(key=lambda gi: (single(gs[gi], gs[gi].best) / gs[gi].n, gs[gi].key))
            for gi in groups:
                for c in gs[gi].cs:
                    if c.r not in used:
                        used.add(c.r)
                        sel[gi] = [c.r]
                        break
        if time.perf_counter() < ddl:
            sel = expand(gs, sel, ddl, 10, 900, 170)
        if time.perf_counter() < ddl:
            sel = reassign(gs, sel, ddl, 3, 150)
        if time.perf_counter() < ddl:
            sel = move_extra(gs, sel, ddl, 130)
        if time.perf_counter() < ddl:
            sel = swap_riders(gs, sel, ddl, 130)
        v = value(gs, sel)
        if better(v, bv):
            best, bv = sel, v
    return best if best is not None else {}

# ---- component: cpsat_offer_seed ----
def cpsat_offer_seed(gs, rids, ddl):
    if time.perf_counter() >= ddl:
        return {}
    try:
        from ortools.sat.python import cp_model
    except Exception:
        return {}
    offers = []
    nt = task_count(gs)
    rider_count = len(rids)
    rich = rider_count >= int(nt * 1.45)
    for gi, g in enumerate(gs):
        if (gi & 31) == 0 and time.perf_counter() >= ddl:
            break
        if g.n > 4:
            continue
        pool, seen = [], set()
        lim = 8 if rich else 7
        lists = (
            g.cs[:lim],
            sorted(g.cs, key=lambda c: (single(g, c), c.s, -c.p))[:lim],
            sorted(g.cs, key=lambda c: (-c.p, c.s))[:max(5, lim - 2)],
        )
        for arr in lists:
            for c in arr:
                if c.r not in seen:
                    seen.add(c.r)
                    pool.append(c)
        if not pool:
            continue
        maxr = 3 if rich else 2
        cfg, seen_rs = [], set()
        for k in range(1, min(maxr, len(pool)) + 1):
            for comb in itertools.combinations(pool[:lim], k):
                rs = tuple(sorted(c.r for c in comb))
                if rs in seen_rs:
                    continue
                seen_rs.add(rs)
                cfg.append((gcost_cands(g, comb), rs))
        cfg.sort(key=lambda x: (x[0] / g.n, len(x[1]), x[0]))
        cap = 4 if g.n == 1 and rich else 3
        for cst, rs in cfg[:cap]:
            offers.append((gi, g.mask, rs, cst, g.n))
    if not offers or time.perf_counter() >= ddl:
        return {}
    model = cp_model.CpModel()
    xs = [model.NewBoolVar("o%d" % i) for i in range(len(offers))]
    for b in range(nt):
        arr = [xs[i] for i, of in enumerate(offers) if of[1] & (1 << b)]
        if arr:
            model.Add(sum(arr) <= 1)
    for r in range(rider_count):
        arr = [xs[i] for i, of in enumerate(offers) if r in of[2]]
        if arr:
            model.Add(sum(arr) <= 1)
    big, scale = 100000000, 1000.0
    model.Maximize(sum(xs[i] * (offers[i][4] * big - int(offers[i][3] * scale + 0.5)) for i in range(len(offers))))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.05, ddl - time.perf_counter() - 0.03)
    solver.parameters.num_search_workers = 1
    try:
        status = solver.Solve(model)
    except Exception:
        return {}
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {}
    sel = {}
    for i, (gi, _, rs, _, _) in enumerate(offers):
        if solver.Value(xs[i]):
            sel[gi] = list(rs)
    return sel

# ---- component: bundle_plan_seed ----
def bundle_plan_seed(gs, rids, target_rows, ddl, cache, qoff=0, jitter=0.0, seed=0):
    tasks, sg, _ = task_maps(gs)
    nt = len(tasks)
    if nt == 0 or time.perf_counter() >= ddl:
        return {}
    rows = max(1, min(target_rows, nt))
    q = int(float(len(rids)) / rows + 0.5)
    q = max(1, min(10, q + qoff))
    sc = {}
    for t in tasks:
        sc[t] = approx_k_cost(gs, sg[t], q, cache)
    rng = random.Random(seed) if jitter > 0.0 else None
    cand = []
    for gi, g in enumerate(gs):
        if (gi & 63) == 0 and time.perf_counter() >= ddl:
            break
        if g.n <= 1 or g.n > 4:
            continue
        base, ok = 0.0, True
        for t in g.tasks:
            if t not in sg:
                ok = False
                break
            base += sc[t]
        if not ok:
            continue
        sav = base - approx_k_cost(gs, gi, q, cache)
        noise = rng.random() * jitter if rng else 0.0
        cand.append((sav / max(1, g.n - 1) + noise, sav + noise, g.n, gi))
    cand.sort(reverse=True)
    chosen, used_mask, cur_rows = [], 0, nt
    for _, sav, _, gi in cand:
        if time.perf_counter() >= ddl:
            break
        g = gs[gi]
        if g.mask & used_mask:
            continue
        if cur_rows <= rows and sav <= 0.0:
            continue
        chosen.append(gi)
        used_mask |= g.mask
        cur_rows -= g.n - 1
        if cur_rows <= rows and sav <= 0.0:
            break
    groups = list(chosen)
    for t in tasks:
        gi = sg[t]
        if not (gs[gi].mask & used_mask):
            groups.append(gi)
    if len(groups) > len(rids):
        return {}
    sel = assign_first_mcf(gs, groups, len(rids), ddl)
    if len(sel) != len(groups):
        used, sel = set(), {}
        groups.sort(key=lambda gi: (single(gs[gi], gs[gi].best) / gs[gi].n, gs[gi].key))
        for gi in groups:
            for c in gs[gi].cs:
                if c.r not in used:
                    used.add(c.r)
                    sel[gi] = [c.r]
                    break
    if time.perf_counter() < ddl:
        sel = expand(gs, sel, ddl, 8, 650, 150)
    if time.perf_counter() < ddl:
        sel = reassign(gs, sel, ddl, 2, 140)
    if time.perf_counter() < ddl:
        sel = move_extra(gs, sel, ddl, 120)
    if time.perf_counter() < ddl:
        sel = swap_riders(gs, sel, ddl, 120)
    return sel
