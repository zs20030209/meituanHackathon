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

# ---- component: anneal ----
def anneal(gs, sel, ddl, rng):
    sel = copy(sel)
    items = [g for g in sel if sel[g]]
    if len(items) < 2:
        return sel
    cost = {g: gcost(gs, g, sel[g]) for g in items}
    cur = sum(cost.values())
    best, bv = copy(sel), cur
    st = time.perf_counter()
    span = max(1e-6, ddl - st)
    steps = 0
    while time.perf_counter() < ddl:
        steps += 1
        t = 0.01 + 1.8 * max(0.0, (ddl - time.perf_counter()) / span)
        if rng.random() < 0.55:
            a, b = rng.sample(items, 2)
            if len(sel[a]) <= 1:
                continue
            r = rng.choice(sel[a])
            if r in sel[b] or r not in gs[b].by:
                continue
            ra = list(sel[a]); ra.remove(r)
            rb = sel[b] + [r]
            na, nb = gcost(gs, a, ra), gcost(gs, b, rb)
            d = na + nb - cost[a] - cost[b]
            if d < -EPS or rng.random() < math.exp(-d / t):
                sel[a], sel[b], cost[a], cost[b], cur = ra, rb, na, nb, cur + d
        else:
            a, b = rng.sample(items, 2)
            ra, rb = rng.choice(sel[a]), rng.choice(sel[b])
            if ra == rb or ra not in gs[b].by or rb not in gs[a].by:
                continue
            ta = [rb if x == ra else x for x in sel[a]]
            tb = [ra if x == rb else x for x in sel[b]]
            if len(set(ta)) < len(ta) or len(set(tb)) < len(tb):
                continue
            na, nb = gcost(gs, a, ta), gcost(gs, b, tb)
            d = na + nb - cost[a] - cost[b]
            if d < -EPS or rng.random() < math.exp(-d / t):
                sel[a], sel[b], cost[a], cost[b], cur = ta, tb, na, nb, cur + d
        if cur < bv - EPS:
            best, bv = copy(sel), cur
    return best

# ---- component: anneal_steps ----
def anneal_steps(gs, sel, ddl, rng, steps):
    sel = copy(sel)
    items = [g for g in sel if sel[g]]
    if len(items) < 2:
        return sel
    cost = {g: gcost(gs, g, sel[g]) for g in items}
    cur = sum(cost.values())
    best, bv = copy(sel), cur
    for step in range(steps):
        if (step & 2047) == 0 and time.perf_counter() >= ddl:
            break
        t = 0.006 + 1.65 * (1.0 - float(step) / max(1, steps))
        if rng.random() < 0.55:
            a, b = rng.sample(items, 2)
            if len(sel[a]) <= 1:
                continue
            r = rng.choice(sel[a])
            if r in sel[b] or r not in gs[b].by:
                continue
            ra = list(sel[a])
            ra.remove(r)
            rb = sel[b] + [r]
            na, nb = gcost(gs, a, ra), gcost(gs, b, rb)
            d = na + nb - cost[a] - cost[b]
            if d < -EPS or rng.random() < math.exp(-d / t):
                sel[a], sel[b], cost[a], cost[b], cur = ra, rb, na, nb, cur + d
        else:
            a, b = rng.sample(items, 2)
            ra, rb = rng.choice(sel[a]), rng.choice(sel[b])
            if ra == rb or ra not in gs[b].by or rb not in gs[a].by:
                continue
            ta = [rb if x == ra else x for x in sel[a]]
            tb = [ra if x == rb else x for x in sel[b]]
            if len(set(ta)) < len(ta) or len(set(tb)) < len(tb):
                continue
            na, nb = gcost(gs, a, ta), gcost(gs, b, tb)
            d = na + nb - cost[a] - cost[b]
            if d < -EPS or rng.random() < math.exp(-d / t):
                sel[a], sel[b], cost[a], cost[b], cur = ta, tb, na, nb, cur + d
        if cur < bv - EPS:
            best, bv = copy(sel), cur
    return best

# ---- component: generated_style_seed ----
def generated_style_seed(gs, rids, cs, ddl):
    nt, nr = task_count(gs), len(rids)
    if nr < int(nt * 1.45):
        return {}
    avgp = sum(c.p for c in cs) / max(1, len(cs))
    if nr <= int(nt * 0.75) and nt >= 20 and avgp >= 0.16:
        return {}
    best_box = [None, None]

    def keep(s):
        if not s:
            return
        v = value(gs, s)
        if better(v, best_box[1]):
            best_box[0], best_box[1] = copy(s), v

    if time.perf_counter() < ddl:
        keep(greedy(gs, cs, 0, ddl))
    if nt >= 22 and time.perf_counter() < ddl:
        s = singleton_seed(gs, min(ddl, time.perf_counter() + 0.70))
        keep(s)
        sv = value(gs, s) if s else (0, 1e100)
        if nt < 35 and s and sv[1] / max(1, nt) > DENSE30 and time.perf_counter() < ddl:
            keep(mcf_singleton_seed(gs, rids, min(ddl, time.perf_counter() + 0.70)))
    if time.perf_counter() < ddl:
        cache = {}
        rem = ddl - time.perf_counter()
        soft = min(ddl, time.perf_counter() + min(1.55 if nt <= 15 else 2.65, max(0.55, rem * 0.72)))
        for k in range(nt // 2, -1, -1):
            if time.perf_counter() >= soft:
                break
            keep(pair_plan_seed(gs, rids, k, soft, cache, True))
        if time.perf_counter() < soft:
            for k in (nt // 2, max(0, nt // 2 - 1), 0):
                if time.perf_counter() >= soft:
                    break
                keep(stochastic_pair_plan_seed(gs, rids, k, soft, cache, 0, 20, 4, 22.0, 5100 + k))
    if best_box[0] and nt >= 22 and time.perf_counter() < ddl and ddl - time.perf_counter() > 0.85:
        keep(cpsat_offer_seed(gs, rids, min(ddl, time.perf_counter() + min(1.2, (ddl - time.perf_counter()) * 0.45))))
    if best_box[0] and time.perf_counter() < ddl:
        s = best_box[0]
        s = cycle_riders(gs, s, ddl, 50)
        keep(s)
        if time.perf_counter() < ddl:
            s = path_riders(gs, s, ddl, 40)
            keep(s)
        if time.perf_counter() < ddl:
            s = config_polish(gs, s, min(ddl, time.perf_counter() + 0.25), 7, 2)
            keep(s)
        if time.perf_counter() < ddl:
            s = pair_kick(gs, s, ddl, 14, 2)
            keep(s)
    return best_box[0] if best_box[0] is not None else {}

# ---- component: search ----
def search(gs, rids, cs, ddl):
    rng = random.Random(20260604 + len(gs) * 19 + len(cs))
    best_box = [None, None]

    def keep(s):
        if not s:
            return
        v = value(gs, s)
        if better(v, best_box[1]):
            best_box[0], best_box[1] = copy(s), v

    nt, nr = task_count(gs), len(rids)
    avgp = sum(c.p for c in cs) / max(1, len(cs))
    start = time.perf_counter()

    sgp = []
    for g in gs:
        if g.n == 1 and g.cs:
            sgp.append(max(c.p for c in g.cs))
    best1 = sum(sgp) / len(sgp) if sgp else 1.0
    low_case = avgp < 0.22 or best1 < 0.55

    if nt <= 15 and time.perf_counter() < ddl:
        slice_end = min(ddl, time.perf_counter() + (1.45 if nt <= 8 else 1.55))
        keep(exact_cover_seed(gs, rids, slice_end, 32 if nt <= 8 else 18, 4 if nt <= 8 else 3, 220 if nt <= 8 else 90, 450000 if nt <= 8 else 260000))

    gen_allowed = (9 <= nt <= 15) and nr >= int(nt * 1.45)
    if gen_allowed and time.perf_counter() < ddl:
        cap = 1.45
        keep(generated_style_seed(gs, rids, cs, min(ddl, time.perf_counter() + cap)))

    if low_case and nt >= 10 and time.perf_counter() < ddl:
        est = int(nt - float(nr) / 4.35 + 0.5)
        ks = []
        span = 6 if nt <= 32 else 5
        for d in [0] + [x for y in range(1, span + 1) for x in (-y, y)]:
            k = est + d
            if 0 <= k <= nt // 2 and k not in ks:
                ks.append(k)
        for k in (nt // 2, max(0, nt // 2 - 1), max(0, nt // 2 - 2), 0):
            if k not in ks:
                ks.append(k)
        cache = {}
        soft = min(ddl, start + min(6.55, max(1.8, (ddl - start) * 0.72)))
        plan_best, plan_val = None, None
        anchors = (
            (nt // 2, -1, 1951, 5.0),
            (nt // 2, -1, 2301, 5.0),
            (est, -1, 1951, 5.0),
            (max(0, nt // 2 - 1), -1, 1951, 5.0),
            (nt // 2, 0, 31, 12.0),
            (nt // 2, 2, 43, 10.0),
            (est, 2, 59, 14.0),
            (est + 1, -2, 1951, 5.0),
            (est + 2, -2, 1651, 8.0),
        )
        for k, qo, seed, jit in anchors:
            if time.perf_counter() >= soft:
                break
            if 0 <= k <= nt // 2:
                s = pair_plan_seed(gs, rids, k, soft, cache, True, qo, jit, seed)
                keep(s)
                v = value(gs, s)
                if better(v, plan_val):
                    plan_best, plan_val = copy(s), v
        for k in ks:
            if time.perf_counter() >= soft:
                break
            s = pair_plan_seed(gs, rids, k, soft, cache, True)
            keep(s)
            v = value(gs, s)
            if better(v, plan_val):
                plan_best, plan_val = copy(s), v
            if time.perf_counter() < soft and k in (est, nt // 2, max(0, nt // 2 - 1)):
                for qo in (-2, -1, 1):
                    if time.perf_counter() >= soft:
                        break
                    s = pair_plan_seed(gs, rids, k, soft, cache, True, qo)
                    keep(s)
                    v = value(gs, s)
                    if better(v, plan_val):
                        plan_best, plan_val = copy(s), v
        for k in (est, est - 1, est + 1, est - 2, est + 2, nt // 2, max(0, nt // 2 - 1)):
            if time.perf_counter() >= soft:
                break
            if k < 0 or k > nt // 2:
                continue
            for seed, jit, qo in ((1, 5.0, -1), (2, 5.0, -1), (3, 10.0, 0), (4, 16.0, 0), (5, 24.0, 1), (6, 32.0, 0), (7, 18.0, 2), (8, 26.0, 2)):
                if time.perf_counter() >= soft:
                    break
                s = pair_plan_seed(gs, rids, k, soft, cache, True, qo, jit, seed + nt * 13 + k)
                keep(s)
                v = value(gs, s)
                if better(v, plan_val):
                    plan_best, plan_val = copy(s), v
        if time.perf_counter() < soft:
            for k, qo, sd in ((est, -1, 4201), (est + 1, -1, 4211), (nt // 2, 0, 4221), (max(0, nt // 2 - 1), 1, 4231)):
                if time.perf_counter() >= soft:
                    break
                if 0 <= k <= nt // 2:
                    s = stochastic_pair_plan_seed(gs, rids, k, soft, cache, qo, 30, 5, 26.0, sd)
                    keep(s)
                    v = value(gs, s)
                    if better(v, plan_val):
                        plan_best, plan_val = copy(s), v
        if plan_best and time.perf_counter() < ddl:
            aend = min(ddl, time.perf_counter() + min(1.85, max(0.18, (ddl - time.perf_counter()) * 0.30)))
            s = anneal(gs, plan_best, aend, random.Random(31 + nt * 7))
            keep(s)

    elif nr >= int(nt * 1.45) and nt >= 22 and time.perf_counter() < ddl:
        base_send = min(ddl, start + min(3.35, max(0.7, (ddl - start) * 0.46)))
        send = base_send
        s = singleton_seed(gs, min(send, time.perf_counter() + 0.8))
        keep(s)
        sm = None
        s_density = value(gs, s)[1] / max(1, nt) if s else 1e100
        if nt < 35 and s_density > DENSE30:
            send = min(ddl, start + min(4.05, max(0.8, (ddl - start) * 0.50)))
        if nt < 35 and s_density > DENSE30 and time.perf_counter() < send:
            sm = mcf_singleton_seed(gs, rids, min(send, time.perf_counter() + 0.72))
            if gain_over(gs, sm, s) > 0.35:
                keep(sm)
        base = sm if sm and gain_over(gs, sm, s) > 0.75 else s
        if base and time.perf_counter() < send:
            keep(anneal_steps(gs, base, send, random.Random(27), 100000))
            if time.perf_counter() < send:
                keep(anneal_steps(gs, base, send, random.Random(35), 180000))
            keep(anneal_steps(gs, base, send, random.Random(5), 220000))
            if time.perf_counter() < send:
                keep(anneal_steps(gs, base, send, random.Random(3), 300000))
            qend = min(send, time.perf_counter() + 1.25)
            if time.perf_counter() < qend:
                keep(anneal(gs, base, qend, random.Random(9)))
            steps = 650000 if nt >= 30 else 260000
            if time.perf_counter() < send:
                keep(anneal_steps(gs, base, send, random.Random(9), steps))
            if time.perf_counter() < send - 0.25:
                if sm and nt < 35:
                    aend = min(send, time.perf_counter() + 0.42)
                    keep(anneal_steps(gs, s, aend, random.Random(1), max(160000, steps // 3)))
                    if time.perf_counter() < send - 0.12:
                        keep(anneal_steps(gs, base, send, random.Random(101), max(140000, steps // 3)))
                else:
                    keep(anneal_steps(gs, base, send, random.Random(1), max(180000, steps // 2)))

    elif nr < int(nt * 1.45) and time.perf_counter() < ddl and nt >= 10:
        est = int(nt - float(nr) / 1.17 + 0.5)
        lb = max(0, nt - nr)
        ks = []
        for base in (lb, est, nt // 2):
            for d in (0, 1, -1, 2, -2, 3, -3):
                k = base + d
                if 0 <= k <= nt // 2 and k not in ks:
                    ks.append(k)
        for k in range(lb, min(nt // 2, lb + 5) + 1):
            if 0 <= k <= nt // 2 and k not in ks:
                ks.append(k)
        cache = {}
        if nr <= int(nt * 0.85):
            soft = min(ddl, start + min(2.45, max(0.75, (ddl - start) * 0.26)))
        else:
            soft = min(ddl, start + min(1.55, max(0.45, (ddl - start) * 0.16)))
        for k in ks:
            if time.perf_counter() >= soft:
                break
            keep(pair_plan_seed(gs, rids, k, soft, cache, True))
            if time.perf_counter() < soft and k in (lb, est, nt // 2):
                keep(pair_plan_seed(gs, rids, k, soft, cache, True, 1))
            if time.perf_counter() < soft and k in (lb, est):
                keep(pair_plan_seed(gs, rids, k, soft, cache, True, 0, 9.0, 3109 + k))
        if time.perf_counter() < soft:
            for k in (lb, est, nt // 2, lb + 1, est + 1):
                if time.perf_counter() >= soft:
                    break
                if 0 <= k <= nt // 2:
                    keep(stochastic_pair_plan_seed(gs, rids, k, soft, cache, 0, 28 if nr <= int(nt * 0.85) else 16, 5 if nr <= int(nt * 0.85) else 3, 22.0, 5100 + k))

    gen_allowed = (22 <= nt < 35) and nr >= int(nt * 1.45)
    if gen_allowed and time.perf_counter() < ddl:
        rem = ddl - time.perf_counter()
        if rem > 0.85:
            keep(generated_style_seed(gs, rids, cs, min(ddl, time.perf_counter() + min(3.15, max(0.85, rem * 0.78)))))

    scarce_tail = nr <= int(nt * 0.75) and nt >= 20 and avgp >= 0.16
    lddl = ddl
    if scarce_tail:
        rem = ddl - time.perf_counter()
        if rem > 3.0:
            lddl = ddl - min(2.0, max(1.0, rem * 0.22))
    elif ddl - time.perf_counter() > 0.45:
        lddl = ddl - 0.12

    if time.perf_counter() < lddl:
        keep(legacy_search(gs, cs, lddl))
    if scarce_tail and best_box[0] is not None and time.perf_counter() < ddl:
        if time.perf_counter() + 0.25 < ddl:
            s = skeleton_mcf_search(gs, rids, best_box[0], min(ddl, time.perf_counter() + 0.45))
            keep(s)
        bddl = min(ddl, time.perf_counter() + 0.82)
        bcache = {}
        rows = []
        for tr in (int(float(nr) / 1.17 + 0.5), nr, nr - 4, int(float(nt) * 0.62 + 0.5)):
            tr = max(1, min(nt, nr, tr))
            if tr not in rows:
                rows.append(tr)
        cfgs = ((0, 0.0, 0), (1, 0.0, 0), (-1, 0.0, 0), (0, 7.0, 741), (1, 9.0, 947))
        for tr in rows:
            for qoff, jit, seed in cfgs:
                if time.perf_counter() >= bddl:
                    break
                keep(bundle_plan_seed(gs, rids, tr, bddl, bcache, qoff, jit, seed + tr * 13))
            if time.perf_counter() >= bddl:
                break
        mid = min(ddl, time.perf_counter() + 0.35)
        s = pair2_swap(gs, best_box[0], mid)
        keep(s)
        if time.perf_counter() < ddl:
            s = scarce_lns(gs, s, ddl)
            if time.perf_counter() < ddl:
                s = pair2_swap(gs, s, ddl)
            keep(s)
    if best_box[0] is not None and time.perf_counter() < ddl and (low_case or scarce_tail):
        rem = ddl - time.perf_counter()
        if rem > 0.9:
            keep(cpsat_offer_seed(gs, rids, min(ddl, time.perf_counter() + min(1.35, rem * 0.45))))
    if best_box[0] is not None and time.perf_counter() < ddl:
        s = cycle_riders(gs, best_box[0], ddl)
        if time.perf_counter() < ddl:
            s = path_riders(gs, s, ddl)
        if time.perf_counter() < ddl:
            s = cycle_riders(gs, s, ddl, 20)
        if time.perf_counter() < ddl:
            s = config_polish(gs, s, min(ddl, time.perf_counter() + 0.10), 6, 2)
        keep(s)

    return best_box[0] if best_box[0] is not None else greedy(gs, cs, 0, ddl)

# ---- component: legacy_search ----
def legacy_search(gs, cs, ddl):
    rng = random.Random(20260604 + len(gs) * 17 + len(cs))
    best, bv = {}, None

    def keep(s):
        nonlocal_best[0] += 1
        v = value(gs, s)
        if better(v, nonlocal_best[2]):
            nonlocal_best[1], nonlocal_best[2] = copy(s), v

    nonlocal_best = [0, best, bv]

    for m in (0, 2, 4, 1, 3):
        if time.perf_counter() >= ddl:
            break
        if best and ddl - time.perf_counter() < 3.0:
            break
        s = greedy(gs, cs, m, ddl)
        s = pair_replace(gs, cs, s, ddl, 300, 3)
        s = polish(gs, cs, s, ddl, 2)
        keep(s)

    best, bv = nonlocal_best[1], nonlocal_best[2]
    if best and ddl - time.perf_counter() > 2.2:
        s = lns(gs, best, min(ddl, time.perf_counter() + 0.55), rng)
        keep(s)
        best, bv = nonlocal_best[1], nonlocal_best[2]

    if best and time.perf_counter() < ddl:
        short = min(ddl, time.perf_counter() + 0.45)
        s = expand(gs, best, short, 6, 200, 120)
        s = reassign(gs, s, short, 2, 120)
        s = move_extra(gs, s, short, 90)
        s = swap_riders(gs, s, short, 90)
        keep(s)
        best, bv = nonlocal_best[1], nonlocal_best[2]

    seeds = (15, 11, 10, 1, 2, 3)
    for i, seed in enumerate(seeds):
        if not best:
            break
        rem = ddl - time.perf_counter()
        if rem <= 0.05:
            break
        sec = max(0.12, rem / (len(seeds) - i))
        s = anneal(gs, best, min(ddl, time.perf_counter() + sec), random.Random(seed + len(gs) * 31))
        if time.perf_counter() < ddl:
            s = swap_riders(gs, s, min(ddl, time.perf_counter() + 0.05), 40)
        keep(s)
        best, bv = nonlocal_best[1], nonlocal_best[2]

    return best if best else greedy(gs, cs, 0, ddl)
