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

# ---- component: expand ----
def expand(gs, sel, ddl, max_total=5, max_add=100, lim=90):
    sel = copy(sel)
    used = set()
    for rs in sel.values():
        used.update(rs)
    add = 0
    while add < max_add and time.perf_counter() < ddl:
        best = None
        for gi, rs in list(sel.items()):
            if len(rs) >= max_total:
                continue
            g = gs[gi]
            base = gcost(gs, gi, rs)
            cur = set(rs)
            for c in g.cs[:lim]:
                if c.r in cur or c.r in used:
                    continue
                nr = rs + [c.r]
                d = gcost(gs, gi, nr) - base
                if d < -1e-6:
                    k = (d, c.s, -c.p)
                    if best is None or k < best[0]:
                        best = (k, gi, c.r)
        if best is None:
            break
        _, gi, r = best
        sel[gi].append(r)
        used.add(r)
        add += 1
    return sel

# ---- component: reassign ----
def reassign(gs, sel, ddl, passes=2, lim=90):
    sel = copy(sel)
    used = set()
    for rs in sel.values():
        used.update(rs)
    for _ in range(passes):
        ok = False
        for gi, rs in list(sel.items()):
            if time.perf_counter() >= ddl:
                return sel
            g, base, cur = gs[gi], gcost(gs, gi, rs), set(rs)
            for pos, old in enumerate(list(rs)):
                best = None
                for c in g.cs[:lim]:
                    if c.r == old or c.r in cur or c.r in used:
                        continue
                    tr = list(rs)
                    tr[pos] = c.r
                    d = gcost(gs, gi, tr) - base
                    if d < -1e-6 and (best is None or d < best[0]):
                        best = (d, pos, c.r)
                if best:
                    _, pos, nr = best
                    used.discard(rs[pos])
                    rs[pos] = nr
                    used.add(nr)
                    ok = True
                    base = gcost(gs, gi, rs)
        if not ok:
            break
    return sel

# ---- component: single_replace ----
def single_replace(gs, cs, sel, ddl, lim=6000, passes=2):
    sel = copy(sel)
    for _ in range(passes):
        mask, used, val = stats(gs, sel)
        curcost, ok = val[1], False
        for i, c in enumerate(cs[:min(lim, len(cs))]):
            if i & 255 == 0 and time.perf_counter() >= ddl:
                return sel
            g = gs[c.g]
            conf = set()
            for gi, rs in sel.items():
                if gs[gi].mask & g.mask or c.r in rs:
                    conf.add(gi)
            if not conf and ((g.mask & mask) or c.r in used):
                continue
            remcov = sum(gs[x].n for x in conf)
            remcost = sum(gcost(gs, x, sel[x]) for x in conf)
            nv = (val[0] - remcov + g.n, curcost - remcost + single(g, c))
            if better(nv, val):
                for x in conf:
                    sel.pop(x, None)
                sel[c.g] = [c.r]
                ok = True
                break
        if not ok:
            break
    return sel

# ---- component: pair_replace ----
def pair_replace(gs, cs, sel, ddl, pooln=300, loops=2):
    sel = copy(sel)
    for _ in range(loops):
        seen, pool = set(), []
        for c in sorted(cs, key=lambda c: (single(gs[c.g], c) / gs[c.g].n, c.s)):
            if c.g in seen:
                continue
            seen.add(c.g)
            pool.append(c)
            if len(pool) >= pooln:
                break
        mask, used, val = stats(gs, sel)
        best = None
        for i, a in enumerate(pool):
            if i & 15 == 0 and time.perf_counter() >= ddl:
                return sel
            ga = gs[a.g]
            for b in pool[i + 1:]:
                if a.r == b.r:
                    continue
                gb = gs[b.g]
                if ga.mask & gb.mask:
                    continue
                conf = set()
                both = ga.mask | gb.mask
                for gi, rs in sel.items():
                    if gs[gi].mask & both or a.r in rs or b.r in rs:
                        conf.add(gi)
                nv = (
                    val[0] - sum(gs[x].n for x in conf) + ga.n + gb.n,
                    val[1] - sum(gcost(gs, x, sel[x]) for x in conf) + single(ga, a) + single(gb, b),
                )
                if better(nv, val):
                    gain = (nv[0] - val[0], val[1] - nv[1])
                    k = (-gain[0], -gain[1])
                    if best is None or k < best[0]:
                        best = (k, conf, a, b)
        if not best:
            break
        _, conf, a, b = best
        for x in conf:
            sel.pop(x, None)
        sel[a.g] = [a.r]
        sel[b.g] = [b.r]
    return sel

# ---- component: move_extra ----
def move_extra(gs, sel, ddl, moves=80):
    sel = copy(sel)
    n = 0
    while n < moves and time.perf_counter() < ddl:
        items = list(sel.items())
        costs = {gi: gcost(gs, gi, rs) for gi, rs in items}
        best = None
        for a, ar in items:
            if len(ar) <= 1:
                continue
            for r in list(ar):
                ra = [x for x in ar if x != r]
                da = gcost(gs, a, ra) - costs[a]
                for b, br in items:
                    if a == b or r in br or r not in gs[b].by:
                        continue
                    tr = br + [r]
                    d = da + gcost(gs, b, tr) - costs[b]
                    if d < -1e-6 and (best is None or d < best[0]):
                        best = (d, a, b, r)
        if not best:
            break
        _, a, b, r = best
        sel[a] = [x for x in sel[a] if x != r]
        sel[b].append(r)
        n += 1
    return sel

# ---- component: swap_riders ----
def swap_riders(gs, sel, ddl, moves=80):
    sel = copy(sel)
    n = 0
    while n < moves and time.perf_counter() < ddl:
        items = list(sel.items())
        costs = {gi: gcost(gs, gi, rs) for gi, rs in items}
        best = None
        for i, (a, ar) in enumerate(items):
            if i & 7 == 0 and time.perf_counter() >= ddl:
                return sel
            for b, br in items[i + 1:]:
                for pa, ra in enumerate(ar):
                    if ra not in gs[b].by:
                        continue
                    for pb, rb in enumerate(br):
                        if rb == ra or rb not in gs[a].by:
                            continue
                        ta, tb = list(ar), list(br)
                        ta[pa], tb[pb] = rb, ra
                        if len(set(ta)) < len(ta) or len(set(tb)) < len(tb):
                            continue
                        d = gcost(gs, a, ta) + gcost(gs, b, tb) - costs[a] - costs[b]
                        if d < -1e-6 and (best is None or d < best[0]):
                            best = (d, a, b, ta, tb)
        if not best:
            break
        _, a, b, ta, tb = best
        sel[a], sel[b] = ta, tb
        n += 1
    return sel

# ---- component: cycle_riders ----
def cycle_riders(gs, sel, ddl, max_iter=80):
    sel = copy(sel)
    for _ in range(max_iter):
        if time.perf_counter() >= ddl:
            break
        owner = {}
        for gi, rs in sel.items():
            for r in rs:
                owner[r] = gi
        riders = list(owner)
        n = len(riders)
        if n < 3:
            break
        idx = {r: i for i, r in enumerate(riders)}
        base = {gi: gcost(gs, gi, rs) for gi, rs in sel.items()}
        edges = []
        for q in riders:
            gi = owner[q]
            g = gs[gi]
            cur = sel[gi]
            b = base[gi]
            for r in riders:
                if r == q or r in cur or r not in g.by:
                    continue
                nr = [r if x == q else x for x in cur]
                d = gcost(gs, gi, nr) - b
                if d < 30.0:
                    edges.append((idx[r], idx[q], d, r, q, gi))
        if not edges:
            break
        dist = [0.0] * n
        pre = [None] * n
        x = -1
        for _ in range(n):
            x = -1
            for e in edges:
                u, v, w, _, _, _ = e
                if dist[u] + w < dist[v] - 1e-11:
                    dist[v] = dist[u] + w
                    pre[v] = e
                    x = v
            if x < 0:
                break
        if x < 0:
            break
        y = x
        for _ in range(n):
            if pre[y] is None:
                y = -1
                break
            y = pre[y][0]
        if y < 0:
            break
        cyc, seen, cur = [], {}, y
        while cur not in seen and pre[cur] is not None:
            seen[cur] = len(cyc)
            e = pre[cur]
            cyc.append(e)
            cur = e[0]
        if cur not in seen:
            break
        cyc = cyc[seen[cur]:]
        ns = copy(sel)
        ok = True
        for _, _, _, r, q, gi in cyc:
            if q not in ns.get(gi, ()) or r in ns.get(gi, ()):
                ok = False
                break
            ns[gi] = [r if x == q else x for x in ns[gi]]
        if ok:
            used = []
            for rs in ns.values():
                used.extend(rs)
            if len(used) != len(set(used)):
                ok = False
        if ok and better(value(gs, ns), value(gs, sel)):
            sel = ns
        else:
            break
    return sel

# ---- component: path_riders ----
def path_riders(gs, sel, ddl, max_iter=60):
    sel = copy(sel)
    for _ in range(max_iter):
        if time.perf_counter() >= ddl:
            break
        owner = {}
        for gi, rs in sel.items():
            for r in rs:
                owner[r] = gi
        riders = list(owner)
        n = len(riders)
        if n < 2:
            break
        idx = {r: i for i, r in enumerate(riders)}
        base = {gi: gcost(gs, gi, rs) for gi, rs in sel.items()}
        dist = [1e100] * n
        pre = [None] * n
        for q in riders:
            gi = owner[q]
            cur = sel[gi]
            if len(cur) <= 1:
                continue
            nr = [x for x in cur if x != q]
            d = gcost(gs, gi, nr) - base[gi]
            qi = idx[q]
            if d < dist[qi]:
                dist[qi] = d
                pre[qi] = ("rem", q, gi)
        edges = []
        for q in riders:
            gi = owner[q]
            g = gs[gi]
            cur = sel[gi]
            b = base[gi]
            for r in riders:
                if r == q or r in cur or r not in g.by:
                    continue
                nr = [r if x == q else x for x in cur]
                d = gcost(gs, gi, nr) - b
                if d < 30.0:
                    edges.append((idx[r], idx[q], d, r, q, gi))
        for _ in range(n - 1):
            ok = False
            for u, v, w, r, q, gi in edges:
                if dist[u] + w < dist[v] - 1e-11:
                    dist[v] = dist[u] + w
                    pre[v] = ("rep", r, q, gi, u)
                    ok = True
            if not ok:
                break
        best = None
        for r in riders:
            ri = idx[r]
            if dist[ri] >= 1e90:
                continue
            for gi, rs in sel.items():
                if r in rs or r not in gs[gi].by:
                    continue
                nr = rs + [r]
                d = gcost(gs, gi, nr) - base[gi]
                total = dist[ri] + d
                if total < -1e-9 and (best is None or total < best[0]):
                    best = (total, r, gi)
        if best is None:
            break
        _, r, add_gi = best
        ops = [("add", r, add_gi)]
        cur = idx[r]
        seen = set()
        ok = True
        while True:
            if cur in seen or pre[cur] is None:
                ok = False
                break
            seen.add(cur)
            p = pre[cur]
            if p[0] == "rem":
                _, q, gi = p
                ops.append(("rem", q, gi))
                break
            _, rin, q, gi, cur = p
            ops.append(("rep", rin, q, gi))
        if not ok:
            break
        ops.reverse()
        ns = copy(sel)
        for op in ops:
            if op[0] == "rem":
                _, q, gi = op
                if q not in ns.get(gi, ()) or len(ns[gi]) <= 1:
                    ok = False
                    break
                ns[gi] = [x for x in ns[gi] if x != q]
            elif op[0] == "rep":
                _, rin, q, gi = op
                if q not in ns.get(gi, ()) or rin in ns.get(gi, ()) or rin not in gs[gi].by:
                    ok = False
                    break
                ns[gi] = [rin if x == q else x for x in ns[gi]]
            else:
                _, rin, gi = op
                if rin in ns.get(gi, ()) or rin not in gs[gi].by:
                    ok = False
                    break
                ns[gi] = ns[gi] + [rin]
        if ok:
            used = []
            for rs in ns.values():
                used.extend(rs)
            if len(used) != len(set(used)):
                ok = False
        if ok and better(value(gs, ns), value(gs, sel)):
            sel = ns
        else:
            break
    return sel

# ---- component: config_polish ----
def config_polish(gs, sel, ddl, width=6, rounds=3, pool_lim=9, cfg_lim=22):
    best = copy(sel)
    bv = value(gs, best)
    for off in range(rounds):
        if time.perf_counter() >= ddl or not best:
            break
        items = sorted(best, key=lambda gi: (gcost(gs, gi, best[gi]) / gs[gi].n, gcost(gs, gi, best[gi])), reverse=True)
        win = items[off:off + width]
        if len(win) < 2:
            break
        fixed = set()
        for gi, rs in best.items():
            if gi not in win:
                fixed.update(rs)
        old = sum(gcost(gs, gi, best[gi]) for gi in win)
        opts = []
        bad = False
        for gi in win:
            g, cur = gs[gi], best[gi]
            pool, seen = [], set()
            for r in cur:
                if r not in fixed and r in g.by and r not in seen:
                    seen.add(r)
                    pool.append(g.by[r])
            lists = (
                g.cs[:pool_lim],
                sorted(g.cs, key=lambda c: (single(g, c), c.s, -c.p))[:pool_lim],
                sorted(g.cs, key=lambda c: (-c.p, c.s))[:max(5, pool_lim - 2)],
            )
            for arr in lists:
                for c in arr:
                    if c.r not in fixed and c.r not in seen:
                        seen.add(c.r)
                        pool.append(c)
            if not pool:
                bad = True
                break
            maxr = min(4, max(1, len(cur) + 1), len(pool))
            cfg, seen_rs = [], set()
            for k in range(1, maxr + 1):
                for comb in itertools.combinations(pool[:pool_lim], k):
                    rs = tuple(sorted(c.r for c in comb))
                    if rs in seen_rs:
                        continue
                    seen_rs.add(rs)
                    cfg.append((gcost_cands(g, comb), rs))
            cur_rs = tuple(sorted(r for r in cur if r in g.by and r not in fixed))
            if cur_rs and cur_rs not in seen_rs:
                cfg.append((gcost(gs, gi, cur_rs), cur_rs))
            cfg.sort(key=lambda x: (x[0], len(x[1])))
            if not cfg:
                bad = True
                break
            opts.append((gi, cfg[:cfg_lim]))
        if bad:
            continue
        opts.sort(key=lambda x: len(x[1]))
        mins = [o[1][0][0] for o in opts]
        suf = [0.0] * (len(mins) + 1)
        for i in range(len(mins) - 1, -1, -1):
            suf[i] = suf[i + 1] + mins[i]
        box = [old, None]

        def dfs(i, used, cost, chosen):
            if time.perf_counter() >= ddl or cost + suf[i] >= box[0] - EPS:
                return
            if i == len(opts):
                box[0], box[1] = cost, list(chosen)
                return
            gi, cfgs = opts[i]
            for cst, rs in cfgs:
                ok = True
                for r in rs:
                    if r in used:
                        ok = False
                        break
                if not ok:
                    continue
                for r in rs:
                    used.add(r)
                chosen.append((gi, rs))
                dfs(i + 1, used, cost + cst, chosen)
                chosen.pop()
                for r in rs:
                    used.remove(r)

        dfs(0, set(), 0.0, [])
        if box[1] is not None:
            cand = copy(best)
            for gi, rs in box[1]:
                cand[gi] = list(rs)
            cv = value(gs, cand)
            if better(cv, bv):
                best, bv = cand, cv
    return best

# ---- component: pair_kick ----
def pair_kick(gs, sel, ddl, topn=12, cand_lim=3):
    bit_single = {}
    for gi, g in enumerate(gs):
        if g.n == 1:
            bit_single[(g.mask & -g.mask).bit_length() - 1] = gi
    pair_cfgs = []
    for gi, g in enumerate(gs):
        if g.n != 2 or not g.cs:
            continue
        pool, seen, lim = [], set(), 10
        lists = (
            g.cs[:lim],
            sorted(g.cs, key=lambda c: (single(g, c), c.s, -c.p))[:lim],
            sorted(g.cs, key=lambda c: (-c.p, c.s))[:8],
        )
        for arr in lists:
            for c in arr:
                if c.r not in seen:
                    seen.add(c.r)
                    pool.append(c)
        cfg, seen_rs = [], set()
        for c in pool[:lim]:
            rs = (c.r,)
            seen_rs.add(rs)
            cfg.append((gcost_cands(g, [c]), rs))
        for a, b in itertools.combinations(pool[:lim], 2):
            rs = tuple(sorted((a.r, b.r)))
            if rs not in seen_rs:
                seen_rs.add(rs)
                cfg.append((gcost_cands(g, [a, b]), rs))
        for comb in itertools.combinations(pool[:7], 3):
            rs = tuple(sorted(c.r for c in comb))
            if rs not in seen_rs:
                seen_rs.add(rs)
                cfg.append((gcost_cands(g, comb), rs))
        cfg.sort(key=lambda x: (x[0], len(x[1])))
        if cfg:
            pair_cfgs.append((cfg[0][0] / g.n, gi, cfg[:cand_lim]))
    pair_cfgs.sort()
    best = copy(sel)
    bv = value(gs, best)

    def cover_extra(cur, extra):
        used = set()
        for rs in cur.values():
            used.update(rs)
        b, m = 0, extra
        while m:
            if m & 1:
                gi = bit_single.get(b)
                if gi is None:
                    return None
                for c in gs[gi].cs:
                    if c.r not in used:
                        cur[gi] = [c.r]
                        used.add(c.r)
                        break
                if gi not in cur:
                    return None
            b += 1
            m >>= 1
        return cur

    improved, rounds = True, 0
    while improved and rounds < 4 and time.perf_counter() < ddl:
        improved = False
        rounds += 1
        for _, gi, cfgs in pair_cfgs[:topn]:
            if time.perf_counter() >= ddl:
                break
            if gi in best:
                continue
            g = gs[gi]
            for _, rs in cfgs:
                if time.perf_counter() >= ddl:
                    break
                cur = copy(best)
                task_owner, rider_owner = {}, {}
                for x, xs in cur.items():
                    b, m = 0, gs[x].mask
                    while m:
                        if m & 1:
                            task_owner[b] = x
                        b += 1
                        m >>= 1
                    for r in xs:
                        rider_owner[r] = x
                rem = set()
                b, m = 0, g.mask
                while m:
                    if m & 1 and b in task_owner:
                        rem.add(task_owner[b])
                    b += 1
                    m >>= 1
                for r in rs:
                    if r in rider_owner:
                        rem.add(rider_owner[r])
                um = 0
                for x in rem:
                    um |= gs[x].mask
                for x in rem:
                    cur.pop(x, None)
                used, ok = set(), True
                for xs in cur.values():
                    used.update(xs)
                for r in rs:
                    if r in used or r not in g.by:
                        ok = False
                        break
                if not ok:
                    continue
                cur[gi] = list(rs)
                extra = um & ~g.mask
                if extra:
                    cur = cover_extra(cur, extra)
                    if cur is None:
                        continue
                short = min(ddl, time.perf_counter() + 0.10)
                cur = expand(gs, cur, short, 8, 80, 160)
                if time.perf_counter() < short:
                    cur = reassign(gs, cur, short, 2, 120)
                if time.perf_counter() < short:
                    cur = move_extra(gs, cur, short, 80)
                if time.perf_counter() < short:
                    cur = swap_riders(gs, cur, short, 80)
                if time.perf_counter() < ddl:
                    cur = cycle_riders(gs, cur, min(ddl, time.perf_counter() + 0.035), 15)
                if time.perf_counter() < ddl:
                    cur = path_riders(gs, cur, min(ddl, time.perf_counter() + 0.045), 15)
                if time.perf_counter() < ddl:
                    cur = cycle_riders(gs, cur, min(ddl, time.perf_counter() + 0.025), 10)
                v = value(gs, cur)
                if better(v, bv):
                    best, bv = cur, v
                    improved = True
                    break
            if improved:
                break
    return best

# ---- component: repair_window ----
def repair_window(gs, sel, rem, ddl, keep=16, take=12, maxr=4):
    fixed = {g: list(rs) for g, rs in sel.items() if g not in rem}
    U, old = 0, 0.0
    for g in rem:
        if g in sel:
            U |= gs[g].mask
            old += gcost(gs, g, sel[g])
    if not U:
        return sel
    fm, fu = 0, set()
    for g, rs in fixed.items():
        fm |= gs[g].mask
        fu.update(rs)
    bits, m, b = [], U, 0
    while m:
        if m & 1:
            bits.append(b)
        b += 1
        m >>= 1
    bybit = {}
    for gi, g in enumerate(gs):
        if g.mask & fm or g.mask & ~U:
            continue
        cands = [c for c in g.cs if c.r not in fu]
        if not cands:
            continue
        pool, seen = [], set()
        lists = (
            sorted(cands, key=lambda c: (single(g, c), c.s, -c.p))[:take],
            sorted(cands, key=lambda c: (c.s, -c.p))[:max(4, take // 2)],
            sorted(cands, key=lambda c: (-c.p, c.s))[:max(4, take // 2)],
        )
        for arr in lists:
            for c in arr:
                if c.r not in seen:
                    seen.add(c.r)
                    pool.append(c)
        cfgs = []
        for r in range(1, min(maxr, len(pool)) + 1):
            for comb in itertools.combinations(pool, r):
                cfgs.append((gcost_cands(g, comb), tuple(c.r for c in comb), gi, g.mask))
        cfgs.sort(key=lambda x: (x[0], len(x[1])))
        cfgs = cfgs[:keep]
        mm, bit = g.mask, 0
        while mm:
            if mm & 1:
                bybit.setdefault(bit, []).extend(cfgs)
            bit += 1
            mm >>= 1
    for bit in bybit:
        bybit[bit].sort(key=lambda x: (x[0] / pc(x[3]), len(x[1])))
    best_box, memo, loops = [old, None], {}, [0]
    def first(mask):
        return (mask & -mask).bit_length() - 1
    def dfs(mask, used, cost, chosen):
        if time.perf_counter() >= ddl:
            return
        loops[0] += 1
        if cost >= best_box[0] - EPS:
            return
        if mask == 0:
            best_box[0], best_box[1] = cost, list(chosen)
            return
        if len(used) < 14:
            key = (mask, tuple(sorted(used)))
            if memo.get(key, 1e100) <= cost + EPS:
                return
            memo[key] = cost
        bit = first(mask)
        for cfg in bybit.get(bit, ()):
            cst, rs, gi, gm = cfg
            if gm & ~mask:
                continue
            bad = False
            for r in rs:
                if r in used:
                    bad = True
                    break
            if bad:
                continue
            for r in rs:
                used.add(r)
            chosen.append(cfg)
            dfs(mask ^ gm, used, cost + cst, chosen)
            chosen.pop()
            for r in rs:
                used.remove(r)
    dfs(U, set(), 0.0, [])
    if best_box[1] is None:
        return sel
    out = fixed
    for _, rs, gi, _ in best_box[1]:
        out[gi] = list(rs)
    return out

# ---- component: lns ----
def lns(gs, sel, ddl, rng):
    best = copy(sel)
    bv = value(gs, best)
    no = 0
    while time.perf_counter() < ddl:
        items = list(best)
        if not items:
            break
        dens = sorted([(gcost(gs, g, best[g]) / gs[g].n, g) for g in items], reverse=True)
        rem = set()
        if no & 1:
            rem.add(rng.choice(items))
        else:
            rem.add(dens[min(no, len(dens) - 1)][1])
        for _, g in dens:
            if len(rem) >= 5:
                break
            if g not in rem and (len(rem) < 2 or rng.random() < 0.45):
                rem.add(g)
        while len(rem) < min(6, len(items)) and rng.random() < 0.4:
            rem.add(rng.choice(items))
        cand = repair_window(gs, best, rem, ddl)
        cv = value(gs, cand)
        if better(cv, bv):
            best, bv, no = cand, cv, 0
        else:
            no += 1
        if no > 80:
            break
    return best

# ---- component: scarce_lns ----
def scarce_lns(gs, sel, ddl):
    best = copy(sel)
    bv = value(gs, best)
    if not best:
        return best
    rng = random.Random(911 + len(best) * 17)
    no, it = 0, 0
    while time.perf_counter() < ddl and no < 10:
        items = list(best)
        if len(items) <= 2:
            break
        costs = {g: gcost(gs, g, best[g]) for g in items}
        dens = sorted(items, key=lambda g: (costs[g] / gs[g].n, costs[g]), reverse=True)
        rem = set()
        if it % 3 == 0:
            rem.update(dens[:min(4, len(dens))])
        else:
            rem.add(dens[min(it, len(dens) - 1)])
        top = dens[:min(20, len(dens))]
        target = min(8, len(items))
        while len(rem) < target:
            rem.add(rng.choice(top))
        cand = repair_window(gs, best, rem, ddl, keep=36, take=18, maxr=3)
        cv = value(gs, cand)
        if better(cv, bv):
            best, bv, no = cand, cv, 0
        else:
            no += 1
        it += 1
    return best

# ---- component: pair2_swap ----
def pair2_swap(gs, sel, ddl, max_moves=80):
    sel = copy(sel)
    pmap = {}
    for gi, g in enumerate(gs):
        if g.n == 2:
            bits, m, b = [], g.mask, 0
            while m:
                if m & 1:
                    bits.append(b)
                b += 1
                m >>= 1
            if len(bits) == 2:
                pmap[(bits[0], bits[1])] = gi

    def bits2(mask):
        out, b = [], 0
        while mask:
            if mask & 1:
                out.append(b)
            b += 1
            mask >>= 1
        return out

    moves = 0
    while moves < max_moves and time.perf_counter() < ddl:
        items = list(sel.items())
        costs = {g: gcost(gs, g, rs) for g, rs in items}
        best = None
        for i, (g1, rs1) in enumerate(items):
            if (i & 7) == 0 and time.perf_counter() >= ddl:
                return sel
            if gs[g1].n != 2 or len(rs1) != 1:
                continue
            a, b = bits2(gs[g1].mask)
            r1 = rs1[0]
            for g2, rs2 in items[i + 1:]:
                if gs[g2].n != 2 or len(rs2) != 1:
                    continue
                c, d = bits2(gs[g2].mask)
                r2 = rs2[0]
                base = costs[g1] + costs[g2]
                for x, y, u, v in ((a, c, b, d), (a, d, b, c)):
                    if x > y:
                        x, y = y, x
                    if u > v:
                        u, v = v, u
                    h1, h2 = pmap.get((x, y)), pmap.get((u, v))
                    if h1 is None or h2 is None:
                        continue
                    if r1 in gs[h1].by and r2 in gs[h2].by:
                        nc = gcost(gs, h1, [r1]) + gcost(gs, h2, [r2])
                        dlt = nc - base
                        if dlt < -1e-9 and (best is None or dlt < best[0]):
                            best = (dlt, g1, g2, h1, [r1], h2, [r2])
                    if r2 in gs[h1].by and r1 in gs[h2].by:
                        nc = gcost(gs, h1, [r2]) + gcost(gs, h2, [r1])
                        dlt = nc - base
                        if dlt < -1e-9 and (best is None or dlt < best[0]):
                            best = (dlt, g1, g2, h1, [r2], h2, [r1])
        if best is None:
            break
        _, g1, g2, h1, r1, h2, r2 = best
        sel.pop(g1, None)
        sel.pop(g2, None)
        sel[h1] = r1
        sel[h2] = r2
        moves += 1
    return sel

# ---- component: polish ----
def polish(gs, cs, sel, ddl, level=1):
    if time.perf_counter() < ddl:
        sel = single_replace(gs, cs, sel, ddl, 4000 + level * 2000, 1)
    if time.perf_counter() < ddl:
        sel = pair_replace(gs, cs, sel, ddl, 230 + level * 40, 1 + (level > 1))
    if time.perf_counter() < ddl:
        sel = expand(gs, sel, ddl, 3 + level, 20 + 20 * level, 80)
    if time.perf_counter() < ddl:
        sel = reassign(gs, sel, ddl, 1 + (level > 1), 80)
    if time.perf_counter() < ddl:
        sel = move_extra(gs, sel, ddl, 20 + level * 20)
    if time.perf_counter() < ddl:
        sel = swap_riders(gs, sel, ddl, 20 + level * 20)
    return sel
