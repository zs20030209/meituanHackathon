import math
import random
import time
import itertools
import heapq


TL = 9.65
EPS = 1e-9
PEN = 100.0
DENSE30 = 16.75


class C:
    __slots__ = ("g", "r", "s", "p")
    def __init__(self, g, r, s, p):
        self.g = g
        self.r = r
        self.s = s
        self.p = p


class G:
    __slots__ = ("key", "tasks", "mask", "n", "cs", "by", "best")
    def __init__(self, key, tasks, mask):
        self.key = key
        self.tasks = tasks
        self.mask = mask
        self.n = len(tasks)
        self.cs = []
        self.by = {}
        self.best = None


if hasattr(int, "bit_count"):
    def pc(x):
        return x.bit_count()
else:
    def pc(x):
        n = 0
        while x:
            x &= x - 1
            n += 1
        return n


def better(a, b):
    return b is None or a[0] > b[0] or (a[0] == b[0] and a[1] < b[1] - EPS)


def parse(txt):
    ls = txt.strip().splitlines()
    if not ls:
        return [], [], [], []
    st = 1 if ls[0].lower().startswith("task_id_list") else 0
    ti, ri, rids, gmap, gs, best = {}, {}, [], {}, [], {}
    for line in ls[st:]:
        p = line.split("\t")
        if len(p) < 4:
            continue
        key, rid = p[0].strip(), p[1].strip()
        try:
            s, w = float(p[2]), float(p[3])
        except Exception:
            continue
        if not key or not rid or not math.isfinite(s):
            continue
        if not math.isfinite(w):
            w = 0.0
        w = 0.0 if w < 0.0 else 1.0 if w > 1.0 else w
        ts = tuple(t.strip() for t in key.split(",") if t.strip())
        if not ts:
            continue
        mask = 0
        for t in ts:
            if t not in ti:
                ti[t] = len(ti)
            mask |= 1 << ti[t]
        ckey = ",".join(ts)
        if ckey not in gmap:
            gmap[ckey] = len(gs)
            gs.append(G(ckey, ts, mask))
        if rid not in ri:
            ri[rid] = len(rids)
            rids.append(rid)
        g, r = gmap[ckey], ri[rid]
        old = best.get((g, r))
        if old is None or s < old[0] - EPS:
            best[(g, r)] = (s, w)
    allc = []
    for (g, r), (s, w) in best.items():
        c = C(g, r, s, w)
        gs[g].cs.append(c)
        gs[g].by[r] = c
        allc.append(c)
    alive, remap = [], {}
    for i, g in enumerate(gs):
        if g.cs:
            remap[i] = len(alive)
            alive.append(g)
    if len(alive) != len(gs):
        nc = []
        for c in allc:
            if c.g in remap:
                c.g = remap[c.g]
                nc.append(c)
        gs, allc = alive, nc
    for g in gs:
        g.cs.sort(key=lambda c: (c.s, -c.p))
        g.best = g.cs[0]
    allc.sort(key=lambda c: (c.s, -c.p, -gs[c.g].n))
    return gs, rids, allc


def gcost_cands(g, cs, pen=PEN):
    if not cs:
        return pen * g.n
    rej, sw, ss = 1.0, 0.0, 0.0
    for c in cs:
        rej *= 1.0 - c.p
        sw += c.p
        ss += c.p * c.s
    acc = 1.0 - rej
    exp = (min(c.s for c in cs) * acc) if sw <= 1e-12 else acc * ss / sw
    return exp + pen * g.n * rej


def gcost(gs, gi, rs, pen=PEN):
    g = gs[gi]
    return gcost_cands(g, [g.by[r] for r in rs if r in g.by], pen)


def single(g, c, pen=PEN):
    return c.p * c.s + pen * g.n * (1.0 - c.p)


def value(gs, sel):
    cov, cost = 0, 0.0
    for gi, rs in sel.items():
        if rs:
            cov += gs[gi].n
            cost += gcost(gs, gi, rs)
    return cov, cost


def gain_over(gs, cand, base):
    if not cand:
        return -1e100
    cv = value(gs, cand)
    bv = value(gs, base)
    if cv[0] != bv[0]:
        return (cv[0] - bv[0]) * 100000.0 + (bv[1] - cv[1])
    return bv[1] - cv[1]


def copy(sel):
    return {g: list(rs) for g, rs in sel.items() if rs}


def task_count(gs):
    m = 0
    for g in gs:
        m |= g.mask
    return pc(m)


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


def stats(gs, sel):
    m, ur, cost = 0, set(), 0.0
    for gi, rs in sel.items():
        m |= gs[gi].mask
        ur.update(rs)
        cost += gcost(gs, gi, rs)
    return m, ur, (pc(m), cost)


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


def skeleton_mcf_search(gs, rids, sel, ddl, max_eval=260):
    tasks, sg, pg = task_maps(gs)
    nt = len(tasks)
    pair_by_g = {v: k for k, v in pg.items()}
    pairs = []
    for gi in sel:
        p = pair_by_g.get(gi)
        if p is not None:
            pairs.append(p)
    k = len(pairs)
    if k <= 1:
        return sel

    def build(ps):
        used, groups = set(), []
        for a, b in ps:
            if a > b:
                a, b = b, a
            gi = pg.get((a, b))
            if gi is None or a in used or b in used:
                return None
            used.add(a)
            used.add(b)
            groups.append(gi)
        for i, t in enumerate(tasks):
            if i not in used:
                if t not in sg:
                    return None
                groups.append(sg[t])
        return groups

    best = copy(sel)
    bv = value(gs, best)
    rng = random.Random(9021 + nt * 17 + len(rids) * 13 + k)
    edges = list(pg.keys())
    evals = 0
    while time.perf_counter() < ddl and evals < max_eval:
        ps = list(pairs)
        typ = rng.random()
        if typ < 0.45:
            used = set()
            for a, b in ps:
                used.add(a)
                used.add(b)
            un = [i for i in range(nt) if i not in used]
            if not un:
                continue
            idx = rng.randrange(k)
            a, b = ps[idx]
            u = rng.choice(un)
            np = (a, u) if rng.random() < 0.5 else (b, u)
            if np[0] > np[1]:
                np = (np[1], np[0])
            if np not in pg:
                continue
            ps[idx] = np
        elif typ < 0.85 and k >= 2:
            i, j = rng.sample(range(k), 2)
            a, b = ps[i]
            c, d = ps[j]
            opts = []
            for e1, e2 in (((a, c), (b, d)), ((a, d), (b, c))):
                x1 = tuple(sorted(e1))
                x2 = tuple(sorted(e2))
                if x1 in pg and x2 in pg and len(set(x1 + x2)) == 4:
                    opts.append((x1, x2))
            if not opts:
                continue
            ps[i], ps[j] = rng.choice(opts)
        else:
            idx = rng.randrange(k)
            used = set()
            for q, p in enumerate(ps):
                if q != idx:
                    used.add(p[0])
                    used.add(p[1])
            cand = [e for e in edges if e[0] not in used and e[1] not in used]
            if not cand:
                continue
            ps[idx] = rng.choice(cand)
        ps = [tuple(sorted(p)) for p in ps]
        if len(set(ps)) < len(ps):
            continue
        flat = []
        for p in ps:
            flat.extend(p)
        if len(set(flat)) < len(flat):
            continue
        groups = build(ps)
        if groups is None:
            continue
        evals += 1
        s = assign_first_mcf(gs, groups, len(rids), ddl)
        if not s:
            continue
        v = value(gs, s)
        if v[0] == bv[0] and v[1] < bv[1] - 1.0:
            pairs, best, bv = ps, s, v
    return best


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


def task_maps(gs):
    tasks, sg, pg = [], {}, {}
    for i, g in enumerate(gs):
        if g.n == 1:
            sg[g.tasks[0]] = i
            tasks.append(g.tasks[0])
    tasks.sort()
    idx = {t: i for i, t in enumerate(tasks)}
    for i, g in enumerate(gs):
        if g.n == 2:
            a, b = sorted(g.tasks)
            if a in idx and b in idx:
                pg[(idx[a], idx[b])] = i
    return tasks, sg, pg


def approx_k_cost(gs, gi, k, cache):
    if k <= 0:
        return 0.0
    key = (gi, k)
    if key in cache:
        return cache[key]
    g = gs[gi]
    rkey = ("r", gi)
    arr = cache.get(rkey)
    if arr is None:
        lim = 18
        seen, arr = set(), []
        lists = (
            g.cs[:lim],
            sorted(g.cs, key=lambda c: (single(g, c), c.s, -c.p))[:lim],
            sorted(g.cs, key=lambda c: (-c.p, c.s))[:max(6, lim // 2)],
            sorted(g.cs, key=lambda c: (c.s, -c.p))[:max(6, lim // 2)],
        )
        for xs in lists:
            for c in xs:
                if c.r not in seen:
                    seen.add(c.r)
                    arr.append(c)
        cache[rkey] = arr
    if not arr:
        cache[key] = 1e100
        return cache[key]
    k = min(k, len(arr))
    best = 1e100
    for seed in arr[:min(8, len(arr))]:
        cur, used = [seed], {seed.r}
        cv = gcost_cands(g, cur)
        while len(cur) < k:
            bd = None
            for c in arr:
                if c.r in used:
                    continue
                nv = gcost_cands(g, cur + [c])
                if bd is None or nv < bd[0]:
                    bd = (nv, c)
            if bd is None:
                break
            cv, c = bd
            cur.append(c)
            used.add(c.r)
        if len(cur) == k and cv < best:
            best = cv
    cache[key] = best
    return best


def choose_pairs(edges, weights, pg, nt, k):
    edges = sorted(edges, reverse=True)
    used, pairs = set(), []
    for w, a, b in edges:
        if len(pairs) >= k:
            break
        if a not in used and b not in used:
            used.add(a)
            used.add(b)
            pairs.append([a, b, w])
    if not pairs:
        return []
    for _ in range(12):
        ok = False
        un = [i for i in range(nt) if i not in used]
        for p in pairs:
            a, b, w = p
            for u in un:
                x, y = (a, u) if a < u else (u, a)
                nw = weights.get((x, y))
                if nw is not None and nw > w + 1e-9:
                    used.remove(b)
                    used.add(u)
                    p[0], p[1], p[2] = x, y, nw
                    ok = True
                    break
                x, y = (b, u) if b < u else (u, b)
                nw = weights.get((x, y))
                if nw is not None and nw > w + 1e-9:
                    used.remove(a)
                    used.add(u)
                    p[0], p[1], p[2] = x, y, nw
                    ok = True
                    break
            if ok:
                break
        if ok:
            continue
        for i in range(len(pairs)):
            a, b, w1 = pairs[i]
            for j in range(i + 1, len(pairs)):
                c, d, w2 = pairs[j]
                cur = w1 + w2
                opts = (((a, c), (b, d)), ((a, d), (b, c)))
                for e1, e2 in opts:
                    x1, y1 = sorted(e1)
                    x2, y2 = sorted(e2)
                    if (x1, y1) not in pg or (x2, y2) not in pg:
                        continue
                    nw = weights.get((x1, y1), -1e100) + weights.get((x2, y2), -1e100)
                    if nw > cur + 1e-9:
                        pairs[i] = [x1, y1, weights[(x1, y1)]]
                        pairs[j] = [x2, y2, weights[(x2, y2)]]
                        ok = True
                        break
                if ok:
                    break
            if ok:
                break
        if not ok:
            break
    return pairs


def assign_first_mcf(gs, groups, nr, ddl):
    ng = len(groups)
    if ng == 0 or nr == 0 or ng > nr or time.perf_counter() >= ddl:
        return {}
    src, sink = 0, 1 + ng + nr
    n = sink + 1
    net = [[] for _ in range(n)]

    def add(u, v, cap, cost):
        net[u].append([v, cap, cost, len(net[v])])
        net[v].append([u, 0, -cost, len(net[u]) - 1])

    for i, gi in enumerate(groups):
        u = 1 + i
        add(src, u, 1, 0.0)
        g = gs[gi]
        for c in g.cs:
            add(u, 1 + ng + c.r, 1, single(g, c))
    for r in range(nr):
        add(1 + ng + r, sink, 1, 0.0)

    pot = [0.0] * n
    flow = 0
    while flow < ng and time.perf_counter() < ddl:
        dist = [1e100] * n
        pv = [-1] * n
        pe = [-1] * n
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]:
                continue
            for ei, e in enumerate(net[u]):
                if e[1] <= 0:
                    continue
                v = e[0]
                nd = d + e[2] + pot[u] - pot[v]
                if nd < dist[v] - 1e-12:
                    dist[v], pv[v], pe[v] = nd, u, ei
                    heapq.heappush(pq, (nd, v))
        if pv[sink] < 0:
            break
        for i in range(n):
            if dist[i] < 1e90:
                pot[i] += dist[i]
        v = sink
        while v != src:
            e = net[pv[v]][pe[v]]
            e[1] -= 1
            net[v][e[3]][1] += 1
            v = pv[v]
        flow += 1
    if flow < ng:
        return {}
    sel = {}
    base = 1 + ng
    for i, gi in enumerate(groups):
        u = 1 + i
        for e in net[u]:
            v = e[0]
            if base <= v < base + nr and e[1] == 0:
                sel[gi] = [v - base]
                break
    return sel


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


def clean(gs, rids, sel):
    out, tm, ur = [], 0, set()
    for gi in sorted(sel, key=lambda x: gs[x].key):
        g = gs[gi]
        if g.mask & tm:
            continue
        row, seen = [], set()
        for r in sorted(sel[gi], key=lambda x: (gs[gi].by[x].s if x in gs[gi].by else 1e99, rids[x])):
            if r in seen or r in ur or r not in g.by:
                continue
            seen.add(r)
            ur.add(r)
            row.append(rids[r])
        if row:
            tm |= g.mask
            out.append((g.key, row))
    return out


def solve(input_text: str) -> list:
    if not isinstance(input_text, str):
        return []
    try:
        st = time.perf_counter()
        gs, rids, cs = parse(input_text)
    except Exception:
        return []
    if not gs or not cs:
        return []
    try:
        n, m = task_count(gs), len(cs)
        bud = 0.1 if n <= 2 else 1.8 if n <= 8 or m <= 800 else 4.8 if n <= 15 else 6.0 if n <= 25 else TL
        ddl = st + min(TL, bud)
        reserve = 0.0
        kick_topn, kick_cand = 12, 3
        tail_low = False
        if n >= 22 and len(rids) >= int(n * 1.45):
            avgp = sum(c.p for c in cs) / max(1, len(cs))
            sgp = []
            for g in gs:
                if g.n == 1 and g.cs:
                    sgp.append(max(c.p for c in g.cs))
            best1 = sum(sgp) / len(sgp) if sgp else 1.0
            tail_low = avgp < 0.22 or best1 < 0.55
            if avgp >= 0.22 and bud > 4.0:
                reserve = min(2.25, max(0.0, (ddl - st) * 0.24))
                if n >= 35:
                    kick_topn, kick_cand = 12, 3
                elif best1 < 0.55:
                    kick_topn, kick_cand = 16, 4
            elif bud > 4.0:
                reserve = min(0.55, max(0.0, (ddl - st) * 0.06))
        sel = search(gs, rids, cs, ddl - reserve)
        if reserve > 0.0 and n >= 35 and len(rids) >= int(n * 1.45):
            base_v = value(gs, sel)
            if base_v[1] / max(1, n) > 16.0:
                kick_topn, kick_cand = 12, 1
        if reserve > 0.0 and time.perf_counter() < ddl:
            sel = pair_kick(gs, sel, ddl, kick_topn, kick_cand)
            tail_density = value(gs, sel)[1] / max(1, n)
            wide_tail = 22 <= n < 35 and (not tail_low) and tail_density > DENSE30
            if wide_tail and time.perf_counter() < ddl:
                alt_top = 18
                alt_cand = 4
                cand = pair_kick(gs, sel, ddl, alt_top, alt_cand)
                if gain_over(gs, cand, sel) > 0.45:
                    sel = cand
            if time.perf_counter() < ddl:
                sel = cycle_riders(gs, sel, min(ddl, time.perf_counter() + 0.08), 24)
            if time.perf_counter() < ddl:
                sel = path_riders(gs, sel, min(ddl, time.perf_counter() + 0.08), 20)
            if time.perf_counter() < ddl:
                sel = config_polish(gs, sel, min(ddl, time.perf_counter() + 0.08), 5, 1)
    except Exception:
        try:
            sel = greedy(gs, cs, 0, time.perf_counter() + 0.05)
        except Exception:
            return []
    try:
        return clean(gs, rids, sel)
    except Exception:
        return []

