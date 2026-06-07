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

def copy(sel):
    return {g: list(rs) for g, rs in sel.items() if rs}

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

def better(a, b):
    return b is None or a[0] > b[0] or (a[0] == b[0] and a[1] < b[1] - EPS)

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

def task_count(gs):
    m = 0
    for g in gs:
        m |= g.mask
    return pc(m)

def pc(x):
    c = 0
    while x:
        c += 1
        x &= x - 1
    return c

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

def single_replace(gs, cs, sel, ddl, lim=6000, passes=2):
    sel = copy(sel)
    for _ in range(passes):
        if time.perf_counter() >= ddl:
            break
        used = set()
        for rs in sel.values():
            used.update(rs)
        items = list(sel.items())
        for gi, rs in items:
            if time.perf_counter() >= ddl:
                return sel
            g, base = gs[gi], gcost(gs, gi, rs)
            cur_rs = set(rs)
            for pos, old in enumerate(list(rs)):
                best = None
                for c in g.cs[:min(len(g.cs), lim)]:
                    if c.r == old or c.r in cur_rs or c.r in used:
                        continue
                    nr = list(rs)
                    nr[pos] = c.r
                    d = gcost(gs, gi, nr) - base
                    if d < -1e-6 and (best is None or d < best[0]):
                        best = (d, pos, c.r)
                if best:
                    _, pos, nr = best
                    used.discard(rs[pos])
                    rs[pos] = nr
                    used.add(nr)
                    base = gcost(gs, gi, rs)
    return sel

def pair_replace(gs, cs, sel, ddl, pooln=300, loops=2):
    sel = copy(sel)
    used = set()
    for rs in sel.values():
        used.update(rs)
    for _ in range(loops):
        if time.perf_counter() >= ddl:
            break
        items = [(gi, rs) for gi, rs in sel.items() if len(rs) >= 1]
        changed = True
        while changed and time.perf_counter() < ddl:
            changed = False
            for gi, rs in items:
                if time.perf_counter() >= ddl:
                    break
                g = gs[gi]
                base_cost = gcost(gs, gi, rs)
                pool = [c for c in g.cs[:pooln] if c.r not in used or c.r in rs]
                best_move = None
                for c in pool:
                    if c.r in rs:
                        continue
                    for i in range(len(rs)):
                        nr = list(rs)
                        nr[i] = c.r
                        if len(set(nr)) < len(nr):
                            continue
                        d = gcost(gs, gi, nr) - base_cost
                        if d < -1e-7 and (best_move is None or d < best_move[0]):
                            best_move = (d, gi, i, c.r)
                if best_move:
                    _, gi, i, nr = best_move
                    used.discard(rs[i])
                    rs[i] = nr
                    used.add(nr)
                    changed = True
    return sel

def pair_plan_seed(gs, rids, k, ddl, cache, mcf=False, qoff=0, jitter=0.0, seed=0):
    tasks, sg, pg = task_maps(gs)
    nt = len(tasks)
    if k <= 0 or k > nt // 2 or nt < 2:
        return {}
    # compute pair weights
    weights = {}
    for (a, b), gi in pg.items():
        g = gs[gi]
        best_cost = min(single(g, c) for c in g.cs) if g.cs else PEN * g.n
        weights[(a, b)] = -best_cost + jitter
    edges = [(weights[(a, b)], a, b) for (a, b) in weights]
    pairs = choose_pairs(edges, weights, pg, nt, k)
    if not pairs:
        return {}
    groups = []
    sel = {}
    used_tasks = set()
    for a, b, w in pairs:
        gi = pg.get((a, b))
        if gi is None:
            continue
        groups.append(gi)
    if mcf:
        sel = assign_first_mcf(gs, groups, len(rids), ddl)
    else:
        # greedy assignment per group
        for gi in groups:
            g = gs[gi]
            for c in g.cs:
                if c.r not in used_tasks:
                    sel[gi] = [c.r]
                    used_tasks.add(c.r)
                    break
    return sel

def stochastic_pair_plan_seed(gs, rids, k, ddl, cache, qoff=0, tries=24, evals=4, jitter=18.0, seed=0):
    rng = random.Random(seed)
    best = None
    for _ in range(tries):
        if time.perf_counter() >= ddl:
            break
        s = pair_plan_seed(gs, rids, k, ddl, cache, True, qoff, rng.uniform(0, jitter), rng.randint(0, 100000))
        if s:
            v = value(gs, s)
            if best is None or v[0] > best[0] or (v[0] == best[0] and v[1] < best[1] - EPS):
                best = (v[0], v[1], copy(s))
    return best[2] if best else {}

def singleton_seed(gs, ddl):
    sel = {}
    used = set()
    for g in gs:
        if time.perf_counter() >= ddl:
            break
        if g.n == 1 and g.cs:
            for c in g.cs:
                if c.r not in used:
                    sel[g.key] = [c.r]
                    used.add(c.r)
                    break
    # rebuild using group indices
    out = {}
    for gi, g in enumerate(gs):
        if g.n == 1 and g.key in sel:
            out[gi] = sel[g.key]
    return out

def mcf_singleton_seed(gs, rids, ddl):
    singles = [gi for gi, g in enumerate(gs) if g.n == 1 and g.cs]
    if not singles:
        return {}
    return assign_first_mcf(gs, singles, len(rids), ddl)

def bundle_plan_seed(gs, rids, target_rows, ddl, cache, qoff=0, jitter=0.0, seed=0):
    # Simplified: build seed by selecting groups up to target_rows using greedy
    rng = random.Random(seed) if seed else random.Random()
    sel = {}
    used_tasks = 0
    used_riders = set()
    for g in gs:
        if time.perf_counter() >= ddl:
            break
        if g.mask & used_tasks:
            continue
        for c in g.cs:
            if c.r in used_riders:
                continue
            sel[g.key] = [c.r]
            used_tasks |= g.mask
            used_riders.add(c.r)
            break
        if len(sel) >= target_rows:
            break
    out = {}
    for gi, g in enumerate(gs):
        if g.key in sel:
            out[gi] = sel[g.key]
    return out

def pair2_swap(gs, sel, ddl, max_moves=80):
    sel = copy(sel)
    n = 0
    while n < max_moves and time.perf_counter() < ddl:
        items = list(sel.items())
        costs = {gi: gcost(gs, gi, rs) for gi, rs in items}
        best = None
        for i, (a, ar) in enumerate(items):
            if len(ar) != 1:
                continue
            for j, (b, br) in enumerate(items):
                if i >= j or len(br) != 1:
                    continue
                a_r = ar[0]
                b_r = br[0]
                if a_r in gs[b].by and b_r in gs[a].by:
                    ta, tb = [b_r], [a_r]
                    d = gcost(gs, a, ta) + gcost(gs, b, tb) - costs[a] - costs[b]
                    if d < -1e-6 and (best is None or d < best[0]):
                        best = (d, a, b, ta, tb)
        if not best:
            break
        _, a, b, ta, tb = best
        sel[a], sel[b] = ta, tb
        n += 1
    return sel

def scarce_lns(gs, sel, ddl):
    sel = copy(sel)
    if not sel:
        return sel
    items = list(sel.items())
    if not items:
        return sel
    costs = [(gcost(gs, gi, rs) / gs[gi].n, gi) for gi, rs in items]
    costs.sort(reverse=True)
    top = max(1, len(items) // 3)
    target = set(gi for _, gi in costs[:top])
    new_sel = {}
    um = 0
    ur = set()
    for gi, rs in sel.items():
        if gi in target:
            continue
        g = gs[gi]
        if g.mask & um:
            continue
        new_sel[gi] = rs
        um |= g.mask
        ur.update(rs)
    for gi in target:
        if time.perf_counter() >= ddl:
            break
        g = gs[gi]
        if g.mask & um:
            continue
        for c in g.cs:
            if c.r not in ur:
                new_sel[gi] = [c.r]
                um |= g.mask
                ur.add(c.r)
                break
    if not new_sel:
        return sel
    new_sel = expand(gs, new_sel, ddl, 4, 50, 80)
    if better(value(gs, new_sel), value(gs, sel)):
        return new_sel
    return sel

def lns(gs, sel, ddl, rng):
    sel = copy(sel)
    if not sel:
        return sel
    items = list(sel.items())
    if not items:
        return sel
    # destroy a random subset of groups and rebuild using expand
    destroy = []
    for gi, rs in items:
        if rng.random() < 0.3:
            destroy.append(gi)
    if not destroy:
        destroy = [items[0][0]]
    new_sel = {}
    um, ur = 0, set()
    for gi, rs in sel.items():
        if gi in destroy:
            continue
        new_sel[gi] = rs
        um |= gs[gi].mask
        ur.update(rs)
    new_sel = expand(gs, new_sel, ddl, 4, 100, 100)
    if better(value(gs, new_sel), value(gs, sel)):
        return new_sel
    return sel

def pair_kick(gs, sel, ddl, topn=12, cand_lim=3):
    sel = copy(sel)
    if not sel:
        return sel
    items = list(sel.items())
    items.sort(key=lambda x: gcost(gs, x[0], x[1]) / gs[x[0]].n, reverse=True)
    top = items[:min(topn, len(items))]
    for gi, rs in top:
        if time.perf_counter() >= ddl:
            break
        g = gs[gi]
        base = gcost(gs, gi, rs)
        best = None
        for c in g.cs[:cand_lim * 30]:
            if c.r in rs:
                continue
            nr = [c.r]
            d = gcost(gs, gi, nr) - base
            if d < -1e-6 and (best is None or d < best[0]):
                best = (d, gi, c.r)
        if best:
            _, gi, r = best
            sel[gi] = [r]
    return sel

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
    while time.perf_counter() < ddl:
        t = 0.01 + 1.8 * max(0.0, (ddl - time.perf_counter()) / span)
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

def legacy_search(gs, cs, ddl):
    rng = random.Random(20260604 + len(gs) * 17 + len(cs))
    best = {}
    bv = None

    def keep(s):
        nonlocal best, bv
        if not s:
            return
        v = value(gs, s)
        if better(v, bv):
            best, bv = copy(s), v

    for m in (0, 2, 4, 1, 3):
        if time.perf_counter() >= ddl:
            break
        if best and ddl - time.perf_counter() < 3.0:
            break
        s = greedy(gs, cs, m, ddl)
        s = pair_replace(gs, cs, s if s else {}, ddl, 300, 3)
        # polish equivalent
        if time.perf_counter() < ddl:
            s = single_replace(gs, cs, s if s else {}, ddl, 5000, 1)
        if time.perf_counter() < ddl:
            s = expand(gs, s if s else {}, ddl, 4, 50, 80)
        if time.perf_counter() < ddl:
            s = reassign(gs, s if s else {}, ddl, 2, 80)
        if time.perf_counter() < ddl:
            s = move_extra(gs, s if s else {}, ddl, 50)
        if time.perf_counter() < ddl:
            s = swap_riders(gs, s if s else {}, ddl, 50)
        keep(s)

    if best and ddl - time.perf_counter() > 2.2:
        s = lns(gs, best, min(ddl, time.perf_counter() + 0.55), rng)
        keep(s)
    if best and time.perf_counter() < ddl:
        short = min(ddl, time.perf_counter() + 0.45)
        s = expand(gs, best, short, 6, 200, 120)
        s = reassign(gs, s if s else best, short, 2, 120)
        s = move_extra(gs, s if s else best, short, 90)
        s = swap_riders(gs, s if s else best, short, 90)
        keep(s)

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
            s = swap_riders(gs, s if s else best, min(ddl, time.perf_counter() + 0.05), 40)
        keep(s)
    return best if best else greedy(gs, cs, 0, ddl)

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

    low_case = avgp < 0.22
    sgp = []
    for g in gs:
        if g.n == 1 and g.cs:
            sgp.append(max(c.p for c in g.cs))
    best1 = sum(sgp) / len(sgp) if sgp else 1.0

    if low_case and nt >= 10 and time.perf_counter() < ddl:
        lddl = min(ddl, start + max(0.8, (ddl - start) * 0.20))
        est = int(nt - float(nr) / 4.35 + 0.5)
        ks = []
        for d in [0] + [x for y in range(1, 5) for x in (-y, y)]:
            k = est + d
            if 0 <= k <= nt // 2 and k not in ks:
                ks.append(k)
        cache = {}
        for k in [est, nt // 2, max(0, nt // 2 - 1)]:
            if time.perf_counter() >= lddl:
                break
            if 0 <= k <= nt // 2:
                s = pair_plan_seed(gs, rids, k, lddl, cache, True)
                keep(s)
        if time.perf_counter() < lddl:
            s = greedy(gs, cs, 3, lddl)
            keep(s)

    if nr >= int(nt * 1.45) and nt >= 22 and time.perf_counter() < ddl:
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
            if sm and gain_over(gs, sm, s) > 0.35:
                keep(sm)
        base = sm if sm and gain_over(gs, sm, s) > 0.75 else s
        if base and time.perf_counter() < send:
            keep(anneal_steps(gs, base, send, random.Random(27), 100000))
            if time.perf_counter() < send:
                keep(anneal_steps(gs, base, send, random.Random(35), 180000))
            if time.perf_counter() < send:
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
        s = greedy(gs, cs, 0, ddl)
        keep(s)
        if time.perf_counter() < ddl:
            s = expand(gs, s if s else {}, ddl, 3, 30, 80)
            keep(s)
        if time.perf_counter() < ddl:
            s = reassign(gs, s if s else {}, ddl, 1, 80)
            keep(s)

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
            s = scarce_lns(gs, s if s else best_box[0], ddl)
            if time.perf_counter() < ddl:
                s = pair2_swap(gs, s if s else best_box[0], ddl)
            keep(s)
    if best_box[0] is not None and time.perf_counter() < ddl:
        s = cycle_riders(gs, best_box[0], ddl)
        if time.perf_counter() < ddl:
            s = path_riders(gs, s if s else best_box[0], ddl)
        if time.perf_counter() < ddl:
            s = cycle_riders(gs, s if s else best_box[0], ddl, 20)
        if time.perf_counter() < ddl:
            s = config_polish(gs, s if s else best_box[0], min(ddl, time.perf_counter() + 0.10), 6, 2)
        keep(s)

    if best_box[0] is None:
        return greedy(gs, cs, 0, ddl)
    return best_box[0]


def solve(input_text):
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
        sel = search(gs, rids, cs, ddl)
        if sel is None:
            sel = {}
        if not sel:
            sel = greedy(gs, cs, 0, time.perf_counter() + 0.1)
        result = clean(gs, rids, sel)
        if not result:
            for g in gs:
                if g.cs:
                    c = g.cs[0]
                    if c.r < len(rids):
                        result = [(g.key, [rids[c.r]])]
                        break
        return result
    except Exception:
        try:
            sel = greedy(gs, cs, 0, time.perf_counter() + 0.1)
            return clean(gs, rids, sel)
        except Exception:
            return []
