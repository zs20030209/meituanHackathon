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

# ---- component: task_maps ----
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

# ---- component: approx_k_cost ----
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

# ---- component: choose_pairs ----
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

# ---- component: assign_first_mcf ----
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

# ---- component: skeleton_mcf_search ----
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
