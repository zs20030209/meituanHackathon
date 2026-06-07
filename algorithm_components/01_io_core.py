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

# ---- component: C ----
class C:
    __slots__ = ("g", "r", "s", "p")
    def __init__(self, g, r, s, p):
        self.g = g
        self.r = r
        self.s = s
        self.p = p

# ---- component: G ----
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

# ---- component: parse ----
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

# ---- component: copy ----
def copy(sel):
    return {g: list(rs) for g, rs in sel.items() if rs}

# ---- component: clean ----
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

# ---- component: solve ----
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
