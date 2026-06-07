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

# ---- component: better ----
def better(a, b):
    return b is None or a[0] > b[0] or (a[0] == b[0] and a[1] < b[1] - EPS)

# ---- component: gcost_cands ----
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

# ---- component: gcost ----
def gcost(gs, gi, rs, pen=PEN):
    g = gs[gi]
    return gcost_cands(g, [g.by[r] for r in rs if r in g.by], pen)

# ---- component: single ----
def single(g, c, pen=PEN):
    return c.p * c.s + pen * g.n * (1.0 - c.p)

# ---- component: value ----
def value(gs, sel):
    cov, cost = 0, 0.0
    for gi, rs in sel.items():
        if rs:
            cov += gs[gi].n
            cost += gcost(gs, gi, rs)
    return cov, cost

# ---- component: gain_over ----
def gain_over(gs, cand, base):
    if not cand:
        return -1e100
    cv = value(gs, cand)
    bv = value(gs, base)
    if cv[0] != bv[0]:
        return (cv[0] - bv[0]) * 100000.0 + (bv[1] - cv[1])
    return bv[1] - cv[1]

# ---- component: task_count ----
def task_count(gs):
    m = 0
    for g in gs:
        m |= g.mask
    return pc(m)

# ---- component: stats ----
def stats(gs, sel):
    m, ur, cost = 0, set(), 0.0
    for gi, rs in sel.items():
        m |= gs[gi].mask
        ur.update(rs)
        cost += gcost(gs, gi, rs)
    return m, ur, (pc(m), cost)
