from __future__ import annotations

import argparse
import ast
import copy as _copy
import datetime as _dt
import io
import json
import math
import os
import py_compile
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PENALTY = 100.0
EPS = 1e-9
DEFAULT_TIMEOUT = 10.0
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_API_KEY_FILE = "deepseek_api_key.txt"
DEFAULT_REFERENCE_SOLVER = ""


def elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:8.2f}s"


def log(args: argparse.Namespace, start: float, message: str, *, verbose_only: bool = False, error: bool = False) -> None:
    if getattr(args, "quiet", False):
        return
    if verbose_only and not getattr(args, "verbose", False):
        return
    if error:
        message = f"ERROR: {message}"
    stream = sys.stdout
    print(f"[{elapsed(start)}] {message}", file=stream, flush=True)


@dataclass
class Cand:
    group_key: str
    courier_id: str
    score: float
    willingness: float


@dataclass
class Group:
    key: str
    tasks: tuple[str, ...]
    by_courier: dict[str, Cand] = field(default_factory=dict)


@dataclass
class CaseData:
    name: str
    text: str
    groups: dict[str, Group]
    tasks: set[str]
    couriers: set[str]
    candidate_count: int
    candidates: list[Cand]


@dataclass
class EvalCase:
    name: str
    text: str | None = None
    path: Path | None = None
    data: CaseData | None = None
    expect_empty: bool = False
    non_string: bool = False
    primary: bool = False


@dataclass
class CaseResult:
    name: str
    ok: bool
    duration: float
    covered: int = 0
    cost: float = math.inf
    raw_cost: float = math.inf
    missing_count: int = 0
    missing_penalty: float = 0.0
    task_count: int = 0
    complete: bool = False
    completion_rate: float = 0.0
    rows: int = 0
    stdout: str = ""
    stderr: str = ""
    error: str = ""

    def value(self) -> tuple[int, float]:
        return self.covered, self.cost


@dataclass
class SolverEval:
    solver_path: Path
    ok: bool
    results: list[CaseResult]
    primary: CaseResult | None
    error: str = ""

    def primary_value(self) -> tuple[int, float]:
        if not self.primary:
            return 0, math.inf
        return self.primary.value()

    def aggregate_value(self) -> tuple[int, float]:
        if not self.ok:
            return 0, math.inf
        return sum(r.covered for r in self.results), sum(r.cost for r in self.results)


@dataclass
class ComponentUnit:
    file_name: str
    name: str
    kind: str
    category: str
    signature: str
    source: str


def canonical_key(raw: str) -> tuple[str, tuple[str, ...]]:
    tasks = tuple(t.strip() for t in raw.split(",") if t.strip())
    return ",".join(tasks), tasks


def parse_case(text: str, name: str = "case") -> CaseData:
    groups: dict[str, Group] = {}
    tasks: set[str] = set()
    couriers: set[str] = set()
    candidates: list[Cand] = []
    lines = text.strip().splitlines()
    if not lines:
        return CaseData(name, text, groups, tasks, couriers, 0, candidates)
    start = 1 if lines[0].lower().startswith("task_id_list") else 0
    best: dict[tuple[str, str], tuple[float, float]] = {}
    group_tasks: dict[str, tuple[str, ...]] = {}
    for line in lines[start:]:
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        key, ts = canonical_key(parts[0])
        courier_id = parts[1].strip()
        if not key or not courier_id:
            continue
        try:
            score = float(parts[2])
            willingness = float(parts[3])
        except Exception:
            continue
        if not math.isfinite(score):
            continue
        if not math.isfinite(willingness):
            willingness = 0.0
        willingness = min(1.0, max(0.0, willingness))
        old = best.get((key, courier_id))
        if old is None or score < old[0] - EPS:
            best[(key, courier_id)] = (score, willingness)
            group_tasks[key] = ts
    for (key, courier_id), (score, willingness) in best.items():
        ts = group_tasks[key]
        group = groups.setdefault(key, Group(key=key, tasks=ts))
        cand = Cand(key, courier_id, score, willingness)
        group.by_courier[courier_id] = cand
        candidates.append(cand)
        tasks.update(ts)
        couriers.add(courier_id)
    return CaseData(name, text, groups, tasks, couriers, len(candidates), candidates)


def group_cost(group: Group, couriers: list[str]) -> float:
    cands = [group.by_courier[c] for c in couriers if c in group.by_courier]
    if not cands:
        return PENALTY * len(group.tasks)
    reject = 1.0
    weighted_score = 0.0
    total_will = 0.0
    for cand in cands:
        reject *= 1.0 - cand.willingness
        total_will += cand.willingness
        weighted_score += cand.willingness * cand.score
    accept = 1.0 - reject
    if total_will <= 1e-12:
        expected_score = min(c.score for c in cands) * accept
    else:
        expected_score = accept * weighted_score / total_will
    return expected_score + PENALTY * len(group.tasks) * reject


def evaluate_solution(data: CaseData, result: Any) -> tuple[bool, int, float, list[str]]:
    errors: list[str] = []
    if not isinstance(result, list):
        return False, 0, math.inf, ["solve() did not return a list"]
    used_tasks: set[str] = set()
    used_couriers: set[str] = set()
    seen_groups: set[str] = set()
    covered = 0
    cost = 0.0
    for i, row in enumerate(result):
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            errors.append(f"row {i} is not a pair")
            continue
        raw_key, raw_couriers = row
        if not isinstance(raw_key, str):
            errors.append(f"row {i} task key is not a string")
            continue
        key, tasks = canonical_key(raw_key)
        if key in seen_groups:
            errors.append(f"group repeated: {key}")
            continue
        seen_groups.add(key)
        group = data.groups.get(key)
        if group is None:
            errors.append(f"group missing from input: {key}")
            continue
        if not isinstance(raw_couriers, list):
            errors.append(f"couriers for {key} are not a list")
            continue
        if not raw_couriers:
            errors.append(f"group {key} has no couriers")
            continue
        valid_couriers: list[str] = []
        for task_id in tasks:
            if task_id in used_tasks:
                errors.append(f"task repeated: {task_id}")
            used_tasks.add(task_id)
        for courier_id in raw_couriers:
            if not isinstance(courier_id, str):
                errors.append(f"non-string courier in {key}")
                continue
            if courier_id in used_couriers:
                errors.append(f"courier repeated: {courier_id}")
            if courier_id not in group.by_courier:
                errors.append(f"missing input pair: ({key}, {courier_id})")
            used_couriers.add(courier_id)
            valid_couriers.append(courier_id)
        covered += len(tasks)
        cost += group_cost(group, valid_couriers)
    return not errors, covered, cost, errors


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def profile_case(data: CaseData) -> dict[str, Any]:
    task_count = len(data.tasks)
    group_count = len(data.groups)
    courier_count = len(data.couriers)
    single_groups = sum(1 for g in data.groups.values() if len(g.tasks) == 1)
    pair_groups = sum(1 for g in data.groups.values() if len(g.tasks) == 2)
    multi_groups = group_count - single_groups - pair_groups
    scores = [c.score for c in data.candidates]
    wills = [c.willingness for c in data.candidates]
    single_best_wills: list[float] = []
    for group in data.groups.values():
        if len(group.tasks) == 1 and group.by_courier:
            single_best_wills.append(max(c.willingness for c in group.by_courier.values()))
    avg_will = sum(wills) / len(wills) if wills else 0.0
    best_single_will = sum(single_best_wills) / len(single_best_wills) if single_best_wills else 1.0
    rider_ratio = courier_count / max(1, task_count)
    candidate_density = data.candidate_count / max(1, group_count)
    if task_count <= 8:
        size_label = "tiny"
    elif task_count <= 15:
        size_label = "small"
    elif task_count <= 35:
        size_label = "medium"
    else:
        size_label = "large"
    labels = [size_label]
    if avg_will < 0.22 or best_single_will < 0.55:
        labels.append("low_willingness")
    if rider_ratio < 0.85:
        labels.append("scarce_couriers")
    elif rider_ratio < 1.45:
        labels.append("tight_couriers")
    else:
        labels.append("rider_rich")
    if group_count and pair_groups / group_count >= 0.5:
        labels.append("pair_heavy")
    if multi_groups:
        labels.append("multi_bundle_available")
    return {
        "name": data.name,
        "labels": labels,
        "task_count": task_count,
        "group_count": group_count,
        "courier_count": courier_count,
        "candidate_count": data.candidate_count,
        "single_group_count": single_groups,
        "pair_group_count": pair_groups,
        "multi_group_count": multi_groups,
        "max_group_size": max((len(g.tasks) for g in data.groups.values()), default=0),
        "avg_willingness": round(avg_will, 6),
        "best_single_willingness_avg": round(best_single_will, 6),
        "rider_ratio": round(rider_ratio, 6),
        "candidate_density": round(candidate_density, 6),
        "score_min": percentile(scores, 0.0),
        "score_p25": percentile(scores, 0.25),
        "score_median": percentile(scores, 0.5),
        "score_p75": percentile(scores, 0.75),
        "score_max": percentile(scores, 1.0),
    }


RUNNER_CODE = r"""
import contextlib
import importlib.util
import io
import json
import sys
import traceback

solver_path = sys.argv[1]
mode = sys.argv[2]
arg = sys.argv[3] if len(sys.argv) > 3 else ""
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        spec = importlib.util.spec_from_file_location("candidate_solver", solver_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if mode == "nonstring":
            result = module.solve(None)
        else:
            with open(arg, "r", encoding="utf-8") as f:
                text = f.read()
            result = module.solve(text)
    payload = {"ok": True, "result": result, "stdout": buf.getvalue()}
except BaseException as exc:
    payload = {
        "ok": False,
        "error": repr(exc),
        "stdout": buf.getvalue(),
        "traceback": traceback.format_exc(limit=8),
    }
print(json.dumps(payload, ensure_ascii=False))
"""


def sanitized_env() -> dict[str, str]:
    keep = ("SystemRoot", "WINDIR", "PATH", "TEMP", "TMP")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def python_command(raw: str | None) -> list[str]:
    if not raw:
        return [sys.executable]
    raw = str(raw).strip()
    if not raw:
        return [sys.executable]
    path = Path(raw.strip("\"'"))
    if path.exists():
        return [str(path)]
    try:
        parts = shlex.split(raw, posix=False)
    except ValueError:
        parts = raw.split()
    return [p.strip("\"'") for p in parts] or [sys.executable]


def compile_with_python(solver_path: Path, judge_python: str | None, timeout: float) -> tuple[bool, str]:
    cmd = python_command(judge_python)
    if cmd == [sys.executable]:
        cfile = ""
        try:
            fd, cfile = tempfile.mkstemp(prefix="autosolver_compile_", suffix=".pyc")
            os.close(fd)
            py_compile.compile(str(solver_path), cfile=cfile, doraise=True)
            return True, ""
        except py_compile.PyCompileError as exc:
            return False, str(exc)
        finally:
            if cfile:
                try:
                    os.unlink(cfile)
                except OSError:
                    pass
    compile_code = (
        "import os,py_compile,sys,tempfile\n"
        "fd,cfile=tempfile.mkstemp(prefix='autosolver_compile_',suffix='.pyc')\n"
        "os.close(fd)\n"
        "try:\n"
        "    py_compile.compile(sys.argv[1], cfile=cfile, doraise=True)\n"
        "finally:\n"
        "    try:\n"
        "        os.unlink(cfile)\n"
        "    except OSError:\n"
        "        pass\n"
    )
    try:
        proc = subprocess.run(
            cmd + ["-c", compile_code, str(solver_path)],
            cwd=str(Path.cwd()),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=max(5.0, min(timeout, 10.0)),
            env=sanitized_env(),
        )
    except OSError as exc:
        return False, f"judge python not found: {exc}"
    except subprocess.TimeoutExpired:
        return False, "judge python py_compile timed out"
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or f"py_compile exited {proc.returncode}").strip()
    return True, ""


def write_case_file(case: EvalCase, case_dir: Path) -> Path | None:
    if case.non_string:
        return None
    if case.path:
        return case.path
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / f"{case.name}.tsv"
    path.write_text(case.text or "", encoding="utf-8")
    case.path = path
    return path


def run_solver_case(solver_path: Path, case: EvalCase, case_dir: Path, timeout: float, judge_python: str | None = None) -> CaseResult:
    start = time.perf_counter()
    cmd = python_command(judge_python)
    try:
        if case.non_string:
            args = cmd + ["-c", RUNNER_CODE, str(solver_path), "nonstring", ""]
        else:
            case_path = write_case_file(case, case_dir)
            args = cmd + ["-c", RUNNER_CODE, str(solver_path), "file", str(case_path)]
        proc = subprocess.run(
            args,
            cwd=str(Path.cwd()),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout + 2.0,
            env=sanitized_env(),
        )
    except OSError as exc:
        return CaseResult(case.name, False, 0.0, error=f"judge python not found: {exc}")
    except subprocess.TimeoutExpired:
        return CaseResult(case.name, False, timeout + 2.0, error=f"timeout over {timeout:.2f}s")
    duration = time.perf_counter() - start
    if duration > timeout + EPS:
        return CaseResult(case.name, False, duration, error=f"runtime {duration:.3f}s exceeds {timeout:.3f}s")
    if proc.returncode != 0:
        return CaseResult(case.name, False, duration, stderr=proc.stderr, error=f"process exited {proc.returncode}")
    try:
        payload = json.loads(proc.stdout)
    except Exception:
        return CaseResult(case.name, False, duration, stdout=proc.stdout, stderr=proc.stderr, error="runner did not return JSON")
    captured_stdout = payload.get("stdout") or ""
    if proc.stderr.strip():
        return CaseResult(case.name, False, duration, stdout=captured_stdout, stderr=proc.stderr, error="solver wrote stderr")
    if captured_stdout.strip():
        return CaseResult(case.name, False, duration, stdout=captured_stdout, error="solver wrote stdout")
    if not payload.get("ok"):
        return CaseResult(case.name, False, duration, error=payload.get("error") or payload.get("traceback") or "solver failed")
    result = payload.get("result")
    if case.expect_empty:
        if result != []:
            return CaseResult(case.name, False, duration, rows=len(result) if isinstance(result, list) else 0, error="expected []")
        return CaseResult(case.name, True, duration, rows=0, covered=0, cost=0.0, raw_cost=0.0)
    if case.data is None:
        return CaseResult(case.name, True, duration, rows=len(result) if isinstance(result, list) else 0)
    ok, covered, cost, errors = evaluate_solution(case.data, result)
    task_total = len(case.data.tasks)
    missing = max(0, task_total - covered)
    missing_penalty = PENALTY * missing
    penalized_cost = cost + missing_penalty if math.isfinite(cost) else cost
    return CaseResult(
        case.name,
        ok,
        duration,
        covered=covered,
        cost=penalized_cost,
        raw_cost=cost,
        missing_count=missing,
        missing_penalty=missing_penalty,
        rows=len(result) if isinstance(result, list) else 0,
        error="; ".join(errors[:6]),
    )


def evaluate_solver(solver_path: Path, cases: list[EvalCase], case_dir: Path, timeout: float, judge_python: str | None = None) -> SolverEval:
    ok, error = compile_with_python(solver_path, judge_python, timeout)
    if not ok:
        return SolverEval(solver_path, False, [], None, error=error)
    results: list[CaseResult] = []
    for case in cases:
        result = run_solver_case(solver_path, case, case_dir, timeout, judge_python)
        if case.data is not None:
            result.task_count = len(case.data.tasks)
            if result.task_count > 0:
                result.completion_rate = max(0.0, min(1.0, result.covered / float(result.task_count)))
                result.complete = result.ok and result.covered >= result.task_count
            else:
                result.complete = result.ok
                result.completion_rate = 1.0 if result.ok else 0.0
        results.append(result)
    primary = next((r for r in results if any(c.primary and c.name == r.name for c in cases)), results[0] if results else None)
    ok = all(r.ok for r in results)
    return SolverEval(solver_path, ok, results, primary)


ALLOWED_IMPORT_ROOTS = {
    "array",
    "bisect",
    "collections",
    "copy",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "random",
    "statistics",
    "time",
    "typing",
    "ortools",
}

BANNED_IMPORT_ROOTS = {
    "hashlib",
    "http",
    "importlib",
    "marshal",
    "multiprocessing",
    "openai",
    "os",
    "pathlib",
    "pickle",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "tempfile",
    "urllib",
}

BANNED_CALL_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
    "print",
}

LIKELY_HARDCODE_RE = re.compile(r"\b[TC]\d{3,}\b")
MAX_SOLVER_BYTES = 100 * 1024
REFERENCE_REQUIRED_DEFS = (
    "generated_style_seed",
    "skeleton_mcf_search",
    "stochastic_pair_plan_seed",
    "bundle_plan_seed",
    "config_polish",
    "pair_kick",
)
REFERENCE_LIGHTWEIGHT_BYTES = 45000
PY36_BIT_COUNT_HELPER = """def pc(x):
    try:
        return x.bit_count()
    except AttributeError:
        c = 0
        while x:
            x &= x - 1
            c += 1
        return c
"""
PY36_UNSUPPORTED_DOTTED = {
    "functools.cache",
    "functools.cached_property",
    "itertools.pairwise",
    "math.comb",
    "math.dist",
    "math.isqrt",
    "math.perm",
    "math.prod",
    "statistics.fmean",
    "time.time_ns",
}
PY36_UNSUPPORTED_ATTRS = {
    "bit_count",
    "removeprefix",
    "removesuffix",
}
PY36_BUILTIN_GENERIC_NAMES = {"dict", "frozenset", "list", "set", "tuple", "type"}


def normalize_solver_source(source: str) -> str:
    """Make common LLM output shapes safer for the Python 3.6 judge."""
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("from __future__ import") and "annotations" in stripped:
            continue
        lines.append(line)
    source = "\n".join(lines).strip() + "\n"
    source = source.replace("def pc(x):\n    return x.bit_count()\n", PY36_BIT_COUNT_HELPER)
    return source


def solver_size_bytes(source: str) -> int:
    return len(source.encode("utf-8"))


def write_solver_source(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(normalize_solver_source(source).encode("utf-8"))


def parse_python36(source: str) -> tuple[ast.AST | None, list[str]]:
    try:
        return ast.parse(source, feature_version=(3, 6)), []
    except TypeError:
        try:
            return ast.parse(source), []
        except SyntaxError as exc:
            return None, [f"syntax error: {exc}"]
    except SyntaxError as exc:
        return None, [f"Python 3.6 syntax error: {exc}"]


def annotation_nodes(node: ast.AST) -> list[ast.AST]:
    anns: list[ast.AST] = []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        all_args = list(getattr(args, "posonlyargs", [])) + list(args.args) + list(args.kwonlyargs)
        if args.vararg:
            all_args.append(args.vararg)
        if args.kwarg:
            all_args.append(args.kwarg)
        anns.extend(arg.annotation for arg in all_args if arg.annotation is not None)
        if node.returns is not None:
            anns.append(node.returns)
    elif isinstance(node, ast.AnnAssign):
        anns.append(node.annotation)
    return anns


def annotation_is_py36_unsafe(annotation: ast.AST) -> bool:
    for node in ast.walk(annotation):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return True
        if isinstance(node, ast.Subscript):
            value = node.value
            if isinstance(value, ast.Name) and value.id in PY36_BUILTIN_GENERIC_NAMES:
                return True
    return False


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def string_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Str):
        return node.s
    return None


def is_hasattr_int_bit_count(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Name) or node.func.id != "hasattr":
        return False
    if len(node.args) != 2:
        return False
    first = node.args[0]
    return isinstance(first, ast.Name) and first.id == "int" and string_literal(node.args[1]) == "bit_count"


def under_bit_count_feature_gate(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    current = node
    while current in parents:
        previous = current
        current = parents[current]
        if isinstance(current, ast.If) and is_hasattr_int_bit_count(current.test):
            return previous in current.body
    return False


def static_guard(source: str) -> list[str]:
    errors: list[str] = []
    source = normalize_solver_source(source)
    byte_count = solver_size_bytes(source)
    if byte_count > MAX_SOLVER_BYTES:
        errors.append(f"solver file exceeds 100KB limit: {byte_count} bytes")
    guard_source = source.replace(PY36_BIT_COUNT_HELPER, "def pc(x):\n    return 0\n")
    tree, parse_errors = parse_python36(guard_source)
    if parse_errors:
        return parse_errors + errors
    if tree is None:
        return errors or ["syntax error"]
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    solve_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "solve"]
    if not solve_defs:
        errors.append("missing solve(input_text: str) function")
    else:
        args = solve_defs[0].args
        if len(args.args) != 1 or args.args[0].arg != "input_text" or args.vararg or args.kwarg:
            errors.append("solve signature must be solve(input_text)")
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            errors.append("top-level function call is not allowed")
    for node in ast.walk(tree):
        for annotation in annotation_nodes(node):
            if annotation_is_py36_unsafe(annotation):
                errors.append("Python 3.6 incompatible annotation; use plain list/dict or typing.List/Dict")
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root == "dataclasses":
                    errors.append("Python 3.6 incompatible import: dataclasses")
                if root in BANNED_IMPORT_ROOTS or root not in ALLOWED_IMPORT_ROOTS:
                    errors.append(f"disallowed import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            if module == "__future__" and any(alias.name == "annotations" for alias in node.names):
                errors.append("Python 3.6 incompatible future import: annotations")
            if root == "dataclasses":
                errors.append("Python 3.6 incompatible import: dataclasses")
            if root in BANNED_IMPORT_ROOTS or root not in ALLOWED_IMPORT_ROOTS:
                errors.append(f"disallowed import: {module}")
        elif isinstance(node, ast.Call):
            name = dotted_name(node.func)
            root = name.split(".")[0]
            guarded_bit_count = name.endswith(".bit_count") and under_bit_count_feature_gate(node, parents)
            if name in PY36_UNSUPPORTED_DOTTED and not guarded_bit_count:
                errors.append(f"Python 3.6 incompatible call: {name}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in PY36_UNSUPPORTED_ATTRS and not guarded_bit_count:
                errors.append(f"Python 3.6 incompatible method: {node.func.attr}()")
            if name in BANNED_CALL_NAMES or root in BANNED_CALL_NAMES:
                errors.append(f"disallowed call: {name}")
        elif isinstance(node, ast.Attribute):
            name = dotted_name(node)
            guarded_bit_count = name.endswith(".bit_count") and under_bit_count_feature_gate(node, parents)
            if name in PY36_UNSUPPORTED_DOTTED and not guarded_bit_count:
                errors.append(f"Python 3.6 incompatible attribute: {name}")
            if node.attr in PY36_UNSUPPORTED_ATTRS and not guarded_bit_count:
                errors.append(f"Python 3.6 incompatible method: {node.attr}()")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for deco in getattr(node, "decorator_list", []):
                name = dotted_name(deco)
                if name in {"dataclass", "dataclasses.dataclass", "cached_property", "functools.cached_property"}:
                    errors.append(f"Python 3.6 incompatible decorator: {name}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value
            low = s.lower()
            if LIKELY_HARDCODE_RE.search(s) or "large_seed" in low or "fingerprint" in low:
                errors.append("likely hardcoded case identifier in string literal")
    return sorted(set(errors))


def reference_guard_applies(target_profiles: list[dict[str, Any]]) -> bool:
    if not target_profiles:
        return False
    labels: set[str] = set()
    max_tasks = 0
    for prof in target_profiles:
        max_tasks = max(max_tasks, int(prof.get("task_count") or 0))
        labels.update(str(label) for label in prof.get("labels") or [])
    if len(target_profiles) >= 4 and max_tasks >= 30:
        return True
    return bool(labels.intersection({"large", "low_willingness", "scarce_couriers"}))


def has_function_def(source: str, name: str) -> bool:
    pattern = r"(?m)^\s*def\s+" + re.escape(name) + r"\s*\("
    return re.search(pattern, source) is not None


def reference_architecture_guard(source: str, target_profiles: list[dict[str, Any]]) -> list[str]:
    if not reference_guard_applies(target_profiles):
        return []
    source = normalize_solver_source(source)
    missing = [name for name in REFERENCE_REQUIRED_DEFS if not has_function_def(source, name)]
    if not missing:
        return []
    errors = ["missing reference architecture components: " + ", ".join(missing)]
    byte_count = solver_size_bytes(source)
    if byte_count < REFERENCE_LIGHTWEIGHT_BYTES:
        errors.append(
            f"candidate looks like a lightweight rewrite ({byte_count} bytes); preserve the 701.97-style portfolio before optimizing"
        )
    return errors


def read_api_key(api_key_file: str) -> str:
    path = Path(api_key_file)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key = line.strip()
            if key:
                return key
    env_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if env_key:
        return env_key
    raise RuntimeError(f"DeepSeek API key not found. Put it in {api_key_file} or set DEEPSEEK_API_KEY.")


def ensure_openai_sdk() -> None:
    try:
        from openai import OpenAI  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("OpenAI SDK is required. Install it with: pip install openai") from exc


def call_llm(
    messages: list[dict[str, str]],
    timeout: float,
    api_key_file: str,
    model: str,
    base_url: str,
    reasoning_effort: str,
    thinking_type: str,
    max_tokens: int,
) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=read_api_key(api_key_file), base_url=base_url, timeout=timeout, max_retries=0)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if thinking_type:
        kwargs["extra_body"] = {"thinking": {"type": thinking_type}}
    if max_tokens > 0:
        kwargs["max_tokens"] = max_tokens
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def call_llm_with_progress(
    args: argparse.Namespace,
    run_start: float,
    round_no: int,
    messages: list[dict[str, str]],
) -> str:
    if args.quiet or args.progress_interval <= 0:
        return call_llm(
            messages,
            args.llm_timeout,
            args.api_key_file,
            args.model,
            args.base_url,
            args.reasoning_effort,
            args.thinking,
            args.max_tokens,
        )

    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            result_queue.put(
                (
                    "ok",
                    call_llm(
                        messages,
                        args.llm_timeout,
                        args.api_key_file,
                        args.model,
                        args.base_url,
                        args.reasoning_effort,
                        args.thinking,
                        args.max_tokens,
                    ),
                )
            )
        except BaseException as exc:
            result_queue.put(("error", exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    call_start = time.perf_counter()
    while True:
        try:
            status, payload = result_queue.get(timeout=args.progress_interval)
        except queue.Empty:
            waited = time.perf_counter() - call_start
            log(
                args,
                run_start,
                f"Round {round_no}: still waiting for DeepSeek response ({waited:.0f}s elapsed, sdk timeout={args.llm_timeout:.0f}s)",
            )
            if waited > args.llm_timeout + args.progress_interval:
                raise RuntimeError(f"DeepSeek call exceeded visible wait limit ({waited:.1f}s)")
            continue
        if status == "ok":
            return str(payload)
        raise payload


def extract_fenced(text: str, language: str | None = None) -> str | None:
    if language:
        pattern = rf"```{language}\s*(.*?)```"
    else:
        pattern = r"```\w*\s*(.*?)```"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def extract_llm_result(content: str) -> tuple[dict[str, Any], str | None]:
    meta: dict[str, Any] = {}
    candidates = [content]
    fenced_json = extract_fenced(content, "json")
    if fenced_json:
        candidates.insert(0, fenced_json)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            meta_keys = (
                "hypothesis",
                "diagnosis",
                "selected_components",
                "target_case_type",
                "change_plan",
                "expected_case_deltas",
                "rollback_risk",
                "expected_effect",
            )
            meta = {k: obj.get(k) for k in meta_keys if k in obj}
            code = obj.get("code") or obj.get("python_code") or obj.get("solver")
            if isinstance(code, str):
                code = extract_fenced(code) or code
                return meta, code.strip()
    fenced_py = extract_fenced(content, "python") or extract_fenced(content)
    if fenced_py and "def solve" in fenced_py:
        return meta, fenced_py.strip()
    if "def solve" in content:
        return meta, content.strip()
    return meta, None


def truncate_for_prompt(text: str, max_chars: int, label: str) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep_head = max(0, int(max_chars * 0.65))
    keep_tail = max(0, max_chars - keep_head)
    return (
        text[:keep_head]
        + f"\n\n# ... {label} truncated to keep the LLM prompt small ...\n\n"
        + text[-keep_tail:]
    )


def summarize_eval(evaluation: SolverEval) -> dict[str, Any]:
    primary = evaluation.primary
    aggregate = evaluation.aggregate_value()
    return {
        "ok": evaluation.ok,
        "solver_path": str(evaluation.solver_path),
        "aggregate": {
            "covered": aggregate[0],
            "cost": aggregate[1],
        },
        "primary": {
            "name": primary.name if primary else None,
            "ok": primary.ok if primary else False,
            "covered": primary.covered if primary else 0,
            "cost": primary.cost if primary else math.inf,
            "rows": primary.rows if primary else 0,
            "duration": primary.duration if primary else 0.0,
            "error": primary.error if primary else evaluation.error,
        },
        "cases": [
            {
                "name": r.name,
                "ok": r.ok,
            "covered": r.covered,
            "cost": r.cost,
            "raw_cost": r.raw_cost,
            "missing_count": r.missing_count,
            "missing_penalty": r.missing_penalty,
            "task_count": r.task_count,
            "complete": r.complete,
                "completion_rate": r.completion_rate,
                "rows": r.rows,
                "duration": r.duration,
                "error": r.error,
            }
            for r in evaluation.results
        ],
    }


def better_value(candidate: tuple[int, float], incumbent: tuple[int, float]) -> bool:
    candidate_cost = finite_number(candidate[1])
    incumbent_cost = finite_number(incumbent[1])
    if candidate_cost is None:
        return False
    if incumbent_cost is None:
        return True
    return candidate_cost < incumbent_cost - EPS


def case_profile_key(name: Any) -> str:
    raw = str(name or "").strip()
    stem = Path(raw).stem
    return stem or raw


def profile_lookup(target_profiles: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for prof in target_profiles or []:
        key = case_profile_key(prof.get("name"))
        if key:
            out[key] = prof
    return out


def finite_number(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def rounded_number(value: Any, digits: int = 3) -> float | None:
    out = finite_number(value)
    return round(out, digits) if out is not None else None


def prompt_digest(value: Any, max_chars: int = 700) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return value[:max_chars]
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        text = str(value)
    return text[:max_chars]


def task_count_for_name(name: Any, profiles: list[dict[str, Any]] | None) -> int:
    prof = profile_lookup(profiles).get(case_profile_key(name), {})
    try:
        return int(prof.get("task_count") or 0)
    except Exception:
        return 0


def result_case_brief(result: CaseResult, profiles: list[dict[str, Any]] | None) -> dict[str, Any]:
    task_count = int(result.task_count or task_count_for_name(result.name, profiles) or 0)
    completion_rate = result.completion_rate
    if task_count > 0:
        completion_rate = max(0.0, min(1.0, result.covered / float(task_count)))
    complete = bool(result.complete or (result.ok and task_count > 0 and result.covered >= task_count))
    cost = finite_number(result.cost)
    denom = max(1, task_count or result.covered)
    cost_per_task = cost / denom if cost is not None else None
    return {
        "name": result.name,
        "ok": result.ok,
        "covered": result.covered,
        "task_count": task_count,
        "complete": complete,
        "completion_rate": round(completion_rate, 4),
        "cost": rounded_number(cost, 3),
        "raw_cost": rounded_number(result.raw_cost, 3),
        "missing_count": result.missing_count,
        "missing_penalty": rounded_number(result.missing_penalty, 3),
        "cost_per_task": rounded_number(cost_per_task, 3),
        "duration_ms": int(round(result.duration * 1000.0)),
        "rows": result.rows,
        "error": result.error,
    }


def eval_case_briefs(evaluation: SolverEval, profiles: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [result_case_brief(result, profiles) for result in evaluation.results]


def compare_evals(
    candidate: SolverEval,
    incumbent: SolverEval | None,
    profiles: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if incumbent is None or not incumbent.results:
        return {"has_incumbent": False}
    cand_value = candidate.aggregate_value()
    best_value = incumbent.aggregate_value()
    cand_cost = finite_number(cand_value[1])
    best_cost = finite_number(best_value[1])
    aggregate_delta = {
        "candidate_covered": cand_value[0],
        "best_covered": best_value[0],
        "delta_covered": cand_value[0] - best_value[0],
        "candidate_cost": rounded_number(cand_cost, 6),
        "best_cost": rounded_number(best_cost, 6),
        "delta_cost": rounded_number(cand_cost - best_cost, 6) if cand_cost is not None and best_cost is not None else None,
    }
    best_by_key = {case_profile_key(result.name): result for result in incumbent.results}
    case_deltas: list[dict[str, Any]] = []
    for result in candidate.results:
        key = case_profile_key(result.name)
        best_result = best_by_key.get(key)
        task_count = int(result.task_count or (best_result.task_count if best_result else 0) or task_count_for_name(result.name, profiles) or 0)
        cand_cost_case = finite_number(result.cost)
        best_cost_case = finite_number(best_result.cost) if best_result else None
        delta_cost = cand_cost_case - best_cost_case if cand_cost_case is not None and best_cost_case is not None else None
        cand_complete = bool(result.complete or (result.ok and task_count > 0 and result.covered >= task_count))
        best_complete = bool(best_result and (best_result.complete or (best_result.ok and task_count > 0 and best_result.covered >= task_count)))
        case_deltas.append(
            {
                "name": result.name,
                "task_count": task_count,
                "candidate_covered": result.covered,
                "best_covered": best_result.covered if best_result else 0,
                "delta_covered": result.covered - (best_result.covered if best_result else 0),
                "candidate_complete": cand_complete,
                "best_complete": best_complete,
                "candidate_missing": result.missing_count,
                "best_missing": best_result.missing_count if best_result else None,
                "candidate_cost": rounded_number(cand_cost_case, 3),
                "best_cost": rounded_number(best_cost_case, 3),
                "delta_cost": rounded_number(delta_cost, 3),
                "candidate_duration_ms": int(round(result.duration * 1000.0)),
                "best_duration_ms": int(round(best_result.duration * 1000.0)) if best_result else None,
                "candidate_error": result.error,
            }
        )

    def regression_key(row: dict[str, Any]) -> tuple[int, float]:
        delta_cost_value = finite_number(row.get("delta_cost")) or 0.0
        return (int(row.get("delta_covered") or 0), -delta_cost_value)

    def improvement_key(row: dict[str, Any]) -> tuple[int, float]:
        delta_cost_value = finite_number(row.get("delta_cost")) or 0.0
        return (int(row.get("delta_covered") or 0), -delta_cost_value)

    regressions = [
        row
        for row in case_deltas
        if int(row.get("delta_covered") or 0) < 0
        or (int(row.get("delta_covered") or 0) == 0 and (finite_number(row.get("delta_cost")) or 0.0) > EPS)
        or (row.get("best_complete") and not row.get("candidate_complete"))
    ]
    improvements = [
        row
        for row in case_deltas
        if int(row.get("delta_covered") or 0) > 0
        or (int(row.get("delta_covered") or 0) == 0 and (finite_number(row.get("delta_cost")) or 0.0) < -EPS)
    ]
    regressions.sort(key=regression_key)
    improvements.sort(key=improvement_key, reverse=True)
    behavior_duplicate = (
        int(aggregate_delta.get("delta_covered") or 0) == 0
        and abs(finite_number(aggregate_delta.get("delta_cost")) or 0.0) <= EPS
        and not regressions
        and not improvements
    )
    return {
        "has_incumbent": True,
        "aggregate_delta": aggregate_delta,
        "case_deltas": case_deltas,
        "regressions": regressions[:5],
        "improvements": improvements[:5],
        "behavior_duplicate": behavior_duplicate,
    }


def compact_record_for_prompt(record: dict[str, Any]) -> dict[str, Any]:
    evaluation = record.get("evaluation") if isinstance(record.get("evaluation"), dict) else {}
    aggregate = evaluation.get("aggregate") if isinstance(evaluation.get("aggregate"), dict) else {}
    comparison = record.get("comparison_to_best") if isinstance(record.get("comparison_to_best"), dict) else {}
    compact = {
        "round": record.get("round"),
        "status": record.get("status"),
        "accepted": bool(record.get("accepted")),
        "prompt_mode": str(record.get("prompt_mode") or "")[:120],
        "exploration_mode": str(record.get("exploration_mode") or "")[:80],
        "aggregate": {
            "covered": aggregate.get("covered"),
            "cost": rounded_number(aggregate.get("cost"), 3),
        },
        "hypothesis": str(record.get("hypothesis") or "")[:500],
        "diagnosis": prompt_digest(record.get("diagnosis"), 700),
        "selected_components": prompt_digest(record.get("selected_components"), 400),
        "target_case_type": str(record.get("target_case_type") or "")[:160],
        "change_plan": prompt_digest(record.get("change_plan"), 700),
        "expected_case_deltas": prompt_digest(record.get("expected_case_deltas"), 700),
        "rollback_risk": prompt_digest(record.get("rollback_risk"), 300),
        "reason": str(record.get("reason") or "")[:300],
    }
    if comparison:
        compact["comparison_to_best"] = {
            "aggregate_delta": comparison.get("aggregate_delta"),
            "regressions": comparison.get("regressions", [])[:3],
            "improvements": comparison.get("improvements", [])[:3],
            "behavior_duplicate": bool(comparison.get("behavior_duplicate")),
        }
    variants = record.get("variants") if isinstance(record.get("variants"), list) else []
    if variants:
        compact["variant_decision"] = record.get("variant_decision") if isinstance(record.get("variant_decision"), dict) else {}
        compact["variants"] = []
        for variant in variants[:3]:
            if not isinstance(variant, dict):
                continue
            vev = variant.get("evaluation") if isinstance(variant.get("evaluation"), dict) else {}
            vagg = vev.get("aggregate") if isinstance(vev.get("aggregate"), dict) else {}
            vcomp = variant.get("comparison_to_best") if isinstance(variant.get("comparison_to_best"), dict) else {}
            compact["variants"].append(
                {
                    "variant": variant.get("variant"),
                    "status": variant.get("status"),
                    "duel_winner": bool(variant.get("duel_winner")),
                    "prompt_mode": variant.get("prompt_mode"),
                    "aggregate": {
                        "covered": vagg.get("covered"),
                        "cost": rounded_number(vagg.get("cost"), 3),
                    },
                    "aggregate_delta": (
                        vcomp.get("aggregate_delta")
                        if isinstance(vcomp.get("aggregate_delta"), dict)
                        else None
                    ),
                    "reason": str(variant.get("reason") or "")[:220],
                }
            )
    return compact


def failure_tag(record: dict[str, Any]) -> str:
    comparison = record.get("comparison_to_best") if isinstance(record.get("comparison_to_best"), dict) else {}
    delta = comparison.get("aggregate_delta") if isinstance(comparison.get("aggregate_delta"), dict) else {}
    delta_covered = int(delta.get("delta_covered") or 0)
    delta_cost = finite_number(delta.get("delta_cost")) or 0.0
    reason = str(record.get("reason") or "").lower()
    status = str(record.get("status") or "").lower()
    if comparison.get("behavior_duplicate") or (
        comparison
        and int(delta.get("delta_covered") or 0) == 0
        and abs(delta_cost) <= EPS
        and not comparison.get("regressions")
        and not comparison.get("improvements")
    ):
        return "behavior_duplicate"
    if "disallowed import" in reason:
        return "banned_import"
    if "syntax" in reason or "nonlocal" in reason:
        return "python36_syntax"
    if "timeout" in reason or "exceeds" in reason:
        return "timeout"
    if "did not contain solver code" in reason:
        return "invalid_llm_response"
    if "reference architecture" in reason or "lightweight rewrite" in reason:
        return "missing_reference_architecture"
    if "zero covered" in reason or "empty" in reason:
        return "empty_or_zero_solution"
    if delta_covered < 0:
        return "coverage_regression"
    if delta_covered == 0 and delta_cost > EPS:
        return "cost_regression"
    if "no improvement" in reason:
        return "no_improvement"
    if status in {"failed", "rejected", "skipped"}:
        return status
    return "none"


def rounds_since_last_accept(records: list[dict[str, Any]]) -> int:
    positive = [rec for rec in records if isinstance(rec.get("round"), int) and rec.get("round", 0) > 0]
    accepted_rounds = [int(rec.get("round") or 0) for rec in positive if rec.get("accepted")]
    if not accepted_rounds:
        return 0
    last_accept = max(accepted_rounds)
    return sum(1 for rec in positive if int(rec.get("round") or 0) > last_accept)


def recent_failure_counts(records: list[dict[str, Any]], after_last_accept: bool = False) -> dict[str, int]:
    positive = [rec for rec in records if isinstance(rec.get("round"), int) and rec.get("round", 0) > 0]
    if after_last_accept:
        accepted_rounds = [int(rec.get("round") or 0) for rec in positive if rec.get("accepted")]
        if accepted_rounds:
            last_accept = max(accepted_rounds)
            positive = [rec for rec in positive if int(rec.get("round") or 0) > last_accept]
    counts: dict[str, int] = {}
    for rec in positive:
        tag = failure_tag(rec)
        if tag != "none":
            counts[tag] = counts.get(tag, 0) + 1
    return counts


def build_experiment_directive(
    best_cases: list[dict[str, Any]],
    records: list[dict[str, Any]],
    failure_counts: dict[str, int],
) -> dict[str, Any]:
    attempts_since_best = rounds_since_last_accept(records)
    post_best_failures = recent_failure_counts(records, after_last_accept=True)
    behavior_duplicates = int(post_best_failures.get("behavior_duplicate", 0))
    cost_regressions = int(post_best_failures.get("cost_regression", 0))
    coverage_regressions = int(post_best_failures.get("coverage_regression", 0))
    focus_names = [str(row.get("name")) for row in best_cases[:5]]
    exploration_mode = "refine"
    if attempts_since_best >= 2 or cost_regressions >= 2 or behavior_duplicates:
        exploration_mode = "challenger"
    if attempts_since_best >= 4 or coverage_regressions:
        exploration_mode = "radical"
    agendas = [
        {
            "name": "low_willingness_incumbent_portfolio",
            "objective": "Reduce penalized score on low-willingness profiles.",
            "profile_gate": "avg_willingness < 0.22 or best_single_willingness_avg < 0.55",
            "must_do": [
                "Keep the incumbent solution path as base.",
                "Build one alternate low-willingness candidate using a different multi-courier or pair/bundle seed, not just a sort-key tweak.",
                "Compute the same penalized proxy value for base and alternate inside solve(), then return the lower-score solution.",
            ],
            "must_not": [
                "Do not replace the global greedy mode for all profiles.",
                "Do not repeat a pure high-willingness greedy-mode change if it already caused cost regression.",
            ],
        },
        {
            "name": "scarce_courier_incumbent_portfolio",
            "objective": "Reduce penalized score on scarce-courier large profiles.",
            "profile_gate": "courier_count / task_count < 0.9 and task_count >= 30",
            "must_do": [
                "Keep the incumbent solution path as base.",
                "Try an alternate bundle or pair skeleton plus a short repair/polish pass only under the scarce-courier gate.",
                "Return min(base, alternate) by penalized proxy score; missing tasks already add penalty.",
            ],
            "must_not": [
                "Do not spend scarce-courier search time on rider-rich profiles.",
                "Do not hide missing tasks; score them with the missing-task penalty.",
            ],
        },
        {
            "name": "small_exact_or_config_cleanup",
            "objective": "Use exact or bounded config search on tiny/small profiles to lower cost without touching large-case behavior.",
            "profile_gate": "task_count <= 15",
            "must_do": [
                "Keep incumbent result as base.",
                "Run a bounded exact/config search only for tiny/small profiles and compare against base.",
                "Use tight node/time limits so larger profiles are unaffected.",
            ],
            "must_not": [
                "Do not change medium or large search schedules in this experiment.",
                "Do not add Python 3.7+ syntax.",
            ],
        },
        {
            "name": "medium_cost_polish_portfolio",
            "objective": "Lower penalized score on medium rider-rich profiles.",
            "profile_gate": "25 <= task_count <= 35 and avg_willingness >= 0.30 and courier_count / task_count >= 1.4",
            "must_do": [
                "Keep incumbent result as base.",
                "Try one bounded rider reassignment/config polish branch and return the better solution.",
                "Target penalized score; a small coverage loss can be acceptable only if the missing-task penalty is still outweighed.",
            ],
            "must_not": [
                "Do not alter low-willingness or scarce-courier branches in this experiment.",
                "Do not change parser/objective/output code.",
            ],
        },
        {
            "name": "time_budget_rebalance_without_algorithm_change",
            "objective": "When prior experiments regress, improve by reallocating time budgets among existing incumbent phases.",
            "profile_gate": "use the same profile gates already present in incumbent code",
            "must_do": [
                "Do not add a new algorithm family in this experiment.",
                "Move small amounts of time from phases that produced no improvements toward the focused high-cost profile.",
                "Keep base logic available and avoid behavior changes outside the target profile.",
            ],
            "must_not": [
                "Do not rewrite helper functions.",
                "Do not introduce another greedy sort mode.",
            ],
        },
        {
            "name": "diverse_challenger_rewrite",
            "objective": "Create a meaningfully different standalone challenger rather than another small incumbent patch.",
            "profile_gate": "all profiles; use computed profile gates inside the challenger for low_willingness, scarce_couriers, tiny/small, and rider-rich cases",
            "must_do": [
                "Use the best evaluation only as the score target; do not copy the incumbent code structure.",
                "Choose a different seed/search schedule from the last two rejected attempts.",
                "Keep all hard constraints, but allow a larger algorithmic change because the candidate will be locally rejected if its penalized score is worse.",
                "Target at least one high-cost focus case with a distinct mechanism and keep other profiles robust.",
            ],
            "must_not": [
                "Do not merely add one more sort mode or one more call to an existing failed branch.",
                "Do not hardcode case names, task IDs, courier IDs, fingerprints, or precomputed answers.",
            ],
        },
        {
            "name": "large_case_cost_challenger",
            "objective": "Attack aggregate cost through large/medium profiles instead of only low-willingness or scarce-courier tails.",
            "profile_gate": "task_count >= 30 and courier_count / task_count >= 1.3",
            "must_do": [
                "Try a distinct assignment/polish schedule for rider-rich medium/large cases.",
                "Use penalized-score internal comparison if a base branch is kept.",
                "Avoid spending most of the budget on low_willingness or scarce_courier profiles this round.",
            ],
            "must_not": [
                "Do not repeat the prior low_willingness bundle/stochastic pair attempts.",
                "Do not modify tiny/small exact behavior in this experiment.",
            ],
        },
    ]
    agenda_index = attempts_since_best % len(agendas)
    if coverage_regressions:
        agenda_index = 4
    elif exploration_mode == "radical":
        agenda_index = 5
    elif exploration_mode == "challenger" and cost_regressions >= 2:
        agenda_index = 6
    elif exploration_mode == "challenger":
        agenda_index = 5
    elif behavior_duplicates:
        agenda_index = (agenda_index + 1) % len(agendas)
    elif cost_regressions and agenda_index == 0:
        agenda_index = 1
    directive = dict(agendas[agenda_index])
    directive["exploration_mode"] = exploration_mode
    directive["available_families"] = [
        {
            "name": agenda.get("name"),
            "objective": agenda.get("objective"),
            "profile_gate": agenda.get("profile_gate"),
        }
        for agenda in agendas
    ]
    directive["selection_contract"] = {
        "role": "recommendation_not_command",
        "recommended_family": directive.get("name"),
        "exploration_pressure": exploration_mode,
        "rule": (
            "The model should choose the experiment family that best explains the recent case deltas. "
            "It may follow recommended_family or choose any available_families entry, but it must explain the choice "
            "in JSON metadata and avoid recently failed families."
        ),
    }
    directive["stagnation"] = {
        "attempts_since_best": attempts_since_best,
        "post_best_failures": post_best_failures,
        "all_recent_failures": failure_counts,
        "focused_best_cases": focus_names,
    }
    directive["global_rules"] = [
        "The next candidate must be behaviorally different from the incumbent unless the alternate branch genuinely loses after internal comparison.",
        "Prefer an incumbent-preserving portfolio: compute base, compute a targeted alternate, compare penalized score, return the lower-score one.",
        "If runtime is tight, run the alternate only under the directive profile_gate.",
        "Implementation gates must use computed profile signals, never case names or IDs.",
    ]
    return directive


def next_curriculum_stage(records: list[dict[str, Any]], best_eval: SolverEval | None) -> dict[str, Any]:
    positive_rounds = [rec for rec in records if isinstance(rec.get("round"), int) and rec.get("round", 0) > 0]
    next_round = len(positive_rounds) + 1
    stage_no = min(next_round, 5)
    stages = [
        {
            "stage": 1,
            "name": "coverage_foundation",
            "goal": "Generate a valid runnable skeleton with parsing, objective math, greedy, pair seed, and simple repair.",
            "components_to_use": ["parse", "gcost", "value", "greedy", "pair_plan_seed", "expand", "reassign", "swap_riders"],
            "expected_progress": "Get a generated baseline. It may miss a few tasks; missing tasks are already penalized in score.",
        },
        {
            "stage": 2,
            "name": "generated_style_tiny_small",
            "goal": "Unlock generated_style_seed plus tiny/small exact and repair components.",
            "components_to_use": ["generated_style_seed", "exact_cover_seed", "single_replace", "pair_replace", "repair_window"],
            "expected_progress": "Improve tiny/small and rider-rich cases while keeping the stage-1 baseline as fallback.",
        },
        {
            "stage": 3,
            "name": "scarce_skeleton_mcf",
            "goal": "Unlock bundle_plan_seed, skeleton_mcf_search, and scarce-courier branches.",
            "components_to_use": ["bundle_plan_seed", "skeleton_mcf_search", "pair2_swap", "scarce_lns"],
            "expected_progress": "Reduce scarce-courier missing penalty or cost under a profile gate.",
        },
        {
            "stage": 4,
            "name": "local_polish_anneal",
            "goal": "Unlock config_polish, pair_kick, anneal, and local enhancement components.",
            "components_to_use": ["config_polish", "pair_kick", "anneal", "anneal_steps", "cycle_riders", "path_riders"],
            "expected_progress": "Lower penalized score by grafting local improvement branches onto the incumbent.",
        },
        {
            "stage": 5,
            "name": "full_orchestration",
            "goal": "Unlock full search/solve orchestration and allow global schedule rewrites or near-miss grafts.",
            "components_to_use": ["search", "legacy_search", "solve", "next_experiment_directive", "near_misses"],
            "expected_progress": "Approach or beat the teacher-style aggregate by combining all learned branches.",
        },
    ]
    return stages[stage_no - 1]


def curriculum_stage_records(records: list[dict[str, Any]], stage_no: int) -> list[dict[str, Any]]:
    out = []
    stage_no = int(stage_no)
    prefixes = (
        f"curriculum-stage{stage_no}",
        f"adaptive-level{stage_no}",
        f"open-level{stage_no}",
    )
    for rec in records:
        if not isinstance(rec.get("round"), int) or rec.get("round", 0) <= 0:
            continue
        prompt_mode = str(rec.get("prompt_mode") or "")
        if any(prompt_mode.startswith(prefix) for prefix in prefixes):
            out.append(rec)
    return out


def curriculum_stage5_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return curriculum_stage_records(records, 5)


def curriculum_stage_intro_due(records: list[dict[str, Any]], stage_no: int, best_eval: SolverEval | None) -> bool:
    if best_eval is None or stage_no < 2:
        return False
    return not curriculum_stage_records(records, stage_no)


def orchestration_rebuild_due(records: list[dict[str, Any]], stage_no: int, best_eval: SolverEval | None) -> bool:
    if stage_no < 5 or best_eval is None:
        return False
    stage5_records = curriculum_stage5_records(records)
    rebuild_positions = [
        idx
        for idx, rec in enumerate(stage5_records)
        if "orchestration-rebuild" in str(rec.get("prompt_mode") or "")
    ]
    if not rebuild_positions:
        return True
    if any(
        rec.get("accepted")
        for rec in stage5_records
        if "orchestration-rebuild" in str(rec.get("prompt_mode") or "")
    ):
        return False
    return len(stage5_records) - rebuild_positions[-1] >= 2


def apply_stage_recompose_directive(learning_brief: dict[str, Any], records: list[dict[str, Any]], stage_no: int) -> None:
    if stage_no >= 5:
        apply_orchestration_rebuild_directive(learning_brief, records)
        return
    curriculum_stage = learning_brief.get("curriculum_stage") if isinstance(learning_brief.get("curriculum_stage"), dict) else {}
    learning_brief["exploration_mode"] = "recompose"
    learning_brief["stage_recompose"] = {
        "reason": "New curriculum components are visible. Test whether rebuilding the main search with all currently visible components beats a small incumbent graft.",
        "stage": stage_no,
        "stage_name": curriculum_stage.get("name"),
        "new_components": curriculum_stage.get("components_to_use", []),
        "best_code_policy": "Do not show incumbent source in this variant; use incumbent evaluation only as the score target.",
        "required_behavior": [
            "Synthesize a fresh main search from every component visible through this stage.",
            "Use the newly unlocked components in the primary schedule, not as dead code.",
            "Keep a simple fallback and Python 3.6 compatibility.",
            "Use profile-derived gates instead of case names, IDs, or input fingerprints.",
        ],
    }
    learning_brief["instructions"] = [
        "This is the recomposition challenger for a newly unlocked component stage.",
        "Treat the current best as a benchmark by per-case score, not as code to preserve.",
        "Use all currently visible components to rebuild the main search schedule.",
        "Optimize penalized aggregate score; coverage is diagnostic because missing tasks are already priced.",
    ]
    directive = learning_brief.get("next_experiment_directive")
    if isinstance(directive, dict):
        directive.update(
            {
                "name": f"stage{stage_no}_component_recompose",
                "objective": "Build a fresh solver from the full visible component set for this stage and compete it against the incumbent-graft variant.",
                "profile_gate": "global portfolio; internal branches may use task count, courier ratio, willingness, group-size mix, and candidate density.",
                "exploration_mode": "recompose",
                "must_do": [
                    "Use newly unlocked components in the main schedule.",
                    "Generate a full standalone solver rather than a one-function patch.",
                    "Compare seeds and repair outputs with the same penalized objective helper.",
                    "Keep robust fallback behavior for profiles where the new schedule loses.",
                ],
                "must_not": [
                    "Do not merely paste a helper that solve() never calls.",
                    "Do not make case-specific gates or constants.",
                    "Do not spend the round only on parameter nits.",
                ],
            }
        )


def apply_orchestration_rebuild_directive(learning_brief: dict[str, Any], records: list[dict[str, Any]]) -> None:
    stage5_count = len(curriculum_stage5_records(records))
    learning_brief["exploration_mode"] = "orchestration_rebuild"
    learning_brief["orchestration_unlock"] = {
        "reason": "Full search/solve orchestration is now visible. The next round should test a fresh global schedule instead of patching the lightweight incumbent.",
        "stage5_rounds_seen": stage5_count,
        "best_code_policy": "Do not show incumbent source in this prompt; use incumbent evaluation only as the score target.",
        "required_behavior": [
            "Use search/solve orchestration components as the primary architecture.",
            "Spend real time budget on medium/large/low-willingness/scarce profiles; do not collapse back to a tiny greedy portfolio.",
            "Keep Python 3.6 compatibility, fallback output, and profile-derived gates.",
            "Local evaluation may reject this challenger, so prefer a genuinely different global schedule over a cosmetic tweak.",
        ],
    }
    learning_brief["instructions"] = [
        "This is a full-orchestration assimilation round, not a small incumbent patch.",
        "Treat the current best as a benchmark by per-case score, not as code to preserve.",
        "Optimize penalized aggregate score; coverage is diagnostic because missing tasks are already priced.",
        "Use profile-derived gates, never case names, task IDs, courier IDs, or input fingerprints.",
    ]
    directive = learning_brief.get("next_experiment_directive")
    if isinstance(directive, dict):
        directive.update(
            {
                "name": "full_orchestration_rebuild",
                "objective": "Rebuild the solver around the newly unlocked full search/solve orchestration and try to jump toward the teacher-style score basin.",
                "profile_gate": "global portfolio; internal branches may use task count, courier ratio, willingness, group-size mix, and candidate density.",
                "exploration_mode": "orchestration_rebuild",
                "must_do": [
                    "Use the visible search/solve orchestration as the main schedule or recreate an equivalent global schedule.",
                    "Include an explicit multi-stage search function, not only a helper-level alternate branch.",
                    "Compare candidate seeds and local-search outputs with the same penalized objective helper.",
                    "Use larger budgets on hard profiles while preserving a simple fallback.",
                ],
                "must_not": [
                    "Do not merely add one more low_will_alternate or scarce_alternate to the incumbent.",
                    "Do not paste case-specific constants, case names, IDs, or precomputed answers.",
                    "Do not reduce the round to parameter nits or comment-only changes.",
                ],
            }
        )


def teacher_benchmark_brief(
    teacher_eval: SolverEval | None,
    best_eval: SolverEval | None,
    target_profiles: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if teacher_eval is None:
        return None
    teacher_value = teacher_eval.aggregate_value()
    brief: dict[str, Any] = {
        "role": "benchmark_only_not_incumbent",
        "aggregate_covered": teacher_value[0],
        "aggregate_cost": rounded_number(teacher_value[1], 6),
        "cases": eval_case_briefs(teacher_eval, target_profiles),
    }
    if best_eval is not None:
        comparison = compare_evals(best_eval, teacher_eval, target_profiles)
        brief["generated_best_vs_teacher"] = comparison
        gaps = []
        for row in comparison.get("case_deltas", []):
            delta_covered = int(row.get("delta_covered") or 0)
            delta_cost = finite_number(row.get("delta_cost")) or 0.0
            if delta_covered < 0 or delta_cost > EPS:
                gaps.append(row)
        brief["largest_remaining_gaps"] = sorted(
            gaps,
            key=lambda row: (int(row.get("delta_covered") or 0), finite_number(row.get("delta_cost")) or 0.0),
        )[:5]
    else:
        brief["generated_best_vs_teacher"] = None
        brief["largest_remaining_gaps"] = brief["cases"][:5]
    return brief


def build_learning_brief(
    target_profiles: list[dict[str, Any]],
    best_eval: SolverEval | None,
    records: list[dict[str, Any]],
    teacher_eval: SolverEval | None = None,
    max_records: int = 5,
) -> dict[str, Any]:
    recent_records = [rec for rec in records if isinstance(rec.get("round"), int) and rec.get("round", 0) > 0][-max_records:]
    failure_counts: dict[str, int] = {}
    recent_trials: list[dict[str, Any]] = []
    near_misses: list[dict[str, Any]] = []
    for rec in recent_records:
        tag = failure_tag(rec)
        if tag != "none":
            failure_counts[tag] = failure_counts.get(tag, 0) + 1
        compact = compact_record_for_prompt(rec)
        recent_trials.append(compact)
        comparison = rec.get("comparison_to_best") if isinstance(rec.get("comparison_to_best"), dict) else {}
        delta = comparison.get("aggregate_delta") if isinstance(comparison.get("aggregate_delta"), dict) else {}
        improvements = comparison.get("improvements") if isinstance(comparison.get("improvements"), list) else []
        if not rec.get("accepted") and improvements and int(delta.get("delta_covered") or 0) >= -2:
            near_misses.append(
                {
                    "round": rec.get("round"),
                    "aggregate_delta": delta,
                    "useful_case_improvements": improvements[:3],
                    "why_rejected": str(rec.get("reason") or "")[:240],
                }
            )
        variants = rec.get("variants") if isinstance(rec.get("variants"), list) else []
        for variant in variants:
            if not isinstance(variant, dict) or variant.get("duel_winner"):
                continue
            vcomparison = variant.get("comparison_to_best") if isinstance(variant.get("comparison_to_best"), dict) else {}
            vdelta = vcomparison.get("aggregate_delta") if isinstance(vcomparison.get("aggregate_delta"), dict) else {}
            vimprovements = vcomparison.get("improvements") if isinstance(vcomparison.get("improvements"), list) else []
            if vimprovements and int(vdelta.get("delta_covered") or 0) >= -2:
                near_misses.append(
                    {
                        "round": rec.get("round"),
                        "variant": variant.get("variant"),
                        "aggregate_delta": vdelta,
                        "useful_case_improvements": vimprovements[:3],
                        "why_rejected": str(variant.get("reason") or "lost variant duel")[:240],
                    }
                )

    curriculum_stage = next_curriculum_stage(records, best_eval)
    teacher_brief = teacher_benchmark_brief(teacher_eval, best_eval, target_profiles)

    if best_eval is None:
        return {
            "mode": "initial_generation",
            "curriculum_stage": curriculum_stage,
            "teacher_benchmark": teacher_brief,
            "best_snapshot": None,
            "optimization_focus": [
                {
                    "name": prof.get("name"),
                    "labels": prof.get("labels", []),
                    "task_count": prof.get("task_count"),
                    "courier_count": prof.get("courier_count"),
                    "candidate_density": prof.get("candidate_density"),
                }
                for prof in target_profiles
            ],
            "recent_trials": recent_trials,
            "avoid_repeating": failure_counts,
            "instructions": [
                "Create the first valid all-case solver.",
                "Minimize penalized score: selected assignment cost plus 100 for each uncovered task.",
                "Include a simple greedy fallback; empty output is legal but usually scores poorly because every task is penalized.",
                "Use the teacher only as a benchmark and component source; do not treat it as an incumbent best.",
            ],
        }

    best_value = best_eval.aggregate_value()
    best_cases = eval_case_briefs(best_eval, target_profiles)
    incomplete = [row for row in best_cases if not row.get("complete")]
    if incomplete:
        focus_cases = sorted(
            incomplete,
            key=lambda row: ((int(row.get("task_count") or 0) - int(row.get("covered") or 0)), finite_number(row.get("cost")) or 0.0),
            reverse=True,
        )[:5]
        focus_reason = "Missing tasks already add penalty to cost; focus on the largest penalized-score contributors."
    else:
        focus_cases = sorted(
            best_cases,
            key=lambda row: (finite_number(row.get("cost_per_task")) or 0.0, finite_number(row.get("cost")) or 0.0),
            reverse=True,
        )[:5]
        focus_reason = "All tracked cases are complete; reduce high cost-per-task cases."
    experiment_directive = build_experiment_directive(focus_cases, records, failure_counts)

    return {
        "mode": "refinement",
        "exploration_mode": experiment_directive.get("exploration_mode", "refine"),
        "curriculum_stage": curriculum_stage,
        "teacher_benchmark": teacher_brief,
        "best_snapshot": {
            "aggregate_covered": best_value[0],
            "aggregate_cost": rounded_number(best_value[1], 6),
            "complete_cases": sum(1 for row in best_cases if row.get("complete")),
            "total_cases": len(best_cases),
            "cases": best_cases,
        },
        "optimization_focus": {
            "reason": focus_reason,
            "cases": focus_cases,
        },
        "next_experiment_directive": experiment_directive,
        "near_misses": near_misses[-4:],
        "recent_trials": recent_trials,
        "avoid_repeating": failure_counts,
        "instructions": [
            "Start from the current best code and make one or two targeted changes.",
            "Accept or reject ideas by penalized aggregate score; coverage is diagnostic, not a hard gate.",
            "Use profile-derived gates, not case names, task IDs, courier IDs, or input fingerprints.",
            "Do not repeat failed patterns listed in avoid_repeating.",
        ],
    }


def component_category(name: str) -> str:
    if name in {"C", "G", "parse", "clean", "solve", "pc"}:
        return "io_and_core_structures"
    if name in {"gcost_cands", "gcost", "single", "value", "gain_over", "better", "stats", "task_count"}:
        return "objective_and_evaluation"
    if name in {"greedy", "exact_cover_seed", "singleton_seed", "mcf_singleton_seed", "pair_plan_seed", "stochastic_pair_plan_seed", "bundle_plan_seed", "cpsat_offer_seed", "generated_style_seed"}:
        return "initial_solution_builders"
    if name in {"assign_first_mcf", "choose_pairs", "task_maps", "approx_k_cost", "skeleton_mcf_search"}:
        return "matching_and_skeleton_helpers"
    if name in {"expand", "reassign", "single_replace", "pair_replace", "move_extra", "swap_riders", "cycle_riders", "path_riders", "config_polish", "pair_kick", "repair_window", "lns", "scarce_lns", "pair2_swap", "polish"}:
        return "local_search_and_repair"
    if name in {"anneal", "anneal_steps", "search", "legacy_search"}:
        return "search_orchestration"
    return "other"


def first_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def shared_component_header(code: str) -> str:
    lines: list[str] = []
    for line in code.splitlines():
        if line.startswith("# ---- component:"):
            break
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def parse_component_units(path: Path, code: str) -> list[ComponentUnit]:
    code = code.lstrip("\ufeff")
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    lines = code.splitlines()
    units: list[ComponentUnit] = []
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            continue
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            continue
        source = "\n".join(lines[node.lineno - 1 : node.end_lineno])
        units.append(
            ComponentUnit(
                file_name=path.name,
                name=node.name,
                kind="class" if isinstance(node, ast.ClassDef) else "def",
                category=component_category(node.name),
                signature=first_line(source),
                source=source,
            )
        )
    return units


def target_label_set(target_profiles: list[dict[str, Any]] | None) -> set[str]:
    labels: set[str] = set()
    for prof in target_profiles or []:
        labels.update(str(x) for x in prof.get("labels", []))
    return labels


def select_component_names(target_profiles: list[dict[str, Any]] | None) -> list[str]:
    labels = target_label_set(target_profiles)
    max_tasks = max((int(p.get("task_count", 0)) for p in target_profiles or []), default=0)
    names = [
        "C",
        "G",
        "parse",
        "copy",
        "clean",
        "better",
        "gcost_cands",
        "gcost",
        "single",
        "value",
        "gain_over",
        "task_count",
        "stats",
        "greedy",
        "task_maps",
        "approx_k_cost",
        "choose_pairs",
        "assign_first_mcf",
        "skeleton_mcf_search",
        "expand",
        "reassign",
        "move_extra",
        "swap_riders",
        "cycle_riders",
        "path_riders",
        "config_polish",
        "polish",
        "anneal",
        "anneal_steps",
        "generated_style_seed",
        "search",
        "legacy_search",
        "solve",
    ]
    if labels.intersection({"tiny", "small"}) or max_tasks <= 15:
        names.extend(["exact_cover_seed", "generated_style_seed", "single_replace", "pair_replace", "repair_window"])
    if labels.intersection({"medium", "large", "pair_heavy"}):
        names.extend(["pair_plan_seed", "stochastic_pair_plan_seed", "generated_style_seed", "lns", "pair_kick"])
    if labels.intersection({"rider_rich", "tight_couriers"}):
        names.extend(["singleton_seed", "mcf_singleton_seed", "pair_kick"])
    if "low_willingness" in labels:
        names.extend(["stochastic_pair_plan_seed", "pair_plan_seed", "pair_kick", "config_polish"])
    if "scarce_couriers" in labels:
        names.extend(["bundle_plan_seed", "skeleton_mcf_search", "pair2_swap", "scarce_lns", "stochastic_pair_plan_seed", "pair_plan_seed"])
    if "multi_bundle_available" in labels:
        names.extend(["bundle_plan_seed", "cpsat_offer_seed"])
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def curriculum_component_names(stage: int, target_profiles: list[dict[str, Any]] | None) -> list[str]:
    stages = [
        [
            "C",
            "G",
            "parse",
            "copy",
            "clean",
            "better",
            "gcost_cands",
            "gcost",
            "single",
            "value",
            "task_count",
            "stats",
            "greedy",
            "task_maps",
            "approx_k_cost",
            "choose_pairs",
            "assign_first_mcf",
            "pair_plan_seed",
            "stochastic_pair_plan_seed",
            "expand",
            "reassign",
            "move_extra",
            "swap_riders",
        ],
        ["exact_cover_seed", "single_replace", "pair_replace", "repair_window", "generated_style_seed"],
        ["bundle_plan_seed", "skeleton_mcf_search", "pair2_swap", "scarce_lns", "bundle_plan_seed"],
        ["cycle_riders", "path_riders", "config_polish", "polish", "pair_kick", "anneal", "anneal_steps", "lns"],
        ["search", "legacy_search", "solve"] + select_component_names(target_profiles),
    ]
    stage_no = max(1, min(int(stage or 1), 5))
    names: list[str] = []
    for block in stages[:stage_no]:
        names.extend(block)
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def adaptive_component_release_plan(
    records: list[dict[str, Any]],
    best_eval: SolverEval | None,
    target_profiles: list[dict[str, Any]] | None,
    max_component_chars: int,
) -> dict[str, Any]:
    positive = [rec for rec in records if isinstance(rec.get("round"), int) and rec.get("round", 0) > 0]
    accepted = [rec for rec in positive if rec.get("accepted")]
    attempts_since_best = rounds_since_last_accept(records)
    post_failures = recent_failure_counts(records, after_last_accept=True)
    labels = target_label_set(target_profiles)
    all_complete = False
    if best_eval is not None and best_eval.ok and best_eval.results:
        all_complete = all(result.complete for result in best_eval.results)

    level = 1
    reasons = []
    if best_eval is None:
        reasons.append("no valid generated best yet: expose only a runnable foundation")
    else:
        level = 2
        reasons.append("valid generated best exists: expose targeted construction/repair components")
        if all_complete:
            level = max(level, 3)
            reasons.append("all tracked cases are complete: shift from coverage to cost reduction")
        else:
            reasons.append("some cases are incomplete: prioritize repair and exact/small-case tools before global search")
        if attempts_since_best >= 2 or post_failures.get("behavior_duplicate") or post_failures.get("cost_regression"):
            level = max(level, 4)
            reasons.append("recent attempts are stale or regressing: increase novelty and local-search budget")
        if (
            all_complete
            and (
                attempts_since_best >= 3
                or len(accepted) >= 3
                or post_failures.get("behavior_duplicate", 0) >= 2
            )
        ):
            level = max(level, 5)
            reasons.append("full coverage plus enough evidence/stagnation: allow global orchestration")

    core = [
        "C",
        "G",
        "parse",
        "copy",
        "clean",
        "better",
        "gcost_cands",
        "gcost",
        "single",
        "value",
        "task_count",
        "stats",
        "greedy",
        "task_maps",
        "approx_k_cost",
        "choose_pairs",
        "assign_first_mcf",
        "pair_plan_seed",
        "expand",
        "reassign",
        "move_extra",
        "swap_riders",
    ]
    blocks: dict[str, list[str]] = {
        "generated_style": ["generated_style_seed", "stochastic_pair_plan_seed", "repair_window"],
        "tiny_small": ["exact_cover_seed", "single_replace", "pair_replace", "repair_window"],
        "scarce": ["bundle_plan_seed", "skeleton_mcf_search", "pair2_swap", "scarce_lns"],
        "low_willingness": ["stochastic_pair_plan_seed", "bundle_plan_seed", "repair_window"],
        "local_polish": ["cycle_riders", "path_riders", "config_polish", "polish", "pair_kick", "anneal", "anneal_steps", "lns"],
        "global_orchestration": ["search", "legacy_search", "solve"],
    }
    names: list[str] = list(core)
    chosen_blocks: list[str] = ["foundation"]
    if level >= 2:
        names.extend(blocks["generated_style"])
        chosen_blocks.append("generated_style")
        if {"tiny", "small"} & labels:
            names.extend(blocks["tiny_small"])
            chosen_blocks.append("tiny_small")
    if level >= 3:
        if "scarce_couriers" in labels:
            names.extend(blocks["scarce"])
            chosen_blocks.append("scarce")
        if "low_willingness" in labels:
            names.extend(blocks["low_willingness"])
            chosen_blocks.append("low_willingness")
        if "scarce_couriers" not in labels and "low_willingness" not in labels:
            rotation = len(positive) % 2
            names.extend(blocks["scarce" if rotation == 0 else "low_willingness"])
            chosen_blocks.append("rotating_sparse_or_low_will")
    if level >= 4:
        names.extend(blocks["local_polish"])
        chosen_blocks.append("local_polish")
    if level >= 5:
        names.extend(blocks["global_orchestration"])
        chosen_blocks.append("global_orchestration")

    seen: set[str] = set()
    deduped: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            deduped.append(name)

    caps = {
        1: 22000,
        2: 34000,
        3: 50000,
        4: 68000,
        5: 90000,
    }
    budget = min(max_component_chars, caps.get(level, 50000)) if max_component_chars > 0 else caps.get(level, 50000)
    return {
        "stage": level,
        "name": f"adaptive_level_{level}",
        "policy": "adaptive",
        "goal": "Expose enough component source to support the next useful LLM-guided experiment without handing over the full solver too early.",
        "components_to_use": deduped,
        "component_blocks": chosen_blocks,
        "component_budget": budget,
        "expected_progress": "Let the model choose the experiment family from diagnostics while the agent controls only source budget and evaluation.",
        "release_reasons": reasons,
        "stagnation": {
            "attempts_since_best": attempts_since_best,
            "post_best_failures": post_failures,
            "accepted_count": len(accepted),
            "all_complete": all_complete,
        },
    }


def curriculum_mandatory_names(stage: int) -> set[str]:
    core = {
        "C",
        "G",
        "parse",
        "copy",
        "clean",
        "better",
        "gcost_cands",
        "gcost",
        "single",
        "value",
        "task_count",
        "stats",
        "greedy",
    }
    stage_no = max(1, min(int(stage or 1), 5))
    if stage_no == 1:
        return core | {"pair_plan_seed", "expand", "reassign", "swap_riders"}
    if stage_no == 2:
        return core | {"pair_plan_seed", "generated_style_seed", "exact_cover_seed", "repair_window"}
    if stage_no == 3:
        return core | {"bundle_plan_seed", "skeleton_mcf_search", "scarce_lns"}
    if stage_no == 4:
        return core | {"config_polish", "pair_kick", "anneal", "anneal_steps"}
    return core | {"search", "solve", "generated_style_seed", "skeleton_mcf_search", "config_polish", "pair_kick"}


def curriculum_component_budget(stage: int, max_component_chars: int) -> int:
    caps = {
        1: 18000,
        2: 26000,
        3: 36000,
        4: 48000,
        5: 90000,
    }
    cap = caps.get(int(stage or 1), max_component_chars)
    return min(max_component_chars, cap) if max_component_chars > 0 else cap


def summarize_components(units: list[ComponentUnit]) -> str:
    lines = ["Component inventory:"]
    for unit in units:
        lines.append(f"- {unit.file_name} / {unit.category}: {unit.signature}")
    return "\n".join(lines)


def append_source_excerpt(
    out: list[str],
    unit: ComponentUnit,
    current_chars: int,
    max_chars: int,
    force: bool,
) -> tuple[int, bool]:
    block = f"## component: {unit.name} ({unit.file_name})\n```python\n{unit.source}\n```\n"
    if not force and max_chars > 0 and current_chars + len(block) > max_chars:
        return current_chars, False
    out.append(block)
    return current_chars + len(block), True


def extract_algorithm_library(
    component_dir: Path,
    prompt_mode: str = "selected",
    target_profiles: list[dict[str, Any]] | None = None,
    max_component_chars: int = 36000,
    selected_names_override: list[str] | None = None,
    curriculum_stage: dict[str, Any] | None = None,
) -> str:
    if not component_dir.exists() or not component_dir.is_dir():
        raise RuntimeError(f"algorithm component directory not found: {component_dir}")
    component_files = sorted(p for p in component_dir.glob("*.py") if p.is_file())
    if not component_files:
        raise RuntimeError(f"no .py algorithm component files found in {component_dir}")
    file_codes = [(path, path.read_text(encoding="utf-8-sig")) for path in component_files]
    units: list[ComponentUnit] = []
    for path, code in file_codes:
        units.extend(parse_component_units(path, code))
    if prompt_mode == "summary":
        return "\n".join(
            [
                f"Algorithm component directory: {component_dir}",
                "Prompt mode: summary. Only component signatures are provided.",
                "Generate a standalone solver by combining the listed ideas; do not import these files.",
                summarize_components(units),
                "",
            ]
        )
    if prompt_mode == "selected":
        selected_names = selected_names_override or select_component_names(target_profiles)
        unit_by_name = {unit.name: unit for unit in units}
        selected_set = set(selected_names)
        visible_units = [unit for unit in units if unit.name in selected_set] if selected_names_override is not None else units
        labels = ",".join(sorted(target_label_set(target_profiles))) or "none"
        stage_desc = ""
        if curriculum_stage:
            stage_desc = (
                f"Curriculum stage {curriculum_stage.get('stage')}: {curriculum_stage.get('name')} - "
                f"{curriculum_stage.get('goal')}"
            )
        out = [
            f"Algorithm component directory: {component_dir}",
            "Prompt mode: selected. Use the inventory plus selected source excerpts below.",
            "These files are reference material only; generate one standalone Python file.",
            "If an excerpt depends on a helper that is not included, rewrite a compact equivalent from the inventory.",
            f"Target labels seen across selected cases: {labels}",
            stage_desc,
            "Curriculum rule: use only the visible component excerpts for this round; later-stage components are intentionally withheld.",
            "",
            summarize_components(visible_units),
            "",
        ]
        header = shared_component_header(file_codes[0][1]) if file_codes else ""
        current_chars = len("\n".join(out))
        if header:
            header_block = f"Shared boilerplate used by components:\n```python\n{header}\n```\n"
            out.append(header_block)
            current_chars += len(header_block)
        stage_number = int((curriculum_stage or {}).get("stage") or 0)
        mandatory = curriculum_mandatory_names(stage_number) if selected_names_override is not None else {
            "C",
            "G",
            "parse",
            "copy",
            "clean",
            "better",
            "gcost_cands",
            "gcost",
            "single",
            "value",
            "task_count",
            "gain_over",
            "pair_plan_seed",
            "stochastic_pair_plan_seed",
            "assign_first_mcf",
            "skeleton_mcf_search",
            "generated_style_seed",
            "anneal",
            "anneal_steps",
            "search",
            "legacy_search",
            "solve",
        }
        omitted: list[str] = []
        for name in selected_names:
            unit = unit_by_name.get(name)
            if unit is None:
                continue
            current_chars, added = append_source_excerpt(out, unit, current_chars, max_component_chars, name in mandatory)
            if not added:
                omitted.append(name)
        if omitted:
            out.append("Source excerpts omitted because of --max-component-chars: " + ", ".join(omitted))
        return "\n".join(out)

    out = [
        f"Algorithm component directory: {component_dir}",
        "Prompt mode: full. Complete component files are included.",
        "Important: these files are the only algorithm material for this run.",
        "They are not incumbent baselines and must not be treated as score targets.",
        "The generated solver must be a standalone single Python file.",
        "You may reuse, simplify, rewrite, or combine selected components from the files below.",
        "",
    ]
    for path, code in file_codes:
        out.append(f"## file: {path.name}")
        out.append("```python")
        out.append(code)
        out.append("```")
        out.append("")
    return "\n".join(out)


def build_messages(
    target_profiles: list[dict[str, Any]],
    best_eval: SolverEval | None,
    recent_records: list[dict[str, Any]],
    algorithm_library: str,
    best_code: str | None,
    prompt_mode: str,
    max_best_code_chars: int,
    learning_brief: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    best_summary = summarize_eval(best_eval) if best_eval is not None else None
    recent = [compact_record_for_prompt(rec) for rec in recent_records[-3:]]
    if learning_brief is None:
        learning_brief = build_learning_brief(target_profiles, best_eval, recent_records)
    exploration_mode = str(learning_brief.get("exploration_mode") or "refine")
    orchestration_rebuild_mode = (
        "orchestration-rebuild" in prompt_mode
        or exploration_mode in {"orchestration_rebuild", "orchestration-rebuild"}
    )
    recompose_mode = (
        orchestration_rebuild_mode
        or "recompose" in prompt_mode
        or exploration_mode == "recompose"
    )
    refinement_mode = (
        best_eval is not None
        and bool(best_code)
        and max_best_code_chars != 0
        and not recompose_mode
    )
    system = (
        "You are an AutoSolver code-generation agent for a delivery assignment contest. "
        "Return a JSON object with keys hypothesis, diagnosis, selected_components, target_case_type, "
        "change_plan, expected_case_deltas, rollback_risk, expected_effect, and code. "
        "The code value must be a complete Python solver file defining solve(input_text: str) -> list. "
        "It must run on Python 3.6, stay under 100KB as UTF-8 source, and finish each case within 10 seconds. "
        "Do not hardcode public cases, task IDs, courier IDs, input fingerprints, or precomputed answers. "
        "Do not use network, filesystem IO, subprocesses, prints, or required third-party packages. "
        "Standard library is allowed; OR-Tools may only be imported optionally inside a guarded try/except path. "
        "Avoid Python 3.7+ syntax/APIs: no dataclasses, no from __future__ annotations, no list[str]/dict[str] "
        "annotations, no int.bit_count without a try/except fallback, no math.prod/isqrt/comb, no functools.cache, "
        "no removeprefix/removesuffix, no walrus operator, and no match/case. "
        "Do not hide normal parse/search bugs with broad except: return [] paths; an empty solution on non-empty cases "
        "is treated as a failed candidate. "
        "Each round must be an LLM-guided experiment: diagnose the latest case deltas, choose components, then generate a full solver."
    )
    best_code_block = ""
    if best_code and max_best_code_chars != 0 and not recompose_mode:
        best_code = truncate_for_prompt(best_code, max_best_code_chars, "current best generated code")
        best_code_block = f"""

Current best generated solver code:
```python
{best_code}
```
"""
    mode_rules = """
Generation mode:
- No valid incumbent exists yet. Compose a robust standalone solver from the algorithm reference.
- Minimize penalized score from the beginning: selected assignment cost plus 100 for each uncovered task.
- Include a simple greedy fallback. Empty output is legal but usually terrible because every task is penalized.
- Treat learning_brief.component_release/curriculum_stage as the visible component envelope for this round. Use the listed components as building blocks, but write one standalone solver file.
- A short greedy-only solver is useful only as a fallback; the main path should combine at least one construction component and one polish/repair component when the case portfolio is medium/large.
"""
    if orchestration_rebuild_mode:
        mode_rules = """
Full orchestration assimilation mode:
- Full search/solve orchestration has just been unlocked or needs a retry. This is a challenger rebuild round, not a small patch.
- Treat the current best evaluation as the score target. The incumbent source is intentionally withheld so it does not anchor the architecture.
- Use the visible `search`, `legacy_search`, `solve`, and local-search orchestration components as the primary design reference; synthesize one complete solver from them.
- The candidate should contain an explicit multi-stage search schedule or equivalent global orchestration. A short greedy/pair portfolio may be the fallback, but it must not be the main path.
- Spend meaningful runtime on hard profiles: medium/large rider-rich, low-willingness, and scarce-courier cases. A per-case runtime in seconds is acceptable under the 10 second limit.
- Preserve the contest contract: Python 3.6, standalone `solve(input_text)`, no hardcoded case names/IDs/fingerprints, no filesystem/network/subprocess, clean fallback.
- A worse challenger is acceptable because local evaluation will reject it; a near-identical incumbent-style patch is not useful in this mode.
- Explain in JSON metadata how the full orchestration differs from the current lightweight best and which profile gates/budgets should close the largest score gaps.
"""
    elif recompose_mode:
        mode_rules = """
Current-stage recomposition mode:
- A new curriculum component set has just been unlocked. This is the fresh rebuild challenger for this stage.
- Treat the current best evaluation as the score target. The incumbent source is intentionally withheld so it does not anchor the architecture.
- Use all visible components through the current stage to synthesize a new main search schedule; the newly unlocked components must be called by solve() or the primary search path.
- Keep the contest contract: Python 3.6, standalone `solve(input_text)`, no hardcoded case names/IDs/fingerprints, no filesystem/network/subprocess, clean fallback.
- A short greedy/pair portfolio may be the fallback, but the main path should be a real recomposition using the stage components.
- Local evaluation will compare this rebuild against the incumbent-graft variant; a worse rebuild is acceptable, but a near-identical patch is not useful.
- Explain in JSON metadata how the new component set changes the search architecture and which profile gates/budgets should close the largest score gaps.
"""
    elif refinement_mode:
        mode_rules = """
Refinement mode:
- Treat the current best solver code as the incumbent to preserve, not as disposable inspiration.
- Treat learning_brief.next_experiment_directive as a recommendation, not a command. Choose the experiment family that best explains the recent case deltas; if you choose another available family, explain why in JSON metadata.
- Make one or two targeted changes by default, preferably profile-gated search schedule, portfolio branch, or time-budget changes. Escalate only when the learning brief shows stagnation or duplicate behavior.
- Do not rewrite parsing, objective math, output formatting, or broad helper structure unless the learning brief proves it is necessary.
- Optimize penalized aggregate score. Coverage is diagnostic; missing tasks are already priced at 100 each.
- Case names may appear in diagnostics, but implementation gates must use computed profile signals such as task count, courier ratio, average willingness, group size mix, or candidate density.
- If a proposed idea is risky, keep the incumbent behavior as the default branch and add the new behavior behind a narrow profile gate.
- Use an incumbent-preserving portfolio whenever possible: compute the incumbent/base solution, compute one targeted alternate, compare penalized score with the same objective helper, and return the better solution.
- A candidate that only renames code, changes comments, or produces exactly the same per-case behavior is a failed experiment. Add a real alternate branch or a measurable budget/selection change.
- If a previous trial regressed or duplicated behavior, do not repeat the same family; choose the directive's new experiment family.
- You choose the experiment from the learning brief; explain the diagnosis and selected components in JSON metadata before the code.
"""
        if exploration_mode in {"challenger", "radical"}:
            mode_rules += """
Challenger escalation:
- The previous post-best attempts did not help enough. Be more aggressive in this round.
- You may create a substantially different challenger solver using the algorithm reference instead of patching the incumbent line by line.
- Use the current best evaluation as the score target, not as a template to copy.
- Prefer a different algorithm family or search schedule from the last two rejected attempts.
- A worse challenger is acceptable because local evaluation will reject it; a near-identical solver is not useful.
"""
        if exploration_mode == "radical":
            mode_rules += """
Radical escalation:
- Make a bold but still valid standalone solver. Do not spend the round on parameter nits.
- Keep Python 3.6 compatibility and the simple fallback, but otherwise explore a different construction/search architecture.
"""
    user = f"""
Problem:
- Input is TSV with columns task_id_list, courier_id, total_score, willingness.
- Output must be [(task_id_list_str, [courier_id, ...]), ...].
- Every output pair must exist in the input.
- A courier can be used at most once globally.
- Selected task groups cannot overlap tasks.
- Minimize penalized proxy score: selected assignment cost plus 100 for each uncovered task.
- All selected local case files are target cases. The generated solver must work for every selected case, not only the first one.
- Hard judge constraints: Python 3.6 runtime, no required third-party package, solver.py <= 100KB, each case <= 10 seconds.
- If you need popcount, use a Python 3.6-safe helper loop or a guarded bit_count fallback.
- Empty output on a non-empty case is allowed but heavily penalized by 100 per uncovered task; include a simple greedy fallback so the baseline is not empty.

Component learning contract:
- Algorithm components are distilled from strong human-tuned solvers, but no complete teacher solver is provided as an incumbent.
- The generated best must come from LLM-returned candidate code in a positive round.
- Use the component inventory and selected source excerpts to synthesize the solver. It is acceptable if the model independently recreates a teacher-like solver through this process.
- Do not hardcode case names, task IDs, courier IDs, input fingerprints, or precomputed answers.
- Every round should describe its diagnosis, chosen components, and why the selected change should improve the current case deltas.

{mode_rules}

Selected target case profiles:
{json.dumps(target_profiles, ensure_ascii=False, indent=2)}

Current best generated-solver evaluation:
{json.dumps(best_summary, ensure_ascii=False, indent=2)}

Learning brief from local experiments:
{json.dumps(learning_brief, ensure_ascii=False, indent=2)}

Recent experiment records:
{json.dumps(recent, ensure_ascii=False, indent=2)}

Allowed strategy families:
- search schedule or threshold tuning by profile
- low-willingness multi-courier assignment
- scarce-courier pair or bundle skeleton search
- local LNS/repair around high proxy-cost groups
- optional CP-SAT offer selection with immediate fallback
- image/profile branch gates that avoid regressions

Prompt mode:
{prompt_mode}

Algorithm reference:
{algorithm_library}
{best_code_block}

Return only JSON. The code field must contain the full candidate solver file.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": textwrap.dedent(user).strip()}]


def generate_and_evaluate_candidate_variant(
    args: argparse.Namespace,
    run_start: float,
    round_no: int,
    variant_name: str,
    profiles: list[dict[str, Any]],
    best_eval: SolverEval | None,
    records: list[dict[str, Any]],
    algorithm_library: str,
    best_code: str | None,
    prompt_mode: str,
    best_code_chars: int,
    learning_brief: dict[str, Any],
    candidate_path: Path,
    raw_response_path: Path,
    cases: list[EvalCase],
    case_dir: Path,
    teacher_eval: SolverEval | None,
) -> tuple[dict[str, Any], SolverEval | None]:
    variant: dict[str, Any] = {
        "variant": variant_name,
        "candidate_path": str(candidate_path),
        "response_path": str(raw_response_path),
        "prompt_mode": prompt_mode,
        "exploration_mode": str(learning_brief.get("exploration_mode") or ""),
        "effective_best_code_chars": best_code_chars if best_eval is not None and best_code_chars != 0 else 0,
    }
    log(args, run_start, f"Round {round_no} [{variant_name}]: building DeepSeek prompt")
    messages = build_messages(
        profiles,
        best_eval,
        records,
        algorithm_library,
        best_code,
        prompt_mode,
        best_code_chars,
        learning_brief,
    )
    prompt_chars = sum(len(m.get("content", "")) for m in messages)
    variant["prompt_chars"] = prompt_chars
    log(args, run_start, f"Round {round_no} [{variant_name}]: prompt size {prompt_chars} chars", verbose_only=True)
    log(args, run_start, f"Round {round_no} [{variant_name}]: calling DeepSeek model={args.model}")
    content = ""
    llm_attempts = max(1, int(getattr(args, "llm_retries", 1)) + 1)
    for llm_attempt in range(1, llm_attempts + 1):
        try:
            content = call_llm_with_progress(args, run_start, round_no, messages)
            if llm_attempt > 1:
                variant["llm_retry_used"] = llm_attempt - 1
            break
        except Exception as exc:
            if llm_attempt >= llm_attempts:
                raise
            variant["llm_retry_used"] = llm_attempt
            log(
                args,
                run_start,
                f"Round {round_no} [{variant_name}]: DeepSeek call failed with {type(exc).__name__}: {exc}; retrying ({llm_attempt}/{llm_attempts - 1})",
                error=True,
            )
            time.sleep(min(8.0, 2.0 * llm_attempt))
    log(args, run_start, f"Round {round_no} [{variant_name}]: DeepSeek response received ({len(content)} chars)")
    raw_response_path.write_text(content, encoding="utf-8")
    log(args, run_start, f"Round {round_no} [{variant_name}]: raw response saved to {raw_response_path}", verbose_only=True)
    meta, code = extract_llm_result(content)
    variant.update(meta)
    if not code:
        variant.update(status="rejected", reason="LLM response did not contain solver code")
        log(args, run_start, f"Round {round_no} [{variant_name}]: rejected, no solver code found")
        return variant, None
    code = normalize_solver_source(code)
    variant["source_bytes"] = solver_size_bytes(code)
    log(args, run_start, f"Round {round_no} [{variant_name}]: extracted candidate code ({len(code)} chars, {variant['source_bytes']} bytes)")
    write_solver_source(candidate_path, code)
    log(args, run_start, f"Round {round_no} [{variant_name}]: candidate saved to {candidate_path}")
    log(args, run_start, f"Round {round_no} [{variant_name}]: running static guard")
    guard_errors = static_guard(code)
    if guard_errors:
        variant.update(status="rejected", reason="; ".join(guard_errors[:8]))
        log(args, run_start, f"Round {round_no} [{variant_name}]: rejected by static guard: {variant['reason']}")
        return variant, None
    log(args, run_start, f"Round {round_no} [{variant_name}]: static guard passed")
    if best_eval is None and getattr(args, "reference_guard", False):
        reference_errors = reference_architecture_guard(code, profiles)
        if reference_errors:
            variant.update(status="rejected", reason="; ".join(reference_errors[:4]))
            log(args, run_start, f"Round {round_no} [{variant_name}]: rejected by reference architecture guard: {variant['reason']}")
            return variant, None
    log(args, run_start, f"Round {round_no} [{variant_name}]: evaluating candidate on {len(cases)} case(s)")
    candidate_eval = evaluate_solver(candidate_path, cases, case_dir, args.timeout, args.judge_python)
    variant["evaluation"] = summarize_eval(candidate_eval)
    if best_eval is not None:
        variant["comparison_to_best"] = compare_evals(candidate_eval, best_eval, profiles)
    if teacher_eval is not None:
        variant["comparison_to_teacher"] = compare_evals(candidate_eval, teacher_eval, profiles)
    if not candidate_eval.ok:
        bad = next((r for r in candidate_eval.results if not r.ok), None)
        variant.update(status="rejected", reason=bad.error if bad else candidate_eval.error)
        log(args, run_start, f"Round {round_no} [{variant_name}]: rejected by evaluation: {variant['reason']}")
        return variant, candidate_eval
    candidate_value = candidate_eval.aggregate_value()
    variant.update(status="evaluated", aggregate={"covered": candidate_value[0], "cost": candidate_value[1]})
    log(args, run_start, f"Round {round_no} [{variant_name}]: aggregate value covered={candidate_value[0]}, cost={candidate_value[1]:.6f}")
    return variant, candidate_eval


def write_history(history_path: Path, record: dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(summary_path: Path, records: list[dict[str, Any]], profiles: list[dict[str, Any]]) -> None:
    lines = [
        "# AutoSolver Agent Summary",
        "",
        f"Updated: {_dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Profiles",
        "",
    ]
    for prof in profiles:
        lines.append(
            f"- {prof['name']}: labels={','.join(prof['labels'])}, "
            f"tasks={prof['task_count']}, couriers={prof['courier_count']}, "
            f"groups={prof['group_count']}, candidates={prof['candidate_count']}, "
            f"avg_will={prof['avg_willingness']}"
        )
    accepted_records = [rec for rec in records if rec.get("accepted")]
    if accepted_records:
        best_rec = accepted_records[-1]
        ev = best_rec.get("evaluation", {})
        aggregate = ev.get("aggregate", {})
        lines.extend(
            [
                "",
                "## Current Best Generated Solver",
                "",
                f"- Best file: `{summary_path.parent / 'best_generated.py'}`",
                f"- Accepted from round: {best_rec.get('round')}",
                f"- Candidate source: `{best_rec.get('candidate_path', '')}`",
                f"- Aggregate covered: {aggregate.get('covered', 0)}",
                f"- Aggregate cost: {float(aggregate.get('cost', math.inf)):.6f}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Current Best Generated Solver",
                "",
                "- No accepted solver yet. `best_generated.py` is not available for this run.",
                "",
            ]
        )
    lines.extend(["", "## Experiments", ""])
    lines.append("| round | status | accepted | winner | variants | cases | aggregate covered | aggregate cost | primary time | file | reason |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|---|---|")
    for rec in records:
        ev = rec.get("evaluation", {})
        aggregate = ev.get("aggregate", {})
        primary = ev.get("primary", {})
        variants = rec.get("variants") if isinstance(rec.get("variants"), list) else []
        variant_decision = rec.get("variant_decision") if isinstance(rec.get("variant_decision"), dict) else {}
        winner = variant_decision.get("winner") or rec.get("variant") or ""
        path = rec.get("candidate_path") or ev.get("solver_path") or ""
        reason = (rec.get("reason") or primary.get("error") or "").replace("|", "/")
        lines.append(
            f"| {rec.get('round', 0)} | {rec.get('status', '')} | {rec.get('accepted', False)} | "
            f"{winner} | {len(variants) if variants else 1} | "
            f"{rec.get('case_count', len(ev.get('cases', [])))} | "
            f"{aggregate.get('covered', 0)} | {float(aggregate.get('cost', math.inf)):.6f} | "
            f"{float(primary.get('duration', 0.0)):.3f} | {path} | {reason[:180]} |"
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_eval_cases(case_paths: list[Path], case_dir: Path) -> tuple[list[EvalCase], list[dict[str, Any]]]:
    cases: list[EvalCase] = []
    profiles: list[dict[str, Any]] = []
    for idx, path in enumerate(case_paths):
        text = path.read_text(encoding="utf-8")
        data = parse_case(text, path.name)
        case = EvalCase(path.stem, text=text, path=path, data=data, primary=(idx == 0))
        cases.append(case)
        profiles.append(profile_case(data))
    for case in cases:
        write_case_file(case, case_dir)
    return cases, profiles


def collect_case_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    case_dir = Path(args.case_dir)
    for raw in args.case or []:
        path = Path(raw)
        if not path.exists() and not path.is_absolute():
            path = case_dir / raw
        paths.append(path)
    if not paths:
        if not case_dir.exists() or not case_dir.is_dir():
            raise RuntimeError(f"case directory not found: {case_dir}; use --case-dir or --case")
        paths.extend(sorted(case_dir.glob("*.txt")))
        paths.extend(sorted(case_dir.glob("*.tsv")))
    if not paths:
        raise RuntimeError(f"no .txt/.tsv case files found in {case_dir}; use --case to select files explicitly")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise RuntimeError(f"case file not found: {', '.join(missing)}")
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def run_agent(args: argparse.Namespace) -> int:
    run_start = time.perf_counter()
    log(args, run_start, "AutoSolver Agent started")
    algorithm_dir = Path(args.algorithm_dir)
    if not algorithm_dir.exists() or not algorithm_dir.is_dir():
        raise RuntimeError(f"algorithm component directory not found: {algorithm_dir}")
    submit_path = Path(args.submit_path)
    out_dir = Path(args.out_dir)
    runs_dir = Path(args.runs_dir)
    case_dir = runs_dir / "cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    history_path = runs_dir / "history.jsonl"
    summary_path = runs_dir / "summary.md"
    best_path = runs_dir / "best_generated.py"

    log(args, run_start, f"Collecting case files from case_dir={args.case_dir}")
    case_paths = collect_case_paths(args)
    log(args, run_start, f"Selected {len(case_paths)} case file(s)")
    for path in case_paths:
        log(args, run_start, f"case: {path}", verbose_only=True)

    log(args, run_start, "Parsing cases and extracting profiles")
    cases, profiles = load_eval_cases(case_paths, case_dir)
    for prof in profiles:
        labels = ",".join(prof["labels"])
        log(
            args,
            run_start,
            f"profile {prof['name']}: labels={labels}, tasks={prof['task_count']}, couriers={prof['courier_count']}, candidates={prof['candidate_count']}",
        )

    log(args, run_start, f"Loading algorithm components from {algorithm_dir} with prompt_mode={args.prompt_mode}")
    algorithm_library = extract_algorithm_library(
        algorithm_dir,
        prompt_mode=args.prompt_mode,
        target_profiles=profiles,
        max_component_chars=args.max_component_chars,
    )
    log(args, run_start, f"Algorithm component prompt size: {len(algorithm_library)} chars", verbose_only=True)
    if args.prompt_mode == "summary":
        algorithm_summary = algorithm_library
    else:
        algorithm_summary = extract_algorithm_library(
            algorithm_dir,
            prompt_mode="summary",
            target_profiles=profiles,
            max_component_chars=args.max_component_chars,
        )
    log(args, run_start, f"Algorithm summary prompt size: {len(algorithm_summary)} chars", verbose_only=True)
    algorithm_selected = extract_algorithm_library(
        algorithm_dir,
        prompt_mode="selected",
        target_profiles=profiles,
        max_component_chars=min(args.max_component_chars, 36000),
    )
    log(args, run_start, f"Algorithm selected prompt size: {len(algorithm_selected)} chars", verbose_only=True)

    records: list[dict[str, Any]] = []
    init_record = {
        "round": 0,
        "status": "initialized",
        "accepted": False,
        "candidate_path": "",
        "case_count": len(cases),
        "reason": f"{algorithm_dir} loaded as algorithm component directory; no baseline evaluation was run",
    }
    records.append(init_record)
    write_history(history_path, init_record)
    write_summary(summary_path, records, profiles)
    log(args, run_start, f"Initialized history={history_path} summary={summary_path}")
    judge_cmd = python_command(getattr(args, "judge_python", ""))
    log(args, run_start, f"Solver judge Python: {' '.join(judge_cmd)}")

    best_eval: SolverEval | None = None
    best_value: tuple[int, float] | None = None
    teacher_eval: SolverEval | None = None
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.rounds > 0:
        try:
            log(args, run_start, f"Checking DeepSeek API key file {args.api_key_file}")
            read_api_key(args.api_key_file)
            log(args, run_start, "Checking OpenAI SDK import")
            ensure_openai_sdk()
            log(
                args,
                run_start,
                "DeepSeek settings: "
                f"model={args.model}, timeout={args.llm_timeout:.0f}s, "
                f"max_tokens={args.max_tokens if args.max_tokens > 0 else 'default'}, "
                f"reasoning_effort={args.reasoning_effort or 'off'}, thinking={args.thinking or 'off'}",
            )
        except RuntimeError as exc:
            record = {
                "round": 1,
                "status": "skipped",
                "accepted": False,
                "candidate_path": "",
                "case_count": len(cases),
                "reason": str(exc),
            }
            records.append(record)
            write_history(history_path, record)
            write_summary(summary_path, records, profiles)
            log(args, run_start, str(exc), error=True)
            return 2
    if args.rounds > 0 and not args.model:
        record = {
            "round": 1,
            "status": "skipped",
            "accepted": False,
            "candidate_path": "",
            "case_count": len(cases),
            "reason": "DeepSeek model is required for LLM rounds",
        }
        records.append(record)
        write_history(history_path, record)
        write_summary(summary_path, records, profiles)
        log(args, run_start, "DeepSeek model is required for LLM rounds.", error=True)
        return 2

    if args.rounds > 0 and not getattr(args, "disable_teacher_benchmark", False) and getattr(args, "reference_solver", ""):
        reference_path = Path(args.reference_solver)
        if reference_path.exists():
            teacher_record: dict[str, Any] = {
                "round": 0,
                "status": "teacher_pending",
                "accepted": False,
                "benchmark": True,
                "candidate_path": str(reference_path),
                "case_count": len(cases),
                "reason": "evaluating teacher solver as benchmark only; it is not an incumbent best",
            }
            try:
                log(args, run_start, f"Evaluating teacher benchmark only: {reference_path}")
                reference_source = normalize_solver_source(reference_path.read_text(encoding="utf-8", errors="replace"))
                guard_errors = static_guard(reference_source)
                if guard_errors:
                    teacher_record.update(status="teacher_rejected", reason="; ".join(guard_errors[:8]))
                    log(args, run_start, f"Teacher benchmark rejected by static guard: {teacher_record['reason']}", error=True)
                else:
                    reference_eval = evaluate_solver(reference_path, cases, case_dir, args.timeout, args.judge_python)
                    teacher_record["evaluation"] = summarize_eval(reference_eval)
                    if reference_eval.ok and reference_eval.aggregate_value()[0] > 0:
                        teacher_eval = reference_eval
                        teacher_value = reference_eval.aggregate_value()
                        teacher_record.update(
                            status="teacher_benchmark",
                            accepted=False,
                            reason="teacher benchmark evaluated; best will still start from generated candidates",
                        )
                        log(
                            args,
                            run_start,
                            f"Teacher benchmark ready: covered={teacher_value[0]}, cost={teacher_value[1]:.6f}",
                        )
                    else:
                        bad = next((r for r in reference_eval.results if not r.ok), None)
                        teacher_record.update(
                            status="teacher_rejected",
                            reason=bad.error if bad else reference_eval.error or "teacher solver did not pass evaluation",
                        )
                        log(args, run_start, f"Teacher benchmark rejected by evaluation: {teacher_record['reason']}", error=True)
            except Exception as exc:
                teacher_record.update(status="teacher_failed", reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(limit=8))
                log(args, run_start, f"Teacher benchmark failed with {type(exc).__name__}: {exc}", error=True)
            records.append(teacher_record)
            write_history(history_path, teacher_record)
            write_summary(summary_path, records, profiles)
        else:
            log(args, run_start, f"Teacher benchmark solver not found, continuing without it: {reference_path}", error=True)

    for round_no in range(1, args.rounds + 1):
        round_start = time.perf_counter()
        best_code = best_path.read_text(encoding="utf-8") if best_eval is not None and best_path.exists() else None
        learning_brief = build_learning_brief(profiles, best_eval, records, teacher_eval)
        refinement_mode = best_eval is not None and bool(best_code) and args.max_best_code_chars != 0
        exploration_mode = str(learning_brief.get("exploration_mode") or "refine")
        curriculum_stage = learning_brief.get("curriculum_stage") if isinstance(learning_brief.get("curriculum_stage"), dict) else {}
        curriculum_enabled = not getattr(args, "no_curriculum", False)
        component_policy = str(getattr(args, "component_policy", "adaptive") or "adaptive")
        stage_no = int(curriculum_stage.get("stage") or 1)
        force_orchestration_rebuild = False
        if curriculum_enabled:
            if component_policy == "open":
                release_plan = {
                    "stage": 5,
                    "name": "open_full_component_set",
                    "policy": "open",
                    "goal": "Expose the full component set immediately. Use only when one-shot strength matters more than visible learning.",
                    "components_to_use": curriculum_component_names(5, profiles),
                    "component_budget": args.max_component_chars,
                    "expected_progress": "Maximize candidate strength without a gradual learning throttle.",
                    "release_reasons": ["manual open policy"],
                }
            elif component_policy == "staged":
                release_plan = dict(curriculum_stage)
                release_plan["policy"] = "staged"
                release_plan["component_budget"] = curriculum_component_budget(stage_no, args.max_component_chars)
                release_plan["components_to_use"] = curriculum_component_names(stage_no, profiles)
            else:
                release_plan = adaptive_component_release_plan(records, best_eval, profiles, args.max_component_chars)
            learning_brief["component_release"] = release_plan
            learning_brief["curriculum_stage"] = release_plan
            stage_no = int(release_plan.get("stage") or stage_no or 1)
            force_orchestration_rebuild = orchestration_rebuild_due(records, stage_no, best_eval)
            selected_names = list(release_plan.get("components_to_use") or curriculum_component_names(stage_no, profiles))
            prompt_algorithm_library = extract_algorithm_library(
                algorithm_dir,
                prompt_mode="selected",
                target_profiles=profiles,
                max_component_chars=int(release_plan.get("component_budget") or curriculum_component_budget(stage_no, args.max_component_chars)),
                selected_names_override=selected_names,
                curriculum_stage=release_plan,
            )
            if component_policy == "staged":
                effective_prompt_mode = f"curriculum-stage{stage_no}"
            else:
                effective_prompt_mode = f"{component_policy}-level{stage_no}"
            if refinement_mode and exploration_mode in {"challenger", "radical"}:
                effective_prompt_mode += f"-{exploration_mode}"
            effective_best_code_chars = args.max_best_code_chars
            if stage_no < 5 and refinement_mode:
                effective_best_code_chars = min(args.max_best_code_chars, 45000)
        elif refinement_mode and exploration_mode in {"challenger", "radical"}:
            prompt_algorithm_library = algorithm_selected
            effective_prompt_mode = f"{exploration_mode}-selected"
            effective_best_code_chars = min(args.max_best_code_chars, 10000 if exploration_mode == "challenger" else 4000)
        else:
            prompt_algorithm_library = algorithm_summary if refinement_mode else algorithm_library
            effective_prompt_mode = "refinement-summary" if refinement_mode else args.prompt_mode
            effective_best_code_chars = args.max_best_code_chars
        duel_policy = str(getattr(args, "duel_policy", "adaptive") or "adaptive")
        if getattr(args, "no_stage_duel", False):
            duel_policy = "off"
        post_best_failures = recent_failure_counts(records, after_last_accept=True)
        adaptive_duel_due = (
            duel_policy == "adaptive"
            and refinement_mode
            and (
                force_orchestration_rebuild
                or rounds_since_last_accept(records) >= 2
                or int(post_best_failures.get("behavior_duplicate", 0)) >= 1
                or int(post_best_failures.get("cost_regression", 0)) >= 2
            )
        )
        stage_intro_duel = (
            duel_policy == "stage"
            and
            curriculum_enabled
            and refinement_mode
            and curriculum_stage_intro_due(records, stage_no, best_eval)
        )
        run_stage_duel = stage_intro_duel or (
            duel_policy in {"stage", "adaptive"}
            and curriculum_enabled
            and refinement_mode
            and (force_orchestration_rebuild or adaptive_duel_due)
        )
        if force_orchestration_rebuild and not run_stage_duel:
            apply_orchestration_rebuild_directive(learning_brief, records)
            exploration_mode = str(learning_brief.get("exploration_mode") or "orchestration_rebuild")
            effective_prompt_mode = f"curriculum-stage{stage_no}-orchestration-rebuild"
            effective_best_code_chars = 0

        variant_specs: list[dict[str, Any]] = [
            {
                "name": "refine" if run_stage_duel else "candidate",
                "learning_brief": learning_brief,
                "prompt_mode": f"{effective_prompt_mode}-refine" if run_stage_duel and curriculum_enabled else effective_prompt_mode,
                "best_code_chars": effective_best_code_chars,
                "algorithm_library": prompt_algorithm_library,
            }
        ]
        if run_stage_duel:
            recompose_brief = _copy.deepcopy(learning_brief)
            apply_stage_recompose_directive(recompose_brief, records, stage_no)
            duel_prefix = f"curriculum-stage{stage_no}" if component_policy == "staged" else f"{component_policy}-level{stage_no}"
            recompose_mode = (
                f"{duel_prefix}-orchestration-rebuild"
                if stage_no >= 5
                else f"{duel_prefix}-recompose"
            )
            variant_specs.append(
                {
                    "name": "recompose",
                    "learning_brief": recompose_brief,
                    "prompt_mode": recompose_mode,
                    "best_code_chars": 0,
                    "algorithm_library": prompt_algorithm_library,
                }
            )

        base_candidate_stem = f"solver_{timestamp}_round{round_no:02d}"
        candidate_path = out_dir / f"{base_candidate_stem}.py"
        record: dict[str, Any] = {
            "round": round_no,
            "status": "pending",
            "accepted": False,
            "candidate_path": str(candidate_path),
            "case_count": len(cases),
            "learning_brief": learning_brief,
        }
        try:
            log(args, run_start, f"Round {round_no}/{args.rounds}: evaluating {len(variant_specs)} candidate variant(s)")
            variant_results: list[tuple[dict[str, Any], dict[str, Any], SolverEval | None]] = []
            for spec in variant_specs:
                suffix = "" if len(variant_specs) == 1 else f"_{spec['name']}"
                variant_candidate_path = out_dir / f"{base_candidate_stem}{suffix}.py"
                variant_response_path = out_dir / f"{base_candidate_stem}{suffix}.response.txt"
                variant_record, variant_eval = generate_and_evaluate_candidate_variant(
                    args,
                    run_start,
                    round_no,
                    str(spec["name"]),
                    profiles,
                    best_eval,
                    records,
                    str(spec["algorithm_library"]),
                    best_code,
                    str(spec["prompt_mode"]),
                    int(spec["best_code_chars"]),
                    spec["learning_brief"],
                    variant_candidate_path,
                    variant_response_path,
                    cases,
                    case_dir,
                    teacher_eval,
                )
                variant_results.append((spec, variant_record, variant_eval))

            valid_variants = [(spec, var, ev) for spec, var, ev in variant_results if ev is not None and ev.ok]
            if not valid_variants:
                record["variants"] = [var for _, var, _ in variant_results]
                first_variant = variant_results[0][1] if variant_results else {}
                if first_variant:
                    record.update({k: v for k, v in first_variant.items() if k not in {"variant"}})
                reasons = [
                    f"{var.get('variant')}: {var.get('reason') or var.get('status') or 'no evaluation'}"
                    for _, var, _ in variant_results
                ]
                record.update(status="rejected", accepted=False, reason="all variants failed or were rejected: " + "; ".join(reasons))
                log(args, run_start, f"Round {round_no}: rejected, all variants failed")
            else:
                winner_spec, winner_record, candidate_eval = valid_variants[0]
                for spec, var, ev in valid_variants[1:]:
                    if better_value(ev.aggregate_value(), candidate_eval.aggregate_value()):
                        winner_spec, winner_record, candidate_eval = spec, var, ev
                for _, var, ev in variant_results:
                    if ev is not None and ev.ok:
                        value = ev.aggregate_value()
                        var["duel_value"] = {"covered": value[0], "cost": value[1]}
                    var["duel_winner"] = var is winner_record
                record["variants"] = [var for _, var, _ in variant_results]
                record["variant_decision"] = {
                    "enabled": len(variant_specs) > 1,
                    "winner": winner_record.get("variant"),
                    "reason": "lowest penalized aggregate among valid variants",
                }
                record["learning_brief"] = winner_spec["learning_brief"]
                record.update({k: v for k, v in winner_record.items() if k not in {"variant", "status", "aggregate"}})
                candidate_path = Path(str(winner_record.get("candidate_path") or candidate_path))
                record["candidate_path"] = str(candidate_path)
                record["evaluation"] = summarize_eval(candidate_eval)
                if best_eval is not None:
                    record["comparison_to_best"] = compare_evals(candidate_eval, best_eval, profiles)
                if teacher_eval is not None:
                    record["comparison_to_teacher"] = compare_evals(candidate_eval, teacher_eval, profiles)
                candidate_value = candidate_eval.aggregate_value()
                log(
                    args,
                    run_start,
                    f"Round {round_no}: variant winner={winner_record.get('variant')} covered={candidate_value[0]}, cost={candidate_value[1]:.6f}",
                )
                if best_value is None:
                    shutil.copy2(candidate_path, best_path)
                    best_eval = candidate_eval
                    best_value = candidate_value
                    record.update(status="accepted", accepted=True, case_count=len(cases), reason="first valid generated solver for all selected cases")
                    log(args, run_start, f"Round {round_no}: accepted as first valid generated solver")
                elif better_value(candidate_value, best_value):
                    shutil.copy2(candidate_path, best_path)
                    best_eval = candidate_eval
                    best_value = candidate_value
                    record.update(status="accepted", accepted=True, case_count=len(cases), reason="penalized aggregate proxy score improved")
                    log(args, run_start, f"Round {round_no}: accepted, improved best to {best_value}")
                else:
                    record.update(
                        status="rejected",
                        case_count=len(cases),
                        reason=f"no penalized-score improvement over best aggregate {best_value}; candidate aggregate {candidate_value}",
                    )
                    log(args, run_start, f"Round {round_no}: rejected, no aggregate improvement")
        except Exception as exc:
            record.update(status="failed", reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(limit=8))
            log(args, run_start, f"Round {round_no}: failed with {type(exc).__name__}: {exc}", error=True)
        records.append(record)
        write_history(history_path, record)
        write_summary(summary_path, records, profiles)
        log(args, run_start, f"Round {round_no}: records updated in {time.perf_counter() - round_start:.2f}s")

    if args.apply_best:
        if best_eval is None or not best_path.exists():
            log(args, run_start, "No generated best solver exists; --apply-best skipped.", error=True)
            return 1
        write_solver_source(submit_path, best_path.read_text(encoding="utf-8"))
        log(args, run_start, f"Applied best generated solver to {submit_path}")
    final_primary = best_eval.primary if best_eval is not None else None
    if final_primary and best_eval is not None:
        aggregate = best_eval.aggregate_value()
        log(
            args,
            run_start,
            f"Best aggregate: covered={aggregate[0]}, cost={aggregate[1]:.6f}, "
            f"primary_time={final_primary.duration:.3f}s, path={best_path}",
        )
    else:
        log(args, run_start, "No valid generated solver has been accepted yet.")
    log(args, run_start, f"History: {history_path}")
    log(args, run_start, f"Summary: {summary_path}")
    return 0 if final_primary is not None or args.rounds == 0 else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline LLM AutoSolver research agent.")
    parser.add_argument(
        "--case",
        action="append",
        help="Case file path, or a filename inside --case-dir. Can be provided multiple times.",
    )
    parser.add_argument(
        "--case-dir",
        default="cases",
        help="Directory containing local .txt/.tsv cases. Used entirely when --case is omitted.",
    )
    parser.add_argument("--rounds", type=int, default=3, help="Number of LLM generation rounds.")
    parser.add_argument("--out-dir", default="generated_solvers", help="Directory for generated solver_*.py files.")
    parser.add_argument("--runs-dir", default="agent_runs", help="Directory for history, summary, and best generated code.")
    parser.add_argument(
        "--algorithm-dir",
        default="algorithm_components",
        help="Directory of .py algorithm component files used for LLM composition.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("selected", "summary", "full"),
        default="selected",
        help="How much algorithm material to send to the LLM. selected is faster; full sends every component file.",
    )
    parser.add_argument(
        "--max-component-chars",
        type=int,
        default=90000,
        help="Approximate character budget for selected component source excerpts; ignored by --prompt-mode full.",
    )
    parser.add_argument(
        "--max-best-code-chars",
        type=int,
        default=70000,
        help="Character budget for including the current best generated solver in later prompts; 0 disables it.",
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Disable staged component unlocking and use --prompt-mode directly from round 1.",
    )
    parser.add_argument(
        "--component-policy",
        choices=("adaptive", "staged", "open"),
        default="adaptive",
        help="How component excerpts are released. adaptive grows by learning state; staged follows fixed round levels; open exposes the full set.",
    )
    parser.add_argument(
        "--duel-policy",
        choices=("adaptive", "stage", "off"),
        default="adaptive",
        help="When to run refine-vs-recompose candidate duels. adaptive uses stagnation/rebuild signals; stage runs on every new level; off disables duels.",
    )
    parser.add_argument(
        "--no-stage-duel",
        action="store_true",
        help="Legacy alias for --duel-policy off.",
    )
    parser.add_argument(
        "--reference-solver",
        default=DEFAULT_REFERENCE_SOLVER,
        help="Optional external teacher benchmark solver. It is evaluated for comparison only and never seeds best.",
    )
    parser.add_argument(
        "--disable-teacher-benchmark",
        action="store_true",
        help="Ignore --reference-solver even when it is provided.",
    )
    parser.add_argument(
        "--reference-guard",
        action="store_true",
        help="Optional strict first-candidate guard requiring key teacher-derived component function names.",
    )
    parser.add_argument(
        "--submit-path",
        default=str(Path("generated_solvers") / "final_generated.py"),
        help="Path overwritten only when --apply-best is used.",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Per-case runtime limit in seconds.")
    parser.add_argument(
        "--judge-python",
        default="",
        help="Python executable or command used to run generated solver.py during evaluation. Empty uses this Python.",
    )
    parser.add_argument("--llm-timeout", type=float, default=180.0, help="DeepSeek API timeout in seconds.")
    parser.add_argument("--llm-retries", type=int, default=1, help="Retry count for transient DeepSeek API failures per round.")
    parser.add_argument("--progress-interval", type=float, default=10.0, help="Seconds between DeepSeek waiting progress messages; <=0 disables heartbeat.")
    parser.add_argument("--api-key-file", default=DEFAULT_API_KEY_FILE, help="Text file containing the DeepSeek API key.")
    parser.add_argument("--base-url", default=DEFAULT_DEEPSEEK_BASE_URL, help="DeepSeek OpenAI-compatible base URL.")
    parser.add_argument("--model", default=DEFAULT_DEEPSEEK_MODEL, help="DeepSeek model name.")
    parser.add_argument("--max-tokens", type=int, default=12000, help="DeepSeek max_tokens value; <=0 uses provider default.")
    parser.add_argument("--reasoning-effort", default="", help="DeepSeek reasoning_effort value; empty string disables it.")
    parser.add_argument("--thinking", default="", help="DeepSeek extra_body thinking type; empty string disables it.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output except fatal argparse errors.")
    parser.add_argument("--verbose", action="store_true", help="Show extra progress details such as selected file paths and prompt sizes.")
    parser.add_argument("--apply-best", action="store_true", help="Copy the best generated code to --submit-path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.rounds < 0:
        parser.error("--rounds must be >= 0")
    if args.max_component_chars < 0:
        parser.error("--max-component-chars must be >= 0")
    if args.max_best_code_chars < 0:
        parser.error("--max-best-code-chars must be >= 0")
    if args.llm_retries < 0:
        parser.error("--llm-retries must be >= 0")
    try:
        return run_agent(args)
    except Exception as exc:
        if not getattr(args, "quiet", False):
            print(f"autosolver_agent failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
