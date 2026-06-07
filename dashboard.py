from __future__ import annotations

import contextlib
import datetime as dt
import html
import importlib.util
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import autosolver_agent as agent


ROOT = Path(__file__).resolve().parent
RUNS_ROOT = ROOT / "agent_runs"
CASE_DIR = ROOT / "cases"
STATE_FILE = "ui_state.json"
LOG_FILE = "agent.log"
DEFAULT_REFRESH_SECONDS = 4

MODEL_PRESETS = {
    "DeepSeek Chat（较快）": {
        "base_url": agent.DEFAULT_DEEPSEEK_BASE_URL,
        "model": "deepseek-chat",
    },
    "DeepSeek Reasoner（较慢）": {
        "base_url": agent.DEFAULT_DEEPSEEK_BASE_URL,
        "model": "deepseek-reasoner",
    },
    "DeepSeek 原默认": {
        "base_url": agent.DEFAULT_DEEPSEEK_BASE_URL,
        "model": agent.DEFAULT_DEEPSEEK_MODEL,
    },
    "自定义": {
        "base_url": agent.DEFAULT_DEEPSEEK_BASE_URL,
        "model": "deepseek-chat",
    },
}

DEFAULT_UI_ROUNDS = 6
DEFAULT_UI_TIMEOUT = agent.DEFAULT_TIMEOUT
DEFAULT_UI_LLM_TIMEOUT = 180.0
DEFAULT_UI_PROMPT_MODE = "selected"
DEFAULT_UI_MAX_TOKENS = 12000
DEFAULT_UI_MAX_COMPONENT_CHARS = 90000
DEFAULT_UI_MAX_BEST_CODE_CHARS = 70000
DEFAULT_JUDGE_PYTHON = os.environ.get("AUTOSOLVER_JUDGE_PYTHON", "")
if not DEFAULT_JUDGE_PYTHON:
    local_py36 = ROOT / ".venv36" / "Scripts" / "python.exe"
    if local_py36.exists():
        DEFAULT_JUDGE_PYTHON = str(local_py36)

def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def resolve_path(raw: str | os.PathLike[str] | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def is_pid_running(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        import ctypes.wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(process_query_limited_information, False, int(pid))
        if not handle:
            return False
        exit_code = ctypes.wintypes.DWORD()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        return bool(ok) and exit_code.value == still_active
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def stop_pid(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        proc = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=10,
        )
        return proc.returncode == 0
    with contextlib.suppress(ProcessLookupError):
        os.killpg(int(pid), signal.SIGTERM)
        return True
    with contextlib.suppress(ProcessLookupError):
        os.kill(int(pid), signal.SIGTERM)
        return True
    return False


def list_case_files() -> list[Path]:
    if not CASE_DIR.exists():
        return []
    return sorted([p for p in CASE_DIR.iterdir() if p.suffix.lower() in {".txt", ".tsv"}])


def default_case_names(case_files: list[Path]) -> list[str]:
    names = [p.name for p in case_files]
    if "tiny_seed42.txt" in names:
        return ["tiny_seed42.txt"]
    return names[:1]


def create_run_dir() -> Path:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("ui_%Y%m%d_%H%M%S")
    run_dir = RUNS_ROOT / stamp
    if not run_dir.exists():
        return run_dir
    return RUNS_ROOT / f"{stamp}_{dt.datetime.now().microsecond:06d}"


def list_run_dirs() -> list[Path]:
    runs: list[Path] = []
    if RUNS_ROOT.exists():
        for path in RUNS_ROOT.iterdir():
            if not path.is_dir():
                continue
            if (path / STATE_FILE).exists() or (path / "history.jsonl").exists():
                runs.append(path)
        if (RUNS_ROOT / "history.jsonl").exists():
            runs.append(RUNS_ROOT)
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def refresh_state(run_dir: Path | None) -> dict[str, Any]:
    if not run_dir:
        return {}
    state_path = run_dir / STATE_FILE
    state = read_json(state_path)
    pid = state.get("pid")
    status = state.get("status")
    if status == "running" and pid and not is_pid_running(int(pid)):
        state["status"] = "finished"
        state["finished_at"] = state.get("finished_at") or now_iso()
        write_json(state_path, state)
    return state


def read_history(run_dir: Path | None) -> list[dict[str, Any]]:
    if not run_dir:
        return []
    history_path = run_dir / "history.jsonl"
    if not history_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in history_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


def read_log_tail(run_dir: Path | None, max_bytes: int = 20000) -> str:
    if not run_dir:
        return ""
    path = run_dir / LOG_FILE
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        return f.read().decode("utf-8", errors="replace")


def parse_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def history_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    best_cost: float | None = None
    best_covered: int | None = None
    for rec in records:
        round_no = rec.get("round")
        if not isinstance(round_no, int):
            continue
        if round_no <= 0:
            continue
        evaluation = rec.get("evaluation") if isinstance(rec.get("evaluation"), dict) else {}
        aggregate = evaluation.get("aggregate") if isinstance(evaluation.get("aggregate"), dict) else {}
        cost = parse_float(aggregate.get("cost"))
        covered = aggregate.get("covered")
        candidate_covered = int(covered or 0)
        if cost is not None and (
            best_cost is None or best_covered is None or agent.better_value((candidate_covered, cost), (best_covered, best_cost))
        ):
            best_cost = cost
            best_covered = candidate_covered
        rows.append(
            {
                "round": round_no,
                "status": rec.get("status", ""),
                "accepted": bool(rec.get("accepted")),
                "candidate_cost": cost,
                "candidate_covered": candidate_covered,
                "best_cost": best_cost,
                "best_covered": best_covered,
                "reason": rec.get("reason", ""),
            }
        )
    return pd.DataFrame(rows)


def best_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for rec in records:
        if rec.get("benchmark") or rec.get("status") == "reference_seed":
            continue
        if not rec.get("accepted"):
            continue
        evaluation = rec.get("evaluation") if isinstance(rec.get("evaluation"), dict) else {}
        aggregate = evaluation.get("aggregate") if isinstance(evaluation.get("aggregate"), dict) else {}
        primary = evaluation.get("primary") if isinstance(evaluation.get("primary"), dict) else {}
        covered = int(aggregate.get("covered") or 0)
        best = {
            "round": rec.get("round"),
            "covered": covered,
            "cost": parse_float(aggregate.get("cost")),
            "primary_time": parse_float(primary.get("duration")),
        }
    return best


def latest_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    return records[-1] if records else {}


def derive_status(state: dict[str, Any], records: list[dict[str, Any]], log_tail: str) -> tuple[str, str]:
    if not state:
        return "Agent 未启动", "网页服务已运行。请点击左侧黄色“启动 Agent”按钮开始生成 solver.py。"
    status = state.get("status", "")
    pid = state.get("pid")
    running = status == "running" and is_pid_running(int(pid or 0))
    latest = latest_record(records)
    last_status = str(latest.get("status", ""))
    lower_log = log_tail.lower()

    if running:
        if "evaluating candidate" in lower_log or "running static guard" in lower_log:
            return "测评中", f"PID {pid} 正在验证候选 solver。"
        if "calling deepseek" in lower_log or "building deepseek prompt" in lower_log or "still waiting" in lower_log:
            return "思考中", f"PID {pid} 正在生成下一版 solver。"
        if latest.get("accepted"):
            return "已接受新最优", "刚刚产生了更优解，Agent 仍在继续。"
        if last_status in {"rejected", "failed"}:
            return "调整策略中", "上一轮未通过或未改进，Agent 正在进入下一轮。"
        return "运行中", f"PID {pid} 正在后台执行。"

    if status == "stopped":
        return "已停止", "后台进程已由页面停止。"
    if last_status in {"failed", "skipped"}:
        return "失败", str(latest.get("reason", "最近一轮失败。"))
    if records:
        return "完成", "后台运行已结束。"
    return "已创建", "运行目录已创建，等待 Agent 写入历史。"


def format_elapsed(started_at: str | None, finished_at: str | None = None) -> str:
    if not started_at:
        return "-"
    try:
        start = dt.datetime.fromisoformat(started_at)
        end = dt.datetime.fromisoformat(finished_at) if finished_at else dt.datetime.now()
    except Exception:
        return "-"
    seconds = max(0, int((end - start).total_seconds()))
    mins, sec = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours}h {mins}m {sec}s"
    if mins:
        return f"{mins}m {sec}s"
    return f"{sec}s"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(math.ceil(seconds)))
    mins, sec = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours}h {mins}m"
    if mins:
        return f"{mins}m {sec}s"
    return f"{sec}s"


def estimate_wait_text(rounds: int, case_count: int, timeout: float, llm_timeout: float) -> str:
    if rounds <= 0:
        return "预计：不调用大模型，只初始化运行记录，通常几秒内完成。"
    per_round_cap = llm_timeout + timeout * max(1, case_count) + 15.0
    total_cap = per_round_cap * rounds
    return (
        f"默认按 {DEFAULT_UI_ROUNDS} 轮、LLM 单次最多 {format_duration(DEFAULT_UI_LLM_TIMEOUT)} 运行。"
        f"当前选择 {case_count} 个 case，"
        f"保守上限约 {format_duration(per_round_cap)}/轮，{rounds} 轮约 {format_duration(total_cap)}。"
    )


def start_agent_run(
    rounds: int,
    selected_cases: list[str],
    model: str,
    base_url: str,
    api_key_file: str,
    timeout: float,
    judge_python: str,
    llm_timeout: float,
    prompt_mode: str,
    max_tokens: int,
    max_component_chars: int,
    max_best_code_chars: int,
    component_policy: str,
    duel_policy: str,
) -> Path:
    run_dir = create_run_dir()
    out_dir = run_dir / "candidates"
    run_dir.mkdir(parents=True, exist_ok=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "autosolver_agent.py"),
        "--rounds",
        str(rounds),
        "--runs-dir",
        str(run_dir),
        "--out-dir",
        str(out_dir),
        "--case-dir",
        str(CASE_DIR),
        "--model",
        model,
        "--base-url",
        base_url,
        "--api-key-file",
        api_key_file,
        "--timeout",
        str(timeout),
        "--llm-timeout",
        str(llm_timeout),
        "--prompt-mode",
        prompt_mode,
        "--max-tokens",
        str(max_tokens),
        "--max-component-chars",
        str(max_component_chars),
        "--max-best-code-chars",
        str(max_best_code_chars),
        "--component-policy",
        component_policy,
        "--duel-policy",
        duel_policy,
        "--progress-interval",
        "5",
        "--verbose",
    ]
    if judge_python.strip():
        command.extend(["--judge-python", judge_python.strip()])
    for case_name in selected_cases:
        command.extend(["--case", case_name])

    log_path = run_dir / LOG_FILE
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        log_file.write(f"[dashboard] started_at={now_iso()}\n")
        log_file.write("[dashboard] command=" + json.dumps(command, ensure_ascii=False) + "\n")
        kwargs: dict[str, Any] = {
            "cwd": str(ROOT),
            "stdin": subprocess.DEVNULL,
            "stdout": log_file,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        }
        if os.name == "nt":
            flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
            kwargs["creationflags"] = flags
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(command, **kwargs)

    state = {
        "status": "running",
        "pid": proc.pid,
        "started_at": now_iso(),
        "run_dir": str(run_dir),
        "cases": selected_cases,
        "rounds": rounds,
        "model": model,
        "base_url": base_url,
        "api_key_file": api_key_file,
        "timeout": timeout,
        "judge_python": judge_python.strip(),
        "llm_timeout": llm_timeout,
        "prompt_mode": prompt_mode,
        "max_tokens": max_tokens,
        "max_component_chars": max_component_chars,
        "max_best_code_chars": max_best_code_chars,
        "component_policy": component_policy,
        "duel_policy": duel_policy,
        "command": command,
    }
    write_json(run_dir / STATE_FILE, state)
    return run_dir


def stop_agent_run(run_dir: Path) -> bool:
    state = read_json(run_dir / STATE_FILE)
    stopped = stop_pid(int(state.get("pid") or 0))
    state["status"] = "stopped"
    state["stopped_at"] = now_iso()
    state["stop_requested"] = True
    write_json(run_dir / STATE_FILE, state)
    with (run_dir / LOG_FILE).open("a", encoding="utf-8") as log_file:
        log_file.write(f"[dashboard] stop_requested_at={state['stopped_at']} stopped={stopped}\n")
    return stopped


def delete_run_dir(run_dir: Path) -> tuple[bool, str]:
    try:
        resolved = run_dir.resolve()
        root = RUNS_ROOT.resolve()
        resolved.relative_to(root)
    except Exception:
        return False, "只能删除 agent_runs 目录下的运行记录。"
    if resolved == root:
        return False, "不能删除 agent_runs 根目录。"
    if not resolved.exists() or not resolved.is_dir():
        return False, "运行目录不存在。"
    state = read_json(resolved / STATE_FILE)
    if state.get("status") == "running" and is_pid_running(int(state.get("pid") or 0)):
        stop_agent_run(resolved)
        time.sleep(0.5)
    shutil.rmtree(resolved)
    if st.session_state.get("selected_run_dir") == str(run_dir):
        st.session_state.pop("selected_run_dir", None)
    return True, "运行记录已删除。"


def make_upload_compatible(source: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    original = source.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"
    source = agent.normalize_solver_source(source)
    remaining = source.replace("        return x.bit_count()\n    except AttributeError:\n", "")
    if ".bit_count(" in remaining:
        warnings.append("仍检测到 bit_count()，Python 3.6 可能不兼容。")
    if agent.solver_size_bytes(source) > agent.MAX_SOLVER_BYTES:
        warnings.append("文件超过 100KB，可能无法通过线上限制。")
    elif source != original:
        warnings.append("已自动规范化为 Python 3.6 上传版本。")
    return source, warnings


def export_best_solver(run_dir: Path | None) -> tuple[bool, str, Path | None]:
    if not run_dir:
        return False, "没有选择运行记录。", None
    metrics = best_metrics(read_history(run_dir))
    if not metrics:
        return False, "当前运行没有正覆盖的 best_generated.py，不能导出上传。", None
    source = run_dir / "best_generated.py"
    if not source.exists():
        return False, "当前运行还没有 best_generated.py。", None
    target = ROOT / "generated_solvers" / "final_generated.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    code, warnings = make_upload_compatible(source.read_text(encoding="utf-8"))
    agent.write_solver_source(target, code)
    size_kb = agent.solver_size_bytes(agent.normalize_solver_source(code)) / 1024.0
    suffix = f" 文件大小 {size_kb:.1f}KB。"
    if warnings:
        suffix += " " + " ".join(warnings)
    return True, f"已导出上传文件：{target}。{suffix}", target


def score_chart(records: list[dict[str, Any]]) -> go.Figure:
    df = history_frame(records)
    fig = go.Figure()
    if df.empty:
        fig.add_annotation(
            text="暂无评测记录。启动 Agent 后，覆盖数和惩罚分会在这里出现。",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            font={"size": 15, "color": "#8a7a62"},
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
    else:
        candidate_df = df.dropna(subset=["candidate_cost"])
        if not candidate_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=candidate_df["round"],
                    y=candidate_df["candidate_cost"],
                    mode="markers",
                    name="候选惩罚分",
                    marker={"size": 8, "color": "#9aa0a6", "symbol": "circle-open", "line": {"width": 1.5}},
                    customdata=candidate_df[["candidate_covered"]],
                    hovertemplate="轮次 %{x}<br>候选覆盖 %{customdata[0]}<br>候选惩罚分 %{y:.3f}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=candidate_df["round"],
                    y=candidate_df["candidate_covered"],
                    mode="lines+markers",
                    name="候选覆盖数",
                    yaxis="y2",
                    line={"color": "#8a6500", "width": 2, "dash": "dot"},
                    marker={"size": 7, "color": "#fff4bd", "line": {"color": "#e2c36a", "width": 1}},
                    hovertemplate="轮次 %{x}<br>候选覆盖 %{y}<extra></extra>",
                )
            )
        best_df = df.dropna(subset=["best_cost"])
        if not best_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=best_df["round"],
                    y=best_df["best_cost"],
                    mode="lines+markers",
                    name="历史最佳惩罚分",
                    line={"color": "#262522", "width": 3, "shape": "hv"},
                    marker={"size": 9, "color": "#262522"},
                    customdata=best_df[["best_covered"]],
                    hovertemplate="轮次 %{x}<br>历史最佳覆盖 %{customdata[0]}<br>历史最佳惩罚分 %{y:.3f}<extra></extra>",
                )
            )
        accepted_df = df[df["accepted"] & df["best_cost"].notna()]
        if not accepted_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=accepted_df["round"],
                    y=accepted_df["best_cost"],
                    mode="markers",
                    name="接受新最优",
                    marker={"size": 13, "color": "#fff4bd", "symbol": "star", "line": {"color": "#e2c36a", "width": 1}},
                    customdata=accepted_df[["best_covered"]],
                    hovertemplate="轮次 %{x}<br>新最优覆盖 %{customdata[0]}<br>新最优惩罚分 %{y:.3f}<extra></extra>",
                )
            )
        fig.update_xaxes(title_text="迭代轮数", dtick=1, gridcolor="#ece7da", zeroline=False)
        fig.update_yaxes(title_text="惩罚总分（越低越好）", gridcolor="#ece7da", zeroline=False)
    fig.update_layout(
        height=350,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(251,250,246,0.72)",
        font={"color": "#262522"},
        yaxis2={
            "title": "覆盖数（优先越高越好）",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "rangemode": "tozero",
            "zeroline": False,
        },
    )
    return fig


def latest_code_path(run_dir: Path | None, records: list[dict[str, Any]]) -> tuple[Path | None, str]:
    if not run_dir:
        return None, ""
    best_path = run_dir / "best_generated.py"
    if best_path.exists() and best_metrics(records):
        return best_path, "当前最佳 solver.py"
    for rec in reversed(records):
        candidate = resolve_path(rec.get("candidate_path"))
        if candidate and candidate.exists():
            return candidate, f"最新候选 solver.py（第 {rec.get('round')} 轮）"
    return None, ""


def read_code(path: Path | None, max_chars: int = 50000) -> str:
    if not path or not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[: max_chars // 2] + "\n\n# ... code truncated for dashboard ...\n\n" + text[-max_chars // 2 :]
    return text


def best_evaluation_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    best_record: dict[str, Any] | None = None
    best_value: tuple[int, float] | None = None
    for rec in records:
        if rec.get("benchmark") or rec.get("status") == "reference_seed":
            continue
        evaluation = rec.get("evaluation") if isinstance(rec.get("evaluation"), dict) else {}
        if not evaluation.get("ok"):
            continue
        aggregate = evaluation.get("aggregate") if isinstance(evaluation.get("aggregate"), dict) else {}
        cost = parse_float(aggregate.get("cost"))
        covered = int(aggregate.get("covered") or 0)
        if cost is None:
            continue
        value = (covered, cost)
        if best_value is None or agent.better_value(value, best_value):
            best_record = rec
            best_value = value
    return best_record


def case_task_count_from_disk(case_name: str, run_dir: Path | None = None) -> int:
    names = [case_name]
    if not Path(case_name).suffix:
        names.extend([case_name + ".txt", case_name + ".tsv"])
    roots: list[Path] = []
    if run_dir:
        roots.append(run_dir / "cases")
    roots.append(CASE_DIR)
    for root in roots:
        for name in names:
            path = root / name
            if not path.exists() or not path.is_file():
                continue
            try:
                data = agent.parse_case(path.read_text(encoding="utf-8"), path.name)
            except Exception:
                continue
            return len(data.tasks)
    return 0


def case_performance_frame(records: list[dict[str, Any]], run_dir: Path | None = None) -> pd.DataFrame:
    best_record = best_evaluation_record(records)
    if not best_record:
        return pd.DataFrame(columns=["case", "covered", "task_count", "coverage_text", "cost", "score_text", "duration_ms", "duration_text", "complete", "ok", "error"])
    evaluation = best_record.get("evaluation") if isinstance(best_record.get("evaluation"), dict) else {}
    rows: list[dict[str, Any]] = []
    for item in evaluation.get("cases", []):
        if not isinstance(item, dict):
            continue
        case_name = str(item.get("name", ""))
        covered = int(item.get("covered") or 0)
        task_count = int(item.get("task_count") or 0)
        if task_count <= 0:
            task_count = case_task_count_from_disk(case_name, run_dir)
        completion_rate = parse_float(item.get("completion_rate"))
        if completion_rate is None:
            completion_rate = covered / float(task_count) if task_count > 0 else None
        complete = bool(item.get("complete")) or (bool(item.get("ok")) and task_count > 0 and covered >= task_count)
        cost = parse_float(item.get("cost"))
        raw_cost = parse_float(item.get("raw_cost"))
        missing_penalty = parse_float(item.get("missing_penalty")) or 0.0
        missing_count = int(item.get("missing_count") or max(0, task_count - covered))
        if raw_cost is None and cost is not None:
            raw_cost = cost - missing_penalty
        duration = parse_float(item.get("duration")) or 0.0
        duration_ms = int(round(duration * 1000.0))
        if task_count > 0 and completion_rate is not None:
            coverage_text = f"{covered}/{task_count}({completion_rate * 100.0:.0f}%)"
        else:
            coverage_text = str(covered)
        rows.append(
            {
                "case": case_name,
                "covered": covered,
                "task_count": task_count,
                "coverage_text": coverage_text,
                "cost": cost,
                "raw_cost": raw_cost,
                "missing_count": missing_count,
                "missing_penalty": missing_penalty,
                "score_text": f"{cost:,.2f}" if cost is not None else "-",
                "missing_text": f"{missing_count} / {missing_penalty:,.0f}",
                "duration_ms": duration_ms,
                "duration_text": f"{duration_ms}ms",
                "complete": complete,
                "ok": bool(item.get("ok")),
                "error": str(item.get("error") or ""),
            }
        )
    df = pd.DataFrame(rows)
    return df


def case_performance_panel(records: list[dict[str, Any]], run_dir: Path | None = None) -> None:
    df = case_performance_frame(records, run_dir)
    if df.empty:
        st.info("暂无 case 级评测数据。首次候选 solver 通过评测后会显示。")
        return
    finite_costs = [float(x) for x in df["cost"].tolist() if isinstance(x, (int, float)) and math.isfinite(float(x))]
    avg_cost = sum(finite_costs) / len(finite_costs) if finite_costs else None
    complete_count = int(df["complete"].sum()) if "complete" in df else 0
    metric_cols = st.columns(2)
    metric_cols[0].metric("平均惩罚分数", f"{avg_cost:,.2f}" if avg_cost is not None else "-")
    metric_cols[1].metric("完成算例", f"{complete_count} / {len(df)}")
    display_df = df[["case", "score_text", "coverage_text", "missing_text", "duration_text"]].rename(
        columns={
            "case": "case",
            "score_text": "分数",
            "coverage_text": "覆盖",
            "missing_text": "缺失/惩罚",
            "duration_text": "耗时",
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=294)
    errors = df[df["error"].astype(str) != ""]
    if not errors.empty:
        with st.expander("未通过或异常 case", expanded=False):
            st.dataframe(errors[["case", "error"]], use_container_width=True, hide_index=True)


def recent_rounds_frame(records: list[dict[str, Any]], limit: int = 8) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in records:
        round_no = rec.get("round")
        if not isinstance(round_no, int) or round_no <= 0:
            continue
        evaluation = rec.get("evaluation") if isinstance(rec.get("evaluation"), dict) else {}
        aggregate = evaluation.get("aggregate") if isinstance(evaluation.get("aggregate"), dict) else {}
        primary = evaluation.get("primary") if isinstance(evaluation.get("primary"), dict) else {}
        variants = rec.get("variants") if isinstance(rec.get("variants"), list) else []
        variant_decision = rec.get("variant_decision") if isinstance(rec.get("variant_decision"), dict) else {}
        rows.append(
            {
                "round": round_no,
                "status": rec.get("status", ""),
                "accepted": bool(rec.get("accepted")),
                "winner": variant_decision.get("winner") or rec.get("variant") or "",
                "variants": len(variants) if variants else 1,
                "covered": int(aggregate.get("covered") or 0),
                "cost": parse_float(aggregate.get("cost")),
                "primary_time": parse_float(primary.get("duration")),
                "diagnosis": agent.prompt_digest(rec.get("diagnosis"), 180),
                "components": agent.prompt_digest(rec.get("selected_components"), 120),
                "reason": str(rec.get("reason") or primary.get("error") or "")[:160],
            }
        )
    return pd.DataFrame(rows[-limit:])


def render_sidebar() -> dict[str, Any]:
    st.sidebar.title("控制与状态")

    missing = [name for name in ("openai", "ortools") if not package_available(name)]
    case_files = list_case_files()
    case_names = [p.name for p in case_files]
    selected_cases = st.sidebar.multiselect(
        "测试用例",
        case_names,
        default=default_case_names(case_files),
        help="默认用 tiny case 做短跑演示；正式运行可选择多个或全部 case。",
    )
    if case_names and len(selected_cases) != len(case_names):
        st.sidebar.caption(f"当前只在 {len(selected_cases)}/{len(case_names)} 个 case 上训练/评测；上传网站通常会评全部 case。")
    if missing:
        st.sidebar.warning("缺少依赖：" + ", ".join(missing) + "。关闭当前启动窗口后，重新双击 start_dashboard.bat 会自动安装。")
    else:
        st.sidebar.success("核心依赖已就绪。")
    rounds = st.sidebar.slider("迭代轮数", 0, 30, DEFAULT_UI_ROUNDS, 1)

    run_dirs = list_run_dirs()
    remembered = st.session_state.get("selected_run_dir")
    options = [""] + [str(p) for p in run_dirs]
    if remembered and remembered not in options and Path(remembered).exists():
        options.insert(1, remembered)
    index = options.index(remembered) if remembered in options else 0
    selected_run = st.sidebar.selectbox(
        "查看运行",
        options,
        index=index,
        format_func=lambda raw: "未选择" if not raw else Path(raw).name,
    )
    if selected_run:
        st.session_state["selected_run_dir"] = selected_run
    selected_run_dir = Path(selected_run) if selected_run else None

    if selected_run_dir:
        confirm_delete = st.sidebar.checkbox("确认删除当前运行记录", value=False)
        delete_clicked = st.sidebar.button("删除当前运行", use_container_width=True, disabled=not confirm_delete)
    else:
        confirm_delete = False
        delete_clicked = False

    col_start, col_stop = st.sidebar.columns(2)
    with col_start:
        start_clicked = st.button("启动 Agent", type="primary", use_container_width=True, disabled=not selected_cases)
    with col_stop:
        stop_clicked = st.button("停止", use_container_width=True, disabled=selected_run_dir is None)

    if missing and rounds > 0:
        st.sidebar.caption("提示：真实生成 solver.py 前需要依赖完整；最新版 start_dashboard.bat 会自动检查并安装。")

    with st.sidebar.expander("高级参数", expanded=False):
        timeout = st.slider("单 case 评测超时（秒）", 0.5, 20.0, float(DEFAULT_UI_TIMEOUT), 0.5)
        judge_python = st.text_input("solver 评测 Python", value=DEFAULT_JUDGE_PYTHON)
        if judge_python.strip():
            st.caption("候选 solver.py 会用这个解释器评测；Agent/网页仍用当前 Python。")
        else:
            st.caption("留空时，候选 solver.py 使用当前启动网页的 Python 评测。")
        llm_timeout = st.slider("大模型最长等待（秒）", 30.0, 3600.0, float(DEFAULT_UI_LLM_TIMEOUT), 30.0)
        preset = st.selectbox("大模型 API", list(MODEL_PRESETS), index=0)
        preset_values = MODEL_PRESETS[preset]
        model = st.text_input("模型名", value=preset_values["model"])
        base_url = st.text_input("Base URL", value=preset_values["base_url"])
        api_key_file = st.text_input("API Key 文件", value=agent.DEFAULT_API_KEY_FILE)
        prompt_options = ["summary", "selected", "full"]
        prompt_mode = st.selectbox("Prompt 模式", prompt_options, index=prompt_options.index(DEFAULT_UI_PROMPT_MODE))
        component_policy = st.selectbox("Component policy", ["adaptive", "staged", "open"], index=0)
        duel_policy = st.selectbox("Duel policy", ["adaptive", "stage", "off"], index=0)
        max_tokens = st.slider("max_tokens", 0, 32000, DEFAULT_UI_MAX_TOKENS, 1000)
        max_component_chars = st.slider("组件代码预算", 0, 120000, DEFAULT_UI_MAX_COMPONENT_CHARS, 1000)
        max_best_code_chars = st.slider("历史最佳代码预算", 0, 100000, DEFAULT_UI_MAX_BEST_CODE_CHARS, 5000)
        refresh = st.checkbox("自动刷新", value=True)
        refresh_seconds = st.slider("刷新间隔（秒）", 2, 15, DEFAULT_REFRESH_SECONDS, 1)

    st.sidebar.caption(
        estimate_wait_text(
            rounds,
            len(selected_cases),
            float(timeout),
            float(llm_timeout),
        )
    )

    if start_clicked:
        run_dir = start_agent_run(
            rounds,
            selected_cases,
            model,
            base_url,
            api_key_file,
            timeout,
            judge_python,
            llm_timeout,
            prompt_mode,
            max_tokens,
            max_component_chars,
            max_best_code_chars,
            component_policy,
            duel_policy,
        )
        st.session_state["selected_run_dir"] = str(run_dir)
        st.toast(f"Agent 已启动：{run_dir.name}")
        st.rerun()

    if stop_clicked and selected_run_dir:
        stop_agent_run(selected_run_dir)
        st.toast("已发送停止请求")
        st.rerun()

    if delete_clicked and selected_run_dir:
        ok, message = delete_run_dir(selected_run_dir)
        if ok:
            st.toast(message)
            st.rerun()
        st.sidebar.error(message)

    return {
        "selected_run_dir": selected_run_dir,
        "selected_cases": selected_cases,
        "refresh": refresh,
        "refresh_seconds": refresh_seconds,
    }


def render_dashboard() -> None:
    st.set_page_config(page_title="AutoSolver Agent Monitor", page_icon="A", layout="wide")
    st.markdown(
        """
        <style>
        :root {
            --md-bg: #f6f4ee;
            --md-surface: #ffffff;
            --md-surface-soft: #fbfaf6;
            --md-border: #e7e1d4;
            --md-ink: #262522;
            --md-muted: #736f66;
            --md-primary: #262522;
            --md-primary-soft: #ebe7dc;
            --md-secondary: #fff4bd;
            --md-secondary-soft: #fff4bd;
            --md-secondary-ink: #6f5200;
            --md-secondary-border: #e2c36a;
            --md-secondary-paper: linear-gradient(180deg, #fff9dc 0%, #fff4bd 100%);
            --md-secondary-paper-hover: var(--md-secondary-paper);
            --md-radius: 18px;
            --md-gap: 14px;
            --md-elevation-1: 0 1px 2px rgba(31, 41, 55, 0.12), 0 1px 3px rgba(31, 41, 55, 0.08);
            --md-elevation-2: 0 4px 10px rgba(31, 41, 55, 0.12), 0 2px 4px rgba(31, 41, 55, 0.08);
            --md-elevation-3: 0 12px 28px rgba(31, 41, 55, 0.16), 0 4px 10px rgba(31, 41, 55, 0.08);
        }
        .stApp {
            background:
                radial-gradient(circle at 10% 0%, rgba(255, 244, 189, 0.72), transparent 22rem),
                radial-gradient(circle at 95% 16%, rgba(38, 37, 34, 0.08), transparent 20rem),
                linear-gradient(180deg, #fffdf5 0%, var(--md-bg) 58%, #ebe6d8 100%);
            color: var(--md-ink);
            font-family: Roboto, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        section.main > div.block-container {
            max-width: 1680px;
            padding: 1.05rem 1.35rem 1.2rem;
        }
        section[data-testid="stSidebar"] {
            background: #f7f5ee;
            border-right: 1px solid var(--md-border);
            box-shadow: 2px 0 10px rgba(31, 41, 55, 0.06);
        }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.58rem;
        }
        div[data-testid="stMetric"],
        div[data-testid="stDataFrame"],
        div[data-testid="stPlotlyChart"],
        div[data-testid="stCodeBlock"],
        div[data-testid="stExpander"] {
            border-radius: var(--md-radius);
            background: var(--md-surface);
            border: 1px solid rgba(223, 229, 238, 0.92);
            box-shadow: var(--md-elevation-1);
            padding: 0.78rem;
            transition: box-shadow 160ms ease, transform 160ms ease, border-color 160ms ease;
        }
        div[data-testid="stMetric"]:hover,
        div[data-testid="stDataFrame"]:hover,
        div[data-testid="stPlotlyChart"]:hover,
        div[data-testid="stCodeBlock"]:hover,
        div[data-testid="stExpander"]:hover {
            box-shadow: var(--md-elevation-2);
            transform: translateY(-1px);
            border-color: rgba(226, 195, 106, 0.58);
        }
        div[data-testid="stMetric"] {
            min-height: 96px;
            background:
                linear-gradient(180deg, rgba(255,255,255,0.98), rgba(251,250,246,0.96)),
                var(--md-surface);
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--md-ink);
            font-weight: 650;
            letter-spacing: 0;
        }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: var(--md-muted);
        }
        div[data-testid="stCodeBlock"] {
            max-height: 420px;
            overflow: auto;
        }
        h1 {
            color: var(--md-ink);
            font-weight: 700;
            letter-spacing: 0;
            margin-bottom: 0.18rem;
        }
        h2, h3 {
            color: var(--md-ink);
            font-weight: 650;
            letter-spacing: 0;
        }
        .stMarkdown h2, .stMarkdown h3 {
            margin-top: 0.45rem;
            margin-bottom: 0.55rem;
        }
        div[data-testid="stSidebar"] button {
            border-radius: 999px;
            border: 1px solid rgba(31, 41, 55, 0.10);
            box-shadow: var(--md-elevation-1);
            transition: transform 140ms ease, box-shadow 140ms ease;
        }
        div[data-testid="stSidebar"] button:hover {
            transform: translateY(-1px);
            box-shadow: var(--md-elevation-2);
        }
        div[data-testid="stSidebar"] button[kind="primary"] {
            background: var(--md-secondary-paper);
            border-color: var(--md-secondary-border);
            color: var(--md-secondary-ink);
        }
        div[data-testid="stDecoration"] {
            background-image: var(--md-secondary-paper) !important;
        }
        button[kind="primary"],
        button[data-testid="stBaseButton-primary"],
        div[data-testid="stButton"] button[kind="primary"],
        div[data-testid="stSidebar"] button[kind="primary"] {
            background: var(--md-secondary-paper) !important;
            border-color: var(--md-secondary-border) !important;
            color: var(--md-secondary-ink) !important;
            box-shadow: var(--md-elevation-2) !important;
        }
        button[kind="primary"]:hover,
        button[data-testid="stBaseButton-primary"]:hover,
        div[data-testid="stButton"] button[kind="primary"]:hover,
        div[data-testid="stSidebar"] button[kind="primary"]:hover {
            background: var(--md-secondary-paper-hover) !important;
        }
        button[kind="secondary"],
        button[data-testid="stBaseButton-secondary"] {
            color: var(--md-ink) !important;
            border-color: var(--md-border) !important;
            background: rgba(255,255,255,0.76) !important;
        }
        div[data-baseweb="select"] > div,
        div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
        input, textarea {
            border-radius: 12px !important;
            border-color: var(--md-border) !important;
        }
        div[data-baseweb="tag"],
        span[data-baseweb="tag"],
        div[data-testid="stMultiSelect"] div[data-baseweb="tag"] {
            background: var(--md-secondary-paper) !important;
            border-color: var(--md-secondary-border) !important;
            color: var(--md-secondary-ink) !important;
            box-shadow: var(--md-elevation-1) !important;
        }
        div[data-baseweb="tag"] svg,
        div[data-baseweb="tag"] button,
        span[data-baseweb="tag"] svg,
        span[data-baseweb="tag"] button {
            color: var(--md-secondary-ink) !important;
            fill: var(--md-secondary-ink) !important;
        }
        div[data-testid="stAlert"] {
            background: var(--md-secondary-paper) !important;
            border: 1px solid var(--md-secondary-border) !important;
            color: var(--md-secondary-ink) !important;
            box-shadow: var(--md-elevation-1);
            border-radius: var(--md-radius);
        }
        div[data-testid="stAlert"] *,
        div[data-testid="stAlert"] svg {
            color: var(--md-secondary-ink) !important;
            fill: var(--md-secondary-ink) !important;
        }
        div[data-testid="stSlider"] [role="slider"] {
            background: var(--md-secondary-paper) !important;
            border-color: var(--md-secondary-border) !important;
            box-shadow: 0 0 0 4px rgba(255, 244, 189, 0.58) !important;
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] div[style*="255, 75, 75"],
        div[data-testid="stSlider"] [data-baseweb="slider"] div[style*="rgb(255, 75, 75)"],
        div[data-testid="stSlider"] [data-baseweb="slider"] div[style*="red"],
        div[data-testid="stSlider"] [data-baseweb="slider"] span[style*="255, 75, 75"],
        div[data-testid="stSlider"] [data-baseweb="slider"] span[style*="rgb(255, 75, 75)"] {
            background: var(--md-secondary-paper) !important;
            color: var(--md-secondary-ink) !important;
            border-color: var(--md-secondary-border) !important;
        }
        [style*="rgb(255, 75, 75)"],
        [style*="rgb(0, 104, 201)"],
        [style*="rgb(9, 171, 59)"],
        [style*="#ff4b4b"],
        [style*="#0068c9"],
        [style*="#09ab3b"] {
            color: var(--md-secondary-ink) !important;
            border-color: var(--md-secondary-border) !important;
        }
        a, a:visited {
            color: #8a6500 !important;
        }
        .status-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.28rem 0.72rem;
            background: var(--md-secondary-paper);
            color: var(--md-secondary-ink);
            font-weight: 700;
            border: 1px solid var(--md-secondary-border);
            box-shadow: var(--md-elevation-1);
        }
        .subtle {
            color: var(--md-muted);
            font-size: 0.92rem;
        }
        .material-hero {
            border-radius: 24px;
            padding: 1rem 1.14rem;
            background:
                radial-gradient(circle at 88% 8%, rgba(255, 244, 189, 0.78), transparent 14rem),
                linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(255, 244, 189, 0.74)),
                var(--md-surface);
            border: 1px solid rgba(38, 37, 34, 0.10);
            box-shadow: var(--md-elevation-2);
            margin-bottom: 0.8rem;
        }
        .material-hero-row {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 1.2rem;
        }
        .material-hero-kicker {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.24rem 0.72rem;
            color: var(--md-ink);
            background: rgba(255,255,255,0.68);
            border: 1px solid rgba(32, 33, 36, 0.08);
            box-shadow: var(--md-elevation-1);
            font-size: 0.86rem;
            margin-bottom: 0.48rem;
        }
        .material-hero-side {
            min-width: 13rem;
            text-align: right;
            color: var(--md-muted);
            font-size: 0.92rem;
        }
        .material-hero-side strong {
            display: block;
            color: var(--md-ink);
            font-size: 1.45rem;
            font-weight: 650;
            margin-bottom: 0.08rem;
        }
        .material-hero h1 {
            margin: 0 0 0.42rem;
            font-size: clamp(2rem, 4vw, 3.1rem);
        }
        .material-section-label {
            display: inline-flex;
            align-items: center;
            gap: 0.42rem;
            color: var(--md-muted);
            font-weight: 650;
            font-size: 0.96rem;
            margin: 0.1rem 0 0.35rem;
        }
        .material-section-label::before {
            content: "";
            width: 0.58rem;
            height: 0.58rem;
            border-radius: 999px;
            background: var(--md-primary);
            box-shadow: 0 0 0 4px var(--md-primary-soft);
        }
        .material-section-label.secondary::before {
            background: var(--md-secondary-paper);
            border: 1px solid var(--md-secondary-border);
            box-shadow: 0 0 0 4px var(--md-secondary-soft);
        }
        @media (prefers-reduced-motion: reduce) {
            div[data-testid="stMetric"],
            div[data-testid="stDataFrame"],
            div[data-testid="stPlotlyChart"],
            div[data-testid="stCodeBlock"],
            div[data-testid="stExpander"],
            div[data-testid="stSidebar"] button {
                transition: none;
                transform: none !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    controls = render_sidebar()
    run_dir = controls["selected_run_dir"]
    state = refresh_state(run_dir)
    records = read_history(run_dir)
    log_tail = read_log_tail(run_dir)
    status_label, status_detail = derive_status(state, records, log_tail)
    best = best_metrics(records)

    run_name = html.escape(run_dir.name if run_dir else "-")
    round_count = sum(1 for r in records if isinstance(r.get("round"), int) and r.get("round", 0) > 0)
    best_covered_text = html.escape(str(best.get("covered", "-") if best else "-"))
    best_cost = best.get("cost") if best else None
    best_cost_text = f"{best_cost:.3f}" if isinstance(best_cost, float) else "-"
    st.markdown(
        f"""
        <div class="material-hero">
            <div class="material-hero-row">
                <div>
                    <div class="material-hero-kicker">AutoSolver Agent Monitor</div>
                    <h1>AutoSolver Agent 监控大屏</h1>
                    <span class="status-pill">{html.escape(status_label)}</span>
                    <span class="subtle">{html.escape(status_detail)}</span>
                </div>
                <div class="material-hero-side">
                    <strong>{best_cost_text}</strong>
                    最佳惩罚分 · 覆盖 {best_covered_text}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns([1.35, 1.0, 1.0, 1.0])
    metric_cols[0].metric("运行目录", run_dir.name if run_dir else "-")
    metric_cols[1].metric("已记录轮次", round_count)
    metric_cols[2].metric("最佳覆盖数", best.get("covered", "-") if best else "-")
    metric_cols[3].metric("最佳惩罚分", f"{best_cost:.3f}" if isinstance(best_cost, float) else "-")

    top_left, top_right = st.columns([1.45, 1.0])
    with top_left:
        st.markdown('<div class="material-section-label">覆盖与惩罚分变化</div>', unsafe_allow_html=True)
        st.plotly_chart(score_chart(records), use_container_width=True)
    with top_right:
        st.markdown('<div class="material-section-label secondary">Case 表现诊断</div>', unsafe_allow_html=True)
        case_performance_panel(records, run_dir)

    lower_left, lower_right = st.columns([1.0, 1.35])
    with lower_left:
        rounds_df = recent_rounds_frame(records)
        if not rounds_df.empty:
            st.markdown('<div class="material-section-label">最近迭代记录</div>', unsafe_allow_html=True)
            st.dataframe(rounds_df, use_container_width=True, hide_index=True, height=238)
        with st.expander("Agent 日志", expanded=False):
            st.text(log_tail or "暂无日志。")

    with lower_right:
        st.markdown('<div class="material-section-label secondary">当前最佳 solver.py</div>', unsafe_allow_html=True)
        code_path, code_label = latest_code_path(run_dir, records)
        if code_path:
            st.caption(f"{code_label}: {code_path}")
            st.code(read_code(code_path), language="python", line_numbers=True)
        else:
            st.info("还没有生成 solver.py。启动 Agent 后，这里会显示最新候选或当前最佳代码。")

    running = state.get("status") == "running" and is_pid_running(int(state.get("pid") or 0))
    if controls["refresh"] and running:
        time.sleep(int(controls["refresh_seconds"]))
        st.rerun()


if __name__ == "__main__":
    render_dashboard()
