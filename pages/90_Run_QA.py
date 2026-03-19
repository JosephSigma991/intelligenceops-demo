from __future__ import annotations

import hashlib
import os
import re
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Run & QA", layout="wide")

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_LOGS_DIR = REPO_ROOT / "artifacts" / "run_logs"
STAMP_IN_FILENAME_RE = re.compile(r"__([0-9]{8}_[0-9]{6}|[0-9]{14})(?:\D|$)")
ABS_WIN_PATH_RE = re.compile(r"^(?:[A-Za-z]:\\|\\\\)")

def safe_read_csv(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    try:
        return pd.read_csv(path), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def maybe_get(d: dict[str, Any], *keys: str) -> Any:
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d:
            return d[k]
    lowered = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        lk = str(k).lower()
        if lk in lowered:
            return lowered[lk]
    return None


def ensure_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    if isinstance(v, tuple):
        return [str(x) for x in v if str(x).strip()]
    return [str(v)] if str(v).strip() else []

def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            while True:
                chunk = f.read(block_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def validate_required_files(required_files: list[str], insights_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for f in required_files:
        p = Path(f)
        full = p if p.is_absolute() else (insights_dir / p)
        exists = full.exists()
        size = full.stat().st_size if exists else 0
        mtime = (
            datetime.fromtimestamp(full.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            if exists
            else None
        )
        rows.append(
            {
                "File": str(p),
                "Exists": exists,
                "NonEmpty": bool(exists and size > 0),
                "SizeBytes": int(size) if exists else 0,
                "LastModified": mtime,
                "FullPath": str(full),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df, df
    fail_df = df[(~df["Exists"]) | (~df["NonEmpty"])].copy()
    return df, fail_df


def path_info(p: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "exists": False,
        "size_bytes": None,
        "mtime_iso": None,
    }
    if p is None:
        return out
    try:
        if not p.exists():
            return out
        stv = p.stat()
        out["exists"] = True
        out["size_bytes"] = int(stv.st_size) if p.is_file() else None
        out["mtime_iso"] = datetime.fromtimestamp(stv.st_mtime).isoformat(timespec="seconds")
        return out
    except Exception:
        return out


def detect_stamp_from_filename(p: Path) -> str | None:
    m = STAMP_IN_FILENAME_RE.search(p.name)
    if not m:
        return None
    return m.group(1)


def normalize_stamp_token(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    m = re.search(r"([0-9]{8}_[0-9]{6}|[0-9]{14})", s)
    if not m:
        return None
    token = m.group(1)
    digits = re.sub(r"\D", "", token)
    return digits if len(digits) == 14 else None


def collect_string_paths(run_stamp: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def walk(node: Any, key_path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                kp = f"{key_path}.{k}" if key_path else str(k)
                walk(v, kp)
            return
        if isinstance(node, list):
            for i, v in enumerate(node):
                kp = f"{key_path}[{i}]" if key_path else f"[{i}]"
                walk(v, kp)
            return
        if isinstance(node, str):
            s = node.strip()
            if not ABS_WIN_PATH_RE.match(s):
                return
            p = Path(s)
            info = path_info(p)
            out.append(
                {
                    "key_path": key_path or "<root>",
                    "value": s,
                    "exists": info["exists"],
                    "size_bytes": info["size_bytes"],
                    "mtime_iso": info["mtime_iso"],
                }
            )

    walk(run_stamp, "")
    return out


def detect_text_encoding(p: Path) -> str:
    try:
        b = p.read_bytes()
    except Exception:
        return "unreadable"
    if len(b) >= 2 and b[0] == 0xFF and b[1] == 0xFE:
        return "utf-16le (bom)"
    if len(b) >= 2 and b[0] == 0xFE and b[1] == 0xFF:
        return "utf-16be (bom)"
    if len(b) >= 3 and b[0] == 0xEF and b[1] == 0xBB and b[2] == 0xBF:
        return "utf-8-sig"
    if len(b) > 0:
        nulls = b.count(0)
        if (nulls / len(b)) > 0.2:
            return "utf-16le (heuristic)"
    return "utf-8"


def tail_text_file(path: Path, lines: int = 200) -> str:
    if not path.exists():
        return f"[missing] {path}"
    try:
        q: deque[str] = deque(maxlen=lines)
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                q.append(line.rstrip("\n"))
        return "\n".join(q) if q else "[empty file]"
    except Exception as e:
        return f"[read error] {type(e).__name__}: {e}"


def latest_logs(run_logs_dir: Path, n: int = 5) -> list[Path]:
    if not run_logs_dir.exists():
        return []
    files = [p for p in run_logs_dir.iterdir() if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:n]


def read_text_robust(p: Path) -> str:
    b = p.read_bytes()
    if len(b) >= 2 and b[0] == 0xFF and b[1] == 0xFE:
        return b.decode("utf-16le", errors="ignore")
    if len(b) >= 2 and b[0] == 0xFE and b[1] == 0xFF:
        return b.decode("utf-16be", errors="ignore")
    if len(b) >= 3 and b[0] == 0xEF and b[1] == 0xBB and b[2] == 0xBF:
        return b.decode("utf-8-sig", errors="ignore")
    nulls = b.count(0)
    if len(b) > 0 and (nulls / len(b)) > 0.2:
        return b.decode("utf-16le", errors="ignore")
    return b.decode("utf-8", errors="ignore")


def find_latest_ops_gate_log(run_logs_dir: Path, region: str, mode: str) -> Path | None:
    if not run_logs_dir.exists():
        return None
    candidates = sorted(
        run_logs_dir.glob("gate_ops_pack__*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    for p in candidates:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        ops_lines = [ln for ln in lines if ln.startswith("OPS_GATE=")]
        if not ops_lines:
            continue
        last = ops_lines[-1]
        if f"Region={region}" in last and f"Mode={mode}" in last:
            return p
    return candidates[0]


def extract_ops_gate_line(p: Path) -> str | None:
    try:
        text = read_text_robust(p)
    except Exception:
        return None
    matches = re.findall(r"(?im)^\s*OPS_GATE=.*$", text)
    if matches:
        return matches[-1].strip()
    for line in reversed(text.splitlines()):
        idx = line.find("OPS_GATE=")
        if idx >= 0:
            return line[idx:].strip()
    return None


def tail_text(p: Path, max_lines: int = 60) -> str:
    try:
        lines = read_text_robust(p).splitlines()
    except Exception as e:
        return f"[read error] {type(e).__name__}: {e}"
    if not lines:
        return "[empty file]"
    return "\n".join(lines[-max_lines:])


st.title("Run & QA")
st.caption("PASS-only provenance panel. Reads published artifacts and run logs only.")

ctx = st.session_state.get("mode_a_ctx")
if not isinstance(ctx, dict):
    st.error("Select a run in the sidebar")
    st.stop()

run_stamp = ctx.get("run_stamp") if isinstance(ctx.get("run_stamp"), dict) else {}
selected_path = Path(ctx.get("run_stamp_json_path", ""))
region = str(ctx.get("region", "")).strip()
mode = str(ctx.get("mode", "")).strip()
stamp = ctx.get("stamp")
git_branch = ctx.get("git_branch")
git_commit = ctx.get("git_commit")
qa_path = ctx.get("qa_path")
required_files = ensure_list(ctx.get("required_files"))
artifacts_by_name = ctx.get("artifacts_by_name", {}) or {}
insights_dir = Path(ctx.get("insights_dir", Path(os.getenv("FLIGHTOPS_INSIGHTS_DIR", r"C:\Users\IT\02_insights\insight_out"))))

if not region or not mode:
    st.error("Run context is missing region/mode.")
    st.stop()
ops_gate_verdict_ref = maybe_get(run_stamp, "ops_gate_pack_verdict", "OpsGatePackVerdict")
ops_gate_log_path_ref = maybe_get(run_stamp, "ops_gate_pack_log_path", "OpsGatePackLogPath")
ops_gate_stamp_ref = maybe_get(run_stamp, "ops_gate_pack_stamp", "OpsGatePackStamp")
ops_gate_log_ref: Path | None = None
if isinstance(ops_gate_log_path_ref, str) and ops_gate_log_path_ref.strip():
    p = Path(ops_gate_log_path_ref.strip())
    ops_gate_log_ref = p if p.is_absolute() else (REPO_ROOT / p)
latest_ops_gate_log = find_latest_ops_gate_log(RUN_LOGS_DIR, region, mode)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Stamp", str(stamp) if stamp else "<missing>")
col2.metric("Git Commit", str(git_commit) if git_commit else "<missing>")
col3.metric("Required Files", str(len(required_files)))
col4.metric("Scope", f"{region}/{mode}")

with st.expander("Run Stamp Details", expanded=True):
    st.write(f"Path: `{selected_path}`")
    st.write(f"SHA256: `{file_sha256(selected_path) or '<unavailable>'}`")
    if qa_path is not None:
        st.write(f"QA Summary Path (resolved): `{qa_path}`")
    else:
        st.write("QA Summary Path (resolved): `<missing>`")
    coverage = {k: v for k, v in run_stamp.items() if "coverage" in str(k).lower()}
    if coverage:
        st.json(coverage)
    st.json(run_stamp)

st.subheader("Evidence & Consistency")
qa_info = path_info(qa_path)
ops_ref_info = path_info(ops_gate_log_ref)
latest_ops_info = path_info(latest_ops_gate_log)
ops_stamp_from_run_stamp = str(ops_gate_stamp_ref).strip() if isinstance(ops_gate_stamp_ref, str) else ""
latest_ops_stamp = detect_stamp_from_filename(latest_ops_gate_log) if latest_ops_gate_log is not None else None
run_ops_stamp_norm = normalize_stamp_token(ops_stamp_from_run_stamp)
latest_ops_stamp_norm = normalize_stamp_token(latest_ops_stamp)

staleness_status = "MISSING"
if ops_ref_info["exists"] and run_ops_stamp_norm is not None and latest_ops_stamp_norm is not None:
    if run_ops_stamp_norm == latest_ops_stamp_norm:
        staleness_status = "OK"
    else:
        staleness_status = "STALE"

evidence_rows = [
    {
        "field": "run_stamp.stamp",
        "value": str(stamp) if stamp else "<missing>",
        "exists": None,
        "size_bytes": None,
        "mtime_iso": None,
    },
    {
        "field": "run_stamp.region",
        "value": str(region) if region else "<missing>",
        "exists": None,
        "size_bytes": None,
        "mtime_iso": None,
    },
    {
        "field": "run_stamp.mode",
        "value": str(mode) if mode else "<missing>",
        "exists": None,
        "size_bytes": None,
        "mtime_iso": None,
    },
    {
        "field": "run_stamp.git_branch",
        "value": str(git_branch) if git_branch else "<missing>",
        "exists": None,
        "size_bytes": None,
        "mtime_iso": None,
    },
    {
        "field": "run_stamp.git_commit",
        "value": str(git_commit) if git_commit else "<missing>",
        "exists": None,
        "size_bytes": None,
        "mtime_iso": None,
    },
    {
        "field": "qa_summary_path",
        "value": str(qa_path) if qa_path is not None else "<missing>",
        "exists": qa_info["exists"],
        "size_bytes": qa_info["size_bytes"],
        "mtime_iso": qa_info["mtime_iso"],
    },
    {
        "field": "ops_gate_pack_log_path",
        "value": str(ops_gate_log_ref) if ops_gate_log_ref is not None else "<missing>",
        "exists": ops_ref_info["exists"],
        "size_bytes": ops_ref_info["size_bytes"],
        "mtime_iso": ops_ref_info["mtime_iso"],
    },
    {
        "field": "ops_gate_pack_stamp",
        "value": ops_stamp_from_run_stamp if ops_stamp_from_run_stamp else "<missing>",
        "exists": None,
        "size_bytes": None,
        "mtime_iso": None,
    },
    {
        "field": "latest_gate_ops_pack_log",
        "value": str(latest_ops_gate_log) if latest_ops_gate_log is not None else "<missing>",
        "exists": latest_ops_info["exists"],
        "size_bytes": latest_ops_info["size_bytes"],
        "mtime_iso": latest_ops_info["mtime_iso"],
    },
    {
        "field": "latest_gate_ops_pack_stamp",
        "value": latest_ops_stamp if latest_ops_stamp else "<missing>",
        "exists": None,
        "size_bytes": None,
        "mtime_iso": None,
    },
]
st.dataframe(pd.DataFrame(evidence_rows), width='stretch', hide_index=True)

if staleness_status == "OK":
    st.success(
        "STALENESS=OK: run_stamp ops_gate_pack_stamp matches latest gate_ops_pack stamp and ops_gate_pack_log_path exists."
    )
elif staleness_status == "STALE":
    st.warning(
        f"STALENESS=STALE: run_stamp ops_gate_pack_stamp={ops_stamp_from_run_stamp or '<missing>'} "
        f"differs from latest gate_ops_pack stamp={latest_ops_stamp or '<missing>'}."
    )
else:
    st.error(
        "STALENESS=MISSING: ops_gate_pack_log_path is missing/non-existent and/or required stamps could not be parsed."
    )

evidence_ops_gate_line: str | None = None
evidence_ops_gate_source = "<none>"
if isinstance(ops_gate_verdict_ref, str) and ops_gate_verdict_ref.strip():
    evidence_ops_gate_line = ops_gate_verdict_ref.strip()
    evidence_ops_gate_source = "run_stamp.ops_gate_pack_verdict"
elif ops_gate_log_ref is not None and ops_gate_log_ref.exists():
    evidence_ops_gate_line = extract_ops_gate_line(ops_gate_log_ref)
    evidence_ops_gate_source = "run_stamp.ops_gate_pack_log_path"
elif latest_ops_gate_log is not None and latest_ops_gate_log.exists():
    evidence_ops_gate_line = extract_ops_gate_line(latest_ops_gate_log)
    evidence_ops_gate_source = "latest gate_ops_pack log"
st.code(evidence_ops_gate_line or "OPS_GATE line not found.", language="text")
st.caption(f"OPS_GATE source: {evidence_ops_gate_source}")

st.subheader("Evidence Paths")
evidence_path_specs = [
    ("b1_refresh_publish", maybe_get(run_stamp, "b1_refresh_publish_log_path", "B1RefreshPublishLogPath", "b1_log_path")),
    ("gate_ops_pack", maybe_get(run_stamp, "ops_gate_pack_log_path", "OpsGatePackLogPath", "gate_log_path")),
    (
        "load_incoming_flights",
        maybe_get(
            run_stamp,
            "load_incoming_flights_log_path",
            "LoadIncomingFlightsLogPath",
            "loader_log_path",
            "LoaderLogPath",
        ),
    ),
]
evidence_rows: list[dict[str, Any]] = []
for label, raw in evidence_path_specs:
    p: Path | None = None
    if isinstance(raw, str) and raw.strip():
        tmp = Path(raw.strip())
        p = tmp if tmp.is_absolute() else (REPO_ROOT / tmp)
    info = path_info(p)
    evidence_rows.append(
        {
            "Evidence": label,
            "Path": str(p) if p is not None else "<missing>",
            "Exists": info["exists"],
            "SizeBytes": info["size_bytes"],
            "LastWriteTime": info["mtime_iso"],
        }
    )
evidence_paths_df = pd.DataFrame(evidence_rows)
st.dataframe(evidence_paths_df, width="stretch", hide_index=True)
for row in evidence_rows:
    p_str = str(row["Path"])
    if p_str == "<missing>":
        st.warning(f"{row['Evidence']}: path missing in run_stamp.")
        continue
    p = Path(p_str)
    if not p.exists():
        st.warning(f"{row['Evidence']}: path does not exist: `{p}`")
        continue
    with st.expander(f"Tail 200 lines: {row['Evidence']} ({p.name})", expanded=False):
        st.code(tail_text(p, 200), language="text")

st.subheader("Required Files Health")
required_health_rows: list[dict[str, Any]] = []
for name in required_files:
    base_name = Path(str(name)).name
    full = Path(artifacts_by_name.get(base_name, insights_dir / base_name))
    info = path_info(full)
    required_health_rows.append(
        {
            "file": base_name,
            "exists": info["exists"],
            "size_bytes": info["size_bytes"],
            "mtime": info["mtime_iso"],
        }
    )
required_health_df = pd.DataFrame(required_health_rows, columns=["file", "exists", "size_bytes", "mtime"])
if required_health_df.empty:
    st.info("No `required_files` entries found in run stamp.")
else:
    st.dataframe(required_health_df, width='stretch', hide_index=True)

missing_required_files = [r["file"] for r in required_health_rows if not r["exists"]]
st.caption(f"RequiredFilesCount={len(required_files)}")
if missing_required_files:
    st.warning(f"MissingRequiredFiles={len(missing_required_files)}")
    st.code("\n".join(missing_required_files), language="text")
else:
    st.success("MissingRequiredFiles=0")

st.subheader("Log References")
all_path_refs = collect_string_paths(run_stamp)
log_ref_rows: list[dict[str, Any]] = []
for row in all_path_refs:
    raw_val = str(row["value"])
    low_val = raw_val.lower()
    low_key = str(row["key_path"]).lower()
    if (
        "\\artifacts\\run_logs\\" in low_val
        or low_val.endswith(".txt")
        or "qa_summary_path" in low_key
    ):
        log_ref_rows.append(
            {
                "key_path": row["key_path"],
                "path": raw_val,
                "exists": row["exists"],
                "size_bytes": row["size_bytes"],
                "mtime_iso": row["mtime_iso"],
            }
        )

log_refs_df = pd.DataFrame(log_ref_rows, columns=["key_path", "path", "exists", "size_bytes", "mtime_iso"])
if log_refs_df.empty:
    st.info("No log/text path references found in run stamp.")
else:
    st.dataframe(log_refs_df, width='stretch', hide_index=True)
    for r in log_ref_rows:
        if not bool(r["exists"]):
            continue
        p = Path(str(r["path"]))
        with st.expander(f"View tail: {p.name} ({r['key_path']})", expanded=False):
            st.caption(f"Encoding detected: `{detect_text_encoding(p)}`")
            st.code(tail_text(p, 60), language="text")

st.subheader("Ops Gate Pack (latest)")
has_ops_ref = (
    (isinstance(ops_gate_verdict_ref, str) and bool(ops_gate_verdict_ref.strip()))
    or (ops_gate_log_ref is not None)
    or (isinstance(ops_gate_stamp_ref, str) and bool(ops_gate_stamp_ref.strip()))
)
if has_ops_ref:
    ops_gate_line: str | None = None
    if isinstance(ops_gate_verdict_ref, str) and ops_gate_verdict_ref.strip():
        ops_gate_line = ops_gate_verdict_ref.strip()
    elif ops_gate_log_ref is not None and ops_gate_log_ref.exists():
        ops_gate_line = extract_ops_gate_line(ops_gate_log_ref)
    st.code(ops_gate_line or "OPS_GATE line not found.", language="text")
    st.caption(f"Log: {ops_gate_log_ref}" if ops_gate_log_ref is not None else "Log: <missing>")
    if isinstance(ops_gate_stamp_ref, str) and ops_gate_stamp_ref.strip():
        st.caption(f"Stamp: {ops_gate_stamp_ref.strip()}")
    if ops_gate_log_ref is not None:
        if ops_gate_log_ref.exists():
            with st.expander("View log tail", expanded=False):
                st.code(tail_text(ops_gate_log_ref, 60), language="text")
        else:
            st.info(f"Ops Gate Pack log path from run_stamp does not exist: `{ops_gate_log_ref}`")
else:
    ops_gate_log = latest_ops_gate_log
    if ops_gate_log is None:
        st.info("Ops Gate Pack log: NONE (run `scripts/gate_ops_pack.ps1` to generate).")
    else:
        ops_gate_line = extract_ops_gate_line(ops_gate_log) or "OPS_GATE line not found."
        st.code(ops_gate_line, language="text")
        st.caption(f"Log: {ops_gate_log}")
        with st.expander("View log tail", expanded=False):
            st.code(tail_text(ops_gate_log, 60), language="text")

st.subheader("Contract Check")
required_df, fail_df = validate_required_files(required_files, insights_dir)
required_count = len(required_files)
fail_count = 0 if fail_df.empty else len(fail_df)

if required_count == 0:
    st.error("FAIL: `required_files` list is missing/empty in run stamp.")
else:
    if fail_count == 0:
        st.success(f"PASS: Required={required_count}, Missing/Empty=0")
    else:
        st.error(f"FAIL: Required={required_count}, Missing/Empty={fail_count}")

if not fail_df.empty:
    st.caption("Failing files only")
    st.dataframe(fail_df, width='stretch', hide_index=True)

with st.expander("All Required Files Status", expanded=False):
    st.dataframe(required_df, width='stretch', hide_index=True)

st.subheader("QA Summary")
if qa_path is None:
    st.error("FAIL: QA summary path could not be resolved from run stamp.")
else:
    st.write(f"Expected/Resolved QA file: `{qa_path}`")
    if qa_path.exists():
        st.caption(
            f"Size={qa_path.stat().st_size:,} bytes | "
            f"Modified={datetime.fromtimestamp(qa_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')} | "
            f"SHA256={file_sha256(qa_path) or '<unavailable>'}"
        )
        qa_df, qa_err = safe_read_csv(qa_path)
        if qa_df is None:
            st.error(f"FAIL: could not load QA summary: `{qa_path}`")
            st.code(qa_err or "unknown read error")
        else:
            st.success(f"PASS: loaded QA summary ({len(qa_df):,} rows)")
            st.write("Top rows")
            st.dataframe(qa_df.head(25), width="stretch", hide_index=True)
            col_map = {str(c).strip().lower(): c for c in qa_df.columns}
            exists_col = col_map.get("exists")
            rows_col = col_map.get("rows")
            cols_col = col_map.get("cols")
            if exists_col is not None and rows_col is not None and cols_col is not None:
                exists_ok = qa_df[exists_col].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
                rows_num = pd.to_numeric(qa_df[rows_col], errors="coerce")
                cols_num = pd.to_numeric(qa_df[cols_col], errors="coerce")
                qa_fails = qa_df[(~exists_ok) | (rows_num <= 0) | (cols_num <= 0)].copy()
                st.write("Failing rows")
                if qa_fails.empty:
                    st.success("No failing QA rows.")
                else:
                    st.dataframe(qa_fails, width="stretch", hide_index=True)
            else:
                st.warning("QA fail filter unavailable: Exists/Rows/Cols columns not found.")
    else:
        st.error(f"FAIL: QA summary file missing: `{qa_path}`")

st.subheader("Latest Logs")
logs = latest_logs(RUN_LOGS_DIR, n=5)
if not logs:
    st.warning(f"No logs found under `{RUN_LOGS_DIR}`")
else:
    logs_table = pd.DataFrame(
        [
            {
                "FileName": p.name,
                "Modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "SizeBytes": int(p.stat().st_size),
                "FullPath": str(p),
            }
            for p in logs
        ]
    )
    st.dataframe(logs_table, width='stretch', hide_index=True)
    for p in logs:
        with st.expander(f"Tail (last 200 lines): {p.name}", expanded=False):
            st.code(tail_text_file(p, lines=200), language="text")


