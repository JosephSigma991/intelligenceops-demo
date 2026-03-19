from __future__ import annotations

import os
import re
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from mode_a_filters import apply_filters
from utils import (
    normalize_name, pick_column, parse_period, ensure_list,
    read_csv_cached, load_csv, resolve_required_file,
    runtime_data_log, maybe_get, show_plotly, render_freshness_banner, render_filter_banner,
)
from kpi_config import (
    PLOTLY_CONFIG,
    WEEKLY_PERIOD_CANDIDATES, MONTHLY_PERIOD_CANDIDATES,
    STATION_CANDIDATES, FLIGHTS_CANDIDATES,
    TOTAL_MIN_CANDIDATES, AVG_DELAY_CANDIDATES, OTP_CANDIDATES,
)


st.set_page_config(page_title="Data Quality / Coverage", layout="wide")

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_text_robust(path: Path) -> str:
    b = path.read_bytes()
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


def tail_text(text: str, max_lines: int = 80) -> str:
    lines = text.splitlines()
    if not lines:
        return "[empty file]"
    q: deque[str] = deque(maxlen=max_lines)
    for ln in lines:
        q.append(ln)
    return "\n".join(q)


def extract_first_line(text: str, prefix: str) -> str | None:
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith(prefix):
            return s
    return None


def extract_top5_block(text: str) -> list[str]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("DB_SANITY_BASE_DTARTIFACTS_TOP5="):
            out = [line.strip()]
            for j in range(i + 1, len(lines)):
                cur = lines[j]
                cur_strip = cur.strip()
                if not cur_strip:
                    break
                if cur_strip.startswith("==="):
                    break
                if re.match(r"^\s+\S+\|\S+", cur):
                    out.append(cur.rstrip())
                    continue
                break
            return out
    return []


def parse_kv_line(line: str | None) -> dict[str, str]:
    if not line:
        return {}
    kv: dict[str, str] = {}
    for m in re.finditer(r"([A-Za-z0-9_]+)=([^\s]+)", line):
        kv[m.group(1)] = m.group(2)
    return kv


def parse_year(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    m = re.search(r"(\d{4})", str(value))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def parse_boolish(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def style_required_health(df: pd.DataFrame) -> Any:
    columns = list(df.columns)

    def row_style(row: pd.Series) -> list[str]:
        styles = [""] * len(columns)
        col_idx = {c: i for i, c in enumerate(columns)}

        exists = bool(row.get("Exists"))
        rows = row.get("Rows")
        file_name = str(row.get("FileName", ""))
        is_2025_named = "2025" in file_name

        if not exists and "Exists" in col_idx:
            styles[col_idx["Exists"]] = "background-color: #f8d7da;"
        if pd.notna(rows) and float(rows) == 0 and "Rows" in col_idx:
            styles[col_idx["Rows"]] = "background-color: #f8d7da;"

        if is_2025_named:
            yr_values = [
                parse_year(row.get("MinYearMonth")),
                parse_year(row.get("MaxYearMonth")),
                parse_year(row.get("MinYearWeek")),
                parse_year(row.get("MaxYearWeek")),
            ]
            if any(y is not None and y != 2025 for y in yr_values):
                for c in ["MinYearMonth", "MaxYearMonth", "MinYearWeek", "MaxYearWeek"]:
                    if c in col_idx:
                        styles[col_idx[c]] = "background-color: #fff3cd;"
        return styles

    return df.style.apply(row_style, axis=1)


def weighted_avg(values: pd.Series, weights: pd.Series) -> float | None:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna()
    if not mask.any():
        return None
    denom = float(w[mask].sum())
    if denom <= 0:
        return None
    return float((v[mask] * w[mask]).sum() / denom)


def _discover_log_path(
    run_stamp: dict,
    repo_root: Path,
    *stamp_keys: str,
    file_prefix: str,
    stamp_val: str | None = None,
) -> Path | None:
    """
    Resolve a log path with 3-tier priority:
      1. Explicit key in run_stamp (existing behaviour — unchanged)
      2. artifacts/run_logs/<prefix>__<stamp>*.txt  (stamp-matched)
      3. Latest  artifacts/run_logs/<prefix>__*.txt  (most-recent fallback)
    """
    # Priority 1 — explicit key in run_stamp
    for k in stamp_keys:
        ref = maybe_get(run_stamp, k)
        if isinstance(ref, str) and ref.strip():
            p = Path(ref.strip())
            return p if p.is_absolute() else (repo_root / p)

    log_dir = repo_root / "artifacts" / "run_logs"
    if not log_dir.exists():
        return None

    # Priority 2 — stamp-matched
    if stamp_val:
        for p in log_dir.glob(f"{file_prefix}__*{stamp_val}*.txt"):
            return p

    # Priority 3 — latest by mtime
    candidates = sorted(log_dir.glob(f"{file_prefix}__*.txt"), key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


st.title("Data Quality / Coverage")
st.caption("PASS-only | Evidence: run_stamp + qa_summary + Ops Gate Pack + B1 publish + A6 runtime proof logs.")

ctx = st.session_state.get("mode_a_ctx")
if not isinstance(ctx, dict):
    st.error("Select a run in the sidebar")
    st.stop()
render_freshness_banner(ctx)
filters = st.session_state.get("mode_a_filters")
if not isinstance(filters, dict):
    st.error("Global filters are missing. Select filters in the sidebar.")
    st.stop()
render_filter_banner(ctx, filters)

stamp_json_path = Path(ctx.get("run_stamp_json_path", ""))
run_stamp = ctx.get("run_stamp") if isinstance(ctx.get("run_stamp"), dict) else {}
region = str(ctx.get("region", "")).strip()
mode = str(ctx.get("mode", "")).strip()
stamp = ctx.get("stamp")
git_branch = ctx.get("git_branch")
git_commit = ctx.get("git_commit")
required_files = ensure_list(ctx.get("required_files"))
insights_dir = Path(ctx.get("insights_dir", Path(os.getenv("FLIGHTOPS_INSIGHTS_DIR", r"C:\Users\IT\02_insights\insight_out"))))
qa_summary_path = ctx.get("qa_path")
artifacts_by_name = ctx.get("artifacts_by_name", {}) or {}

if not region or not mode:
    st.error("Run context is missing region/mode.")
    st.stop()
ops_gate_verdict = maybe_get(run_stamp, "ops_gate_pack_verdict", "OpsGatePackVerdict")
ops_gate_log_path_ref = maybe_get(run_stamp, "ops_gate_pack_log_path", "OpsGatePackLogPath")
ops_gate_stamp = maybe_get(run_stamp, "ops_gate_pack_stamp", "OpsGatePackStamp")

ops_gate_log_path: Path | None = None
if isinstance(ops_gate_log_path_ref, str) and ops_gate_log_path_ref.strip():
    p = Path(ops_gate_log_path_ref.strip())
    ops_gate_log_path = p if p.is_absolute() else (REPO_ROOT / p)

runtime_data_log(f"RunStampPath={stamp_json_path}")
runtime_data_log(f"QaSummaryPath={qa_summary_path}")
runtime_data_log(f"OpsGateLogPath={ops_gate_log_path}")

st.subheader("Section A: Provenance & Gate Evidence")
p1, p2, p3, p4, p5 = st.columns(5)
p1.metric("Stamp", str(stamp) if stamp else "<missing>")
p2.metric("Region", str(region))
p3.metric("Mode", str(mode))
p4.metric("Git Branch", str(git_branch) if git_branch else "<missing>")
p5.metric("Git Commit", str(git_commit) if git_commit else "<missing>")

st.caption(f"Scope: {region} / {mode} | Stamp: {stamp or '<missing>'}")
with st.expander("Evidence paths", expanded=True):
    st.text(f"RunStampJSON : {stamp_json_path}")
    st.text(f"QA Summary   : {qa_summary_path or '<not set>'}")
    st.text(f"OpsGateLog   : {ops_gate_log_path or '<not set>'}")

if isinstance(ops_gate_verdict, str) and ops_gate_verdict.strip():
    st.code(ops_gate_verdict.strip(), language="text")
else:
    st.info("ops_gate_pack_verdict missing from run_stamp")

if isinstance(ops_gate_stamp, str) and ops_gate_stamp.strip():
    st.caption(f"ops_gate_pack_stamp: `{ops_gate_stamp}`")
if ops_gate_log_path is not None:
    st.caption(f"ops_gate_pack_log_path: `{ops_gate_log_path}`")

ops_text: str | None = None
ops_gate_line_from_log: str | None = None
db_sanity_line_from_log: str | None = None
db_top5_block: list[str] = []

if ops_gate_log_path is not None and ops_gate_log_path.exists():
    try:
        ops_text = read_text_robust(ops_gate_log_path)
        ops_gate_line_from_log = extract_first_line(ops_text, "OPS_GATE=")
        db_sanity_line_from_log = extract_first_line(ops_text, "DB_SANITY=")
        db_top5_block = extract_top5_block(ops_text)
    except Exception as e:
        st.error(f"Could not read ops gate log: {type(e).__name__}: {e}")

    st.write("OPS_GATE line (from log)")
    if ops_gate_line_from_log:
        st.code(ops_gate_line_from_log, language="text")
    else:
        st.info("No OPS_GATE= line found in referenced log.")

    st.write("DB_SANITY line (from log)")
    if db_sanity_line_from_log:
        st.code(db_sanity_line_from_log, language="text")
    else:
        st.info("No DB_SANITY= line found in referenced log.")

    if db_top5_block:
        st.write("DB_SANITY_BASE_DTARTIFACTS_TOP5 block")
        st.code("\n".join(db_top5_block), language="text")

    with st.expander("View log tail", expanded=False):
        st.code(tail_text(ops_text, max_lines=80), language="text")
else:
    st.info("ops_gate_pack_log_path missing or not found. Only run_stamp/qa_summary evidence is available.")

b1_log_path = _discover_log_path(
    run_stamp,
    REPO_ROOT,
    "b1_log_path", "LatestB1Log", "B1LogPath",
    file_prefix="b1_refresh_publish",
    stamp_val=str(stamp) if stamp else None,
)
if b1_log_path is not None:
    st.caption(f"B1 publish log: `{b1_log_path}`")
    if b1_log_path.exists():
        with st.expander("View B1 publish log tail", expanded=False):
            try:
                st.code(tail_text(read_text_robust(b1_log_path), max_lines=60), language="text")
            except Exception as e:
                st.error(f"Could not read B1 log: {type(e).__name__}: {e}")
    else:
        st.info("B1 log path resolved but file not found on disk.")
else:
    st.info("B1 publish log not found (no run_stamp key and no matching file in artifacts/run_logs/).")

a6_log_path = _discover_log_path(
    run_stamp,
    REPO_ROOT,
    "a6_runtime_proof_log_path", "LatestA6RuntimeProofLog", "A6RuntimeProofLogPath",
    file_prefix="a6_runtime_proof",
    stamp_val=str(stamp) if stamp else None,
)
if a6_log_path is not None:
    st.caption(f"A6 runtime proof log: `{a6_log_path}`")
    if a6_log_path.exists():
        with st.expander("View A6 runtime proof log tail", expanded=False):
            try:
                st.code(tail_text(read_text_robust(a6_log_path), max_lines=60), language="text")
            except Exception as e:
                st.error(f"Could not read A6 log: {type(e).__name__}: {e}")
    else:
        st.info("A6 log path resolved but file not found on disk.")
else:
    st.info("A6 runtime proof log not found (no run_stamp key and no matching file in artifacts/run_logs/).")

st.subheader("Section B: Required Files Health")
if not required_files:
    st.warning("run_stamp.required_files is missing/empty.")

qa_df: pd.DataFrame | None = None
qa_lookup: dict[str, dict[str, Any]] = {}
if qa_summary_path is not None and qa_summary_path.exists():
    qa_df = load_csv(qa_summary_path)
    if qa_df is None:
        st.error(f"Could not load qa_summary CSV: `{qa_summary_path}`")

if qa_df is not None and not qa_df.empty:
    # ── Poison / Clean flight counts ─────────────────────────────────────────────
    st.subheader("Flight Data Quality Gate")
    _base_col    = pick_column(qa_df, ["Base_Flights_Total",  "base_flights_total"])
    _clean_col   = pick_column(qa_df, ["Clean_Flights_Total", "clean_flights_total"])
    _poison_col  = pick_column(qa_df, ["Poison_Count",        "poison_count"])

    if _base_col and _clean_col and _poison_col:
        _base   = int(pd.to_numeric(qa_df[_base_col],   errors="coerce").dropna().iloc[0]) if not qa_df.empty else None
        _clean  = int(pd.to_numeric(qa_df[_clean_col],  errors="coerce").dropna().iloc[0]) if not qa_df.empty else None
        _poison = int(pd.to_numeric(qa_df[_poison_col], errors="coerce").dropna().iloc[0]) if not qa_df.empty else None
        _pct    = f"{100.0 * _poison / _base:.2f}%" if (_base and _base > 0) else "N/A"
        _m1, _m2, _m3, _m4 = st.columns(4)
        _m1.metric("Base Flights Total",  f"{_base:,}"   if _base   is not None else "N/A")
        _m2.metric("Clean Flights Total", f"{_clean:,}"  if _clean  is not None else "N/A")
        _m3.metric("Poison Count",        f"{_poison:,}" if _poison is not None else "N/A")
        _m4.metric("Poison Rate",         _pct)
        if _poison is not None and _poison > 0:
            st.warning(f"⚠️ {_poison:,} flights excluded by hygiene gates ({_pct} of base).")
        else:
            st.success("✅ No poison flights — all base flights passed hygiene gates.")
    else:
        st.info("Poison/clean flight counts not available in this qa_summary (older stamp).")

    qa_file_col = pick_column(qa_df, ["FileName", "File", "file", "file_name"])
    qa_rows_col = pick_column(qa_df, ["Rows", "rows"])
    qa_cols_col = pick_column(qa_df, ["Cols", "cols", "Columns"])
    qa_hym_col = pick_column(qa_df, ["HasYearMonth", "has_yearmonth"])
    qa_min_ym_col = pick_column(qa_df, ["MinYearMonth", "min_yearmonth"])
    qa_max_ym_col = pick_column(qa_df, ["MaxYearMonth", "max_yearmonth"])
    qa_hyw_col = pick_column(qa_df, ["HasYearWeek", "has_yearweek"])
    qa_min_yw_col = pick_column(qa_df, ["MinYearWeek", "min_yearweek"])
    qa_max_yw_col = pick_column(qa_df, ["MaxYearWeek", "max_yearweek"])

    if qa_file_col is not None:
        for _, r in qa_df.iterrows():
            file_val = r.get(qa_file_col)
            if pd.isna(file_val):
                continue
            key = normalize_name(Path(str(file_val)).name)
            qa_lookup[key] = {
                "Rows": int(r.get(qa_rows_col)) if qa_rows_col and pd.notna(r.get(qa_rows_col)) else None,
                "Cols": int(r.get(qa_cols_col)) if qa_cols_col and pd.notna(r.get(qa_cols_col)) else None,
                "HasYearMonth": parse_boolish(r.get(qa_hym_col)) if qa_hym_col else None,
                "MinYearMonth": str(r.get(qa_min_ym_col)) if qa_min_ym_col and pd.notna(r.get(qa_min_ym_col)) else None,
                "MaxYearMonth": str(r.get(qa_max_ym_col)) if qa_max_ym_col and pd.notna(r.get(qa_max_ym_col)) else None,
                "HasYearWeek": parse_boolish(r.get(qa_hyw_col)) if qa_hyw_col else None,
                "MinYearWeek": str(r.get(qa_min_yw_col)) if qa_min_yw_col and pd.notna(r.get(qa_min_yw_col)) else None,
                "MaxYearWeek": str(r.get(qa_max_yw_col)) if qa_max_yw_col and pd.notna(r.get(qa_max_yw_col)) else None,
            }

health_rows: list[dict[str, Any]] = []
for entry in required_files:
    p = Path(str(entry))
    full = p if p.is_absolute() else (insights_dir / p)
    exists = full.exists()
    size = int(full.stat().st_size) if exists and full.is_file() else 0
    mtime = (
        datetime.fromtimestamp(full.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        if exists
        else None
    )
    key = normalize_name(p.name)
    qa_meta = qa_lookup.get(key, {})
    health_rows.append(
        {
            "FileName": p.name,
            "Exists": bool(exists),
            "SizeBytes": size,
            "Rows": qa_meta.get("Rows"),
            "Cols": qa_meta.get("Cols"),
            "HasYearMonth": qa_meta.get("HasYearMonth"),
            "MinYearMonth": qa_meta.get("MinYearMonth"),
            "MaxYearMonth": qa_meta.get("MaxYearMonth"),
            "HasYearWeek": qa_meta.get("HasYearWeek"),
            "MinYearWeek": qa_meta.get("MinYearWeek"),
            "MaxYearWeek": qa_meta.get("MaxYearWeek"),
            "LastWriteTime": mtime,
        }
    )

health_df = pd.DataFrame(
    health_rows,
    columns=[
        "FileName",
        "Exists",
        "SizeBytes",
        "Rows",
        "Cols",
        "HasYearMonth",
        "MinYearMonth",
        "MaxYearMonth",
        "HasYearWeek",
        "MinYearWeek",
        "MaxYearWeek",
        "LastWriteTime",
    ],
)

if health_df.empty:
    st.info("No required files to display.")
else:
    st.dataframe(style_required_health(health_df), width="stretch", hide_index=True)
    flagged_2025 = 0
    for _, rr in health_df.iterrows():
        if "2025" not in str(rr.get("FileName", "")):
            continue
        years = [parse_year(rr.get("MinYearMonth")), parse_year(rr.get("MaxYearMonth")), parse_year(rr.get("MinYearWeek")), parse_year(rr.get("MaxYearWeek"))]
        if any(y is not None and y != 2025 for y in years):
            flagged_2025 += 1
    if flagged_2025 > 0:
        st.warning(f"Amber note: {flagged_2025} file(s) with 2025 name show period range outside 2025.")

health_csv = health_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Required Files Health CSV",
    data=health_csv,
    file_name=f"required_files_health__{region}__{mode}.csv",
    mime="text/csv",
)

st.subheader("Section C: Coverage by Period")
grain = str(filters.get("grain", "Monthly"))
periods_from_filters = [str(p) for p in filters.get("periods", []) if str(p).strip()]
stations_from_filters = [str(s) for s in filters.get("stations", []) if str(s).strip()]
period_label = periods_from_filters[0] if len(periods_from_filters) == 1 else (f"{periods_from_filters[0]}..{periods_from_filters[-1]} ({len(periods_from_filters)})" if periods_from_filters else "<auto>")
station_label = "NETWORK" if stations_from_filters == ["NETWORK"] else f"{len(stations_from_filters)} selected"
st.caption(f"Filters: Grain={grain} | Periods={period_label} | Stations={station_label}")
kpi_name = f"2025_DEP_Monthly_Station_KPIs__{region}.csv" if grain == "Monthly" else f"2025_DEP_Weekly_Station_KPIs__{region}.csv"
_, kpi_path = resolve_required_file(required_files, kpi_name, insights_dir, artifacts_by_name)

coverage_df: pd.DataFrame | None = None
coverage_raw: pd.DataFrame | None = None
coverage_station_col: str | None = None
coverage_period_col: str | None = None
coverage_flights_col: str | None = None
period_invalid_count: int | None = None
flights_nonpositive_count: int | None = None
duplicates_df: pd.DataFrame | None = None
duplicate_key_count: int | None = None

if not kpi_path.exists():
    st.error(f"KPI file not found for selected grain: `{kpi_path}`")
else:
    cov = load_csv(kpi_path)
    if cov is None:
        st.error(f"Could not load KPI file: `{kpi_path}`")
    else:
        runtime_data_log(f"Grain={grain} StationKpisFile={kpi_path} Rows={len(cov)} Cols={len(cov.columns)}")
        coverage_raw = cov.copy()

        period_col = pick_column(cov, MONTHLY_PERIOD_CANDIDATES if grain == "Monthly" else WEEKLY_PERIOD_CANDIDATES)
        station_col = pick_column(cov, STATION_CANDIDATES, token_groups=[["station"], ["airport"]])
        flights_col = pick_column(cov, FLIGHTS_CANDIDATES, token_groups=[["flight", "oper"]])
        avg_col = pick_column(cov, AVG_DELAY_CANDIDATES, token_groups=[["avg", "delay"], ["per", "flight"]])
        otp_col = pick_column(cov, OTP_CANDIDATES, token_groups=[["otp"], ["d15"]])
        total_col = pick_column(cov, TOTAL_MIN_CANDIDATES, token_groups=[["dep", "delay"], ["min"]])

        coverage_station_col = station_col
        coverage_period_col = period_col
        coverage_flights_col = flights_col

        if period_col is None or station_col is None or flights_col is None:
            st.error(
                "Missing required columns after inference: "
                f"period={period_col}, station={station_col}, flights={flights_col}"
            )
        else:
            work = cov.copy()
            work[flights_col] = pd.to_numeric(work[flights_col], errors="coerce")
            if avg_col is not None:
                work[avg_col] = pd.to_numeric(work[avg_col], errors="coerce")
            if otp_col is not None:
                work[otp_col] = pd.to_numeric(work[otp_col], errors="coerce")
            if total_col is not None:
                work[total_col] = pd.to_numeric(work[total_col], errors="coerce")

            period_pattern = r"^\d{4}-\d{2}$" if grain == "Monthly" else r"^\d{4}-W\d{2}$"
            period_invalid_count = int(
                (~work[period_col].astype(str).str.strip().str.match(period_pattern, na=False)).sum()
            )
            flights_nonpositive_count = int((work[flights_col].fillna(0) <= 0).sum())

            parsed = work[period_col].map(lambda x: parse_period(x, grain))
            work["_period"] = parsed.map(lambda t: t[0])
            work["_sort"] = parsed.map(lambda t: t[1])
            work = work.dropna(subset=["_period", "_sort"]).copy()

            active = work[work[flights_col].fillna(0) > 0].copy()
            active = apply_filters(active, filters, station_col=station_col, period_col="_period")
            if active.empty:
                st.info("No active periods found (Flights Operated > 0).")
            else:
                grouped = active.groupby(["_period", "_sort"], as_index=False)
                cov_rows: list[dict[str, Any]] = []
                otp_method = "none"
                for (period, sort_dt), g in grouped:
                    row: dict[str, Any] = {
                        "Period": str(period),
                        "_sort": sort_dt,
                        "ActiveStations": int(g[station_col].astype(str).nunique()),
                        "Flights": float(pd.to_numeric(g[flights_col], errors="coerce").sum()),
                    }
                    if total_col is not None:
                        total_sum = float(pd.to_numeric(g[total_col], errors="coerce").sum())
                        row["TotalMinutes"] = total_sum
                    else:
                        row["TotalMinutes"] = pd.NA

                    avg_value: float | None = None
                    if avg_col is not None:
                        avg_value = weighted_avg(g[avg_col], g[flights_col])
                    if avg_value is None and total_col is not None:
                        flights = float(row["Flights"])
                        total = float(row["TotalMinutes"]) if pd.notna(row["TotalMinutes"]) else 0.0
                        avg_value = (total / flights) if flights > 0 else None
                    row["AvgDelay"] = avg_value if avg_value is not None else pd.NA

                    otp_value: float | None = None
                    if otp_col is not None:
                        otp_weighted = weighted_avg(g[otp_col], g[flights_col])
                        if otp_weighted is not None:
                            otp_value = otp_weighted
                            otp_method = "weighted"
                        else:
                            otp_simple = pd.to_numeric(g[otp_col], errors="coerce").mean()
                            if pd.notna(otp_simple):
                                otp_value = float(otp_simple)
                                otp_method = "simple_mean"
                    row["OTP"] = otp_value if otp_value is not None else pd.NA

                    for col_name, out_name in [
                        (flights_col, "NullPct_Flights"),
                        (avg_col, "NullPct_AvgDelay"),
                        (otp_col, "NullPct_OTP"),
                        (total_col, "NullPct_TotalMinutes"),
                    ]:
                        if col_name is None:
                            row[out_name] = pd.NA
                        else:
                            row[out_name] = float(pd.to_numeric(g[col_name], errors="coerce").isna().mean() * 100.0)
                    cov_rows.append(row)

                coverage_df = pd.DataFrame(cov_rows).sort_values("_sort").reset_index(drop=True)
                if "OTP" in coverage_df.columns and coverage_df["OTP"].notna().any():
                    if otp_method == "weighted":
                        st.caption("OTP aggregation method: weighted by Flights Operated.")
                    elif otp_method == "simple_mean":
                        st.caption("OTP aggregation method: simple mean (weights unavailable).")

                flights_fig = go.Figure()
                flights_fig.add_trace(
                    go.Scatter(
                        x=coverage_df["Period"].tolist(),
                        y=coverage_df["Flights"].tolist(),
                        mode="lines+markers",
                        name="Flights",
                    )
                )
                flights_fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Flights")
                show_plotly(flights_fig, key="d6__coverage_flights_trend")

                stn_fig = go.Figure()
                stn_fig.add_trace(
                    go.Scatter(
                        x=coverage_df["Period"].tolist(),
                        y=coverage_df["ActiveStations"].tolist(),
                        mode="lines+markers",
                        name="Active Stations",
                    )
                )
                stn_fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Stations")
                show_plotly(stn_fig, key="d6__coverage_stations_trend")

                if "AvgDelay" in coverage_df.columns and coverage_df["AvgDelay"].notna().any():
                    avg_fig = go.Figure()
                    avg_fig.add_trace(
                        go.Scatter(
                            x=coverage_df["Period"].tolist(),
                            y=coverage_df["AvgDelay"].tolist(),
                            mode="lines+markers",
                            name="Avg DEP Delay per Flight (min)",
                        )
                    )
                    avg_fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Minutes")
                    show_plotly(avg_fig, key="d6__coverage_avg_delay_trend")

                show_df = coverage_df.drop(columns=["_sort"], errors="ignore").tail(24)
                st.dataframe(show_df, width="stretch", hide_index=True)
                st.download_button(
                    label="Download Coverage by Period CSV",
                    data=show_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"coverage_by_period__{grain.lower()}__{region}__{mode}.csv",
                    mime="text/csv",
                )

                dup_work = work.copy()
                dup_work["_station"] = dup_work[station_col].astype(str)
                dup_counts = (
                    dup_work.groupby(["_period", "_station"], as_index=False)
                    .size()
                    .rename(columns={"size": "Count"})
                )
                dup_counts = dup_counts[dup_counts["Count"] > 1].sort_values("Count", ascending=False)
                duplicate_key_count = int(len(dup_counts))
                duplicates_df = dup_counts.head(10).copy()

st.subheader("Section D: Data Quality Flags")
db_sanity_kv = parse_kv_line(db_sanity_line_from_log)
if db_sanity_line_from_log:
    st.code(db_sanity_line_from_log, language="text")
else:
    st.info("DB_SANITY evidence not available from referenced ops gate log.")

clean_future = db_sanity_kv.get("CleanFuture")
clean_dup = db_sanity_kv.get("CleanDupExtra")
clean_dt = db_sanity_kv.get("CleanDtArtifacts")
base_future = db_sanity_kv.get("BaseFuture")
base_dup = db_sanity_kv.get("BaseDupExtra")
base_dt_rows = db_sanity_kv.get("BaseDtArtifactsRows") or db_sanity_kv.get("BaseDtArtifacts")
base_dt_occ = db_sanity_kv.get("BaseDtArtifactsOcc")

dq1, dq2, dq3 = st.columns(3)
dq1.metric("CleanFuture", clean_future if clean_future is not None else "<missing>")
dq1.metric("BaseFuture", base_future if base_future is not None else "<missing>")
dq2.metric("CleanDupExtra", clean_dup if clean_dup is not None else "<missing>")
dq2.metric("BaseDupExtra", base_dup if base_dup is not None else "<missing>")
dq3.metric("CleanDtArtifacts", clean_dt if clean_dt is not None else "<missing>")
dq3.metric("BaseDtArtifactsRows", base_dt_rows if base_dt_rows is not None else "<missing>")
st.metric("BaseDtArtifactsOcc", base_dt_occ if base_dt_occ is not None else "<missing>")

if db_top5_block:
    top_rows: list[dict[str, Any]] = []
    for ln in db_top5_block[1:]:
        s = ln.strip()
        if "|" not in s:
            continue
        col, cnt = s.split("|", 1)
        top_rows.append({"Column": col.strip(), "BadRows": cnt.strip()})
    if top_rows:
        st.write("Top5 base datetime artifact columns")
        st.dataframe(pd.DataFrame(top_rows), width="stretch", hide_index=True)

if coverage_raw is None or coverage_period_col is None or coverage_station_col is None or coverage_flights_col is None:
    st.info("Station KPI computed checks unavailable because required coverage columns could not be resolved.")
else:
    checks1, checks2, checks3 = st.columns(3)
    checks1.metric(
        "Duplicate (Period, Station) Rows",
        str(duplicate_key_count) if duplicate_key_count is not None else "0",
    )
    checks2.metric("Invalid Period Format Rows", str(period_invalid_count) if period_invalid_count is not None else "<missing>")
    checks3.metric("Flights <= 0 Rows", str(flights_nonpositive_count) if flights_nonpositive_count is not None else "<missing>")

    if duplicates_df is not None and not duplicates_df.empty:
        st.write("Duplicate sample (top 10)")
        st.dataframe(duplicates_df, width="stretch", hide_index=True)
