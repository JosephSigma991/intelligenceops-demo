from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from utils import normalize_name, maybe_get, ensure_list, runtime_data_log, read_csv_cached, render_freshness_banner, render_filter_banner


st.set_page_config(page_title="Dictionary / Policy", layout="wide")


def safe_read_csv(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.exists() or not path.is_file():
        return None, "missing"
    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = None
    df = read_csv_cached(str(path), mtime)
    if df is None:
        return None, "read_error"
    return df, None


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = list(df.columns)
    if not cols:
        return None
    for c in candidates:
        if c in cols:
            return c
    low_lookup = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low_lookup:
            return low_lookup[c.lower()]
    norm_lookup = {normalize_name(c): c for c in cols}
    for c in candidates:
        nc = normalize_name(c)
        if nc in norm_lookup:
            return norm_lookup[nc]
    return None


def baseline_required_files(region: str, mode: str) -> list[str]:
    return [
        f"2025_DEP_Monthly_Station_KPIs__{region}.csv",
        f"2025_DEP_Weekly_Station_KPIs__{region}.csv",
        f"2025_DEP_Owner_Minutes__{region}_NORM.csv",
        f"2025_DEP_TopDelayCodes__{region}_NORM.csv",
        f"2025_DEP_DelayCategory_Minutes__{region}_NORM__MONTHLY.csv",
        f"2025_DEP_DelayCategory_Minutes__{region}_NORM__WEEKLY.csv",
        f"DelayCode_Pareto__Station_YearMonth__{region}.csv",
        f"DelayCode_Pareto__Station_YearWeek__{region}.csv",
        f"TAT_Station_Period_Summary__YearMonth__{region}.csv",
        f"TAT_Station_Period_Summary__YearWeek__{region}.csv",
        f"TAT_Exceptions__{region}.csv",
        f"2025_DEP_MoM_ChangeLog__Drivers__{region}__{mode}_v1.csv",
        f"2025_DEP_WoW_ChangeLog__Drivers__{region}__{mode}_v1.csv",
    ]


def classify_grain_period(file_name: str) -> tuple[str, str]:
    s = file_name.lower()
    if "__monthly" in s or "monthly" in s or "yearmonth" in s:
        return "Monthly", "YearMonth"
    if "__weekly" in s or "weekly" in s or "yearweek" in s:
        return "Weekly", "YearWeek"
    return "Non-periodized", "none"


def artifact_purpose(file_name: str) -> str:
    s = file_name.lower()
    if "monthly_station_kpis" in s:
        return "Station KPI dataset (monthly grain)."
    if "weekly_station_kpis" in s:
        return "Station KPI dataset (weekly grain)."
    if "owner_minutes" in s:
        return "Accountability minutes by DelayCategory owner basis."
    if "topdelaycodes" in s:
        return "Top delay codes summary."
    if "delaycategory_minutes" in s:
        return "DelayCategory minutes for RCA and contribution analysis."
    if "delaycode_pareto" in s:
        return "Station-period delay code Pareto export."
    if "tat_station_period_summary" in s:
        return "Station-period turnaround summary."
    if "tat_exceptions" in s:
        return "Turnaround exceptions listing."
    if "mom_changelog" in s:
        return "Month-over-month drivers change log."
    if "wow_changelog" in s:
        return "Week-over-week drivers change log."
    return "Published required artifact."


def artifact_notes(file_name: str) -> str:
    s = file_name.lower()
    if "mom_changelog" in s or "wow_changelog" in s:
        return "ChangeLog uses 2025 only."
    if "delaycode_pareto" in s:
        return "Pareto includes 2024-2026 periods per export."
    return ""


def build_kpi_dictionary() -> pd.DataFrame:
    rows = [
        {
            "KPI": "Flights Operated",
            "Unit": "flights",
            "Basis / Definition": "Count of operated departures (default proxy: ATD is not null unless overridden).",
            "Directionality": "UP good",
            "Grain compatibility": "Monthly/Weekly",
        },
        {
            "KPI": "Avg DEP Delay per Flight (min)",
            "Unit": "minutes",
            "Basis / Definition": "Total DEP delay minutes / Flights Operated for the slice.",
            "Directionality": "DOWN good",
            "Grain compatibility": "Monthly/Weekly",
        },
        {
            "KPI": "DEP OTP D15 (%)",
            "Unit": "%",
            "Basis / Definition": "% of departures with DEP delay <= 15 minutes.",
            "Directionality": "UP good",
            "Grain compatibility": "Monthly/Weekly",
        },
        {
            "KPI": "Total DEP Delay Minutes",
            "Unit": "minutes",
            "Basis / Definition": "Total departure delay minutes for slice.",
            "Directionality": "DOWN good",
            "Grain compatibility": "Monthly/Weekly",
        },
        {
            "KPI": "Controllable Minutes",
            "Unit": "minutes",
            "Basis / Definition": "Minutes attributable to controllable causes (per published outputs).",
            "Directionality": "DOWN good",
            "Grain compatibility": "Monthly/Weekly",
        },
        {
            "KPI": "Inherited Minutes (Late Arrival)",
            "Unit": "minutes",
            "Basis / Definition": "Reactionary minutes inherited from inbound lateness.",
            "Directionality": "DOWN good",
            "Grain compatibility": "Monthly/Weekly",
        },
        {
            "KPI": "Ground Ops Minutes",
            "Unit": "minutes",
            "Basis / Definition": "Minutes attributable to Ground Ops category.",
            "Directionality": "DOWN good",
            "Grain compatibility": "Monthly/Weekly",
        },
    ]
    return pd.DataFrame(rows)


def build_qa_lookup(qa_df: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if qa_df is None or qa_df.empty:
        return {}

    file_col = find_column(qa_df, ["FileName", "File", "file", "file_name"])
    if file_col is None:
        return {}

    rows_col = find_column(qa_df, ["Rows", "rows"])
    cols_col = find_column(qa_df, ["Cols", "cols", "Columns"])
    min_ym_col = find_column(qa_df, ["MinYearMonth", "min_yearmonth"])
    max_ym_col = find_column(qa_df, ["MaxYearMonth", "max_yearmonth"])
    min_yw_col = find_column(qa_df, ["MinYearWeek", "min_yearweek"])
    max_yw_col = find_column(qa_df, ["MaxYearWeek", "max_yearweek"])

    out: dict[str, dict[str, Any]] = {}
    for _, row in qa_df.iterrows():
        file_val = row.get(file_col)
        if pd.isna(file_val):
            continue
        key = normalize_name(Path(str(file_val)).name)
        if not key:
            continue
        out[key] = {
            "QaRows": int(row.get(rows_col)) if rows_col and pd.notna(row.get(rows_col)) else None,
            "QaCols": int(row.get(cols_col)) if cols_col and pd.notna(row.get(cols_col)) else None,
            "QaMinYearMonth": str(row.get(min_ym_col)) if min_ym_col and pd.notna(row.get(min_ym_col)) else None,
            "QaMaxYearMonth": str(row.get(max_ym_col)) if max_ym_col and pd.notna(row.get(max_ym_col)) else None,
            "QaMinYearWeek": str(row.get(min_yw_col)) if min_yw_col and pd.notna(row.get(min_yw_col)) else None,
            "QaMaxYearWeek": str(row.get(max_yw_col)) if max_yw_col and pd.notna(row.get(max_yw_col)) else None,
        }
    return out


# ── Page header ──────────────────────────────────────────────────────────────

st.title("Dictionary / Policy")
st.caption("PASS-only. Definitions + governance + provenance.")

ctx = st.session_state.get("mode_a_ctx")
if not isinstance(ctx, dict):
    st.error("Select a run in the sidebar")
    st.stop()
render_freshness_banner(ctx)
render_filter_banner(ctx, st.session_state.get("mode_a_filters", {}))

stamp_json_path = Path(ctx.get("run_stamp_json_path", ""))
runtime_data_log(f"ReadRunStamp path={stamp_json_path}")

run_stamp = ctx.get("run_stamp") if isinstance(ctx.get("run_stamp"), dict) else {}
region = str(ctx.get("region", "")).strip()
mode = str(ctx.get("mode", "")).strip()
stamp = ctx.get("stamp")
git_commit = ctx.get("git_commit")
git_branch = ctx.get("git_branch")
insights_dir = Path(ctx.get("insights_dir", Path(os.getenv("FLIGHTOPS_INSIGHTS_DIR", r"C:\Users\IT\02_insights\insight_out"))))
qa_path = ctx.get("qa_path")
runtime_data_log(f"ReadQaSummaryPath path={qa_path}")

ops_gate_line = maybe_get(run_stamp, "ops_gate_pack_verdict", "OpsGatePackVerdict")
ops_gate_log_path = maybe_get(run_stamp, "ops_gate_pack_log_path", "OpsGatePackLogPath")
ops_gate_stamp = maybe_get(run_stamp, "ops_gate_pack_stamp", "OpsGatePackStamp")

required_files = ensure_list(ctx.get("required_files"))
if not required_files:
    st.warning("run_stamp.required_files is missing/empty. Using baseline 13-file contract pattern for dictionary view.")
    required_files = baseline_required_files(region, mode)

# ── Load QA summary (needed by both tabs) ────────────────────────────────────

qa_df: pd.DataFrame | None = None
if qa_path is not None and Path(qa_path).exists():
    qa_df, _ = safe_read_csv(Path(qa_path))

# ── Tab layout ────────────────────────────────────────────────────────────────

tab_kpi, tab_registry = st.tabs(["📋 KPI Definitions", "🗂 Artifact Registry"])


# ── Tab 1: KPI Definitions ────────────────────────────────────────────────────

with tab_kpi:
    st.subheader("KPI Dictionary")
    kpi_dict_df = build_kpi_dictionary()
    st.dataframe(kpi_dict_df, width="stretch", hide_index=True)

    st.subheader("Governance Notes")
    st.markdown("- **Owner = DelayCategory** — single accountability basis. No separate Owner axis.")
    st.markdown("- **POSONLY mode** — only positive delays (>0 min) are included in all KPI computations.")
    st.markdown("- **OTP D15 target: 85 %** — departures ≤ 15 min late count as on-time.")
    st.markdown("- **Aerops exports use UTC timestamps** — no timezone conversion required.")
    st.markdown("- **Grain safety** — never use annual-only driver/category data for monthly or weekly views.")
    st.markdown("- **Gates:** Contract + LoaderGate + LandingDelays + DbSanity → OPS_GATE verdict.")
    st.markdown("- **Evidence** — every run is stamped; logs are timestamped; UTF-8 and UTF-16LE logs handled.")

    st.download_button(
        label="⬇ Download KPI Dictionary CSV",
        data=kpi_dict_df.to_csv(index=False).encode("utf-8"),
        file_name=f"kpi_dictionary__{region}__{mode}.csv",
        mime="text/csv",
    )


# ── Tab 2: Artifact Registry ──────────────────────────────────────────────────

with tab_registry:
    st.subheader("Provenance")
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Stamp", str(stamp) if stamp else "<missing>")
    p2.metric("Region", str(region))
    p3.metric("Mode", str(mode))
    p4.metric("Git Branch", str(git_branch) if git_branch else "<missing>")
    p5.metric("Git Commit", str(git_commit) if git_commit else "<missing>")

    st.write("Ops Gate Line")
    if isinstance(ops_gate_line, str) and ops_gate_line.strip():
        st.code(ops_gate_line.strip(), language="text")
    else:
        st.write("missing in run_stamp")

    if isinstance(ops_gate_stamp, str) and ops_gate_stamp.strip():
        st.caption(f"ops_gate_pack_stamp: `{ops_gate_stamp}`")
    if isinstance(ops_gate_log_path, str) and ops_gate_log_path.strip():
        st.caption(f"ops_gate_pack_log_path: `{ops_gate_log_path}`")

    st.subheader("QA Summary")
    if qa_path is None:
        st.error("qa_summary path: FAIL (cannot infer; missing stamp/qa_summary_path in run_stamp)")
    else:
        st.write(f"qa_summary_path: `{qa_path}`")
        if Path(qa_path).exists():
            st.success("qa_summary existence: PASS")
            if qa_df is not None:
                st.dataframe(qa_df.head(5), width="stretch", hide_index=True)
            else:
                st.error("Could not read qa_summary CSV.")
        else:
            st.error("qa_summary existence: FAIL (file not found)")


    st.subheader("Artifact Dictionary")
    qa_lookup = build_qa_lookup(qa_df)
    artifact_rows: list[dict[str, Any]] = []

    for file_entry in required_files:
        path_obj = Path(file_entry)
        full_path = path_obj if path_obj.is_absolute() else (insights_dir / path_obj)
        file_name = path_obj.name
        grain, period_col = classify_grain_period(file_name)
        key = normalize_name(file_name)
        qa_meta = qa_lookup.get(key, {})
        artifact_rows.append(
            {
                "FileName": file_name,
                "Purpose": artifact_purpose(file_name),
                "Grain": grain,
                "Period column": period_col,
                "Notes": artifact_notes(file_name),
                "Exists": bool(full_path.exists()),
                "FullPath": str(full_path),
                "QaRows": qa_meta.get("QaRows"),
                "QaCols": qa_meta.get("QaCols"),
                "QaMinYearMonth": qa_meta.get("QaMinYearMonth"),
                "QaMaxYearMonth": qa_meta.get("QaMaxYearMonth"),
                "QaMinYearWeek": qa_meta.get("QaMinYearWeek"),
                "QaMaxYearWeek": qa_meta.get("QaMaxYearWeek"),
            }
        )

    artifact_df = pd.DataFrame(artifact_rows)
    st.dataframe(artifact_df, width="stretch", hide_index=True)

    missing_artifacts = artifact_df.loc[~artifact_df["Exists"], "FileName"].tolist() if not artifact_df.empty else []
    if missing_artifacts:
        st.warning(f"Missing artifacts: {len(missing_artifacts)}")
        st.code("\n".join(missing_artifacts), language="text")
    else:
        st.success("All listed artifacts exist under the resolved insights directory.")

    st.download_button(
        label="⬇ Download Artifact Dictionary CSV",
        data=artifact_df.to_csv(index=False).encode("utf-8"),
        file_name=f"artifact_dictionary__{region}__{mode}.csv",
        mime="text/csv",
    )
