import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from mode_a_run_context import get_or_select_run_context
from mode_a_filters import render_global_filters
from utils import (
    show_plotly, runtime_data_log, render_freshness_banner, render_filter_banner,
    render_gate_verdict_banner,
    otp_rag_class, render_kpi_card,
    style_plotly_card, _compute_delta_html,
)


st.set_page_config(page_title="DEP Intelligence | Mode A", layout="wide")

def inject_premium_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Sidebar child styling (bg handled by config.toml) ── */
[data-testid="stSidebar"] * {
    color: #334155;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label {
    font-size: 0.82rem;
    font-weight: 600;
    color: #475569;
}

/* ── Global font (color handled by config.toml) ── */
html, body, [class*="css"] {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* ── Cards (bordered containers) ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04);
    padding: 0.25rem;
}
div[data-testid="stVerticalBlockBorderWrapper"] * {
    color: #0F172A;
}

/* ── KPI card styles ── */
.kpi-label {
    color: #64748B !important;
    font-weight: 600;
    font-size: 11px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
    line-height: 1.2;
}
.kpi-value {
    color: #0F172A !important;
    font-weight: 800;
    font-size: 34px;
    line-height: 1.0;
}
.kpi-delta {
    font-size: 13px;
    font-weight: 600;
    margin-top: 0.2rem;
    line-height: 1.3;
}
.kpi-delta-good { color: #059669 !important; }
.kpi-delta-bad  { color: #E11D48 !important; }
.kpi-delta-flat { color: #94A3B8 !important; }

/* RAG status borders on KPI cards */
.kpi-rag-green  { border-left: 4px solid #059669 !important; }
.kpi-rag-amber  { border-left: 4px solid #D97706 !important; }
.kpi-rag-red    { border-left: 4px solid #E11D48 !important; }
.kpi-rag-neutral { border-left: 4px solid #CBD5E1 !important; }

/* ── Hero title ── */
.hero-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.25;
}
.hero-subtitle {
    margin-top: 0.15rem;
    font-size: 0.88rem;
    color: #64748B;
}

/* ── Filter pills ── */
.pill-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    justify-content: flex-end;
}
.filter-pill {
    display: inline-block;
    border: 1px solid #CBD5E1;
    border-radius: 999px;
    padding: 0.22rem 0.6rem;
    font-size: 0.76rem;
    color: #475569;
    background: #F1F5F9;
    white-space: nowrap;
    font-weight: 500;
}

/* ── Section notes & labels ── */
.card-note {
    margin-top: 0.35rem;
    color: #64748B !important;
    font-weight: 500;
    font-size: 12px;
    line-height: 1.35;
}
.muted-note {
    margin-top: 0.35rem;
    font-size: 0.82rem;
    color: #64748B;
    font-weight: 500;
}

/* ── Concentration panel ── */
.conc-top2 {
    font-size: 0.86rem;
    color: #0F172A;
    margin-bottom: 0.3rem;
}
.conc-red {
    color: #E11D48 !important;
    font-weight: 800;
}

/* ── Decision pill ── */
.decision-pill {
    display: inline-block;
    margin-top: 0.45rem;
    padding: 0.35rem 0.65rem;
    border-radius: 999px;
    background: #059669;
    color: #FFFFFF !important;
    font-size: 0.82rem;
    font-weight: 600;
}

/* ── Donut legend items ── */
.split-legend { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.1rem; }
.split-item { display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.82rem; color: #0F172A; }
.split-dot { width: 10px; height: 10px; border-radius: 999px; display: inline-block; }

/* ── Footer note ── */
.footer-note {
    margin-top: 0.5rem;
    color: #94A3B8;
    font-size: 0.78rem;
    font-weight: 500;
}

/* ── Expander headers — cleaner ── */
[data-testid="stExpander"] summary {
    font-weight: 600;
    color: #334155;
}

/* ── Metric widget override ── */
[data-testid="stMetric"] label {
    color: #64748B !important;
    font-size: 0.78rem;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #0F172A !important;
    font-size: 1.1rem;
}

/* ── Hide hamburger + footer for screenshots ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
        """,
        unsafe_allow_html=True,
    )

@st.cache_data(show_spinner=False)
def load_csv(path_str: str, mtime: float | None) -> pd.DataFrame:
    _ = mtime
    return pd.read_csv(path_str)


def _artifact_name_for_key(key_or_filename: str, region: str) -> str:
    by_key = {
        "monthly": f"2025_DEP_Monthly_Station_KPIs__{region}.csv",
        "weekly": f"2025_DEP_Weekly_Station_KPIs__{region}.csv",
        "topcodes": f"2025_DEP_TopDelayCodes__{region}_NORM.csv",
        "owner": f"2025_DEP_Owner_Minutes__{region}_NORM.csv",
        "category_annual": f"2025_DEP_DelayCategory_Minutes__{region}_NORM.csv",
        "category_monthly": f"2025_DEP_DelayCategory_Minutes__{region}_NORM__MONTHLY.csv",
        "category_weekly": f"2025_DEP_DelayCategory_Minutes__{region}_NORM__WEEKLY.csv",
    }
    return by_key.get(key_or_filename, key_or_filename)


def resolve_artifact_path(key_or_filename: str) -> Path:
    ctx = st.session_state.get("mode_a_ctx")
    if not isinstance(ctx, dict):
        st.error("Run context missing. Select a run in the sidebar.")
        st.stop()

    region = str(ctx.get("region", "")).strip()
    insights_dir = Path(ctx.get("insights_dir", Path(os.getenv("FLIGHTOPS_INSIGHTS_DIR", "demo_data/insight_out"))))
    required_files = [Path(str(x)).name for x in ctx.get("required_files", [])]
    artifacts_by_name = ctx.get("artifacts_by_name", {}) or {}

    preferred = _artifact_name_for_key(key_or_filename, region)
    if preferred in artifacts_by_name:
        return Path(artifacts_by_name[preferred])

    preferred_l = preferred.lower()
    for name, p in artifacts_by_name.items():
        if str(name).lower() == preferred_l:
            return Path(p)

    prefix = preferred.split("__")[0].lower()
    region_l = region.lower()
    for name, p in artifacts_by_name.items():
        n = str(name)
        n_l = n.lower()
        if n_l.startswith(prefix) and region_l in n_l:
            return Path(p)

    for name in required_files:
        n_l = name.lower()
        if n_l.startswith(prefix) and region_l in n_l:
            return insights_dir / name

    return insights_dir / preferred


def read_artifact_csv(key_or_filename: str) -> pd.DataFrame:
    p = resolve_artifact_path(key_or_filename)
    mtime = p.stat().st_mtime if p.exists() else None
    return load_csv(str(p), mtime)



def category_file_key_for_grain(grain_name: str) -> str:
    if grain_name == "Monthly":
        return "category_monthly"
    if grain_name == "Weekly":
        return "category_weekly"
    return "category_annual"

# ---------------- Helpers ----------------
def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def detect_col(df: pd.DataFrame, candidates: list[str], contains_any: list[str] | None = None) -> str | None:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    if contains_any:
        low = {c: c.lower() for c in cols}
        for c in cols:
            cl = low[c]
            if any(tok in cl for tok in contains_any):
                return c
    return None

def pick_minutes_col(df: pd.DataFrame) -> str:
    candidates = [
        "Minutes_NORM_to_DepDelayMin",
        "Minutes_NORM",
        "Minutes_NORM_Total",
        "DelayMin_NORM",
        "DelayMin",
        "Minutes",
        "DepDelayMin_Total",
        "DepDelayMin_Total_NORM",
        "DepDelayMin_Total_PosOnly_Min",
        "DepDelayTotal_PosOnly_Min",
    ]
    c = detect_col(df, candidates, contains_any=["min", "minute"])
    if c is None:
        raise ValueError(f"No minutes-like column found. Columns={list(df.columns)}")
    return c

def station_key_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def safe_filter_station(df: pd.DataFrame, stations_sel: list[str]) -> pd.DataFrame:
    st_col = detect_col(df, ["Station"], contains_any=["station"])
    if st_col is None:
        return df
    sel_keys = {str(v).strip().upper() for v in stations_sel if str(v).strip()}
    if not sel_keys:
        return df.iloc[0:0].copy()
    return df[station_key_series(df[st_col]).isin(sel_keys)].copy()

def available_periods(ts: pd.DataFrame, period_col: str, sort_col: str, stations: list[str]) -> list[str]:
    station_keys = {str(v).strip().upper() for v in stations if str(v).strip()}
    d = ts[station_key_series(ts["Station"]).isin(station_keys)].copy()
    d["Flights_Operated"] = to_num(d["Flights_Operated"]).fillna(0.0)
    # --- Mode A: derive missing sort_col (WeekStart/MonthStart) for Period-based KPI exports ---
    if sort_col not in d.columns:
        if sort_col == "WeekStart":
            src = None
            if "YearWeek" in d.columns:
                src = d["YearWeek"].astype(str)
            elif "Period" in d.columns:
                src = d["Period"].astype(str)
            if src is not None:
                _s = src.str.strip()
                _s = _s.str.replace(r"^(\d{4})-?W?(\d{1,2})$", r"\1-W\2", regex=True)
                _s = _s.str.replace(r"^(\d{4})-(\d{1,2})$", r"\1-W\2", regex=True)
                _s = _s.str.replace(r"^(\d{4})-W(\d{1})$", r"\1-W0\2", regex=True)
                d["WeekStart"] = pd.to_datetime(_s + "-1", format="%G-W%V-%u", errors="coerce")
                if "YearWeek" not in d.columns:
                    d["YearWeek"] = _s
        elif sort_col == "MonthStart":
            src = None
            if "YearMonth" in d.columns:
                src = d["YearMonth"].astype(str)
            elif "Period" in d.columns:
                src = d["Period"].astype(str)
            if src is not None:
                _ym = src.str.strip().str.slice(0, 7)
                d["MonthStart"] = pd.to_datetime(_ym + "-01", format="%Y-%m-%d", errors="coerce")
                if "YearMonth" not in d.columns:
                    d["YearMonth"] = _ym
    # --- end derive ---
    d[sort_col] = pd.to_datetime(d[sort_col], errors="coerce")
    g = d.groupby([period_col, sort_col], as_index=False)["Flights_Operated"].sum()
    g = g.dropna(subset=[sort_col])
    g = g[g["Flights_Operated"] > 0].sort_values(sort_col)
    return g[period_col].astype(str).tolist()

def fmt_int(x) -> str:
    if x is None or pd.isna(x):
        return "&mdash;"
    return f"{int(round(float(x))):,}"

def fmt_pct0(x) -> str:
    if x is None or pd.isna(x):
        return "&mdash;"
    return f"{int(round(float(x)))}%"




def delta_mode(label: str) -> str:
    l = str(label or "").lower()
    if "otp" in l:
        return "normal"
    if "flight" in l and "operat" in l:
        return "normal"
    if any(tok in l for tok in ["avg", "delay", "minutes", "min"]):
        return "inverse"
    return "off"

def build_network_series(ts: pd.DataFrame, period_col: str, sort_col: str) -> pd.DataFrame:
    d = ts.copy() if hasattr(ts, "copy") else ts
    # --- Mode A schema normalization (protect against KPI schema drift) ---
    # Flights
    if "Flights_Operated" not in d.columns and "Flights" in d.columns:
        d["Flights_Operated"] = d["Flights"]

    # OTP aliasing (station-level)
    if "OTP_D15_Pct_basis_PerfEligible" not in d.columns:
        for _c in ["OTP_D15_Pct", "OTP_D15", "OTP_D15_Pct_basis"]:
            if _c in d.columns:
                d["OTP_D15_Pct_basis_PerfEligible"] = d[_c]
                break

    # Delay avg aliasing (station-level)
    if "AvgDepDelayMin_PerFlight_basis_PosOnly" not in d.columns:
        for _c in ["AvgDepDelayMin_PerFlight", "AvgDepDelayMin_PerFlight_basis"]:
            if _c in d.columns:
                d["AvgDepDelayMin_PerFlight_basis_PosOnly"] = d[_c]
                break

    # Monthly period columns
    if "MonthStart" not in d.columns:
        if "YearMonth" in d.columns:
            d["MonthStart"] = pd.to_datetime(d["YearMonth"].astype(str).str.slice(0, 7) + "-01", errors="coerce")
        elif "Period" in d.columns:
            _ym = d["Period"].astype(str).str.slice(0, 7)
            d["YearMonth"] = _ym
            d["MonthStart"] = pd.to_datetime(_ym + "-01", format="%Y-%m-%d", errors="coerce")

    if "YearMonth" not in d.columns and "MonthStart" in d.columns:
        d["YearMonth"] = pd.to_datetime(d["MonthStart"], errors="coerce").dt.strftime("%Y-%m")

    # Weekly period columns (ISO week)
    if "WeekStart" not in d.columns:
        if "YearWeek" in d.columns:
            _s = d["YearWeek"].astype(str).str.strip()
            _s = _s.str.replace(r"^(\d{4})-?W?(\d{1,2})$", r"\1-W\2", regex=True)
            _s = _s.str.replace(r"^(\d{4})-(\d{1,2})$", r"\1-W\2", regex=True)
            _s = _s.str.replace(r"^(\d{4})-W(\d{1})$", r"\1-W0\2", regex=True)
            d["WeekStart"] = pd.to_datetime(_s + "-1", format="%G-W%V-%u", errors="coerce")
        elif "Period" in d.columns:
            _s = d["Period"].astype(str).str.strip()
            _s = _s.str.replace(r"^(\d{4})-?W?(\d{1,2})$", r"\1-W\2", regex=True)
            _s = _s.str.replace(r"^(\d{4})-(\d{1,2})$", r"\1-W\2", regex=True)
            _s = _s.str.replace(r"^(\d{4})-W(\d{1})$", r"\1-W0\2", regex=True)
            d["YearWeek"] = _s
            d["WeekStart"] = pd.to_datetime(_s + "-1", format="%G-W%V-%u", errors="coerce")

    if "YearWeek" not in d.columns and "WeekStart" in d.columns:
        _dt = pd.to_datetime(d["WeekStart"], errors="coerce")
        iso = _dt.dt.isocalendar()
        d["YearWeek"] = iso["year"].astype(str) + "-W" + iso["week"].astype(int).astype(str).str.zfill(2)

    # --- end normalization ---
    # Weighted (network-correct) aggregation
    delay_total_col = detect_col(ts,
        ["DepDelayTotal_PosOnly_Min", "DepDelayMin_Total_PosOnly_Min", "DepDelayMin_Total_PosOnly", "DepDelayMin_Total"],
        contains_any=["depdelaymin_total", "posonly", "pos_only"]
    )
    f_col = "Flights_Operated"
    pe_col = detect_col(ts, ["Flights_PerfEligible", "PerfEligible", "FlightsPerfEligible"], contains_any=["perfel"])
    fail_col = detect_col(ts, ["D15_Fail_Flights", "Fail_D15_Flights"], contains_any=["d15", "fail"])
    cont_col = detect_col(ts, ["Controllable_DEP_Min_NORM", "ControllableMin_NORM", "Controllable_Min_NORM_Total"], contains_any=["controllable"])

    d = ts.copy()
    d[f_col] = to_num(d[f_col]).fillna(0.0)
    if pe_col:
        d[pe_col] = to_num(d[pe_col]).fillna(0.0)
    if fail_col:
        d[fail_col] = to_num(d[fail_col]).fillna(0.0)
    if cont_col:
        d[cont_col] = to_num(d[cont_col]).fillna(0.0)
    if delay_total_col:
        d[delay_total_col] = to_num(d[delay_total_col]).fillna(0.0)

    d[sort_col] = pd.to_datetime(d[sort_col], errors="coerce")
    g = d.groupby([period_col, sort_col], as_index=False).agg({
        f_col: "sum",
        (pe_col or f_col): "sum",
        (fail_col or f_col): "sum",
        (cont_col or f_col): "sum",
        (delay_total_col or f_col): "sum",
    }).sort_values(sort_col)

    g["x_label"] = g[period_col].astype(str)

    # Avg delay (min/flight)
    if delay_total_col:
        g["AvgDelay"] = np.where(g[f_col] > 0, g[delay_total_col] / g[f_col], np.nan)
    else:
        # fallback: mean-of-stations column if present (less ideal, but safe)
        avg_col = detect_col(d, ["AvgDepDelayMin_PerFlight_basis_PosOnly"], contains_any=["avgdepdelay"])
        if avg_col:
            h = d.groupby([period_col, sort_col], as_index=False)[avg_col].mean().sort_values(sort_col)
            g = g.merge(h[[period_col, sort_col, avg_col]], on=[period_col, sort_col], how="left")
            g["AvgDelay"] = g[avg_col]
        else:
            g["AvgDelay"] = np.nan

    # OTP D15% (network)
    if pe_col and fail_col:
        g["OTP"] = np.where(g[pe_col] > 0, 100.0 * (1.0 - (g[fail_col] / g[pe_col])), np.nan)
    else:
        otp_col = detect_col(d, ["OTP_D15_Pct_basis_PerfEligible"], contains_any=["otp", "d15"])
        # --- OTP: compute weighted network OTP (fix KeyError + correct aggregation) ---
    # (removed) import pandas as pd  # keep pandas import at module level
        if otp_col in d.columns:
            _tmp = d[[period_col, sort_col, otp_col, "Flights_Operated"]].copy()
            _tmp[otp_col] = pd.to_numeric(_tmp[otp_col], errors="coerce")
            _tmp["__otp_w"] = _tmp[otp_col] * _tmp["Flights_Operated"]

            otp_w = _tmp.groupby([period_col, sort_col], as_index=False)["__otp_w"].sum()
            flt  = _tmp.groupby([period_col, sort_col], as_index=False)["Flights_Operated"].sum().rename(columns={"Flights_Operated":"__flt"})
            otp_w = otp_w.merge(flt, on=[period_col, sort_col], how="left")
            otp_w["OTP"] = otp_w["__otp_w"] / otp_w["__flt"].replace(0, np.nan)

            g = g.merge(otp_w[[period_col, sort_col, "OTP"]], on=[period_col, sort_col], how="left")
        else:
            # fallback: avoid crash; surfaces missing input column clearly
            g["OTP"] = np.nan
        # --- end OTP patch ---

    g["Controllable"] = g[cont_col] if cont_col else np.nan
    return g

# ---------------- Load & UI ----------------
inject_premium_css()

st.info(
    "**Live demo — synthetic data only.** "
    "Architecture, design, and all code are production-identical. "
    "Station codes, delay minutes, and OTP figures are computer-generated for demonstration purposes. "
    "No real operational data is present in this deployment.",
    icon="ℹ️",
)

ctx = get_or_select_run_context(st, Path(os.getenv("FLIGHTOPS_INSIGHTS_DIR", "demo_data/insight_out")))
st.session_state["mode_a_ctx"] = ctx
render_freshness_banner(ctx)
render_gate_verdict_banner(ctx)

region = str(ctx.get("region", ""))
mode = str(ctx.get("mode", ""))
stamp = ctx.get("stamp")
git_commit = ctx.get("git_commit")
qa_path = ctx.get("qa_path")
qa_df = ctx.get("qa_df")
required_files = ctx.get("required_files", [])

# ── Compact metadata behind expander for management pages ──
with st.expander("📋 Data Source & Pipeline Details", expanded=False):
    hdr_c1, hdr_c2, hdr_c3, hdr_c4 = st.columns(4)
    hdr_c1.metric("Region", f"{region} / {mode}")
    hdr_c2.metric("Build ID", str(stamp) if stamp else "<missing>")
    hdr_c3.metric("Version", str(git_commit)[:8] if git_commit else "<missing>")
    hdr_c4.metric("QA File", Path(str(qa_path)).name if qa_path else "<missing>")

st.caption(f"debug qa_path: {qa_path}")
if not isinstance(qa_df, pd.DataFrame) or qa_df.empty:
    st.error("Contract gate failed: qa_summary is missing or empty.")
    st.stop()

qa_col_map = {str(c).strip().lower(): c for c in qa_df.columns}
file_col = qa_col_map.get("filename")
exists_col = qa_col_map.get("exists")
rows_col = qa_col_map.get("rows")
cols_col = qa_col_map.get("cols")
if file_col is None or exists_col is None or rows_col is None or cols_col is None:
    st.error("Contract gate failed: standardized QA must include FileName, Exists, Rows, Cols.")
    st.stop()

exists_norm = qa_df[exists_col].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
rows_num = pd.to_numeric(qa_df[rows_col], errors="coerce")
cols_num = pd.to_numeric(qa_df[cols_col], errors="coerce")
qa_fail = qa_df[(~exists_norm) | (rows_num <= 0) | (cols_num <= 0)].copy()
if not qa_fail.empty:
    st.error("Contract gate failed: required artifacts have missing files (computed Exists) or invalid Rows/Cols.")
    st.dataframe(qa_fail.head(50), width="stretch", hide_index=True)
    st.stop()

filters = render_global_filters(st, ctx)
st.session_state["mode_a_filters"] = filters
render_filter_banner(ctx, filters)

grain = str(filters.get("grain", "Monthly"))
selected_grain = grain if grain in ["Annual", "Monthly", "Weekly"] else "Annual"
selected_periods_from_global = [str(p) for p in filters.get("periods", []) if str(p).strip()]
station_mode = str(filters.get("station_mode", "NETWORK"))
stations_from_global = [str(s).strip() for s in filters.get("stations", []) if str(s).strip()]
if not selected_periods_from_global:
    selected_periods_from_global = []

if selected_periods_from_global:
    if len(selected_periods_from_global) == 1:
        period_hdr = selected_periods_from_global[0]
    else:
        period_hdr = f"{selected_periods_from_global[0]}..{selected_periods_from_global[-1]} ({len(selected_periods_from_global)})"
else:
    period_hdr = "<auto>"
station_count_hdr = 1 if stations_from_global == ["NETWORK"] else len(stations_from_global)
st.caption(
    f"Filters: Grain={grain} | Periods={period_hdr} | StationMode={station_mode} | Stations={station_count_hdr}"
)

monthly = read_artifact_csv("monthly")
weekly = read_artifact_csv("weekly")
cat_file_key = category_file_key_for_grain(selected_grain)
cat_path = resolve_artifact_path(cat_file_key)
cat_raw = read_artifact_csv(cat_file_key)
codes_raw = read_artifact_csv("topcodes")

ts = monthly if grain == "Monthly" else weekly
period_col = "YearMonth" if grain == "Monthly" else "YearWeek"
sort_col = "MonthStart" if grain == "Monthly" else "WeekStart"
kpis_file_key = "monthly" if grain == "Monthly" else "weekly"
kpis_path = resolve_artifact_path(kpis_file_key)
kpis_df_for_grain = monthly if kpis_file_key == "monthly" else weekly
runtime_data_log(
    f"Grain={grain} StationKpisFile={kpis_path} Rows={len(kpis_df_for_grain)} Cols={len(kpis_df_for_grain.columns)}"
)

stations_all = sorted([s.strip() for s in ts["Station"].astype(str).unique().tolist() if s.strip() and s.strip().upper() != "ALL"])
if station_mode == "NETWORK" or stations_from_global == ["NETWORK"]:
    stations_sel = stations_all
else:
    stations_sel = [s for s in stations_from_global if s in stations_all]
    if not stations_sel:
        stations_sel = stations_all
if not stations_sel:
    st.error("Select at least one station.")
    st.stop()
selected_station_keys = list(dict.fromkeys([s.upper() for s in stations_sel]))
station_display_by_key = {s.upper(): s for s in stations_sel}

# --- Mode A: ensure period_col/sort_col exist on ts (pre-filters) ---
# Weekly KPI exports may contain Period and/or WeekStart but not YearWeek; monthly may contain Period but not YearMonth.
# This block normalizes `ts` so later filters like ts[period_col] and ts[sort_col] never KeyError.

# Ensure period_col exists
if period_col not in ts.columns:
    if "Period" in ts.columns:
        ts[period_col] = ts["Period"].astype(str)
    elif period_col == "YearWeek" and "WeekStart" in ts.columns:
        _dt = pd.to_datetime(ts["WeekStart"], errors="coerce")
        iso = _dt.dt.isocalendar()
        ts["YearWeek"] = iso["year"].astype(str) + "-W" + iso["week"].astype(int).astype(str).str.zfill(2)
    elif period_col == "YearMonth" and "MonthStart" in ts.columns:
        ts["YearMonth"] = pd.to_datetime(ts["MonthStart"], errors="coerce").dt.strftime("%Y-%m")

# Ensure sort_col exists
if sort_col not in ts.columns:
    if sort_col == "WeekStart":
        src = None
        if "YearWeek" in ts.columns:
            src = ts["YearWeek"].astype(str)
        elif "Period" in ts.columns:
            src = ts["Period"].astype(str)
        if src is not None:
            _s = src.str.strip()
            _s = _s.str.replace(r"^(\d{4})-?W?(\d{1,2})$", r"\1-W\2", regex=True)
            _s = _s.str.replace(r"^(\d{4})-(\d{1,2})$", r"\1-W\2", regex=True)
            _s = _s.str.replace(r"^(\d{4})-W(\d{1})$", r"\1-W0\2", regex=True)
            ts["WeekStart"] = pd.to_datetime(_s + "-1", format="%G-W%V-%u", errors="coerce")
            if "YearWeek" not in ts.columns:
                ts["YearWeek"] = _s

    elif sort_col == "MonthStart":
        src = None
        if "YearMonth" in ts.columns:
            src = ts["YearMonth"].astype(str)
        elif "Period" in ts.columns:
            src = ts["Period"].astype(str)
        if src is not None:
            _ym = src.str.strip().str.slice(0, 7)
            ts["MonthStart"] = pd.to_datetime(_ym + "-01", format="%Y-%m-%d", errors="coerce")
            if "YearMonth" not in ts.columns:
                ts["YearMonth"] = _ym

# --- end normalize ---
periods = available_periods(ts, period_col, sort_col, stations_sel)
if not periods:
    st.error("No active operations for selected stations.")
    st.stop()

selected_periods = [p for p in selected_periods_from_global if p in periods]
if not selected_periods:
    selected_periods = [periods[-1]]
period_sel = selected_periods[-1]
focus_options = ["Controllable Minutes", "Total DEP Delay Minutes", "Ground Ops Minutes"]
focus_metric = st.sidebar.selectbox("Focus Metric", focus_options, index=0)

# ---------------- Build network series ----------------
ts_filtered = ts[station_key_series(ts["Station"]).isin(selected_station_keys)].copy()
net = build_network_series(ts_filtered, period_col, sort_col)
net = net[net[period_col].astype(str).isin(selected_periods)].copy()
if net.empty:
    net = build_network_series(ts_filtered, period_col, sort_col)

def val_for(p: str | None, col: str) -> float:
    if p is None:
        return np.nan
    r = net[net[period_col].astype(str) == str(p)]
    if r.empty:
        return np.nan
    return float(r[col].iloc[-1])

delay_total_col = detect_col(
    ts_filtered,
    ["DepDelayTotal_PosOnly_Min", "DepDelayMin_Total_PosOnly_Min", "DepDelayMin_Total_PosOnly", "DepDelayMin_Total"],
    contains_any=["depdelaymin_total", "posonly", "pos_only"]
)
cont_station_col = detect_col(
    ts_filtered,
    ["Controllable_DEP_Min_NORM", "ControllableMin_NORM", "Controllable_Min_NORM_Total"],
    contains_any=["controllable"]
)
gops_station_col = detect_col(
    ts_filtered,
    ["GOPS_Min_NORM_Total", "GOPS_Min_NORM", "GroundOps_Min_NORM_Total", "Ground_Ops_Min_NORM_Total"],
    contains_any=["gops_min", "gops"]
)
if gops_station_col and "share" in gops_station_col.lower():
    gops_station_col = None

inherited_station_col = detect_col(
    ts_filtered,
    ["Inherited_Min_NORM_Total", "LateArrival_Min_NORM_Total", "Late_Arrival_Min_NORM_Total", "Inherited_DEP_Min_NORM"],
    contains_any=["inherited", "latearrival", "late_arrival"]
)
if inherited_station_col:
    low_inh = inherited_station_col.lower()
    if ("share" in low_inh) or ("pct" in low_inh):
        inherited_station_col = None

cur_flights = val_for(period_sel, "Flights_Operated") if "Flights_Operated" in net.columns else np.nan
cur_otp = val_for(period_sel, "OTP")
cur_controllable = val_for(period_sel, "Controllable")
cur_total_dep = val_for(period_sel, delay_total_col) if (delay_total_col and delay_total_col in net.columns) else np.nan

def dataset_supports_selected_period(df: pd.DataFrame, selected_grain_name: str) -> bool:
    if selected_grain_name == "Annual":
        return True
    if selected_grain_name == "Monthly":
        return detect_col(
            df,
            ["YearMonth", "Period", "MonthStart"],
            contains_any=["yearmonth", "period", "monthstart"],
        ) is not None
    if selected_grain_name == "Weekly":
        return detect_col(
            df,
            ["YearWeek", "Period", "WeekStart"],
            contains_any=["yearweek", "period", "weekstart"],
        ) is not None
    return False

def filter_to_selected_period(df: pd.DataFrame, allow_full_fallback: bool = True) -> pd.DataFrame:
    pcol = detect_col(df, [period_col, "Period", "YearMonth", "YearWeek"], contains_any=["period", "yearmonth", "yearweek"])
    if pcol is None:
        return df.copy()
    d = df[df[pcol].astype(str).isin(selected_periods)].copy()
    if d.empty and allow_full_fallback:
        return df.copy()
    return d

ts_period = ts_filtered[ts_filtered[period_col].astype(str).isin(selected_periods)].copy()
if ts_period.empty:
    ts_period = ts_filtered.copy()

station_metrics = pd.DataFrame({"StationKey": selected_station_keys})
station_metrics["Station"] = station_metrics["StationKey"].map(station_display_by_key).fillna(station_metrics["StationKey"])

if delay_total_col and delay_total_col in ts_period.columns:
    delay_map = (
        ts_period.assign(
            _station_key=station_key_series(ts_period["Station"]),
            _m=to_num(ts_period[delay_total_col]).fillna(0.0),
        )
        .groupby("_station_key")["_m"]
        .sum()
    )
    has_station_total_dep = True
else:
    delay_map = pd.Series(dtype="float64")
    has_station_total_dep = False
station_metrics["Total DEP Delay Minutes"] = station_metrics["StationKey"].map(delay_map).fillna(0.0)

if cont_station_col and cont_station_col in ts_period.columns:
    cont_map = (
        ts_period.assign(
            _station_key=station_key_series(ts_period["Station"]),
            _m=to_num(ts_period[cont_station_col]).fillna(0.0),
        )
        .groupby("_station_key")["_m"]
        .sum()
    )
else:
    cont_map = pd.Series(dtype="float64")
station_metrics["Controllable Minutes"] = station_metrics["StationKey"].map(cont_map).fillna(0.0)

cat_df_base = safe_filter_station(cat_raw, stations_sel)
category_supports_period = dataset_supports_selected_period(cat_df_base, selected_grain)
cat_supports_selected_grain = category_supports_period
category_grain_mismatch = (selected_grain in ["Monthly", "Weekly"]) and (not category_supports_period)
codes_supports_period = dataset_supports_selected_period(codes_raw, selected_grain)
codes_split_blocked = (selected_grain in ["Monthly", "Weekly"]) and (not codes_supports_period)
split_insights_blocked = category_grain_mismatch
decision_pack_split_blocked = split_insights_blocked or codes_split_blocked
if category_grain_mismatch:
    runtime_data_log(
        "Grain="
        + str(grain)
        + " DelayCategoryMinutesFile=<DISABLED> Reason=guardrail_category_minutes_not_periodized_for_selected_grain"
    )
else:
    runtime_data_log(
        f"Grain={grain} DelayCategoryMinutesFile={cat_path} Rows={len(cat_df_base)} Cols={len(cat_df_base.columns)}"
    )

if category_grain_mismatch:
    cat_df = cat_df_base.iloc[0:0].copy()
else:
    cat_df = filter_to_selected_period(cat_df_base, allow_full_fallback=False)

cat_col = detect_col(
    cat_df,
    ["DelayCategory_Basis_DepDelayMin", "DelayCategory", "Category"],
    contains_any=["delaycategory", "delay_category", "category"]
)
cat_station_col = detect_col(cat_df, ["Station"], contains_any=["station"])
cat_min_col = pick_minutes_col(cat_df)
cat_df[cat_min_col] = to_num(cat_df[cat_min_col]).fillna(0.0)

if gops_station_col and gops_station_col in ts_period.columns:
    gops_station_map = (
        ts_period.assign(
            _station_key=station_key_series(ts_period["Station"]),
            _m=to_num(ts_period[gops_station_col]).fillna(0.0),
        )
        .groupby("_station_key")["_m"]
        .sum()
    )
else:
    gops_station_map = pd.Series(dtype="float64")

if inherited_station_col and inherited_station_col in ts_period.columns:
    inherited_kpi_station_map = (
        ts_period.assign(
            _station_key=station_key_series(ts_period["Station"]),
            _m=to_num(ts_period[inherited_station_col]).fillna(0.0),
        )
        .groupby("_station_key")["_m"]
        .sum()
    )
else:
    inherited_kpi_station_map = pd.Series(dtype="float64")

if cat_col and cat_col in cat_df.columns:
    cat_norm = cat_df[cat_col].astype(str).str.strip().str.lower()
else:
    cat_norm = pd.Series("", index=cat_df.index, dtype="string")

late_mask = (cat_norm.str.contains("late", na=False) & cat_norm.str.contains("arrival", na=False)) | cat_norm.str.contains("inherited", na=False)
ground_mask = cat_norm.str.contains("ground", na=False) & cat_norm.str.contains("ops", na=False)

if cat_supports_selected_grain and cat_station_col and cat_station_col in cat_df.columns:
    cat_work = cat_df.assign(_station_key=station_key_series(cat_df[cat_station_col]))
    ground_station_map = (
        cat_work.loc[ground_mask]
        .groupby("_station_key")[cat_min_col]
        .sum()
    )
    inherited_station_map = (
        cat_work.loc[late_mask]
        .groupby("_station_key")[cat_min_col]
        .sum()
    )
    total_category_station_map = cat_work.groupby("_station_key")[cat_min_col].sum()
else:
    ground_station_map = gops_station_map
    inherited_station_map = inherited_kpi_station_map
    total_category_station_map = pd.Series(dtype="float64")
station_metrics["Ground Ops Minutes"] = station_metrics["StationKey"].map(ground_station_map).fillna(0.0)

if not has_station_total_dep and cat_supports_selected_grain:
    inherited_station_vals = station_metrics["StationKey"].map(inherited_station_map).fillna(0.0)
    total_category_vals = station_metrics["StationKey"].map(total_category_station_map).fillna(0.0)
    controllable_vals = station_metrics["Controllable Minutes"].fillna(0.0)
    other_station_vals = (total_category_vals - inherited_station_vals - controllable_vals).clip(lower=0.0)
    station_metrics["Total DEP Delay Minutes"] = inherited_station_vals + controllable_vals + other_station_vals

if pd.isna(cur_total_dep) or cur_total_dep <= 0:
    cur_total_dep = float(station_metrics["Total DEP Delay Minutes"].sum())
if pd.isna(cur_controllable) or cur_controllable <= 0:
    cur_controllable = float(station_metrics["Controllable Minutes"].sum())
cur_controllable_share = (100.0 * cur_controllable / cur_total_dep) if cur_total_dep > 0 else np.nan

if split_insights_blocked:
    inherited_minutes = np.nan
    minutes_split_total = float(cur_total_dep) if cur_total_dep > 0 else float(station_metrics["Total DEP Delay Minutes"].sum())
    other_minutes = np.nan
    split_has_inherited_breakdown = False
elif cat_supports_selected_grain:
    inherited_minutes = float(cat_df.loc[late_mask, cat_min_col].sum()) if not cat_df.empty else 0.0
    minutes_split_total = float(cur_total_dep) if cur_total_dep > 0 else float(cat_df[cat_min_col].sum())
    other_minutes = max(minutes_split_total - inherited_minutes - float(cur_controllable), 0.0)
    split_has_inherited_breakdown = True
elif inherited_station_col and inherited_station_col in ts_period.columns:
    inherited_minutes = float(to_num(ts_period[inherited_station_col]).fillna(0.0).sum())
    minutes_split_total = float(cur_total_dep) if cur_total_dep > 0 else float(station_metrics["Total DEP Delay Minutes"].sum())
    other_minutes = max(minutes_split_total - inherited_minutes - float(cur_controllable), 0.0)
    split_has_inherited_breakdown = True
else:
    inherited_minutes = np.nan
    minutes_split_total = float(cur_total_dep) if cur_total_dep > 0 else float(station_metrics["Total DEP Delay Minutes"].sum())
    other_minutes = np.nan
    split_has_inherited_breakdown = False

station_metrics = station_metrics.sort_values("Station").reset_index(drop=True)

focus_col = focus_metric
focus_total = float(station_metrics[focus_col].sum())
top5 = station_metrics.sort_values(focus_col, ascending=False).head(5).copy()
top5["SharePct"] = np.where(focus_total > 0, 100.0 * top5[focus_col] / focus_total, 0.0)

top2_names = station_metrics.sort_values(focus_col, ascending=False)["Station"].head(2).tolist()
top2_mask = station_metrics["Station"].isin(top2_names)

def pct_top2(metric_col: str, denom_override: float | None = None) -> float:
    denom = denom_override
    if denom is None:
        denom = float(station_metrics[metric_col].sum())
    if pd.isna(denom) or float(denom) <= 0:
        return np.nan
    num = float(station_metrics.loc[top2_mask, metric_col].sum())
    return 100.0 * num / float(denom)

pct_top2_total = pct_top2("Total DEP Delay Minutes")
pct_top2_cont = pct_top2("Controllable Minutes")
pct_top2_ground = pct_top2("Ground Ops Minutes")

def fmt_pct1_or_na(x: float) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{float(x):.1f}%"

if cat_supports_selected_grain:
    ground_ops_controllable = float(cat_df.loc[ground_mask & ~late_mask, cat_min_col].sum()) if not cat_df.empty else 0.0
    controllable_total_for_split = float(cur_controllable) if cur_controllable > 0 else float(cat_df.loc[~late_mask, cat_min_col].sum())
    ground_ops_controllable = min(max(ground_ops_controllable, 0.0), max(controllable_total_for_split, 0.0))
    other_controllable_minutes = max(controllable_total_for_split - ground_ops_controllable, 0.0)
else:
    controllable_total_for_split = float(cur_controllable) if cur_controllable > 0 else float(station_metrics["Controllable Minutes"].sum())
    gops_total_for_split = float(station_metrics["Ground Ops Minutes"].sum())
    ground_ops_controllable = min(max(gops_total_for_split, 0.0), max(controllable_total_for_split, 0.0))
    other_controllable_minutes = max(controllable_total_for_split - ground_ops_controllable, 0.0)

flights_series = net["Flights_Operated"].astype(float).tolist() if "Flights_Operated" in net.columns else None
otp_series = net["OTP"].astype(float).tolist() if "OTP" in net.columns else None
total_series = net[delay_total_col].astype(float).tolist() if (delay_total_col and delay_total_col in net.columns) else None
if delay_total_col and delay_total_col in net.columns and "Controllable" in net.columns:
    den = net[delay_total_col].astype(float).to_numpy()
    num = net["Controllable"].astype(float).to_numpy()
    share_series = np.where(den > 0, 100.0 * num / den, np.nan).tolist()
else:
    share_series = None

def compact_station_label(selected: list[str], all_stations: list[str]) -> str:
    if len(selected) == len(all_stations):
        return f"All ({len(selected)})"
    return f"{len(selected)} selected"

header_station = compact_station_label(stations_sel, stations_all)
if len(selected_periods) == 1:
    header_period = selected_periods[0]
else:
    header_period = f"{selected_periods[0]}..{selected_periods[-1]} ({len(selected_periods)})"

if split_insights_blocked:
    st.warning(
        "Category/Owner minutes are annual-only; split insights disabled to prevent distortion."
    )

with st.container(border=True):
    col_h1, col_h2 = st.columns([6, 4])
    with col_h1:
        st.markdown("<div class='hero-title'>Departure Performance — Network Overview</div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-subtitle'>East Africa &amp; Turkey · Operated flights · Positive delays only</div>", unsafe_allow_html=True)
    with col_h2:
        st.markdown(
            (
                "<div class='pill-wrap'>"
                f"<span class='filter-pill'>Station: {header_station}</span>"
                f"<span class='filter-pill'>Period: {header_period}</span>"
                f"<span class='filter-pill'>Metric Basis: NORM</span>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi_card(
        "Flights Operated", fmt_int(cur_flights), flights_series,
        "p1_network__kpi_flights_sparkline",
        rag_class="kpi-rag-neutral",
        delta_html=_compute_delta_html(flights_series, "Flights", ""),
    )
with k2:
    render_kpi_card(
        "On-Time Performance", fmt_pct0(cur_otp), otp_series,
        "p1_network__kpi_otp_sparkline",
        rag_class=otp_rag_class(cur_otp),
        delta_html=_compute_delta_html(otp_series, "OTP", " pp"),
    )
with k3:
    render_kpi_card(
        "Total Delay Minutes", fmt_int(cur_total_dep), total_series,
        "p1_network__kpi_totaldep_sparkline",
        rag_class="kpi-rag-neutral",
        delta_html=_compute_delta_html(total_series, "Delay", " min"),
    )
with k4:
    share_text = "&mdash;" if pd.isna(cur_controllable_share) else f"{cur_controllable_share:.1f}%"
    render_kpi_card(
        "Controllable Share", share_text, share_series,
        "p1_network__kpi_controllable_share_sparkline",
        rag_class="kpi-rag-neutral",
        delta_html=_compute_delta_html(share_series, "Delay", " pp"),
    )

st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
left_col, right_col = st.columns([6, 4], gap="large")

with left_col:
    with st.container(border=True):
        if split_has_inherited_breakdown:
            split_labels = ["Inherited (Late Arrival)", "Controllable", "Other"]
            split_vals = [max(inherited_minutes, 0.0), max(float(cur_controllable), 0.0), max(other_minutes, 0.0)]
            split_total = sum(split_vals)
            fig_split = go.Figure(go.Pie(
                labels=split_labels,
                values=split_vals,
                hole=0.66,
                sort=False,
                textinfo="percent",
                textposition="inside",
                insidetextorientation="horizontal",
                textfont=dict(color="#0B1220", size=12),
                marker=dict(colors=["#94A3B8", "#4F46E5", "#E2E8F0"]),
                hovertemplate="%{label}<br>%{value:,.0f} min<br>%{percent}<extra></extra>",
            ))
            fig_split.add_annotation(
                x=0.5,
                y=0.5,
                showarrow=False,
                text=f"Total<br><b>{split_total:,.0f} min</b>",
                font=dict(color="#0B1220", size=15, family="Inter, Arial, sans-serif"),
                align="center",
            )
            style_plotly_card(fig_split, title="Delay Minutes Breakdown", height=320, margin=dict(l=10, r=10, t=52, b=92))
            fig_split.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="h",
                    x=0,
                    y=-0.05,
                    xanchor="left",
                    yanchor="top",
                    font=dict(color="#0B1220", size=12, family="Inter, Arial, sans-serif"),
                    itemwidth=40,
                ),
                uniformtext_minsize=12,
                uniformtext_mode="hide",
            )
            show_plotly(fig_split, key="p1_network__minutes_split")
            st.markdown(
                "<div class='card-note'>Controllable minutes represent the operator levers; inherited minutes reflect network input.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='kpi-label' style='font-size:14px;'>Minutes Split — Network</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='card-note'>Controllable: <b>{fmt_int(cur_controllable)} min</b> | Total DEP: <b>{fmt_int(cur_total_dep)} min</b> | Split disabled at this grain (Category/Owner minutes are annual-only).</div>",
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        bars = top5.sort_values(focus_col, ascending=False)
        bar_colors = ["#4F46E5"] + ["#A5B4FC"] * max(len(bars) - 1, 0)
        fig_top5 = go.Figure(go.Bar(
            x=bars[focus_col],
            y=bars["Station"],
            orientation="h",
            marker=dict(color=bar_colors),
            text=[f"{v:,.0f} min | {p:.1f}%" for v, p in zip(bars[focus_col], bars["SharePct"])],
            textposition="outside",
            hovertemplate="%{y}<br>Minutes: %{x:.0f}<extra></extra>",
        ))
        style_plotly_card(fig_top5, title="Station Performance Ranking", height=340, margin=dict(l=10, r=40, t=52, b=12))
        fig_top5.update_yaxes(autorange="reversed")
        fig_top5.update_xaxes(showticklabels=False)
        show_plotly(fig_top5, key="p1_network__top5_stations_focus")
        st.markdown(f"<div class='muted-note'>Focus Metric: {focus_metric}</div>", unsafe_allow_html=True)

with right_col:
    with st.container(border=True):
        top2_label = " + ".join(top2_names) if top2_names else "N/A"
        st.markdown(f"<div class='conc-top2'>Top 2: <span class='conc-red'>{top2_label}</span></div>", unsafe_allow_html=True)
        conc_items = ["Total DEP Minutes", "Controllable Minutes", "Ground Ops Minutes"]
        conc_vals = [pct_top2_total, pct_top2_cont, pct_top2_ground]
        conc_vals_plot = [0.0 if pd.isna(v) else float(v) for v in conc_vals]
        conc_vals_text = [fmt_pct1_or_na(v) for v in conc_vals]
        fig_conc = go.Figure(go.Bar(
            x=conc_vals_plot,
            y=conc_items,
            orientation="h",
            marker=dict(color="#C7D2FE"),
            text=conc_vals_text,
            customdata=conc_vals_text,
            textposition="outside",
            textfont=dict(color="#E31E24", size=14),
            hovertemplate="%{y}<br>Top 2 share: %{customdata}<extra></extra>",
        ))
        style_plotly_card(fig_conc, title="Top 2 Stations — Share of Delays", height=290, margin=dict(l=10, r=48, t=52, b=10))
        fig_conc.update_xaxes(range=[0, 100], showticklabels=False)
        fig_conc.update_yaxes(autorange="reversed")
        show_plotly(fig_conc, key="p1_network__concentration_top2")
        st.markdown(
            (
                f"<div class='muted-note'>Top 2 = <span class='conc-red'>{fmt_pct1_or_na(pct_top2_total)}</span> of Total DEP Minutes<br>"
                f"Top 2 = <span class='conc-red'>{fmt_pct1_or_na(pct_top2_cont)}</span> of Controllable Minutes<br>"
                f"Top 2 = <span class='conc-red'>{fmt_pct1_or_na(pct_top2_ground)}</span> of Ground Ops Minutes</div>"
            ),
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        own_labels = ["Ground Ops", "Other Categories"]
        own_vals = [ground_ops_controllable, other_controllable_minutes]
        own_total = sum(own_vals)
        own_pct = [(v / own_total * 100.0) if own_total > 0 else 0.0 for v in own_vals]
        fig_own = go.Figure(go.Pie(
            labels=own_labels,
            values=own_vals,
            hole=0.62,
            sort=False,
            textinfo="percent",
            textposition="inside",
            insidetextorientation="horizontal",
            textfont=dict(color="#0B1220", size=12),
            marker=dict(colors=["#4F46E5", "#E2E8F0"]),
            hovertemplate="%{label}<br>%{value:,.0f} min<br>%{percent}<extra></extra>",
        ))
        fig_own.add_annotation(
            x=0.5,
            y=0.5,
            showarrow=False,
            text=f"Controllable<br><b>{own_total:,.0f} min</b>",
            font=dict(color="#0B1220", size=15, family="Inter, Arial, sans-serif"),
            align="center",
        )
        style_plotly_card(fig_own, title="Controllable Delays — Category Split", height=300, margin=dict(l=10, r=10, t=52, b=92))
        fig_own.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                x=0,
                y=-0.05,
                xanchor="left",
                yanchor="top",
                font=dict(color="#0B1220", size=12, family="Inter, Arial, sans-serif"),
                itemwidth=40,
            ),
            uniformtext_minsize=12,
            uniformtext_mode="hide",
        )
        show_plotly(fig_own, key="p1_network__controllable_ownership")
        st.markdown(
            f"<div class='card-note'>Ground Ops: {fmt_int(own_vals[0])} min ({own_pct[0]:.1f}%)<br>Other Categories: {fmt_int(own_vals[1])} min ({own_pct[1]:.1f}%)</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<span class='decision-pill'>Decision: Prioritize levers where Ground Ops share is highest.</span>",
            unsafe_allow_html=True,
        )

st.divider()
st.markdown(
    "<div class='footer-note'>All metrics reflect filtered operated departures only. Owner dimension is Delay Category.</div>",
    unsafe_allow_html=True,
)

with st.expander("Delay Code Analysis", expanded=False):
    if decision_pack_split_blocked:
        st.warning(
            "Decision Pack category/code split insights are disabled for this grain because Category/Owner minutes are annual-only or code splits are non-periodized."
        )
    else:
        codes_df = safe_filter_station(codes_raw, stations_sel)
        codes_df = filter_to_selected_period(codes_df, allow_full_fallback=False)
        code_col = detect_col(codes_df, ["DelayCode", "Code"], contains_any=["delaycode", "code"])
        codes_cat_col = detect_col(codes_df, ["DelayCategory", "Category"], contains_any=["delaycategory", "delay_category", "category"])
        desc_col = detect_col(codes_df, ["Description", "DelayCode_Desc"], contains_any=["desc"])

        if codes_df.empty:
            st.info("No periodized top-code data available for current filters.")
        else:
            codes_min_col = pick_minutes_col(codes_df)
            codes_df[codes_min_col] = to_num(codes_df[codes_min_col]).fillna(0.0)

            if code_col is None:
                st.warning("Top Delay Codes identity column not found.")
            else:
                top10_all = (
                    codes_df.groupby(code_col, as_index=False)[codes_min_col]
                    .sum()
                    .sort_values(codes_min_col, ascending=False)
                    .head(10)
                )
                if desc_col and desc_col in codes_df.columns:
                    desc_map = codes_df[[code_col, desc_col]].drop_duplicates(subset=[code_col])
                    top10_all = top10_all.merge(desc_map, on=code_col, how="left")
                else:
                    desc_map = None

                if codes_cat_col and codes_cat_col in codes_df.columns:
                    cc = codes_df[codes_cat_col].astype(str).str.strip().str.lower()
                    gmask = cc.eq("ground ops") | cc.eq("ground_ops")
                    if not gmask.any():
                        gmask = cc.str.contains("ground", na=False) & cc.str.contains("ops", na=False)
                    ground_df = codes_df[gmask].copy()
                else:
                    ground_df = codes_df.iloc[0:0].copy()

                top5_ground = (
                    ground_df.groupby(code_col, as_index=False)[codes_min_col]
                    .sum()
                    .sort_values(codes_min_col, ascending=False)
                    .head(5)
                )
                if desc_map is not None and not top5_ground.empty:
                    top5_ground = top5_ground.merge(desc_map, on=code_col, how="left")

                label_all = desc_col if (desc_col and desc_col in top10_all.columns) else code_col
                label_ground = desc_col if (desc_col and desc_col in top5_ground.columns) else code_col
                c_codes1, c_codes2 = st.columns(2)

                with c_codes1:
                    fig_codes_all = go.Figure(go.Bar(
                        x=top10_all[codes_min_col],
                        y=top10_all[label_all],
                        orientation="h",
                        marker=dict(color="#94A3B8"),
                        hovertemplate="%{y}<br>Minutes: %{x:.0f}<extra></extra>",
                    ))
                    style_plotly_card(fig_codes_all, title="Top Delay Codes (Top 10, All Categories)", height=420)
                    fig_codes_all.update_yaxes(autorange="reversed")
                    show_plotly(fig_codes_all, key="p1_network__top_delay_codes_top10_all")

                with c_codes2:
                    if top5_ground.empty:
                        st.info("No Ground Ops delay codes found for current filters.")
                    else:
                        fig_codes_ground = go.Figure(go.Bar(
                            x=top5_ground[codes_min_col],
                            y=top5_ground[label_ground],
                            orientation="h",
                            marker=dict(color="#64748B"),
                            hovertemplate="%{y}<br>Minutes: %{x:.0f}<extra></extra>",
                        ))
                        style_plotly_card(fig_codes_ground, title="Top Delay Codes (Top 5, Ground Ops)", height=420)
                        fig_codes_ground.update_yaxes(autorange="reversed")
                        show_plotly(fig_codes_ground, key="p1_network__top_delay_codes_top5_ground_ops")

