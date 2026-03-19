from __future__ import annotations

import os
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
    weighted_avg, runtime_data_log, show_plotly, render_freshness_banner, render_filter_banner,
    render_gate_verdict_banner,
    otp_rag_class, delay_rag_class, render_kpi_card, _compute_delta_html,
    style_plotly_card, render_situation_summary,
)
from kpi_config import (
    PLOTLY_CONFIG, KPI_CANDIDATES,
    WEEKLY_PERIOD_CANDIDATES, MONTHLY_PERIOD_CANDIDATES,
    OTP_TARGET, COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TARGET_LINE,
    COLOR_GREEN, COLOR_AMBER, COLOR_RED,
)


st.set_page_config(page_title="Network Overview", layout="wide")


def fmt_value(label: str, value: float | None) -> str:
    if value is None or pd.isna(value):
        return "<missing>"
    if label in {"Avg DEP Delay per Flight (min)", "DEP OTP D15 (%)"}:
        return f"{float(value):,.1f}"
    return f"{int(round(float(value))):,}"


def fmt_delta(label: str, cur: float | None, prev: float | None) -> str | None:
    if cur is None or prev is None or pd.isna(cur) or pd.isna(prev):
        return None
    d = float(cur) - float(prev)
    if label == "DEP OTP D15 (%)":
        return f"{d:+.1f} pp"
    if label == "Avg DEP Delay per Flight (min)":
        return f"{d:+.1f}"
    return f"{d:+,.0f}"


def delta_mode(label: str) -> str:
    l = str(label or "").lower()
    if "otp" in l:
        return "normal"
    if "flights" in l:
        return "normal"
    if any(tok in l for tok in ["avg", "delay", "minutes", "min"]):
        return "inverse"
    return "off"



def build_network_frame(df: pd.DataFrame, grain: str, filters: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, str | None], list[str]]:
    warnings: list[str] = []
    period_candidates = WEEKLY_PERIOD_CANDIDATES if grain == "Weekly" else MONTHLY_PERIOD_CANDIDATES
    period_col = pick_column(df, period_candidates)
    station_col = pick_column(df, ["Station", "DepartureAirport", "DepAirport", "Airport"], token_groups=[["station"], ["airport"]])
    if period_col is None:
        return pd.DataFrame(), {}, ["Missing period column for selected grain."]

    token_groups = {
        "Flights Operated": [["flight", "oper"]],
        "Avg DEP Delay per Flight (min)": [["avg", "delay"], ["per", "flight"]],
        "DEP OTP D15 (%)": [["otp"], ["d15"]],
        "Total DEP Delay Minutes": [["dep", "delay"], ["min"]],
        "Controllable Minutes": [["controllable"], ["min"]],
        "Inherited Minutes (Late Arrival)": [["reactionary"], ["late", "arrival"]],
        "Ground Ops Minutes": [["ground", "ops"]],
    }

    col_map: dict[str, str | None] = {}
    for label, cands in KPI_CANDIDATES.items():
        col_map[label] = pick_column(df, cands, token_groups=[token_groups[label]])

    if col_map["Flights Operated"] is None:
        warnings.append("Flights Operated column missing. Active period filtering and weighted metrics are unavailable.")

    d = df.copy()
    parsed = d[period_col].map(lambda v: parse_period(v, grain))
    d["_period"] = parsed.map(lambda t: t[0])
    d["_sort"] = parsed.map(lambda t: t[1])
    d = d.dropna(subset=["_period", "_sort"])
    if d.empty:
        return pd.DataFrame(), col_map, warnings + ["No valid periods parsed from source data."]

    for label, c in col_map.items():
        if c is None:
            warnings.append(f"{label} column missing.")
            continue
        d[c] = pd.to_numeric(d[c], errors="coerce")

    if filters:
        d = apply_filters(d, filters, station_col=station_col, period_col="_period")
        if d.empty:
            warnings.append("No rows remain after global period/station filters.")
            return pd.DataFrame(), col_map, warnings

    rows: list[dict[str, Any]] = []
    for (period, sortv), g in d.groupby(["_period", "_sort"], dropna=True):
        row: dict[str, Any] = {"Period": period, "_sort": sortv}
        flights_col = col_map["Flights Operated"]
        flights_val: float | None = None
        if flights_col is not None:
            flights_val = float(g[flights_col].fillna(0).sum())
        row["Flights Operated"] = flights_val

        otp_col = col_map["DEP OTP D15 (%)"]
        if otp_col is not None and flights_col is not None:
            row["DEP OTP D15 (%)"] = weighted_avg(g[otp_col], g[flights_col])
        elif otp_col is not None:
            row["DEP OTP D15 (%)"] = float(pd.to_numeric(g[otp_col], errors="coerce").mean())
        else:
            row["DEP OTP D15 (%)"] = None

        avg_col = col_map["Avg DEP Delay per Flight (min)"]
        if avg_col is not None and flights_col is not None:
            row["Avg DEP Delay per Flight (min)"] = weighted_avg(g[avg_col], g[flights_col])
        elif avg_col is not None:
            row["Avg DEP Delay per Flight (min)"] = float(pd.to_numeric(g[avg_col], errors="coerce").mean())
        else:
            row["Avg DEP Delay per Flight (min)"] = None

        for label in [
            "Total DEP Delay Minutes",
            "Controllable Minutes",
            "Inherited Minutes (Late Arrival)",
            "Ground Ops Minutes",
        ]:
            c = col_map[label]
            if c is None:
                row[label] = None
            else:
                row[label] = float(g[c].fillna(0).sum())

        rows.append(row)

    net = pd.DataFrame(rows).sort_values("_sort").reset_index(drop=True)
    flights = pd.to_numeric(net["Flights Operated"], errors="coerce")
    net = net[flights > 0].copy()
    if net.empty:
        warnings.append("No active periods found where Flights_Operated > 0.")
    return net, col_map, warnings


st.title("Network Overview")
st.caption("PASS-only artifacts view. Reads run_stamp and published CSVs only.")

ctx = st.session_state.get("mode_a_ctx")
if not isinstance(ctx, dict):
    st.error("Select a run in the sidebar")
    st.stop()
render_freshness_banner(ctx)
render_gate_verdict_banner(ctx)
filters = st.session_state.get("mode_a_filters")
if not isinstance(filters, dict):
    st.error("Global filters are missing. Select filters in the sidebar.")
    st.stop()
render_filter_banner(ctx, filters)

run_stamp = ctx.get("run_stamp") if isinstance(ctx.get("run_stamp"), dict) else {}
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

with st.expander("📋 Data Source Details", expanded=False):
    prov_cols = st.columns(5)
    prov_cols[0].metric("Region/Mode", f"{region}/{mode}")
    prov_cols[1].metric("Build ID", str(stamp) if stamp else "<missing>")
    prov_cols[2].metric("Branch", str(git_branch) if git_branch else "<missing>")
    prov_cols[3].metric("Version", str(git_commit)[:8] if git_commit else "<missing>")
    qa_state = "exists" if (qa_path is not None and qa_path.exists()) else "missing"
    prov_cols[4].metric("Quality Check", qa_state)

grain = str(filters.get("grain", "Monthly"))
periods_from_filters = [str(p) for p in filters.get("periods", []) if str(p).strip()]
stations_from_filters = [str(s) for s in filters.get("stations", []) if str(s).strip()]
station_label = "NETWORK" if stations_from_filters == ["NETWORK"] else f"{len(stations_from_filters)} selected"
period_label = periods_from_filters[0] if len(periods_from_filters) == 1 else (f"{periods_from_filters[0]}..{periods_from_filters[-1]} ({len(periods_from_filters)})" if periods_from_filters else "<auto>")
st.caption(f"Filters: Grain={grain} | Periods={period_label} | Stations={station_label}")

default_monthly = f"2025_DEP_Monthly_Station_KPIs__{region}.csv"
default_weekly = f"2025_DEP_Weekly_Station_KPIs__{region}.csv"
_, monthly_path = resolve_required_file(required_files, default_monthly, insights_dir, artifacts_by_name)
_, weekly_path = resolve_required_file(required_files, default_weekly, insights_dir, artifacts_by_name)
source_path = monthly_path if grain == "Monthly" else weekly_path

df = load_csv(source_path)
if df is None:
    st.error(f"Could not read source CSV for {grain}: `{source_path}`")
    st.stop()

network, col_map, warn_msgs = build_network_frame(df, grain, filters=filters)
for msg in warn_msgs:
    st.warning(msg)

if network.empty:
    st.info("No active network periods available for selected grain.")
    st.stop()

sel = network.sort_values("_sort").reset_index(drop=True)

if sel.empty:
    st.info("No rows in selected period range.")
    st.stop()

# ── Situation summary ────────────────────────────────────────────────────────
_latest = sel.iloc[-1] if not sel.empty else {}
_otp_val = pd.to_numeric(pd.Series([_latest.get("DEP OTP D15 (%)")]), errors="coerce").iloc[0]
_avg_val = pd.to_numeric(pd.Series([_latest.get("Avg DEP Delay per Flight (min)")]), errors="coerce").iloc[0]
_period_name = str(_latest.get("Period", ""))
_parts: list[str] = []
if pd.notna(_otp_val):
    _sev = "good" if _otp_val >= 85 else ("warn" if _otp_val >= 80 else "bad")
    _parts.append(f"OTP at <b>{_otp_val:.1f}%</b>")
else:
    _sev = "info"
if pd.notna(_avg_val):
    _parts.append(f"avg delay <b>{_avg_val:.1f} min/flight</b>")
if _period_name:
    _parts.append(f"for <b>{_period_name}</b>")
if _parts:
    render_situation_summary(" · ".join(_parts), severity=_sev if pd.notna(_otp_val) else "info")

kpi_order = [
    "Flights Operated",
    "Avg DEP Delay per Flight (min)",
    "DEP OTP D15 (%)",
    "Total DEP Delay Minutes",
    "Controllable Minutes",
    "Inherited Minutes (Late Arrival)",
    "Ground Ops Minutes",
]

latest_row = sel.iloc[-1]

def _rag_for_p10(label: str, value) -> str:
    if "OTP" in label:
        return otp_rag_class(value)
    if "Avg" in label and "Delay" in label:
        return delay_rag_class(value)
    return "kpi-rag-neutral"

def _series_for_p10(label: str) -> list | None:
    if label not in sel.columns:
        return None
    return pd.to_numeric(sel[label], errors="coerce").tolist()

def _metric_key_p10(label: str) -> str:
    ll = label.lower()
    if "otp" in ll:
        return "OTP"
    if "flight" in ll:
        return "Flights"
    return "Delay"

def _unit_for_p10(label: str) -> str:
    if "OTP" in label or "(%)" in label:
        return " pp"
    return ""

row1 = st.columns(4)
row2 = st.columns(3)
card_cols = row1 + row2

for i, label in enumerate(kpi_order):
    with card_cols[i]:
        cur = latest_row.get(label)
        series = _series_for_p10(label)
        render_kpi_card(
            label=label,
            value_html=fmt_value(label, cur),
            series=series,
            chart_key=f"p10_overview__kpi_{i}_{sel.iloc[-1].get('Period', 'x')}",
            rag_class=_rag_for_p10(label, cur),
            delta_html=_compute_delta_html(series, _metric_key_p10(label), _unit_for_p10(label)),
        )

st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
st.subheader("Trends")
t1, t2 = st.columns(2)

with t1:
    st.caption("DEP OTP D15 (%)")
    otp_series = pd.to_numeric(sel["DEP OTP D15 (%)"], errors="coerce")
    if otp_series.notna().any():
        marker_colors = [
            COLOR_GREEN if (v >= OTP_TARGET) else (COLOR_AMBER if (v >= 80.0) else COLOR_RED)
            for v in otp_series.fillna(0).tolist()
        ]
        fig_otp = go.Figure(
            go.Scatter(
                x=sel["Period"].astype(str).tolist(),
                y=otp_series.tolist(),
                mode="lines+markers",
                name="DEP OTP D15 (%)",
                marker=dict(color=marker_colors, size=8),
                line=dict(color=COLOR_PRIMARY),
            )
        )
        fig_otp.add_hline(
            y=OTP_TARGET,
            line_dash="dash",
            line_color=COLOR_TARGET_LINE,
            annotation_text=f"Target {OTP_TARGET:.0f}%",
            annotation_position="top right",
        )
        style_plotly_card(fig_otp, height=300, margin=dict(l=20, r=20, t=10, b=20))
        show_plotly(fig_otp, key="p10__otp_trend")
    else:
        st.info("<missing>")

with t2:
    st.caption("Avg DEP Delay per Flight (min)")
    avg_series = pd.to_numeric(sel["Avg DEP Delay per Flight (min)"], errors="coerce")
    if avg_series.notna().any():
        fig_avg = go.Figure(
            go.Scatter(
                x=sel["Period"].astype(str).tolist(),
                y=avg_series.tolist(),
                mode="lines+markers",
                name="Avg DEP Delay per Flight (min)",
                line=dict(color=COLOR_SECONDARY),
                marker=dict(color=COLOR_SECONDARY, size=8),
            )
        )
        valid = avg_series.dropna()
        if len(valid) >= 2:
            last_val = valid.iloc[-1]
            prev_val = valid.iloc[-2]
            last_period = sel["Period"].iloc[valid.index[-1]]
            mom_delta = last_val - prev_val
            fig_avg.add_annotation(
                x=str(last_period),
                y=last_val,
                text=f"MoM {mom_delta:+.1f}",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-30,
                font=dict(size=11),
            )
        style_plotly_card(fig_avg, height=300, margin=dict(l=20, r=20, t=10, b=20))
        show_plotly(fig_avg, key="p10__avg_delay_trend")
    else:
        st.info("<missing>")


