from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from mode_a_filters import apply_filters, weighted_aggregate
from utils import (
    normalize_name, pick_column, parse_period, ensure_list,
    read_csv_cached, load_csv, resolve_required_file,
    runtime_data_log, show_plotly, render_freshness_banner, render_filter_banner,
    render_gate_verdict_banner, style_plotly_card,
)
from kpi_config import (
    PLOTLY_CONFIG, KPI_CANDIDATES,
    WEEKLY_PERIOD_CANDIDATES, MONTHLY_PERIOD_CANDIDATES,
    STATION_CANDIDATES, FLIGHTS_CANDIDATES,
    OTP_TARGET, COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TARGET_LINE,
)


st.set_page_config(page_title="Station Ranking", layout="wide")


def metric_direction(label: str) -> str:
    l = label.lower()
    if "otp" in l:
        return "normal"
    if "flights" in l:
        return "normal"
    if any(tok in l for tok in ["avg", "delay", "minutes", "min"]):
        return "inverse"
    return "off"


st.title("Station Ranking")
st.caption("PASS-only. Reads published artifacts and run logs only.")

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
required_files = ensure_list(ctx.get("required_files"))
insights_dir = Path(ctx.get("insights_dir", Path(os.getenv("FLIGHTOPS_INSIGHTS_DIR", r"C:\Users\IT\02_insights\insight_out"))))
artifacts_by_name = ctx.get("artifacts_by_name", {}) or {}

if not region or not mode:
    st.error("Run context is missing region/mode.")
    st.stop()

monthly_name = f"2025_DEP_Monthly_Station_KPIs__{region}.csv"
weekly_name = f"2025_DEP_Weekly_Station_KPIs__{region}.csv"
monthly_in_req, monthly_path = resolve_required_file(required_files, monthly_name, insights_dir, artifacts_by_name)
weekly_in_req, weekly_path = resolve_required_file(required_files, weekly_name, insights_dir, artifacts_by_name)

if not monthly_in_req:
    st.error(f"Required file missing in run_stamp.required_files: `{monthly_name}`")
if not weekly_in_req:
    st.error(f"Required file missing in run_stamp.required_files: `{weekly_name}`")
if (not monthly_in_req) and (not weekly_in_req):
    st.stop()

with st.expander("📋 Data Source Details", expanded=False):
    hdr = st.columns(4)
    hdr[0].metric("Region/Mode", f"{region}/{mode}")
    hdr[1].metric("Build ID", str(stamp) if stamp else "<missing>")
    hdr[2].metric("Branch", str(git_branch) if git_branch else "<missing>")
    hdr[3].metric("Version", str(git_commit)[:8] if git_commit else "<missing>")
grain = str(filters.get("grain", "Monthly"))
periods_from_filters = [str(p) for p in filters.get("periods", []) if str(p).strip()]
stations_from_filters = [str(s) for s in filters.get("stations", []) if str(s).strip()]
period_label = periods_from_filters[0] if len(periods_from_filters) == 1 else (f"{periods_from_filters[0]}..{periods_from_filters[-1]} ({len(periods_from_filters)})" if periods_from_filters else "<auto>")
station_label = "NETWORK" if stations_from_filters == ["NETWORK"] else f"{len(stations_from_filters)} selected"
st.caption(f"Filters: Grain={grain} | Periods={period_label} | Stations={station_label}")

kpi_path = monthly_path if grain == "Monthly" else weekly_path
if not kpi_path.exists():
    st.error(f"Selected grain artifact not found: `{kpi_path}`")
    st.stop()

df = load_csv(kpi_path)
if df is None:
    st.error(f"Could not read station KPI CSV: `{kpi_path}`")
    st.stop()
runtime_data_log(f"Grain={grain} StationKpisFile={kpi_path} Rows={len(df)} Cols={len(df.columns)}")

period_col = pick_column(
    df,
    MONTHLY_PERIOD_CANDIDATES if grain == "Monthly" else WEEKLY_PERIOD_CANDIDATES,
)
station_col = pick_column(df, STATION_CANDIDATES, token_groups=[["station"], ["airport"]])
flights_col = pick_column(df, FLIGHTS_CANDIDATES, token_groups=[["flight", "oper"]])

if period_col is None:
    st.error("Missing period column in station KPI artifact for selected grain.")
    st.stop()
if station_col is None:
    st.error("Missing station column in station KPI artifact.")
    st.stop()
if flights_col is None:
    st.error("Missing Flights Operated column in station KPI artifact.")
    st.stop()

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
for label, candidates in KPI_CANDIDATES.items():
    col_map[label] = pick_column(df, candidates, token_groups=[token_groups[label]])

for label, c in col_map.items():
    if c is None:
        st.warning(f"{label} column is missing and will be skipped.")

work = df.copy()
parsed = work[period_col].map(lambda v: parse_period(v, grain))
work["_period"] = parsed.map(lambda t: t[0])
work["_sort"] = parsed.map(lambda t: t[1])
work = work.dropna(subset=["_period", "_sort"]).copy()

work[flights_col] = pd.to_numeric(work[flights_col], errors="coerce")
active = (
    work.groupby(["_period", "_sort"], as_index=False)[flights_col]
    .sum()
    .query(f"{flights_col} > 0")
    .sort_values("_sort")
)
period_options = active["_period"].astype(str).tolist()
if not period_options:
    st.error("No active periods found where Flights Operated > 0.")
    st.stop()

work = apply_filters(work, filters, station_col=station_col, period_col="_period")
work = work[pd.to_numeric(work[flights_col], errors="coerce").fillna(0) > 0].copy()
if work.empty:
    st.info("No station rows remain after global filters.")
    st.stop()

for label, c in col_map.items():
    if c is not None:
        work[c] = pd.to_numeric(work[c], errors="coerce")

stations = sorted(work[station_col].astype(str).dropna().unique().tolist())
rows: list[dict[str, Any]] = []
for station_name, g in work.groupby(station_col, dropna=True):
    agg = weighted_aggregate(g, flights_col)
    row = {"Station (DepartureAirport)": str(station_name)}
    for label in [
        "Flights Operated",
        "Avg DEP Delay per Flight (min)",
        "DEP OTP D15 (%)",
        "Total DEP Delay Minutes",
        "Controllable Minutes",
        "Inherited Minutes (Late Arrival)",
        "Ground Ops Minutes",
    ]:
        c = col_map.get(label)
        row[label] = agg.get(c) if c else pd.NA
    rows.append(row)
table_raw = pd.DataFrame(rows)
if table_raw.empty:
    st.info("No ranking rows after aggregation.")
    st.stop()

present_kpis: list[str] = []
for label in [
    "Flights Operated",
    "Avg DEP Delay per Flight (min)",
    "DEP OTP D15 (%)",
    "Total DEP Delay Minutes",
    "Controllable Minutes",
    "Inherited Minutes (Late Arrival)",
    "Ground Ops Minutes",
]:
    if label not in table_raw.columns:
        continue
    present_kpis.append(label)

if table_raw.empty or not present_kpis:
    st.error("No ranking KPIs available from selected artifact columns.")
    st.stop()

default_sort_metric = "Avg DEP Delay per Flight (min)" if "Avg DEP Delay per Flight (min)" in present_kpis else present_kpis[0]
ctrl1, ctrl2, ctrl3 = st.columns(3)
sort_metric = ctrl1.selectbox("Sort Metric", options=present_kpis, index=present_kpis.index(default_sort_metric))
sort_dir = ctrl2.selectbox("Sort Direction", options=["Descending", "Ascending"], index=0)
top_bottom = ctrl3.selectbox("Slice", options=["Top 10 (best)", "Bottom 10 (worst)"], index=0)

ascending = sort_dir == "Ascending"
ranked = table_raw.sort_values(sort_metric, ascending=ascending, na_position="last")
if top_bottom.startswith("Top"):
    ranked = ranked.head(10)
else:
    ranked = ranked.tail(10)

fmt_df = ranked.copy()
for c in fmt_df.columns:
    if c == "Station (DepartureAirport)":
        continue
    if c == "Flights Operated" or c.endswith("Minutes"):
        fmt_df[c] = pd.to_numeric(fmt_df[c], errors="coerce").map(lambda v: "<missing>" if pd.isna(v) else f"{int(round(float(v))):,}")
    elif "(%)" in c or "(min)" in c:
        fmt_df[c] = pd.to_numeric(fmt_df[c], errors="coerce").map(lambda v: "<missing>" if pd.isna(v) else f"{float(v):,.1f}")
    else:
        fmt_df[c] = pd.to_numeric(fmt_df[c], errors="coerce").map(lambda v: "<missing>" if pd.isna(v) else f"{float(v):,.2f}")

st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
st.subheader("Station Ranking")
st.caption(f"Grain={grain} | Periods={period_label} | Rows={len(ranked)}")
st.dataframe(fmt_df, width="stretch", hide_index=True)

st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
st.subheader("KPI Matrix")
heat_metrics = [c for c in present_kpis if c in table_raw.columns]
matrix_src = table_raw[["Station (DepartureAirport)"] + heat_metrics].copy()
matrix_src = matrix_src.dropna(subset=["Station (DepartureAirport)"])
if matrix_src.empty or not heat_metrics:
    st.info("KPI matrix not available for current selection.")
else:
    score_df = pd.DataFrame(index=matrix_src.index)
    score_df["Station"] = matrix_src["Station (DepartureAirport)"].astype(str)
    for k in heat_metrics:
        vals = pd.to_numeric(matrix_src[k], errors="coerce")
        vmin = vals.min(skipna=True)
        vmax = vals.max(skipna=True)
        if pd.isna(vmin) or pd.isna(vmax):
            norm = pd.Series([0.5] * len(vals), index=vals.index, dtype="float64")
        elif float(vmax) == float(vmin):
            norm = pd.Series([0.5] * len(vals), index=vals.index, dtype="float64")
        else:
            norm = (vals - vmin) / (vmax - vmin)

        direction = metric_direction(k)
        if direction == "inverse":
            score = 1.0 - norm
        else:
            score = norm
        score_df[k] = score.fillna(0.5)

    heat_y = score_df["Station"].tolist()
    heat_x = heat_metrics
    heat_z = score_df[heat_metrics].to_numpy()
    fig_heat = go.Figure(
        data=go.Heatmap(
            x=heat_x,
            y=heat_y,
            z=heat_z,
            zmin=0,
            zmax=1,
            colorscale="RdYlGn",
            colorbar=dict(title="Score"),
        )
    )
    style_plotly_card(fig_heat, height=max(280, 28 * len(heat_y) + 80), margin=dict(l=10, r=10, t=20, b=10))
    show_plotly(fig_heat, key="p20__kpi_matrix_heatmap")

st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
st.subheader("Trend Preview")
selected_filter_stations = [s for s in stations_from_filters if s.upper() != "NETWORK"]
if len(selected_filter_stations) != 1:
    st.info("Set Station Mode=Single in global filters to view station trend preview.")
else:
    station_choice = selected_filter_stations[0]
    hist = work[work[station_col].astype(str) == station_choice].copy()
    hist = hist[pd.to_numeric(hist[flights_col], errors="coerce").fillna(0) > 0].copy()
    hist = hist.sort_values("_sort")
    if hist.empty:
        st.info("No active periods available for selected station.")
    else:
        tail_n = 12
        hist = hist.tail(tail_n)
        otp_col = col_map.get("DEP OTP D15 (%)")
        avg_col = col_map.get("Avg DEP Delay per Flight (min)")
        if otp_col is None and avg_col is None:
            st.info("Trend preview not available because OTP and Avg Delay columns are missing.")
        else:
            x = hist["_period"].astype(str).tolist()
            fig_trend = go.Figure()
            if otp_col is not None:
                y_otp = pd.to_numeric(hist[otp_col], errors="coerce")
                fig_trend.add_trace(
                    go.Scatter(
                        x=x,
                        y=y_otp.tolist(),
                        mode="lines+markers",
                        name="DEP OTP D15 (%)",
                        yaxis="y1",
                        line=dict(color=COLOR_PRIMARY),
                        marker=dict(color=COLOR_PRIMARY, size=8),
                    )
                )
                fig_trend.add_hline(
                    y=OTP_TARGET,
                    line_dash="dash",
                    line_color=COLOR_TARGET_LINE,
                    annotation_text=f"Target {OTP_TARGET:.0f}%",
                    annotation_position="top right",
                )
            if avg_col is not None:
                y_avg = pd.to_numeric(hist[avg_col], errors="coerce")
                fig_trend.add_trace(
                    go.Scatter(
                        x=x,
                        y=y_avg.tolist(),
                        mode="lines+markers",
                        name="Avg DEP Delay per Flight (min)",
                        yaxis="y2",
                        line=dict(color=COLOR_SECONDARY),
                        marker=dict(color=COLOR_SECONDARY, size=8),
                    )
                )
            style_plotly_card(fig_trend, height=320, margin=dict(l=20, r=20, t=20, b=20))
            fig_trend.update_layout(
                xaxis=dict(title="Period"),
                yaxis=dict(title="DEP OTP D15 (%)"),
                yaxis2=dict(title="Avg DEP Delay per Flight (min)", overlaying="y", side="right"),
                showlegend=True,
                legend=dict(orientation="h"),
            )
            show_plotly(fig_trend, key="p20__station_trend_preview")


