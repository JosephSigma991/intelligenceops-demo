from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from mode_a_filters import apply_filters
from utils import (
    normalize_name, pick_column, parse_period, ensure_list,
    read_csv_cached, load_csv, resolve_required_file,
    runtime_data_log, show_plotly, render_freshness_banner, render_filter_banner,
    render_gate_verdict_banner, fit_otp_model,
    otp_rag_class, delay_rag_class, render_kpi_card, style_plotly_card,
    render_situation_summary,
)
from kpi_config import (
    PLOTLY_CONFIG,
    WEEKLY_PERIOD_CANDIDATES, MONTHLY_PERIOD_CANDIDATES,
    STATION_CANDIDATES, FLIGHTS_CANDIDATES,
    TOTAL_MIN_CANDIDATES, AVG_DELAY_CANDIDATES, OTP_CANDIDATES,
    CONTROLLABLE_CANDIDATES, INHERITED_CANDIDATES, GROUND_CANDIDATES,
    OTP_MODEL_MIN_OBS, OTP_MODEL_MIN_R2_WARN, OTP_MODEL_FLIGHTS_FLOOR,
    ACCENT_PRIMARY, ACCENT_SECONDARY, COLOR_MUTED,
)


st.set_page_config(page_title="Scenarios / Levers", layout="wide")


def fmt_num(v: float | None, decimals: int = 0) -> str:
    if v is None or pd.isna(v):
        return "<missing>"
    if decimals <= 0:
        return f"{int(round(float(v))):,}"
    return f"{float(v):,.{decimals}f}"


def fmt_delta(v: float | None, decimals: int = 0) -> str | None:
    if v is None or pd.isna(v):
        return None
    if decimals <= 0:
        return f"{int(round(float(v))):,}"
    return f"{float(v):,.{decimals}f}"


st.title("Scenarios / Levers")
st.caption("PASS-only. Deterministic simulation on published KPIs (minutes → avg delay → estimated OTP).")

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
git_commit = ctx.get("git_commit")
required_files = ensure_list(ctx.get("required_files"))
insights_dir = Path(ctx.get("insights_dir", Path(os.getenv("FLIGHTOPS_INSIGHTS_DIR", r"C:\Users\IT\02_insights\insight_out"))))
artifacts_by_name = ctx.get("artifacts_by_name", {}) or {}

if not region or not mode:
    st.error("Run context is missing region/mode.")
    st.stop()

grain = str(filters.get("grain", "Monthly"))
periods_from_filters = [str(p) for p in filters.get("periods", []) if str(p).strip()]
stations_from_filters = [str(s) for s in filters.get("stations", []) if str(s).strip()]
period_label = periods_from_filters[0] if len(periods_from_filters) == 1 else (f"{periods_from_filters[0]}..{periods_from_filters[-1]} ({len(periods_from_filters)})" if periods_from_filters else "<auto>")
station_label = "NETWORK" if stations_from_filters == ["NETWORK"] else f"{len(stations_from_filters)} selected"
st.caption(f"Filters: Grain={grain} | Periods={period_label} | Stations={station_label}")

kpi_name = f"2025_DEP_Monthly_Station_KPIs__{region}.csv" if grain == "Monthly" else f"2025_DEP_Weekly_Station_KPIs__{region}.csv"
kpi_listed, kpi_path = resolve_required_file(required_files, kpi_name, insights_dir, artifacts_by_name)
if not required_files:
    st.warning("run_stamp.required_files is missing/empty; proceeding with on-disk file validation.")
elif not kpi_listed:
    st.warning(f"KPI file is not listed in run_stamp.required_files: `{kpi_name}`")

if not kpi_path.exists():
    st.error(f"KPI file not found for selected grain: `{kpi_path}`")
    st.stop()

df = load_csv(kpi_path)
if df is None:
    st.error(f"Could not load KPI file: `{kpi_path}`")
    st.stop()
runtime_data_log(f"Grain={grain} StationKpisFile={kpi_path} Rows={len(df)} Cols={len(df.columns)}")

period_col = pick_column(df, MONTHLY_PERIOD_CANDIDATES if grain == "Monthly" else WEEKLY_PERIOD_CANDIDATES)
station_col = pick_column(df, STATION_CANDIDATES, token_groups=[["station"], ["airport"]])
flights_col = pick_column(df, FLIGHTS_CANDIDATES, token_groups=[["flight", "oper"]])
total_col = pick_column(df, TOTAL_MIN_CANDIDATES, token_groups=[["dep", "delay"], ["min"]])
cont_col = pick_column(df, CONTROLLABLE_CANDIDATES, token_groups=[["controllable"], ["min"]])
inh_col = pick_column(df, INHERITED_CANDIDATES, token_groups=[["reactionary"], ["late", "arrival"]])
ground_col = pick_column(df, GROUND_CANDIDATES, token_groups=[["ground", "ops"]])
avg_col = pick_column(df, AVG_DELAY_CANDIDATES, token_groups=[["avg", "delay"], ["per", "flight"]])
otp_col = pick_column(df, OTP_CANDIDATES, token_groups=[["otp"], ["d15"]])

if period_col is None or station_col is None or flights_col is None or total_col is None:
    st.error(
        "Missing required columns after inference: "
        f"period={period_col}, station={station_col}, flights={flights_col}, total_minutes={total_col}"
    )
    st.stop()

work = df.copy()
parsed = work[period_col].map(lambda v: parse_period(v, grain))
work["_period"] = parsed.map(lambda t: t[0])
work["_sort"] = parsed.map(lambda t: t[1])
work = work.dropna(subset=["_period", "_sort"]).copy()

num_cols = [flights_col, total_col, cont_col, inh_col, ground_col, avg_col, otp_col]
for c in num_cols:
    if c is not None:
        work[c] = pd.to_numeric(work[c], errors="coerce")

agg_spec: dict[str, Any] = {
    flights_col: "sum",
    total_col: "sum",
}
if cont_col is not None:
    agg_spec[cont_col] = "sum"
if inh_col is not None:
    agg_spec[inh_col] = "sum"
if ground_col is not None:
    agg_spec[ground_col] = "sum"

grouped = work.groupby(["_period", "_sort", station_col], as_index=False).agg(agg_spec)

if avg_col is not None:
    tmp = work[["_period", "_sort", station_col, avg_col, flights_col]].copy()
    tmp = tmp[tmp[avg_col].notna()].copy()  # exclude NaN KPI rows from weighting
    tmp["__w"] = tmp[avg_col] * tmp[flights_col]
    avg_w = tmp.groupby(["_period", "_sort", station_col], as_index=False)[["__w", flights_col]].sum()
    avg_w["avg_reported_weighted"] = avg_w["__w"] / avg_w[flights_col].replace(0, pd.NA)
    grouped = grouped.merge(avg_w[["_period", "_sort", station_col, "avg_reported_weighted"]], on=["_period", "_sort", station_col], how="left")
else:
    grouped["avg_reported_weighted"] = pd.NA

if otp_col is not None:
    tmpo = work[["_period", "_sort", station_col, otp_col, flights_col]].copy()
    tmpo = tmpo[tmpo[otp_col].notna()].copy()  # exclude NaN KPI rows from weighting
    tmpo["__w"] = tmpo[otp_col] * tmpo[flights_col]
    otp_w = tmpo.groupby(["_period", "_sort", station_col], as_index=False)[["__w", flights_col]].sum()
    otp_w["otp_weighted"] = otp_w["__w"] / otp_w[flights_col].replace(0, pd.NA)
    grouped = grouped.merge(otp_w[["_period", "_sort", station_col, "otp_weighted"]], on=["_period", "_sort", station_col], how="left")
else:
    grouped["otp_weighted"] = pd.NA

grouped = grouped[grouped[flights_col].fillna(0) > 0].copy()
if grouped.empty:
    st.error("No active rows where Flights Operated > 0.")
    st.stop()

active_periods = (
    grouped.groupby(["_period", "_sort"], as_index=False)[flights_col]
    .sum()
    .sort_values("_sort")
)
period_options = active_periods["_period"].astype(str).tolist()
if not period_options:
    st.error("No active periods available in KPI data.")
    st.stop()

filtered_grouped = apply_filters(grouped, filters, station_col=station_col, period_col="_period")
filtered_grouped = filtered_grouped[filtered_grouped[flights_col].fillna(0) > 0].copy()
if filtered_grouped.empty:
    st.error("No rows remain after global filters.")
    st.stop()

F = float(pd.to_numeric(filtered_grouped[flights_col], errors="coerce").sum())
T = float(pd.to_numeric(filtered_grouped[total_col], errors="coerce").sum())
C = float(pd.to_numeric(filtered_grouped[cont_col], errors="coerce").sum()) if cont_col is not None else None
I = float(pd.to_numeric(filtered_grouped[inh_col], errors="coerce").sum()) if inh_col is not None else None
G = float(pd.to_numeric(filtered_grouped[ground_col], errors="coerce").sum()) if ground_col is not None else None

avg_reported = None
if "avg_reported_weighted" in filtered_grouped.columns and F > 0:
    w = pd.to_numeric(filtered_grouped[flights_col], errors="coerce")
    v = pd.to_numeric(filtered_grouped["avg_reported_weighted"], errors="coerce")
    mask = v.notna() & w.notna()
    denom = float(w[mask].sum())
    if denom > 0:
        avg_reported = float((v[mask] * w[mask]).sum() / denom)

# OTP baseline — weighted average of reported OTP for the selected filter scope
otp_baseline = None
if "otp_weighted" in filtered_grouped.columns and F > 0:
    w = pd.to_numeric(filtered_grouped[flights_col], errors="coerce")
    v = pd.to_numeric(filtered_grouped["otp_weighted"], errors="coerce")
    mask = v.notna() & w.notna()
    denom = float(w[mask].sum())
    if denom > 0:
        otp_baseline = float((v[mask] * w[mask]).sum() / denom)

avg_baseline = (T / F) if F > 0 else None

if avg_reported is not None and avg_baseline is not None:
    if abs(avg_reported - avg_baseline) > 0.05:
        st.warning(
            f"Avg delay mismatch detected: reported={avg_reported:.3f} vs computed={avg_baseline:.3f}. Using computed T/F for simulation."
        )

# ── OTP regression model ─────────────────────────────────────────────────────
otp_model = None
if otp_col is not None and avg_col is not None and len(filtered_grouped) >= OTP_MODEL_MIN_OBS:
    otp_model = fit_otp_model(
        avg_delay=filtered_grouped["avg_reported_weighted"],
        otp=filtered_grouped["otp_weighted"],
        flights=filtered_grouped[flights_col],
        min_obs=OTP_MODEL_MIN_OBS,
        flights_floor=OTP_MODEL_FLIGHTS_FLOOR,
    )
    if otp_model is not None:
        runtime_data_log(
            f"OTP model fitted: intercept={otp_model['intercept']:.2f}, "
            f"slope={otp_model['slope']:.3f}, R²={otp_model['r2']:.3f}, "
            f"n={otp_model['n_obs']}, delay_range=[{otp_model['delay_min']:.1f}, {otp_model['delay_max']:.1f}]"
        )

lever_options = ["Reduce Total DEP Delay Minutes"]
if C is not None:
    lever_options.append("Reduce Controllable Minutes")
if G is not None:
    lever_options.append("Reduce Ground Ops Minutes")
ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 2])
lever_sel = ctrl1.radio("Lever", options=lever_options, index=0)
pct_reduction = int(ctrl2.slider("Percent reduction", min_value=0, max_value=50, value=10, step=1))
abs_minutes = float(ctrl3.number_input("Absolute minutes reduction (optional, overrides % if > 0)", min_value=0.0, value=0.0, step=1.0))

if lever_sel == "Reduce Total DEP Delay Minutes":
    bucket = T
elif lever_sel == "Reduce Controllable Minutes":
    bucket = C if C is not None else 0.0
else:
    bucket = G if G is not None else 0.0

delta_raw = float(abs_minutes) if float(abs_minutes) > 0 else (float(bucket) * pct_reduction / 100.0)
delta_minutes = max(0.0, min(float(bucket), delta_raw))

if lever_sel in {"Reduce Controllable Minutes", "Reduce Ground Ops Minutes"}:
    st.info("Assumption: selected lever bucket is a subset of Total DEP Delay Minutes, so reducing this bucket reduces Total by the same minutes.")

T2 = max(0.0, T - delta_minutes)
avg2 = (T2 / F) if F > 0 else None
C2 = max(0.0, C - delta_minutes) if (C is not None and lever_sel == "Reduce Controllable Minutes") else C
G2 = max(0.0, G - delta_minutes) if (G is not None and lever_sel == "Reduce Ground Ops Minutes") else G
I2 = I

# ── OTP predictions ──────────────────────────────────────────────────────────
otp_predicted_baseline = None
otp_predicted_scenario = None
otp_delta = None
otp_extrapolation = False

if otp_model is not None and avg_baseline is not None and avg2 is not None:
    otp_predicted_baseline = max(0.0, min(100.0, otp_model["intercept"] + otp_model["slope"] * avg_baseline))
    otp_predicted_scenario = max(0.0, min(100.0, otp_model["intercept"] + otp_model["slope"] * avg2))
    otp_delta = otp_predicted_scenario - otp_predicted_baseline
    d_min = otp_model["delay_min"]
    d_max = otp_model["delay_max"]
    if avg2 < d_min or avg2 > d_max or avg_baseline < d_min or avg_baseline > d_max:
        otp_extrapolation = True

st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
st.subheader("Scenario Output")

_scenario_cards = [
    ("Flights Operated", fmt_num(F, 0), "kpi-rag-neutral"),
    ("Total DEP Delay Minutes", fmt_num(T2, 0), "kpi-rag-neutral"),
    ("Avg DEP Delay per Flight (min)", fmt_num(avg2, 2),
     delay_rag_class(avg2) if avg2 is not None else "kpi-rag-neutral"),
    ("Controllable Minutes", fmt_num(C2, 0) if C is not None else "&mdash;", "kpi-rag-neutral"),
    ("Inherited Minutes (Late Arrival)", fmt_num(I2, 0) if I is not None else "&mdash;", "kpi-rag-neutral"),
    ("Ground Ops Minutes", fmt_num(G2, 0) if G is not None else "&mdash;", "kpi-rag-neutral"),
]
if otp_model is not None and otp_predicted_scenario is not None:
    _scenario_cards.append((
        "DEP OTP D15 (%) [estimated]",
        f"{otp_predicted_scenario:.1f}%",
        otp_rag_class(otp_predicted_scenario),
    ))

kcols = st.columns(len(_scenario_cards))
for idx, (label, value, rag) in enumerate(_scenario_cards):
    with kcols[idx]:
        render_kpi_card(
            label=label,
            value_html=value,
            series=None,
            chart_key=f"p40__scenario_card_{idx}",
            rag_class=rag,
            delta_html="",
        )

# ── Situation summary ────────────────────────────────────────────────────────
if avg_baseline is not None and avg2 is not None:
    _reduction_pct = ((avg_baseline - avg2) / avg_baseline * 100) if avg_baseline > 0 else 0
    _otp_text = ""
    if otp_model is not None and otp_delta is not None:
        _otp_text = f" · OTP impact: <b>{otp_delta:+.1f} pp</b>"
    render_situation_summary(
        f"Scenario: avg delay <b>{avg_baseline:.1f} → {avg2:.1f} min</b> "
        f"(−{_reduction_pct:.1f}%){_otp_text}",
        severity="good" if avg2 < avg_baseline else "info",
    )

# OTP model confidence & status
if otp_model is not None:
    r2 = otp_model["r2"]
    n = otp_model["n_obs"]
    model_label = f"OTP estimated via linear regression (R² = {r2:.2f}, n = {n} obs)"
    has_warning = False
    if r2 < OTP_MODEL_MIN_R2_WARN:
        st.warning(f"⚠️ Low model confidence — {model_label}. Treat OTP estimate with caution.")
        has_warning = True
    if otp_extrapolation:
        st.warning(
            f"⚠️ Model extrapolation — avg delay outside training range "
            f"({otp_model['delay_min']:.1f}–{otp_model['delay_max']:.1f} min). {model_label}"
        )
        has_warning = True
    if not has_warning:
        st.caption(f"📈 {model_label}")
elif otp_col is None:
    st.caption("OTP column not found in data — OTP impact not modeled.")
else:
    st.caption("OTP impact not modeled — insufficient data for regression (need ≥ {} obs).".format(OTP_MODEL_MIN_OBS))

bc1, bc2 = st.columns(2)
with bc1:
    fig_total = go.Figure(go.Bar(x=["Before", "After"], y=[T, T2], marker=dict(color=[COLOR_MUTED, ACCENT_PRIMARY])))
    style_plotly_card(fig_total, title="Total DEP Delay Minutes", height=280, margin=dict(l=20, r=20, t=48, b=20))
    fig_total.update_layout(yaxis_title="Minutes")
    show_plotly(fig_total, key="p40__scenario_total_bar")
with bc2:
    fig_avg = go.Figure(go.Bar(x=["Before", "After"], y=[avg_baseline or 0.0, avg2 or 0.0], marker=dict(color=[COLOR_MUTED, ACCENT_SECONDARY])))
    style_plotly_card(fig_avg, title="Avg DEP Delay per Flight", height=280, margin=dict(l=20, r=20, t=48, b=20))
    fig_avg.update_layout(yaxis_title="Minutes / flight")
    show_plotly(fig_avg, key="p40__scenario_avg_bar")

st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
st.subheader("Required Reduction Solver")
target_default = max(0.0, (avg_baseline or 0.0) - 1.0)
target_avg = st.number_input("Target Avg DEP Delay per Flight (min)", min_value=0.0, value=float(target_default), step=0.1)
required_delta = max(0.0, T - (target_avg * F))
required_delta_clamped = min(required_delta, float(bucket))
required_pct_bucket = (100.0 * required_delta_clamped / float(bucket)) if float(bucket) > 0 else None

s1, s2, s3 = st.columns(3)
s1.metric("Required Delta Minutes", fmt_num(required_delta, 0), delta=None, delta_color="off")
s2.metric("Required Delta (clamped to lever)", fmt_num(required_delta_clamped, 0), delta=None, delta_color="off")
s3.metric("Required % of Lever Bucket", fmt_num(required_pct_bucket, 1) + "%" if required_pct_bucket is not None else "<missing>", delta=None, delta_color="off")

st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
st.subheader("Apply % Across Recent Periods (what-if preview)")
station_only_filters = dict(filters)
station_only_filters["periods"] = []
hist = apply_filters(grouped, station_only_filters, station_col=station_col, period_col=None).copy()
hist = hist.groupby(["_period", "_sort"], as_index=False).agg({flights_col: "sum", total_col: "sum"}).sort_values("_sort").tail(12)
if hist.empty:
    st.info("No recent period history available for selected global station filter.")
else:
    hist_total = pd.to_numeric(hist[total_col], errors="coerce").fillna(0.0)
    hist_flights = pd.to_numeric(hist[flights_col], errors="coerce").replace(0, pd.NA)
    hist_avg = hist_total / hist_flights
    sim_total = (hist_total * (1.0 - (pct_reduction / 100.0))).clip(lower=0.0)
    sim_avg = sim_total / hist_flights

    fig_preview = go.Figure()
    fig_preview.add_trace(go.Scatter(x=hist["_period"].astype(str).tolist(), y=hist_avg.tolist(), mode="lines+markers", name="Baseline Avg"))
    fig_preview.add_trace(go.Scatter(x=hist["_period"].astype(str).tolist(), y=sim_avg.tolist(), mode="lines+markers", name=f"What-if Avg ({pct_reduction}% reduction)"))
    style_plotly_card(fig_preview, height=320, margin=dict(l=20, r=20, t=20, b=20))
    fig_preview.update_layout(
        yaxis_title="Avg DEP Delay per Flight (min)",
        xaxis_title="Period",
        showlegend=True,
    )
    show_plotly(fig_preview, key="p40__whatif_preview")

with st.expander("📋 Evidence & Source Details", expanded=False):
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Build ID", str(stamp) if stamp else "<missing>")
    e2.metric("Region", str(region))
    e3.metric("Mode", str(mode))
    e4.metric("Version", str(git_commit)[:8] if git_commit else "<missing>")

    evidence_row = {
        "Period": period_label,
        "Station": station_label,
        "Flights Operated": F,
        "Total DEP Delay Minutes": T,
        "Controllable Minutes": C,
        "Inherited Minutes (Late Arrival)": I,
        "Ground Ops Minutes": G,
        "Avg DEP Delay per Flight (computed)": avg_baseline,
        "Avg DEP Delay per Flight (reported weighted)": avg_reported,
        "DEP OTP D15 (%) [reported]": otp_baseline,
        "DEP OTP D15 (%) [model baseline]": otp_predicted_baseline,
        "DEP OTP D15 (%) [model scenario]": otp_predicted_scenario,
        "OTP Δ (pp)": otp_delta,
    }
    if otp_model is not None:
        evidence_row["Model R²"] = otp_model["r2"]
        evidence_row["Model n_obs"] = otp_model["n_obs"]
        evidence_row["Model delay range"] = f"{otp_model['delay_min']:.1f}–{otp_model['delay_max']:.1f} min"
    selected_row_df = pd.DataFrame([evidence_row])
    st.dataframe(selected_row_df, width="stretch", hide_index=True)


