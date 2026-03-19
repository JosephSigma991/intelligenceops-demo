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
    render_gate_verdict_banner, style_plotly_card, render_situation_summary,
)
from kpi_config import (
    PLOTLY_CONFIG,
    WEEKLY_PERIOD_CANDIDATES, MONTHLY_PERIOD_CANDIDATES,
    STATION_CANDIDATES, MINUTES_CANDIDATES, CATEGORY_CANDIDATES,
    COLOR_PRIMARY, COLOR_SECONDARY,
)


st.set_page_config(page_title="Drivers / RCA", layout="wide")


def normalize_category(v: Any) -> str:
    return normalize_name(v).replace("_", "")


st.title("Drivers / RCA")
st.caption("PASS-only. Accountability (DelayCategory) -> drilldown to codes. No SQL.")

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
required_files = ensure_list(ctx.get("required_files"))
insights_dir = Path(ctx.get("insights_dir", Path(os.getenv("FLIGHTOPS_INSIGHTS_DIR", r"C:\Users\IT\02_insights\insight_out"))))
artifacts_by_name = ctx.get("artifacts_by_name", {}) or {}

if not region or not mode:
    st.error("Run context is missing region/mode.")
    st.stop()

with st.expander("📋 Data Source Details", expanded=False):
    st.caption(f"Region: {region}/{mode} · Build ID: {stamp}")
grain = str(filters.get("grain", "Monthly"))
periods_from_filters = [str(p) for p in filters.get("periods", []) if str(p).strip()]
stations_from_filters = [str(s) for s in filters.get("stations", []) if str(s).strip()]
period_label = periods_from_filters[0] if len(periods_from_filters) == 1 else (f"{periods_from_filters[0]}..{periods_from_filters[-1]} ({len(periods_from_filters)})" if periods_from_filters else "<auto>")
station_label = "NETWORK" if stations_from_filters == ["NETWORK"] else f"{len(stations_from_filters)} selected"
st.caption(f"Filters: Grain={grain} | Periods={period_label} | Stations={station_label}")

cat_monthly_name = f"2025_DEP_DelayCategory_Minutes__{region}_NORM__MONTHLY.csv"
cat_weekly_name = f"2025_DEP_DelayCategory_Minutes__{region}_NORM__WEEKLY.csv"
top_codes_name = f"2025_DEP_TopDelayCodes__{region}_NORM.csv"
owner_name = f"2025_DEP_Owner_Minutes__{region}_NORM.csv"

cat_name = cat_monthly_name if grain == "Monthly" else cat_weekly_name
cat_ok, cat_path = resolve_required_file(required_files, cat_name, insights_dir, artifacts_by_name)
top_ok, top_path = resolve_required_file(required_files, top_codes_name, insights_dir, artifacts_by_name)
owner_ok, owner_path = resolve_required_file(required_files, owner_name, insights_dir, artifacts_by_name)

missing_contract: list[str] = []
if not top_ok:
    missing_contract.append(top_codes_name)
if not owner_ok:
    missing_contract.append(owner_name)
if missing_contract:
    st.error("Missing required artifact name(s) in run_stamp.required_files:")
    st.code("\n".join(missing_contract), language="text")
    st.stop()

missing_paths = [str(p) for p in [top_path, owner_path] if not p.exists()]
if missing_paths:
    st.error("Required artifact path(s) do not exist:")
    st.code("\n".join(missing_paths), language="text")
    st.stop()

cat_available = cat_ok and cat_path.exists()
if not cat_available:
    st.warning(
        f"Periodized delay category file not found for grain={grain}. (Expected: `{cat_name}`)\n\n"
        "Category Pareto and trend sections cannot render without this file. "
        "Owner cross-check and top delay codes (annual scope) are shown below."
    )

cat_df = load_csv(cat_path) if cat_available else None
top_df = load_csv(top_path)
owner_df = load_csv(owner_path)
if top_df is None or owner_df is None:
    st.error("Failed to load one or more required artifacts.")
    st.stop()

if cat_df is not None:
    runtime_data_log(f"Grain={grain} DelayCategoryFile={cat_path} Rows={len(cat_df)} Cols={len(cat_df.columns)}")
runtime_data_log(f"Grain={grain} TopDelayCodesFile={top_path} Rows={len(top_df)} Cols={len(top_df.columns)}")
runtime_data_log(f"Grain={grain} OwnerMinutesFile={owner_path} Rows={len(owner_df)} Cols={len(owner_df.columns)}")

period_candidates = MONTHLY_PERIOD_CANDIDATES if grain == "Monthly" else WEEKLY_PERIOD_CANDIDATES

# Sentinels used by owner section when cat_df is unavailable
delay_category_sel: str | None = None
category_summary: pd.DataFrame | None = None

if cat_df is not None:
    cat_period_col = pick_column(cat_df, period_candidates)
    cat_category_col = pick_column(cat_df, CATEGORY_CANDIDATES, token_groups=[["delay", "category"], ["owner"]])
    cat_minutes_col = pick_column(cat_df, MINUTES_CANDIDATES, token_groups=[["min"], ["minute"]])
    cat_station_col = pick_column(cat_df, STATION_CANDIDATES, token_groups=[["station"], ["airport"]])

    if cat_period_col is None or cat_category_col is None or cat_minutes_col is None:
        st.error(
            "DelayCategory artifact missing required columns for period/category/minutes inference "
            f"(period={cat_period_col}, category={cat_category_col}, minutes={cat_minutes_col})."
        )
        st.stop()

    cat = cat_df.copy()
    cat["_period"], cat["_sort"] = zip(*cat[cat_period_col].map(lambda v: parse_period(v, grain)))
    cat[cat_minutes_col] = pd.to_numeric(cat[cat_minutes_col], errors="coerce").fillna(0.0)
    cat = cat.dropna(subset=["_period", "_sort"]).copy()

    active_periods_df = (
        cat.groupby(["_period", "_sort"], as_index=False)[cat_minutes_col]
        .sum()
        .query(f"{cat_minutes_col} > 0")
        .sort_values("_sort")
    )
    period_options = active_periods_df["_period"].astype(str).tolist()
    if not period_options:
        st.error("No active periods found where total minutes > 0.")
        st.stop()
    slice_df = apply_filters(cat, filters, station_col=cat_station_col, period_col="_period")
    if slice_df.empty:
        st.info("No records for selected period/station slice.")
        st.stop()

    category_summary = (
        slice_df.groupby(cat_category_col, as_index=False)[cat_minutes_col]
        .sum()
        .rename(columns={cat_category_col: "DelayCategory", cat_minutes_col: "Minutes"})
        .sort_values("Minutes", ascending=False)
    )
    if category_summary.empty:
        st.info("No category minutes available for selected slice.")
        st.stop()
    category_summary["SharePct"] = 100.0 * category_summary["Minutes"] / category_summary["Minutes"].sum()

    if category_summary is not None and not category_summary.empty:
        _top_cat = str(category_summary.iloc[0]["DelayCategory"])
        _top_share = float(category_summary.iloc[0]["SharePct"])
        _sev = "bad" if _top_share > 50 else ("warn" if _top_share > 35 else "info")
        render_situation_summary(
            f"Top driver: <b>{_top_cat}</b> at <b>{_top_share:.1f}%</b> of total delay.",
            severity=_sev,
        )

    default_category = str(category_summary.iloc[0]["DelayCategory"])
    ctrl1, ctrl2 = st.columns(2)
    delay_category_sel = ctrl1.selectbox(
        "DelayCategory",
        options=category_summary["DelayCategory"].astype(str).tolist(),
        index=0,
    )
    top_n = ctrl2.slider("Pareto Top N", min_value=5, max_value=20, value=10, step=1)

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
    st.subheader("DelayCategory Pareto")
    pareto = category_summary.head(top_n).copy()
    if len(category_summary) > top_n:
        other_minutes = float(category_summary.iloc[top_n:]["Minutes"].sum())
        other_share = float(category_summary.iloc[top_n:]["SharePct"].sum())
        pareto = pd.concat(
            [
                pareto,
                pd.DataFrame([{"DelayCategory": "Other", "Minutes": other_minutes, "SharePct": other_share}]),
            ],
            ignore_index=True,
        )

    fig_pareto = go.Figure(
        go.Bar(
            x=pareto["DelayCategory"].astype(str).tolist(),
            y=pareto["Minutes"].tolist(),
            marker=dict(color=COLOR_PRIMARY),
            hovertemplate="%{x}<br>Minutes: %{y:,.0f}<extra></extra>",
        )
    )
    style_plotly_card(fig_pareto, height=340, margin=dict(l=20, r=20, t=20, b=20))
    fig_pareto.update_layout(xaxis_title="DelayCategory", yaxis_title="Minutes")
    show_plotly(fig_pareto, key="p30__category_pareto")

    pareto_table = pareto.copy()
    pareto_table["Minutes"] = pareto_table["Minutes"].map(lambda v: f"{float(v):,.0f}")
    pareto_table["Share %"] = pareto_table["SharePct"].map(lambda v: f"{float(v):.1f}%")
    st.dataframe(pareto_table[["DelayCategory", "Minutes", "Share %"]], width="stretch", hide_index=True)

    # ── Computed insight annotations ─────────────────────────────────────────
    _insights: list[str] = []

    # Insight 1: top driver share
    _top = category_summary.iloc[0]
    _insights.append(
        f"**{_top['DelayCategory']}** is the top delay driver — "
        f"**{_top['SharePct']:.1f}%** of total delay minutes "
        f"({float(_top['Minutes']):,.0f} min)."
    )

    # Insight 2: top-2 concentration
    if len(category_summary) >= 2:
        _top2_share = float(category_summary.head(2)["SharePct"].sum())
        _top2_cats = " + ".join(category_summary.head(2)["DelayCategory"].astype(str).tolist())
        _insights.append(
            f"Top 2 categories ({_top2_cats}) account for **{_top2_share:.1f}%** of total delay."
        )

    # Insight 3: controllable share (excludes LATE ARRIVAL and REACTIONARY/INHERITED categories)
    _NON_CTRL_TOKENS = {"latearrival", "reactionary", "inherited", "lateacft", "lateaircraft"}
    _ctrl_mask = category_summary["DelayCategory"].astype(str).map(
        lambda v: not any(t in normalize_name(v) for t in _NON_CTRL_TOKENS)
    )
    _ctrl_share = float(category_summary.loc[_ctrl_mask, "SharePct"].sum())
    if _ctrl_share > 0:
        _insights.append(
            f"Controllable categories (excluding Late Arrival / Reactionary) account for "
            f"**{_ctrl_share:.1f}%** of total delay."
        )

    # Insight 4: MoM trend for selected category (computed after trend df is built — injected below)
    # (placeholder index so we can insert it in position)
    _mom_insight_idx = len(_insights)
    _insights.append("")   # filled after trend computation

    # Insight 5: station filter context
    if stations_from_filters and stations_from_filters != ["NETWORK"]:
        _stn_txt = stations_from_filters[0] if len(stations_from_filters) == 1 else f"{len(stations_from_filters)} stations"
        _insights.append(f"Filter active — showing **{_stn_txt}** only (not full network).")

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
    st.subheader("Trend (last 12 periods)")
    trend = cat.copy()
    trend = trend[trend[cat_category_col].astype(str).map(normalize_category) == normalize_category(delay_category_sel)].copy()
    trend = apply_filters(trend, filters, station_col=cat_station_col, period_col="_period")

    trend = (
        trend.groupby(["_period", "_sort"], as_index=False)[cat_minutes_col]
        .sum()
        .sort_values("_sort")
        .tail(12)
    )
    if trend.empty:
        st.info("No trend data for selected DelayCategory and station filter.")
    else:
        trend_y = pd.to_numeric(trend[cat_minutes_col], errors="coerce")
        fig_trend = go.Figure(
            go.Scatter(
                x=trend["_period"].astype(str).tolist(),
                y=trend_y.tolist(),
                mode="lines+markers",
                name=f"{delay_category_sel} Minutes",
                line=dict(color=COLOR_PRIMARY),
                marker=dict(color=COLOR_PRIMARY, size=8),
            )
        )
        valid = trend_y.dropna()
        if len(valid) >= 2:
            last_val = valid.iloc[-1]
            prev_val = valid.iloc[-2]
            last_period = trend["_period"].iloc[valid.index[-1]]
            mom_delta = last_val - prev_val
            mom_pct = (mom_delta / prev_val * 100.0) if prev_val != 0 else 0.0
            fig_trend.add_annotation(
                x=str(last_period),
                y=last_val,
                text=f"MoM {mom_delta:+.0f}",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-30,
                font=dict(size=11),
            )
            # Fill in MoM insight placeholder
            _direction = "▲ up" if mom_delta > 0 else "▼ down"
            _insights[_mom_insight_idx] = (
                f"**{delay_category_sel}** is {_direction} **{mom_delta:+,.0f} min** "
                f"({mom_pct:+.1f}%) vs prior period ({last_period})."
            )
        else:
            _insights[_mom_insight_idx] = (
                f"Insufficient trend data to compute MoM change for **{delay_category_sel}**."
            )
        style_plotly_card(fig_trend, height=320, margin=dict(l=20, r=20, t=20, b=20))
        fig_trend.update_layout(xaxis_title="Period", yaxis_title="Minutes")
        show_plotly(fig_trend, key="p30__category_trend")

    # ── Render Key Insights box ───────────────────────────────────────────────
    _final_insights = [s for s in _insights if s.strip()]
    if _final_insights:
        _bullet_md = "\n".join(f"- {s}" for s in _final_insights)
        st.info(f"**💡 Key Insights — {delay_category_sel} ({grain})**\n\n{_bullet_md}")

st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
st.subheader("Drilldown: Top Delay Codes")
if grain != "Annual":
    st.info(
        "Top delay codes data is annual-scope (no period dimension) — "
        "period filter is not applied to code drill-down. Switch to Annual grain to compare."
    )
top_code_col = pick_column(top_df, ["DelayCode", "Code", "Delay_Code"], token_groups=[["code"]])
top_minutes_col = pick_column(top_df, MINUTES_CANDIDATES, token_groups=[["min"], ["minute"]])
top_category_col = pick_column(top_df, ["DelayCategory", "Category", "Owner_Basis_DepDelayMin"], token_groups=[["category"], ["owner"]])
top_period_col = pick_column(top_df, period_candidates)
top_station_col = pick_column(top_df, STATION_CANDIDATES, token_groups=[["station"], ["airport"]])
top_count_col = pick_column(top_df, ["Flights_Affected", "Count", "EventCount", "Occurrences"])

if top_code_col is None or top_minutes_col is None:
    st.warning("Top delay codes artifact is missing code/minutes columns for drilldown.")
else:
    drill = top_df.copy()
    drill[top_minutes_col] = pd.to_numeric(drill[top_minutes_col], errors="coerce").fillna(0.0)

    scope_label_parts: list[str] = []
    if top_period_col is not None:
        drill["_period"], drill["_sort"] = zip(*drill[top_period_col].map(lambda v: parse_period(v, grain)))
        drill = apply_filters(drill, filters, station_col=None, period_col="_period")
        scope_label_parts.append(f"Periods={period_label}")
    else:
        scope_label_parts.append("overall (not periodized)")

    if top_station_col is not None:
        drill = apply_filters(drill, filters, station_col=top_station_col, period_col=None)
        if stations_from_filters and stations_from_filters != ["NETWORK"]:
            scope_label_parts.append(f"Stations={len(stations_from_filters)}")

    has_category_mapping = top_category_col is not None and delay_category_sel is not None
    if has_category_mapping:
        selected_norm = normalize_category(delay_category_sel)
        drill_cat_norm = drill[top_category_col].astype(str).map(normalize_category)
        drill = drill[drill_cat_norm == selected_norm].copy()
    elif top_category_col is None:
        st.warning(
            "Category -> Code mapping not available in published artifacts; requires adding a dedicated exporter later."
        )

    title_prefix = "Top 10 codes"
    if has_category_mapping:
        title_prefix = f"Top 10 codes for DelayCategory: {delay_category_sel}"
    else:
        title_prefix = "Top 10 codes (overall, not category-specific)"

    if drill.empty:
        st.info("No code rows available for selected drilldown slice.")
    else:
        group_cols = [top_code_col]
        agg = drill.groupby(group_cols, as_index=False)[top_minutes_col].sum()
        if top_count_col is not None:
            drill[top_count_col] = pd.to_numeric(drill[top_count_col], errors="coerce").fillna(0.0)
            cnt = drill.groupby(group_cols, as_index=False)[top_count_col].sum()
            agg = agg.merge(cnt, on=group_cols, how="left")
        agg = agg.sort_values(top_minutes_col, ascending=False).head(10)

        fig_codes = go.Figure(
            go.Bar(
                x=agg[top_code_col].astype(str).tolist(),
                y=agg[top_minutes_col].tolist(),
                marker=dict(color=COLOR_SECONDARY),
                hovertemplate="%{x}<br>Minutes: %{y:,.0f}<extra></extra>",
            )
        )
        style_plotly_card(fig_codes, height=320, margin=dict(l=20, r=20, t=20, b=20))
        fig_codes.update_layout(xaxis_title="Delay Code", yaxis_title="Minutes")
        st.caption(f"{title_prefix} | {' | '.join(scope_label_parts)}")
        show_plotly(fig_codes, key="p30__top_delay_codes_bar")

        codes_table = pd.DataFrame(
            {
                "DelayCode": agg[top_code_col].astype(str),
                "Minutes": agg[top_minutes_col].map(lambda v: f"{float(v):,.0f}"),
            }
        )
        if top_count_col is not None and top_count_col in agg.columns:
            codes_table["Count"] = agg[top_count_col].map(lambda v: f"{float(v):,.0f}")
        st.dataframe(codes_table, width="stretch", hide_index=True)

st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
st.subheader("Owner Cross-check")
if grain != "Annual":
    st.info(
        "Owner cross-check data is annual-scope (no period dimension) — "
        "period filter is not applied here. Switch to Annual grain to compare."
    )
owner_category_col = pick_column(owner_df, CATEGORY_CANDIDATES, token_groups=[["owner"], ["category"]])
owner_minutes_col = pick_column(owner_df, MINUTES_CANDIDATES, token_groups=[["min"], ["minute"]])
owner_period_col = pick_column(owner_df, period_candidates)
owner_station_col = pick_column(owner_df, STATION_CANDIDATES, token_groups=[["station"], ["airport"]])

if owner_category_col is None or owner_minutes_col is None:
    st.warning("Owner artifact missing category/minutes columns for cross-check.")
else:
    own = owner_df.copy()
    own[owner_minutes_col] = pd.to_numeric(own[owner_minutes_col], errors="coerce").fillna(0.0)
    owner_scope = "overall (not periodized)"
    if owner_period_col is not None:
        own["_period"], own["_sort"] = zip(*own[owner_period_col].map(lambda v: parse_period(v, grain)))
        own = apply_filters(own, filters, station_col=None, period_col="_period")
        owner_scope = f"periodized ({period_label})"
    if owner_station_col is not None:
        own = apply_filters(own, filters, station_col=owner_station_col, period_col=None)
        if stations_from_filters and stations_from_filters != ["NETWORK"]:
            owner_scope += f", stations={len(stations_from_filters)}"

    owner_sum = (
        own.groupby(owner_category_col, as_index=False)[owner_minutes_col]
        .sum()
        .rename(columns={owner_category_col: "OwnerCategory", owner_minutes_col: "Minutes"})
        .sort_values("Minutes", ascending=False)
    )
    if category_summary is not None and delay_category_sel is not None:
        selected_cat_minutes = float(
            category_summary.loc[
                category_summary["DelayCategory"].astype(str).map(normalize_category) == normalize_category(delay_category_sel),
                "Minutes",
            ].sum()
        )
        owner_selected_minutes = float(
            owner_sum.loc[
                owner_sum["OwnerCategory"].astype(str).map(normalize_category) == normalize_category(delay_category_sel),
                "Minutes",
            ].sum()
        ) if not owner_sum.empty else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Selected Category Minutes", f"{selected_cat_minutes:,.0f}")
        c2.metric("Owner Cross-check Minutes", f"{owner_selected_minutes:,.0f}")
        c3.metric("Owner Scope", owner_scope)
    else:
        st.caption(f"Owner Scope: {owner_scope}")

    if owner_sum.empty:
        st.info("No owner rows available for current cross-check scope.")
    else:
        own_tbl = owner_sum.head(10).copy()
        own_tbl["Minutes"] = own_tbl["Minutes"].map(lambda v: f"{float(v):,.0f}")
        st.dataframe(own_tbl, width="stretch", hide_index=True)


