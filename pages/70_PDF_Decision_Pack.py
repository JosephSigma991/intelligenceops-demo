from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from mode_a_run_context import get_run_context_for_region_mode
from mode_a_filters import apply_filters
from utils import (
    normalize_name, pick_column, parse_period, ensure_list,
    read_csv_cached, load_csv, resolve_required_file,
    runtime_data_log, weighted_avg, maybe_get, render_freshness_banner, render_filter_banner,
)
from kpi_config import (
    MONTHLY_PERIOD_CANDIDATES, WEEKLY_PERIOD_CANDIDATES,
    STATION_CANDIDATES, FLIGHTS_CANDIDATES,
    TOTAL_MIN_CANDIDATES, AVG_DELAY_CANDIDATES, OTP_CANDIDATES,
    CATEGORY_CANDIDATES, MINUTES_CANDIDATES,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SCOPE_LABEL = "Synthetic Demo"
PUBLIC_PDF_FILENAME = "IntelligenceOps_Synthetic_DecisionPack.pdf"


def default_insights_dir() -> Path:
    return Path(os.getenv("FLIGHTOPS_INSIGHTS_DIR", REPO_ROOT / "demo_data" / "insight_out"))


def _detect_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        try:
            from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx  # type: ignore
        except Exception:
            return False
    try:
        return get_script_run_ctx() is not None
    except Exception:
        return False


HAS_STREAMLIT_RUNTIME = _detect_streamlit_runtime()


def resolve_qa_path(run_stamp: dict[str, Any], insights_dir: Path, region: str, mode: str) -> Path | None:
    from_json = maybe_get(run_stamp, "qa_summary_path", "QaSummaryPath")
    if isinstance(from_json, str) and from_json.strip():
        p = Path(from_json.strip())
        return p if p.is_absolute() else (insights_dir / p)
    stamp = maybe_get(run_stamp, "stamp", "Stamp")
    if isinstance(stamp, str) and stamp.strip():
        return insights_dir / f"qa_summary__{region}__{mode}__{stamp.strip()}.csv"
    return None


def _kaleido_available() -> bool:
    try:
        import kaleido  # noqa: F401
        return True
    except Exception:
        return False


def sanitize_token(v: Any, default: str = "NA") -> str:
    s = str(v).strip() if v is not None else ""
    if not s:
        s = default
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def build_otp_trend_png(
    kpi_work: pd.DataFrame,
    kpi_mapping: dict[str, str | None],
    grain: str,
) -> bytes | None:
    """Render OTP D15 trend as PNG bytes. Returns None if kaleido unavailable or data missing."""
    if not _kaleido_available():
        return None
    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        otp_col = kpi_mapping.get("otp")
        flights_col = kpi_mapping.get("flights")
        if otp_col is None or otp_col not in kpi_work.columns:
            return None

        # Aggregate network-level weighted OTP per period
        work = kpi_work.copy()
        work[otp_col] = pd.to_numeric(work[otp_col], errors="coerce")
        work[flights_col] = pd.to_numeric(work[flights_col], errors="coerce").fillna(0)
        work["_otp_w"] = work[otp_col] * work[flights_col]

        grp = (
            work.dropna(subset=["_period", "_sort"])
            .groupby(["_period", "_sort"], as_index=False)
            .agg(_otp_w=("_otp_w", "sum"), _flights=(flights_col, "sum"))
            .sort_values("_sort")
        )
        grp["OTP"] = grp["_otp_w"] / grp["_flights"].replace(0, float("nan"))
        grp = grp.dropna(subset=["OTP"])
        if grp.empty:
            return None

        x = grp["_period"].astype(str).tolist()
        y = grp["OTP"].tolist()

        # Color-code markers: green>=85, amber 80-85, red<80
        marker_colors = [
            "#22c55e" if v >= 85 else ("#f59e0b" if v >= 80 else "#ef4444")
            for v in y
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers",
            line=dict(color="#1E3A5F", width=2),
            marker=dict(color=marker_colors, size=7),
            name="OTP D15 %",
            hovertemplate="%{x}<br>OTP: %{y:.1f}%<extra></extra>",
        ))
        # Dashed 85% target line
        fig.add_hline(
            y=85,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="Target 85%",
            annotation_position="top right",
            annotation_font=dict(size=10, color="#ef4444"),
        )
        fig.update_layout(
            title=dict(text="DEP OTP D15 (%) — Network Trend", font=dict(size=13, color="#0B1220")),
            height=220, width=500,
            margin=dict(l=40, r=20, t=40, b=40),
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            font=dict(color="#0B1220", family="Arial, sans-serif", size=11),
            showlegend=False,
            yaxis=dict(range=[max(0, min(y) - 5), 100], ticksuffix="%", gridcolor="#E5E7EB"),
            xaxis=dict(tickangle=-45, gridcolor="#E5E7EB"),
        )
        png_bytes = pio.to_image(fig, format="png", width=500, height=220, scale=2)
        return png_bytes
    except Exception as exc:
        runtime_data_log(f"OTP_TREND_PNG_FAIL {type(exc).__name__}: {exc}")
        return None


def build_pareto_png(drivers_df: pd.DataFrame) -> bytes | None:
    """Render delay category pareto as PNG bytes. Returns None if kaleido unavailable or data missing."""
    if not _kaleido_available():
        return None
    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        if drivers_df is None or drivers_df.empty:
            return None
        if "DelayCategory" not in drivers_df.columns or "Minutes" not in drivers_df.columns:
            return None

        df = drivers_df.head(10).copy()
        df["Minutes"] = pd.to_numeric(df["Minutes"], errors="coerce").fillna(0)
        df = df.sort_values("Minutes", ascending=True)  # ascending for horizontal bar readability

        x = df["Minutes"].tolist()
        y = df["DelayCategory"].astype(str).tolist()
        text = [f"{int(v):,} min" for v in x]

        fig = go.Figure(go.Bar(
            x=x, y=y,
            orientation="h",
            marker=dict(color="#1E3A5F"),
            text=text, textposition="outside",
            textfont=dict(size=9, color="#0B1220"),
            hovertemplate="%{y}<br>%{x:,.0f} min<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text="Top Delay Categories — Minutes (Pareto)", font=dict(size=13, color="#0B1220")),
            height=260, width=500,
            margin=dict(l=120, r=80, t=40, b=30),
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            font=dict(color="#0B1220", family="Arial, sans-serif", size=10),
            showlegend=False,
            xaxis=dict(gridcolor="#E5E7EB", tickformat=","),
            yaxis=dict(tickfont=dict(size=9)),
        )
        png_bytes = pio.to_image(fig, format="png", width=500, height=260, scale=2)
        return png_bytes
    except Exception as exc:
        runtime_data_log(f"PARETO_PNG_FAIL {type(exc).__name__}: {exc}")
        return None


def otp_is_percent(col_name: str | None, values: pd.Series | None) -> bool:
    if col_name is None or values is None:
        return False
    name = col_name.lower()
    if "%" in name or "pct" in name or "percent" in name:
        return True
    num = pd.to_numeric(values, errors="coerce").dropna()
    if num.empty:
        return False
    q95 = float(num.quantile(0.95))
    return 1.5 < q95 <= 100.0


def fmt_int(v: float | int | None) -> str:
    if v is None or pd.isna(v):
        return "not available"
    return f"{int(round(float(v))):,}"


def fmt_float(v: float | int | None, digits: int = 2) -> str:
    if v is None or pd.isna(v):
        return "not available"
    return f"{float(v):,.{digits}f}"


def build_station_kpi_work(
    df: pd.DataFrame,
    grain: str,
) -> tuple[pd.DataFrame, dict[str, str | None], str | None]:
    period_col = pick_column(df, MONTHLY_PERIOD_CANDIDATES if grain == "Monthly" else WEEKLY_PERIOD_CANDIDATES)
    station_col = pick_column(df, STATION_CANDIDATES, token_groups=[["station"], ["airport"]])
    flights_col = pick_column(df, FLIGHTS_CANDIDATES, token_groups=[["flight", "oper"]])
    total_col = pick_column(df, TOTAL_MIN_CANDIDATES, token_groups=[["dep", "delay"], ["min"]])
    avg_col = pick_column(df, AVG_DELAY_CANDIDATES, token_groups=[["avg", "delay"], ["per", "flight"]])
    otp_col = pick_column(df, OTP_CANDIDATES, token_groups=[["otp"], ["d15"]])

    mapping: dict[str, str | None] = {
        "period": period_col,
        "station": station_col,
        "flights": flights_col,
        "total": total_col,
        "avg": avg_col,
        "otp": otp_col,
    }
    if period_col is None or station_col is None or flights_col is None:
        return pd.DataFrame(), mapping, "Missing required station KPI columns (period/station/flights)."

    work = df.copy()
    parsed = work[period_col].map(lambda v: parse_period(v, grain))
    work["_period"] = parsed.map(lambda t: t[0])
    work["_sort"] = parsed.map(lambda t: t[1])
    work = work.dropna(subset=["_period", "_sort"]).copy()

    work[flights_col] = pd.to_numeric(work[flights_col], errors="coerce")
    if total_col is not None:
        work[total_col] = pd.to_numeric(work[total_col], errors="coerce")
    if avg_col is not None:
        work[avg_col] = pd.to_numeric(work[avg_col], errors="coerce")
    if otp_col is not None:
        work[otp_col] = pd.to_numeric(work[otp_col], errors="coerce")

    return work, mapping, None


def build_delay_minutes_work(
    df: pd.DataFrame,
    grain: str,
) -> tuple[pd.DataFrame, dict[str, str | None], str | None]:
    period_col = pick_column(df, MONTHLY_PERIOD_CANDIDATES if grain == "Monthly" else WEEKLY_PERIOD_CANDIDATES)
    station_col = pick_column(df, STATION_CANDIDATES, token_groups=[["station"], ["airport"]])
    category_col = pick_column(df, CATEGORY_CANDIDATES, token_groups=[["delay", "category"], ["owner"]])
    minutes_col = pick_column(df, MINUTES_CANDIDATES, token_groups=[["min"]])

    mapping: dict[str, str | None] = {
        "period": period_col,
        "station": station_col,
        "category": category_col,
        "minutes": minutes_col,
    }
    if period_col is None or category_col is None or minutes_col is None:
        return pd.DataFrame(), mapping, "Missing required delay category columns (period/category/minutes)."

    work = df.copy()
    parsed = work[period_col].map(lambda v: parse_period(v, grain))
    work["_period"] = parsed.map(lambda t: t[0])
    work["_sort"] = parsed.map(lambda t: t[1])
    work = work.dropna(subset=["_period", "_sort"]).copy()
    work[minutes_col] = pd.to_numeric(work[minutes_col], errors="coerce")
    return work, mapping, None


def build_snapshot(
    period_df: pd.DataFrame,
    station_selection: str,
    mapping: dict[str, str | None],
) -> dict[str, Any]:
    flights_col = mapping["flights"]
    total_col = mapping["total"]
    avg_col = mapping["avg"]
    otp_col = mapping["otp"]
    station_col = mapping["station"]

    if station_selection != "NETWORK" and station_col is not None:
        sdf = period_df[period_df[station_col].astype(str) == station_selection].copy()
    else:
        sdf = period_df.copy()

    flights = float(pd.to_numeric(sdf[flights_col], errors="coerce").sum()) if flights_col else 0.0
    total = float(pd.to_numeric(sdf[total_col], errors="coerce").sum()) if total_col else None

    avg_delay = None
    avg_basis = "not available"
    if total is not None and flights > 0:
        avg_delay = total / flights
        avg_basis = "computed (total/flights)"
    elif avg_col is not None:
        avg_delay = weighted_avg(sdf[avg_col], sdf[flights_col])
        if avg_delay is not None:
            avg_basis = "weighted avg from avg column"

    otp = None
    otp_basis = "not available"
    if otp_col is not None and otp_is_percent(otp_col, sdf[otp_col]):
        otp = weighted_avg(sdf[otp_col], sdf[flights_col])
        if otp is not None:
            otp_basis = "weighted avg (percent column)"

    return {
        "Flights": flights,
        "TotalMinutes": total,
        "AvgDelay": avg_delay,
        "AvgDelayBasis": avg_basis,
        "OTP": otp,
        "OTPBasis": otp_basis,
    }


def compute_coverage_flights_operated(
    station_kpi_df: pd.DataFrame,
    grain: str,
    period_selected: str,
    station_selected: str,
) -> int:
    if station_kpi_df is None or station_kpi_df.empty:
        return 0

    cols = list(station_kpi_df.columns)
    flights_col: str | None = None
    for c in ["Flights_Operated", "FlightsOperated"]:
        if c in cols:
            flights_col = c
            break
    if flights_col is None:
        for c in cols:
            lc = c.lower().replace(" ", "_")
            if "flights_operated" in lc:
                flights_col = c
                break
    if flights_col is None:
        for c in cols:
            lc = c.lower()
            if ("flights" in lc) and (("operated" in lc) or ("ops" in lc)):
                flights_col = c
                break
    if flights_col is None:
        return 0

    period_candidates = MONTHLY_PERIOD_CANDIDATES if grain == "Monthly" else WEEKLY_PERIOD_CANDIDATES
    period_col = pick_column(station_kpi_df, period_candidates, token_groups=[["period"]])
    station_col = pick_column(station_kpi_df, STATION_CANDIDATES, token_groups=[["station"], ["airport"]])

    d = station_kpi_df.copy()
    d[flights_col] = pd.to_numeric(d[flights_col], errors="coerce").fillna(0)

    if period_col is not None:
        target_period, _ = parse_period(period_selected, grain)
        target_period = target_period or str(period_selected)
        parsed_period = d[period_col].map(lambda v: parse_period(v, grain)[0] or str(v).strip())
        d = d[parsed_period.astype(str) == str(target_period)].copy()

    if station_col is not None:
        station_series = d[station_col].astype(str).str.strip().str.upper()
        if station_selected != "NETWORK":
            d = d[station_series == str(station_selected).strip().upper()].copy()
        else:
            network_rows = d[station_series == "NETWORK"].copy()
            if not network_rows.empty:
                d = network_rows

    val = pd.to_numeric(d[flights_col], errors="coerce").fillna(0).sum()
    try:
        return max(0, int(round(float(val))))
    except Exception:
        return 0


def build_station_ranking(period_df: pd.DataFrame, mapping: dict[str, str | None]) -> tuple[pd.DataFrame, pd.DataFrame]:
    station_col = mapping["station"]
    flights_col = mapping["flights"]
    total_col = mapping["total"]
    avg_col = mapping["avg"]
    if station_col is None or flights_col is None:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for station, g in period_df.groupby(station_col):
        flights = float(pd.to_numeric(g[flights_col], errors="coerce").sum())
        if flights <= 0:
            continue
        total = float(pd.to_numeric(g[total_col], errors="coerce").sum()) if total_col is not None else None
        if total is not None:
            avg_delay = total / flights
        else:
            avg_delay = weighted_avg(g[avg_col], g[flights_col]) if avg_col is not None else None
        if avg_delay is None or pd.isna(avg_delay):
            continue
        rows.append({"Station": str(station), "AvgDelayMin": float(avg_delay), "Flights": flights, "TotalMinutes": total})

    rank_df = pd.DataFrame(rows)
    if rank_df.empty:
        return rank_df, rank_df
    rank_df = rank_df.sort_values("AvgDelayMin", ascending=False).reset_index(drop=True)
    worst = rank_df.head(10).copy()
    best = rank_df.sort_values("AvgDelayMin", ascending=True).head(10).copy()
    return worst, best


def build_drivers_summary(
    delay_df: pd.DataFrame,
    period: str,
    station_selection: str,
    mapping: dict[str, str | None],
) -> tuple[pd.DataFrame, str]:
    period_col = mapping["period"]
    station_col = mapping["station"]
    category_col = mapping["category"]
    minutes_col = mapping["minutes"]
    if period_col is None or category_col is None or minutes_col is None:
        return pd.DataFrame(), "not available"

    d = delay_df[delay_df["_period"] == period].copy()
    if d.empty:
        return pd.DataFrame(), "no rows for selected period"

    scope_note = "station column present"
    if station_selection != "NETWORK":
        if station_col is not None:
            d = d[d[station_col].astype(str) == station_selection].copy()
            scope_note = "station-filtered"
        else:
            scope_note = "station column missing in delay minutes file; treated as network-only"

    if d.empty:
        return pd.DataFrame(), scope_note

    grp = (
        d.groupby(category_col, as_index=False)[minutes_col]
        .sum()
        .rename(columns={category_col: "DelayCategory", minutes_col: "Minutes"})
        .sort_values("Minutes", ascending=False)
    )
    total_minutes = float(pd.to_numeric(grp["Minutes"], errors="coerce").sum())
    grp["SharePct"] = (grp["Minutes"] / total_minutes * 100.0) if total_minutes > 0 else 0.0
    return grp.head(10).reset_index(drop=True), scope_note


def _normalized_category_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _delay_bucket(value: Any) -> str:
    label = _normalized_category_label(value)
    if not label:
        return "Other / not classified"
    if "groundops" in label or label == "gops":
        return "Controllable - Ground Ops"
    if (
        "latearrival" in label
        or "reactionary" in label
        or "inherited" in label
        or "lateaircraft" in label
        or "lateacft" in label
    ):
        return "Inherited / reactionary"
    if "weather" in label or "atc" in label or "airport" in label or "uncontrollable" in label:
        return "Other / not classified"
    return "Controllable - Other categories"


def build_accountability_split(
    delay_df: pd.DataFrame,
    period: str,
    station_selection: str,
    mapping: dict[str, str | None],
) -> pd.DataFrame:
    period_col = mapping["period"]
    station_col = mapping["station"]
    category_col = mapping["category"]
    minutes_col = mapping["minutes"]
    if period_col is None or category_col is None or minutes_col is None:
        return pd.DataFrame(columns=["Bucket", "Minutes", "SharePct"])

    d = delay_df[delay_df["_period"] == period].copy()
    if station_selection != "NETWORK" and station_col is not None:
        d = d[d[station_col].astype(str) == station_selection].copy()
    if d.empty:
        return pd.DataFrame(columns=["Bucket", "Minutes", "SharePct"])

    d["_bucket"] = d[category_col].map(_delay_bucket)
    d["_minutes"] = pd.to_numeric(d[minutes_col], errors="coerce").fillna(0.0)
    rows = (
        d.groupby("_bucket", as_index=False)["_minutes"]
        .sum()
        .rename(columns={"_bucket": "Bucket", "_minutes": "Minutes"})
    )

    bucket_order = [
        "Controllable - Ground Ops",
        "Controllable - Other categories",
        "Inherited / reactionary",
        "Other / not classified",
    ]
    rows["_order"] = rows["Bucket"].map({v: i for i, v in enumerate(bucket_order)}).fillna(99)
    rows = rows.sort_values(["_order", "Bucket"]).drop(columns=["_order"]).reset_index(drop=True)
    total = float(pd.to_numeric(rows["Minutes"], errors="coerce").sum())
    rows["SharePct"] = (rows["Minutes"] / total * 100.0) if total > 0 else 0.0
    return rows


def build_executive_findings(
    snapshot: dict[str, Any],
    drivers_df: pd.DataFrame,
    accountability_df: pd.DataFrame,
    station_df: pd.DataFrame,
) -> list[str]:
    findings: list[str] = []

    otp = snapshot.get("OTP")
    avg_delay = snapshot.get("AvgDelay")
    if otp is not None and not pd.isna(otp):
        target_gap = 85.0 - float(otp)
        if target_gap > 0:
            findings.append(f"OTP is {fmt_float(target_gap, 1)} points below the demo target for the selected scope.")
        else:
            findings.append("OTP is at or above the demo target for the selected scope.")
    elif avg_delay is not None and not pd.isna(avg_delay):
        findings.append(f"Average departure delay is {fmt_float(avg_delay, 2)} minutes per operated flight in the selected scope.")
    else:
        findings.append("KPI status is limited because the selected scope does not expose all snapshot fields.")

    if drivers_df is not None and not drivers_df.empty:
        top = drivers_df.iloc[0]
        findings.append(
            f"Top delay category in the selected synthetic scope is {top.get('DelayCategory')} "
            f"at {fmt_float(top.get('SharePct'), 1)}% of category minutes."
        )

    if accountability_df is not None and not accountability_df.empty:
        ctrl_mask = accountability_df["Bucket"].astype(str).str.startswith("Controllable")
        ctrl_share = float(pd.to_numeric(accountability_df.loc[ctrl_mask, "SharePct"], errors="coerce").fillna(0).sum())
        findings.append(f"Controllable categories represent {fmt_float(ctrl_share, 1)}% of categorized delay minutes.")

    if station_df is not None and not station_df.empty:
        findings.append("Station review ranks synthetic stations by average departure delay for follow-up prioritization.")

    return findings[:4]


def _fmt_pct(v: Any, digits: int = 1) -> str:
    if v is None or pd.isna(v):
        return "not available"
    return f"{float(v):,.{digits}f}%"


def _materiality_note(flights: Any) -> str:
    try:
        f = float(flights)
    except Exception:
        return "flight volume not available"
    if f < 30:
        return "materiality caveat: low flight volume may distort average delay"
    return "sufficient synthetic volume for review"


def _station_metric_frame(period_df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    station_col = mapping["station"]
    flights_col = mapping["flights"]
    total_col = mapping["total"]
    avg_col = mapping["avg"]
    otp_col = mapping["otp"]
    if station_col is None or flights_col is None:
        return pd.DataFrame()

    gops_col = pick_column(period_df, ["GOPS_Min_NORM_Total", "GOPS_Min_NORM", "GroundOps_Min_NORM_Total"], token_groups=[["gops"], ["min"]])
    cont_col = pick_column(period_df, ["Controllable_Min_NORM_Total", "Controllable_Min_NORM"], token_groups=[["controllable"], ["min"]])
    inherited_col = pick_column(period_df, ["Reactionary_Min_NORM_Total", "Inherited_Min_NORM_Total"], token_groups=[["reactionary"], ["min"]])

    rows: list[dict[str, Any]] = []
    for station, g in period_df.groupby(station_col):
        flights = float(pd.to_numeric(g[flights_col], errors="coerce").fillna(0).sum())
        if flights <= 0:
            continue
        total = float(pd.to_numeric(g[total_col], errors="coerce").fillna(0).sum()) if total_col else None
        gops = float(pd.to_numeric(g[gops_col], errors="coerce").fillna(0).sum()) if gops_col else None
        controllable = float(pd.to_numeric(g[cont_col], errors="coerce").fillna(0).sum()) if cont_col else None
        inherited = float(pd.to_numeric(g[inherited_col], errors="coerce").fillna(0).sum()) if inherited_col else None
        avg_delay = (total / flights) if total is not None and flights > 0 else weighted_avg(g[avg_col], g[flights_col]) if avg_col else None
        otp = weighted_avg(g[otp_col], g[flights_col]) if otp_col else None
        rows.append(
            {
                "Station": str(station),
                "Flights": flights,
                "OTP": otp,
                "AvgDelayMin": avg_delay,
                "TotalMinutes": total,
                "GOPSMinutes": gops,
                "ControllableMinutes": controllable,
                "InheritedMinutes": inherited,
                "GOPSSharePct": (gops / total * 100.0) if gops is not None and total and total > 0 else None,
                "GOPSAvgPerFlight": (gops / flights) if gops is not None and flights > 0 else None,
            }
        )
    return pd.DataFrame(rows)


def _accountability_value(accountability_df: pd.DataFrame, bucket: str, col: str = "Minutes") -> float | None:
    if accountability_df is None or accountability_df.empty or col not in accountability_df.columns:
        return None
    m = accountability_df[accountability_df["Bucket"].astype(str) == bucket]
    if m.empty:
        return 0.0
    return float(pd.to_numeric(m[col], errors="coerce").fillna(0).sum())


def _largest_bucket(accountability_df: pd.DataFrame) -> tuple[str, float] | None:
    if accountability_df is None or accountability_df.empty:
        return None
    d = accountability_df.copy()
    d["Minutes"] = pd.to_numeric(d["Minutes"], errors="coerce").fillna(0)
    d = d.sort_values("Minutes", ascending=False)
    if d.empty:
        return None
    return str(d.iloc[0]["Bucket"]), float(d.iloc[0]["Minutes"])


def build_ground_ops_impact_stations(station_metrics: pd.DataFrame, max_rows: int = 5) -> pd.DataFrame:
    cols = ["Station", "Flights", "OTP D15", "GOPS min", "GOPS share (%)", "Action / caveat"]
    if station_metrics is None or station_metrics.empty or "GOPSMinutes" not in station_metrics.columns:
        return pd.DataFrame(columns=cols)
    d = station_metrics.copy()
    d["GOPSMinutes"] = pd.to_numeric(d["GOPSMinutes"], errors="coerce").fillna(0)
    d = d[d["GOPSMinutes"] > 0].sort_values("GOPSMinutes", ascending=False).head(max_rows)
    rows = []
    for _, r in d.iterrows():
        rows.append(
            {
                "Station": r.get("Station"),
                "Flights": fmt_int(r.get("Flights")),
                "OTP D15": _fmt_pct(r.get("OTP"), 1),
                "GOPS min": fmt_int(r.get("GOPSMinutes")),
                "GOPS share (%)": _fmt_pct(r.get("GOPSSharePct"), 1),
                "Action / caveat": _materiality_note(r.get("Flights")),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def build_delay_context_indicators(accountability_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = [
        ("Controllable Ground Ops minutes", "Controllable - Ground Ops"),
        ("Controllable Other categories minutes", "Controllable - Other categories"),
        ("Inherited / reactionary minutes", "Inherited / reactionary"),
        ("Other / not classified minutes", "Other / not classified"),
    ]
    for label, bucket in labels:
        minutes = _accountability_value(accountability_df, bucket, "Minutes")
        share = _accountability_value(accountability_df, bucket, "SharePct")
        rows.append(
            {
                "Indicator": label,
                "Minutes": fmt_int(minutes),
                "Share": _fmt_pct(share, 1),
                "Basis": "selected-scope synthetic evidence",
            }
        )
    return pd.DataFrame(rows)


def build_network_kpi_summary(snapshot: dict[str, Any], station_metrics: pd.DataFrame) -> pd.DataFrame:
    top_station = None
    bench_station = None
    if station_metrics is not None and not station_metrics.empty:
        d = station_metrics.dropna(subset=["AvgDelayMin"]).copy()
        if not d.empty:
            top_station = d.sort_values("AvgDelayMin", ascending=False).iloc[0]
            bench_station = d.sort_values("AvgDelayMin", ascending=True).iloc[0]

    rows = [
        {"Metric": "Flights operated", "Value": fmt_int(snapshot.get("Flights")), "Note": "selected synthetic scope"},
        {"Metric": "Avg DEP Delay per Flight", "Value": fmt_float(snapshot.get("AvgDelay"), 2), "Note": str(snapshot.get("AvgDelayBasis", "computed from selected scope"))},
        {"Metric": "DEP OTP D15", "Value": _fmt_pct(snapshot.get("OTP"), 1), "Note": "target benchmark 85.0%"},
    ]
    if top_station is not None:
        rows.append(
            {
                "Metric": "Top station by Avg DEP Delay per Flight",
                "Value": f"{top_station.get('Station')} | {fmt_float(top_station.get('AvgDelayMin'), 2)} min",
                "Note": _materiality_note(top_station.get("Flights")),
            }
        )
    if bench_station is not None:
        rows.append(
            {
                "Metric": "Benchmark station by Avg DEP Delay per Flight",
                "Value": f"{bench_station.get('Station')} | {fmt_float(bench_station.get('AvgDelayMin'), 2)} min",
                "Note": _materiality_note(bench_station.get("Flights")),
            }
        )
    return pd.DataFrame(rows)


def build_station_ranking_snapshot(station_metrics: pd.DataFrame) -> pd.DataFrame:
    cols = ["View", "Station", "Avg delay", "Flights", "Materiality note"]
    if station_metrics is None or station_metrics.empty:
        return pd.DataFrame(columns=cols)
    d = station_metrics.dropna(subset=["AvgDelayMin"]).copy()
    if d.empty:
        return pd.DataFrame(columns=cols)
    top = d.sort_values("AvgDelayMin", ascending=False).iloc[0]
    bench = d.sort_values("AvgDelayMin", ascending=True).iloc[0]
    rows = [
        {
            "View": "Top station by Avg Delay",
            "Station": top.get("Station"),
            "Avg delay": fmt_float(top.get("AvgDelayMin"), 2),
            "Flights": fmt_int(top.get("Flights")),
            "Materiality note": _materiality_note(top.get("Flights")),
        },
        {
            "View": "Benchmark station by Avg Delay",
            "Station": bench.get("Station"),
            "Avg delay": fmt_float(bench.get("AvgDelayMin"), 2),
            "Flights": fmt_int(bench.get("Flights")),
            "Materiality note": _materiality_note(bench.get("Flights")),
        },
    ]
    return pd.DataFrame(rows, columns=cols)


def build_executive_summary_matrix(
    snapshot: dict[str, Any],
    station_metrics: pd.DataFrame,
    action_focus: str,
) -> pd.DataFrame:
    otp = snapshot.get("OTP")
    gap = None if otp is None or pd.isna(otp) else float(otp) - 85.0
    top_gops = None
    if station_metrics is not None and not station_metrics.empty and "GOPSMinutes" in station_metrics.columns:
        g = station_metrics.copy()
        g["GOPSMinutes"] = pd.to_numeric(g["GOPSMinutes"], errors="coerce").fillna(0)
        g = g[g["GOPSMinutes"] > 0].sort_values("GOPSMinutes", ascending=False)
        if not g.empty:
            top_gops = g.iloc[0]
    if top_gops is not None:
        pain_station = f"{top_gops.get('Station')} by Ground Ops minutes"
    else:
        ranked = build_station_ranking_snapshot(station_metrics)
        pain_station = str(ranked.iloc[0]["Station"]) + " by Avg DEP Delay with materiality caveat" if not ranked.empty else "not available"

    rows = [
        {"Signal": "OTP D15", "Value": _fmt_pct(otp, 1), "Decision use": "selected synthetic scope"},
        {"Signal": "Target", "Value": "85.0%", "Decision use": "demo target benchmark"},
        {"Signal": "Target gap", "Value": "not available" if gap is None else f"{gap:+.1f} pts", "Decision use": "positive means above target"},
        {"Signal": "Flights operated", "Value": fmt_int(snapshot.get("Flights")), "Decision use": "materiality context"},
        {"Signal": "Top pain station", "Value": pain_station, "Decision use": "where review starts"},
        {"Signal": "Primary action review", "Value": action_focus, "Decision use": "public-safe next review path"},
    ]
    return pd.DataFrame(rows)


def build_leadership_framing(
    snapshot: dict[str, Any],
    station_metrics: pd.DataFrame,
    accountability_df: pd.DataFrame,
    action_focus: str,
) -> list[str]:
    otp = snapshot.get("OTP")
    gap_text = "not available" if otp is None or pd.isna(otp) else f"{float(otp) - 85.0:+.1f} points vs 85.0% target"
    largest = _largest_bucket(accountability_df)
    bucket_text = f"{largest[0]} is the largest delay context bucket at {fmt_int(largest[1])} minutes" if largest else "delay context bucket unavailable"
    station_text = "station-level pain point unavailable"
    if station_metrics is not None and not station_metrics.empty:
        ranked = station_metrics.dropna(subset=["AvgDelayMin"]).sort_values("AvgDelayMin", ascending=False)
        if not ranked.empty:
            top = ranked.iloc[0]
            station_text = f"{top.get('Station')} has the highest average delay in the selected synthetic scope"
    return [
        "Decision context: convert synthetic operational evidence into a review sequence.",
        f"OTP vs target and gap: {gap_text}.",
        f"Headline operational pain: {station_text}; {bucket_text}.",
        f"Action focus: {action_focus}",
        "Synthetic data note: all figures are computer-generated for public demonstration.",
    ]


def build_action_lanes(
    station_metrics: pd.DataFrame,
    accountability_df: pd.DataFrame,
    top_code_lookup: dict[str, str],
) -> pd.DataFrame:
    lane_cols = ["Lane", "Evidence", "Review focus"]
    gops_station_text = "Ground Ops station evidence not available"
    top_station = None
    if station_metrics is not None and not station_metrics.empty:
        d = station_metrics.copy()
        d["GOPSMinutes"] = pd.to_numeric(d.get("GOPSMinutes"), errors="coerce").fillna(0)
        d = d[d["GOPSMinutes"] > 0].sort_values("GOPSMinutes", ascending=False)
        if not d.empty:
            top_station = d.iloc[0]
            top_two = d.head(2)
            parts = [f"{r.get('Station')} {fmt_int(r.get('GOPSMinutes'))} min" for _, r in top_two.iterrows()]
            gops_station_text = "; ".join(parts)

    if top_station is not None:
        station_key = str(top_station.get("Station"))
        code = top_code_lookup.get(station_key)
        if code:
            gops_focus = f"Review highest-impact synthetic station; start with code {code} as a public demo drilldown."
        else:
            gops_focus = "Review highest-impact synthetic station; start with selected-scope category drilldown."
    else:
        gops_focus = "Start with selected-scope category drilldown."

    largest = _largest_bucket(accountability_df)
    inherited_minutes = _accountability_value(accountability_df, "Inherited / reactionary", "Minutes") or 0.0
    gops_minutes = _accountability_value(accountability_df, "Controllable - Ground Ops", "Minutes") or 0.0
    if largest and largest[0] == "Inherited / reactionary":
        inherited_focus = "Governance review before assigning Ground Ops action."
    elif inherited_minutes > gops_minutes:
        inherited_focus = "Protect reactionary/inherited risk before narrowing Ground Ops accountability."
    else:
        inherited_focus = "Keep inherited/reactionary context visible while reviewing controllable levers."
    inherited_evidence = f"Inherited / reactionary {fmt_int(inherited_minutes)} min; largest bucket {largest[0] if largest else 'not available'}."

    station_focus = "Use local drilldown on the highest-priority synthetic station; avoid over-reading low-volume averages."
    if top_station is not None:
        station_evidence = (
            f"{top_station.get('Station')} | flights {fmt_int(top_station.get('Flights'))} | "
            f"avg delay {fmt_float(top_station.get('AvgDelayMin'), 2)} min | {_materiality_note(top_station.get('Flights'))}."
        )
    else:
        station_evidence = "Station priority not available for the selected synthetic scope."

    rows = [
        {
            "Lane": "A) Ground Ops controllable delay",
            "Evidence": gops_station_text,
            "Review focus": gops_focus,
        },
        {
            "Lane": "B) Reactionary / inherited delay protection",
            "Evidence": inherited_evidence,
            "Review focus": inherited_focus,
        },
        {
            "Lane": "C) Station follow-up / local drilldown",
            "Evidence": station_evidence,
            "Review focus": station_focus,
        },
        {
            "Lane": "D) TAT layer",
            "Evidence": "TAT layer is not included in this public synthetic demo.",
            "Review focus": "In the full methodology, TAT is used as a turnaround execution risk layer.",
        },
    ]
    return pd.DataFrame(rows, columns=lane_cols)


def build_top_code_lookup(top_codes_df: pd.DataFrame | None) -> dict[str, str]:
    if top_codes_df is None or top_codes_df.empty:
        return {}
    station_col = pick_column(top_codes_df, STATION_CANDIDATES, token_groups=[["station"]])
    code_col = pick_column(top_codes_df, ["DelayCode", "Delay Code", "Code"], token_groups=[["code"]])
    minutes_col = pick_column(top_codes_df, MINUTES_CANDIDATES, token_groups=[["min"]])
    if station_col is None or code_col is None:
        return {}
    d = top_codes_df.copy()
    if minutes_col is not None:
        d["_minutes"] = pd.to_numeric(d[minutes_col], errors="coerce").fillna(0)
        d = d.sort_values("_minutes", ascending=False)
    out: dict[str, str] = {}
    for station, g in d.groupby(station_col):
        if g.empty:
            continue
        code = str(g.iloc[0][code_col]).strip()
        if code:
            out[str(station)] = code
    return out


def build_confidence_caveats(include_appendix: bool) -> list[str]:
    appendix_state = "enabled" if include_appendix else "available only when enabled"
    return [
        "Evidence basis: selected-scope synthetic artifacts.",
        "This public demo uses synthetic data only.",
        "Delay Category / Owner basis: DelayCategory.",
        "No employer data, real station figures, internal context, or company identity is present.",
        f"Technical appendix is {appendix_state}.",
    ]


def build_sanitized_appendix(
    required_files: list[str],
    station_kpi_df: pd.DataFrame,
    delay_minutes_df: pd.DataFrame,
    qa_df: pd.DataFrame | None,
) -> pd.DataFrame:
    rows = [
        {
            "Item": "Synthetic data contract",
            "Status": "Available" if required_files else "Not available",
            "Detail": f"{len(required_files)} artifacts registered" if required_files else "No contract registry found",
        },
        {
            "Item": "Station KPI artifact",
            "Status": "Available" if station_kpi_df is not None and not station_kpi_df.empty else "Not available",
            "Detail": f"{len(station_kpi_df):,} rows" if station_kpi_df is not None and not station_kpi_df.empty else "No rows loaded",
        },
        {
            "Item": "Delay category artifact",
            "Status": "Available" if delay_minutes_df is not None and not delay_minutes_df.empty else "Not available",
            "Detail": f"{len(delay_minutes_df):,} rows" if delay_minutes_df is not None and not delay_minutes_df.empty else "No rows loaded",
        },
        {
            "Item": "QA summary",
            "Status": "Available" if qa_df is not None and not qa_df.empty else "Not included",
            "Detail": f"{len(qa_df):,} rows" if qa_df is not None and not qa_df.empty else "Technical appendix only",
        },
    ]
    return pd.DataFrame(rows)


def build_qa_excerpt(qa_df: pd.DataFrame | None, required_files: list[str], insights_dir: Path) -> pd.DataFrame:
    qa_lookup: dict[str, dict[str, Any]] = {}
    qa_file_names: list[str] = []
    if qa_df is not None and not qa_df.empty:
        file_col = pick_column(qa_df, ["FileName", "File", "file", "file_name"])
        rows_col = pick_column(qa_df, ["Rows", "rows"])
        cols_col = pick_column(qa_df, ["Cols", "cols", "Columns"])
        min_ym_col = pick_column(qa_df, ["MinYearMonth", "min_yearmonth"])
        max_ym_col = pick_column(qa_df, ["MaxYearMonth", "max_yearmonth"])
        min_yw_col = pick_column(qa_df, ["MinYearWeek", "min_yearweek"])
        max_yw_col = pick_column(qa_df, ["MaxYearWeek", "max_yearweek"])
        if file_col is not None:
            for _, r in qa_df.iterrows():
                fv = r.get(file_col)
                if pd.isna(fv):
                    continue
                fn = Path(str(fv)).name
                qa_file_names.append(fn)
                qa_lookup[normalize_name(fn)] = {
                    "Rows": int(r.get(rows_col)) if rows_col and pd.notna(r.get(rows_col)) else None,
                    "Cols": int(r.get(cols_col)) if cols_col and pd.notna(r.get(cols_col)) else None,
                    "MinYearMonth": str(r.get(min_ym_col)) if min_ym_col and pd.notna(r.get(min_ym_col)) else None,
                    "MaxYearMonth": str(r.get(max_ym_col)) if max_ym_col and pd.notna(r.get(max_ym_col)) else None,
                    "MinYearWeek": str(r.get(min_yw_col)) if min_yw_col and pd.notna(r.get(min_yw_col)) else None,
                    "MaxYearWeek": str(r.get(max_yw_col)) if max_yw_col and pd.notna(r.get(max_yw_col)) else None,
                }

    names = [Path(str(x)).name for x in required_files] if required_files else sorted(set(qa_file_names))
    rows: list[dict[str, Any]] = []
    if names:
        for fn in names:
            p = insights_dir / fn
            exists = p.exists()
            size = int(p.stat().st_size) if exists and p.is_file() else 0
            key = normalize_name(fn)
            meta = qa_lookup.get(key, {})
            rows.append(
                {
                    "FileName": fn,
                    "Exists": bool(exists),
                    "SizeBytes": size,
                    "Rows": meta.get("Rows"),
                    "Cols": meta.get("Cols"),
                    "MinYearMonth": meta.get("MinYearMonth"),
                    "MaxYearMonth": meta.get("MaxYearMonth"),
                    "MinYearWeek": meta.get("MinYearWeek"),
                    "MaxYearWeek": meta.get("MaxYearWeek"),
                }
            )
    return pd.DataFrame(rows)


def df_for_pdf(df: pd.DataFrame, cols: list[str], max_rows: int = 20) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[cols].head(max_rows).copy()


def build_pdf_bytes(
    title: str,
    grain: str,
    period_selected: str,
    station_selected: str,
    executive_matrix_df: pd.DataFrame,
    leadership_framing: list[str],
    action_lanes_df: pd.DataFrame,
    ground_ops_stations_df: pd.DataFrame,
    delay_context_df: pd.DataFrame,
    network_summary_df: pd.DataFrame,
    station_snapshot_df: pd.DataFrame,
    appendix_df: pd.DataFrame,
    caveats: list[str],
    include_appendix: bool,
    otp_chart_png: bytes | None = None,
    pareto_chart_png: bytes | None = None,
) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception as e:
        raise RuntimeError(f"ReportLab import failed: {type(e).__name__}: {e}") from e

    styles = getSampleStyleSheet()
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=28,
        rightMargin=1.5 * cm,
        topMargin=28,
        bottomMargin=28,
        title=title,
        pageCompression=0,
    )
    story: list[Any] = []

    small_style = styles["BodyText"].clone("SmallBody")
    small_style.fontSize = 8
    small_style.leading = 10
    normal_style = styles["BodyText"].clone("CompactBody")
    normal_style.fontSize = 9
    normal_style.leading = 11

    def _pdf_cell(value: Any, header: bool = False) -> Any:
        txt = "not available" if value is None or pd.isna(value) else str(value)
        txt = txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return Paragraph(txt, styles["Normal"] if header else small_style)

    def _col_widths(columns: list[str]) -> list[float]:
        page_width = A4[0] - doc.leftMargin - doc.rightMargin
        if len(columns) <= 2:
            return [page_width * 0.28, page_width * 0.72]
        weights = []
        for c in columns:
            c_low = str(c).lower()
            if c_low in {"lane", "action / caveat", "review focus", "decision use", "note", "materiality note", "basis"}:
                weights.append(2.2)
            elif c_low in {"evidence", "value"}:
                weights.append(2.0)
            else:
                weights.append(1.0)
        total = sum(weights) or 1.0
        return [page_width * w / total for w in weights]

    def add_table_from_df(section_title: str, tdf: pd.DataFrame) -> None:
        story.append(Paragraph(section_title, styles["Heading2"]))
        story.append(Spacer(1, 6))
        if tdf.empty:
            story.append(Paragraph("Not available.", styles["Normal"]))
            story.append(Spacer(1, 10))
            return
        display = tdf.copy().fillna("not available")
        columns = [str(c) for c in display.columns]
        data = [[_pdf_cell(c, header=True) for c in columns]]
        for _, row in display.iterrows():
            data.append([_pdf_cell(row[c]) for c in display.columns])
        tbl = Table(data, repeatRows=1, colWidths=_col_widths(columns), hAlign="LEFT")
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 9),
                    ("FONTSIZE", (0, 1), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFA")]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(tbl)
        story.append(Spacer(1, 10))

    def add_bullets(section_title: str, lines: list[str]) -> None:
        story.append(Paragraph(section_title, styles["Heading2"]))
        story.append(Spacer(1, 4))
        if not lines:
            story.append(Paragraph("Not available.", styles["Normal"]))
        for line in lines:
            story.append(Paragraph(f"- {line}", normal_style))
            story.append(Spacer(1, 2))
        story.append(Spacer(1, 8))

    story.append(Paragraph("IntelligenceOps Synthetic Decision Pack", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Public demo using synthetic flight operations data.", styles["Heading2"]))
    story.append(Spacer(1, 8))
    cover_lines = [
        f"Scope: {PUBLIC_SCOPE_LABEL}",
        f"Selection: {grain} | {period_selected} | {station_selected}",
        datetime.now().strftime("Generated: %Y-%m-%d %H:%M:%S"),
        "Synthetic note: station codes, delay minutes, OTP figures, trends, and outputs are computer-generated for demonstration.",
        "No employer data, real station figures, internal context, or company identity is present.",
    ]
    for line in cover_lines:
        story.append(Paragraph(line, normal_style))
    story.append(Spacer(1, 12))

    add_table_from_df("Executive Summary", executive_matrix_df)
    add_bullets("Leadership Framing", leadership_framing)
    add_table_from_df("Action Lanes", action_lanes_df)

    if otp_chart_png is not None:
        story.append(Paragraph("OTP D15 Trend", styles["Heading2"]))
        story.append(Spacer(1, 4))
        from io import BytesIO as _BytesIO
        otp_img = Image(_BytesIO(otp_chart_png), width=14 * cm, height=6.2 * cm)
        story.append(otp_img)
        story.append(Spacer(1, 8))

    add_table_from_df("Ground Ops Impact Stations", ground_ops_stations_df)
    add_table_from_df("Delay Context Indicators", delay_context_df)
    add_table_from_df("Network KPI Summary", network_summary_df)
    add_table_from_df("Station Ranking Snapshot", station_snapshot_df)

    if pareto_chart_png is not None:
        story.append(Paragraph("DelayCategory Pareto", styles["Heading2"]))
        story.append(Spacer(1, 6))
        from io import BytesIO as _BytesIO
        pareto_img = Image(_BytesIO(pareto_chart_png), width=14 * cm, height=7.3 * cm)
        story.append(pareto_img)
        story.append(Spacer(1, 8))

    add_bullets("Confidence / Caveats", caveats)

    if include_appendix:
        story.append(PageBreak())
        story.append(Paragraph("Sanitized Technical Appendix", styles["Heading1"]))
        story.append(Paragraph("Appendix contains public-safe contract status only. Local paths and raw run context are intentionally excluded.", styles["Normal"]))
        story.append(Spacer(1, 8))
        appendix_pdf = df_for_pdf(appendix_df, ["Item", "Status", "Detail"], max_rows=20)
        add_table_from_df("Demo Data Contract", appendix_pdf)

    doc.build(story)
    return buf.getvalue()


def export_decision_pack_pdf(
    ctx: dict[str, Any],
    grain: str,
    period: str,
    station: str,
    out_dir: Path,
    include_snapshot: bool = True,
    include_ranking: bool = True,
    include_drivers: bool = True,
    include_qa: bool = False,
) -> Path:
    if not isinstance(ctx, dict):
        raise ValueError("Invalid ctx: expected dict.")

    region = str(ctx.get("region", "")).strip()
    mode = str(ctx.get("mode", "")).strip()
    required_files = ensure_list(ctx.get("required_files"))
    insights_dir = Path(ctx.get("insights_dir", default_insights_dir()))
    qa_summary_path = ctx.get("qa_path")
    artifacts_by_name = ctx.get("artifacts_by_name", {}) or {}

    if not region or not mode:
        raise RuntimeError("Demo data context missing scope metadata.")
    if not required_files:
        raise RuntimeError("Demo data contract registry is missing/empty.")

    station_kpi_name = f"2025_DEP_Monthly_Station_KPIs__{region}.csv" if grain == "Monthly" else f"2025_DEP_Weekly_Station_KPIs__{region}.csv"
    delay_minutes_name = f"2025_DEP_DelayCategory_Minutes__{region}_NORM__MONTHLY.csv" if grain == "Monthly" else f"2025_DEP_DelayCategory_Minutes__{region}_NORM__WEEKLY.csv"
    top_codes_name = f"2025_DEP_TopDelayCodes__{region}_NORM.csv"
    _, station_kpi_path = resolve_required_file(required_files, station_kpi_name, insights_dir, artifacts_by_name)
    _, delay_minutes_path = resolve_required_file(required_files, delay_minutes_name, insights_dir, artifacts_by_name)
    _, top_codes_path = resolve_required_file(required_files, top_codes_name, insights_dir, artifacts_by_name)

    if not station_kpi_path.exists():
        raise FileNotFoundError(f"Required Station KPI file missing: {station_kpi_path}")

    station_kpi_df = load_csv(station_kpi_path)
    if station_kpi_df is None:
        raise RuntimeError(f"Could not load Station KPI file: {station_kpi_path}")
    runtime_data_log(f"Grain={grain} StationKpisFile={station_kpi_path} Rows={len(station_kpi_df)} Cols={len(station_kpi_df.columns)}")

    if delay_minutes_path.exists():
        delay_minutes_df = load_csv(delay_minutes_path)
        if delay_minutes_df is None:
            runtime_data_log(f"WARNING: DelayCategoryMinutesFile could not be loaded: {delay_minutes_path}")
            delay_minutes_df = pd.DataFrame()
        else:
            runtime_data_log(f"Grain={grain} DelayCategoryMinutesFile={delay_minutes_path} Rows={len(delay_minutes_df)} Cols={len(delay_minutes_df.columns)}")
    else:
        runtime_data_log(f"WARNING: Periodized DelayCategory file not found for grain={grain}: {delay_minutes_path} — drivers section will be empty.")
        delay_minutes_df = pd.DataFrame()

    top_codes_df = load_csv(top_codes_path) if top_codes_path.exists() else pd.DataFrame()

    kpi_work, kpi_mapping, kpi_err = build_station_kpi_work(station_kpi_df, grain)
    if kpi_err is not None:
        raise RuntimeError(kpi_err)

    delay_work, delay_mapping, delay_err = build_delay_minutes_work(delay_minutes_df, grain)
    if delay_err is not None:
        runtime_data_log(f"DelayMinutesWarning={delay_err}")

    active_periods_df = (
        kpi_work[kpi_work[kpi_mapping["flights"]].fillna(0) > 0]
        .groupby(["_period", "_sort"], as_index=False)[kpi_mapping["flights"]]
        .sum()
        .sort_values("_sort")
    )
    period_options = active_periods_df["_period"].astype(str).tolist()
    if not period_options:
        raise RuntimeError("No active periods found in Station KPI file (Flights_Operated > 0).")

    if str(period).strip().lower() == "latest":
        period_selected = period_options[-1]
    else:
        parsed_label, _ = parse_period(period, grain)
        candidate = parsed_label or str(period).strip()
        if candidate not in period_options:
            raise ValueError(f"Requested period not found in active periods: {period}")
        period_selected = candidate

    period_slice = kpi_work[kpi_work["_period"] == period_selected].copy()
    active_station_slice = period_slice[period_slice[kpi_mapping["flights"]].fillna(0) > 0].copy()
    station_values = sorted(active_station_slice[kpi_mapping["station"]].astype(str).dropna().unique().tolist())

    if str(station).strip().upper() in {"ALL", "NETWORK", ""}:
        station_selected = "NETWORK"
    else:
        station_selected = None
        for s in station_values:
            if s.upper() == str(station).strip().upper():
                station_selected = s
                break
        if station_selected is None:
            raise ValueError(f"Requested station not found in active stations for period {period_selected}: {station}")

    if station_selected != "NETWORK" and kpi_mapping["station"] is not None:
        station_period_slice = period_slice[period_slice[kpi_mapping["station"]].astype(str) == station_selected].copy()
    else:
        station_period_slice = period_slice.copy()

    snapshot = build_snapshot(period_slice, station_selected, kpi_mapping)
    station_metrics = _station_metric_frame(station_period_slice, kpi_mapping)
    if not delay_work.empty:
        drivers_df, _drivers_scope_note = build_drivers_summary(delay_work, period_selected, station_selected, delay_mapping)
        accountability_df = build_accountability_split(delay_work, period_selected, station_selected, delay_mapping)
    else:
        drivers_df = pd.DataFrame()
        accountability_df = pd.DataFrame(columns=["Bucket", "Minutes", "SharePct"])

    qa_df = load_csv(qa_summary_path) if (include_qa and qa_summary_path is not None and qa_summary_path.exists()) else None
    appendix_df = build_sanitized_appendix(required_files, station_kpi_df, delay_minutes_df, qa_df)
    top_code_lookup = build_top_code_lookup(top_codes_df)
    action_focus = "Review the highest-impact synthetic station using selected-scope Ground Ops and delay-category evidence."
    executive_matrix_df = build_executive_summary_matrix(snapshot, station_metrics, action_focus)
    leadership_framing = build_leadership_framing(snapshot, station_metrics, accountability_df, action_focus)
    action_lanes_df = build_action_lanes(station_metrics, accountability_df, top_code_lookup)
    ground_ops_stations_df = build_ground_ops_impact_stations(station_metrics)
    delay_context_df = build_delay_context_indicators(accountability_df)
    network_summary_df = build_network_kpi_summary(snapshot, station_metrics)
    station_snapshot_df = build_station_ranking_snapshot(station_metrics)
    caveats = build_confidence_caveats(include_qa)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_name = PUBLIC_PDF_FILENAME
    pdf_path = out_dir / pdf_name
    latest_pdf_path = out_dir / "DecisionPack__LATEST.pdf"
    # Build chart PNGs (graceful: None if kaleido unavailable or data insufficient)
    otp_chart_png = build_otp_trend_png(kpi_work, kpi_mapping, grain)
    pareto_chart_png = build_pareto_png(drivers_df) if include_drivers else None

    pdf_bytes = build_pdf_bytes(
        title="IntelligenceOps Synthetic Decision Pack",
        grain=grain,
        period_selected=period_selected,
        station_selected=station_selected,
        executive_matrix_df=executive_matrix_df,
        leadership_framing=leadership_framing,
        action_lanes_df=action_lanes_df,
        ground_ops_stations_df=ground_ops_stations_df,
        delay_context_df=delay_context_df,
        network_summary_df=network_summary_df,
        station_snapshot_df=station_snapshot_df,
        appendix_df=appendix_df,
        caveats=caveats,
        include_appendix=include_qa,
        otp_chart_png=otp_chart_png,
        pareto_chart_png=pareto_chart_png,
    )

    pdf_path.write_bytes(pdf_bytes)
    latest_pdf_path.write_bytes(pdf_bytes)
    return pdf_path


def write_text_log(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def is_streamlit_runtime() -> bool:
    return HAS_STREAMLIT_RUNTIME


def parse_flag_int(v: Any) -> bool:
    try:
        return int(v) != 0
    except Exception:
        return False


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Public-safe synthetic decision pack exporter")
    parser.add_argument("--region", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--grain", required=True, choices=["Monthly", "Weekly"])
    parser.add_argument("--period", required=True, help="latest or explicit period label")
    parser.add_argument("--station", required=True, help="ALL or station code")
    parser.add_argument("--include_snapshot", default="1")
    parser.add_argument("--include_ranking", default="1")
    parser.add_argument("--include_drivers", default="1")
    parser.add_argument("--include_qa", default="0")
    args = parser.parse_args(argv)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_logs_dir = REPO_ROOT / "artifacts" / "run_logs"
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_logs_dir / f"d7_decision_pack_pdf__{ts}.txt"

    try:
        region = str(args.region).strip()
        mode = str(args.mode).strip()
        grain = str(args.grain).strip()
        period_arg = str(args.period).strip()
        station_arg = str(args.station).strip()

        include_snapshot = parse_flag_int(args.include_snapshot)
        include_ranking = parse_flag_int(args.include_ranking)
        include_drivers = parse_flag_int(args.include_drivers)
        include_qa = parse_flag_int(args.include_qa)

        cli_ctx = get_run_context_for_region_mode(default_insights_dir(), region, mode)
        pdf_path = export_decision_pack_pdf(
            ctx=cli_ctx,
            grain=grain,
            period=period_arg,
            station=station_arg,
            out_dir=REPO_ROOT / "artifacts" / "prework_out",
            include_snapshot=include_snapshot,
            include_ranking=include_ranking,
            include_drivers=include_drivers,
            include_qa=include_qa,
        )
        latest_pdf_path = (REPO_ROOT / "artifacts" / "prework_out" / "DecisionPack__LATEST.pdf")

        verdict = "D7_PDF=PASS SyntheticDecisionPackGenerated=1"
        write_text_log(
            log_path,
            [
                f"Timestamp={datetime.now().isoformat(timespec='seconds')}",
                f"Scope={PUBLIC_SCOPE_LABEL}",
                f"Grain={grain}",
                f"Period={period_arg}",
                f"Station={station_arg}",
                verdict,
            ],
        )
        print(verdict)
        return 0
    except Exception as e:
        err = re.sub(r"\s+", " ", f"{type(e).__name__}: {e}").strip()
        verdict = f"D7_PDF=FAIL Error={err}"
        try:
            write_text_log(
                log_path,
                [
                    f"Timestamp={datetime.now().isoformat(timespec='seconds')}",
                    f"Scope={PUBLIC_SCOPE_LABEL}",
                    f"Grain={getattr(args, 'grain', '')}",
                    f"Period={getattr(args, 'period', '')}",
                    f"Station={getattr(args, 'station', '')}",
                    verdict,
                ],
            )
        except Exception:
            pass
        print(verdict)
        return 1


if __name__ == "__main__" and not is_streamlit_runtime():
    sys.exit(cli_main())


if is_streamlit_runtime():
    st.set_page_config(page_title="Synthetic Decision Pack", layout="wide")

    st.title("Synthetic Decision Pack")
    st.caption(
        "Public demo PDF using synthetic flight operations artifacts. "
        "No employer data, real station figures, internal context, or company identity is present."
    )

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

    region = str(ctx.get("region", "")).strip()
    mode = str(ctx.get("mode", "")).strip()
    required_files = ensure_list(ctx.get("required_files"))
    insights_dir = Path(ctx.get("insights_dir", default_insights_dir()))

    if not region or not mode:
        st.error("Demo data context is unavailable.")
        st.stop()

    with st.expander("Demo Data Contract", expanded=False):
        contract_status = "available" if required_files else "not available"
        st.write(f"Scope: {PUBLIC_SCOPE_LABEL}")
        st.write(f"Contract: {contract_status}")
        st.write(f"Registered artifacts: {len(required_files)}")

    if not required_files:
        st.warning("Demo data contract registry is unavailable; generation may fail contract checks.")

    grain = str(filters.get("grain", "Monthly"))
    periods_from_filters = [str(p) for p in filters.get("periods", []) if str(p).strip()]
    stations_from_filters = [str(s) for s in filters.get("stations", []) if str(s).strip()]
    period_summary = periods_from_filters[0] if len(periods_from_filters) == 1 else (f"{periods_from_filters[0]}..{periods_from_filters[-1]} ({len(periods_from_filters)})" if periods_from_filters else "<auto>")
    station_summary = "NETWORK" if stations_from_filters == ["NETWORK"] else f"{len(stations_from_filters)} selected"
    st.caption(f"Filters: Grain={grain} | Periods={period_summary} | Stations={station_summary}")

    st.caption("Leadership sections are included by default: executive summary, action lanes, station impact, delay context, KPI summary, and caveats.")
    include_snapshot = True
    include_ranking = True
    include_drivers = True
    include_qa = bool(st.toggle("Include Technical Appendix", value=False))

    station_kpi_name = (
        f"2025_DEP_Monthly_Station_KPIs__{region}.csv"
        if grain == "Monthly"
        else f"2025_DEP_Weekly_Station_KPIs__{region}.csv"
    )
    artifacts_by_name = ctx.get("artifacts_by_name", {}) or {}
    _, station_kpi_path = resolve_required_file(required_files, station_kpi_name, insights_dir, artifacts_by_name)
    if not station_kpi_path.exists():
        st.error("Required synthetic Station KPI artifact is missing.")
        st.stop()

    station_kpi_df = load_csv(station_kpi_path)
    if station_kpi_df is None:
        st.error("Could not load the synthetic Station KPI artifact.")
        st.stop()
    runtime_data_log(
        f"Grain={grain} StationKpisFile={station_kpi_path} Rows={len(station_kpi_df)} Cols={len(station_kpi_df.columns)}"
    )

    kpi_work, kpi_mapping, kpi_err = build_station_kpi_work(station_kpi_df, grain)
    if kpi_err is not None:
        st.error(kpi_err)
        st.stop()

    active_periods_df = (
        kpi_work[kpi_work[kpi_mapping["flights"]].fillna(0) > 0]
        .groupby(["_period", "_sort"], as_index=False)[kpi_mapping["flights"]]
        .sum()
        .sort_values("_sort")
    )
    period_options = active_periods_df["_period"].astype(str).tolist()
    if not period_options:
        st.error("No active periods found in Station KPI file (Flights_Operated > 0).")
        st.stop()

    selected_periods = [p for p in periods_from_filters if p in period_options]
    if not selected_periods:
        selected_periods = [period_options[-1]]
    period_selected = selected_periods[-1]
    period_slice = apply_filters(kpi_work, filters, station_col=kpi_mapping["station"], period_col="_period")
    period_slice = period_slice[period_slice[kpi_mapping["flights"]].fillna(0) > 0].copy()
    station_selected = "NETWORK"
    if stations_from_filters and stations_from_filters != ["NETWORK"] and len(stations_from_filters) == 1:
        station_selected = stations_from_filters[0]

    snapshot = build_snapshot(period_slice, station_selected, kpi_mapping)
    st.subheader("Preview")
    pv1, pv2, pv3, pv4 = st.columns(4)
    pv1.metric("Flights", fmt_int(snapshot.get("Flights")))
    pv2.metric("Total DEP Delay Minutes", fmt_int(snapshot.get("TotalMinutes")))
    pv3.metric("Avg DEP Delay per Flight (min)", fmt_float(snapshot.get("AvgDelay"), 2))
    pv4.metric("DEP OTP D15 (%)", fmt_float(snapshot.get("OTP"), 2))
    st.caption(f"Avg basis: {snapshot.get('AvgDelayBasis')} | OTP basis: {snapshot.get('OTPBasis')}")

    out_dir = REPO_ROOT / "artifacts" / "prework_out"
    run_logs_dir = REPO_ROOT / "artifacts" / "run_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_logs_dir.mkdir(parents=True, exist_ok=True)

    if "decision_pack_pdf_bytes" not in st.session_state:
        st.session_state["decision_pack_pdf_bytes"] = None
    if "decision_pack_pdf_name" not in st.session_state:
        st.session_state["decision_pack_pdf_name"] = None

    if st.button("Generate PDF", type="primary"):
        try:
            pdf_path = export_decision_pack_pdf(
                ctx={**ctx, "mode_a_filters": filters},
                grain=grain,
                period=period_selected,
                station=station_selected,
                out_dir=out_dir,
                include_snapshot=include_snapshot,
                include_ranking=include_ranking,
                include_drivers=include_drivers,
                include_qa=include_qa,
            )
            latest_pdf_path = out_dir / "DecisionPack__LATEST.pdf"
            pdf_bytes = pdf_path.read_bytes()

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = run_logs_dir / f"decision_pack_pdf__{ts}.txt"
            write_text_log(
                log_path,
                [
                    f"Timestamp={datetime.now().isoformat(timespec='seconds')}",
                    f"Scope={PUBLIC_SCOPE_LABEL}",
                    f"Grain={grain}",
                    f"Period={period_selected}",
                    f"Station={station_selected}",
                ],
            )

            st.session_state["decision_pack_pdf_bytes"] = pdf_bytes
            st.session_state["decision_pack_pdf_name"] = PUBLIC_PDF_FILENAME
            st.success("Synthetic decision pack generated. Use the download button below.")
        except Exception as e:
            st.error(f"PDF generation failed: {type(e).__name__}: {e}")

    if st.session_state.get("decision_pack_pdf_bytes") is not None:
        st.download_button(
            label="Download PDF",
            data=st.session_state["decision_pack_pdf_bytes"],
            file_name=st.session_state.get("decision_pack_pdf_name") or PUBLIC_PDF_FILENAME,
            mime="application/pdf",
        )
