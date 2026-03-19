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


def default_insights_dir() -> Path:
    return Path(os.getenv("FLIGHTOPS_INSIGHTS_DIR", r"C:\Users\IT\02_insights\insight_out"))


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
    provenance_lines: list[str],
    input_lines: list[str],
    snapshot: dict[str, Any],
    worst_df: pd.DataFrame,
    best_df: pd.DataFrame,
    drivers_df: pd.DataFrame,
    drivers_scope_note: str,
    qa_excerpt_df: pd.DataFrame,
    include_snapshot: bool,
    include_ranking: bool,
    include_drivers: bool,
    include_qa: bool,
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
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=28, rightMargin=1.5*cm, topMargin=28, bottomMargin=28, title=title)
    story: list[Any] = []

    def add_table_from_df(section_title: str, tdf: pd.DataFrame) -> None:
        story.append(Paragraph(section_title, styles["Heading2"]))
        story.append(Spacer(1, 6))
        if tdf.empty:
            story.append(Paragraph("Not available.", styles["Normal"]))
            story.append(Spacer(1, 10))
            return
        display = tdf.copy().fillna("not available")
        for c in display.columns:
            display[c] = display[c].map(lambda x: str(x))
        data = [list(display.columns)] + display.values.tolist()
        tbl = Table(data, repeatRows=1)
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
                ]
            )
        )
        story.append(tbl)
        story.append(Spacer(1, 10))

    story.append(Paragraph("Decision Pack", styles["Title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(datetime.now().strftime("Generated: %Y-%m-%d %H:%M:%S"), styles["Normal"]))
    story.append(Spacer(1, 8))
    for line in provenance_lines:
        story.append(Paragraph(line, styles["Normal"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Inputs", styles["Heading2"]))
    for line in input_lines:
        story.append(Paragraph(line, styles["Normal"]))
    story.append(PageBreak())

    if otp_chart_png is not None:
        story.append(Paragraph("OTP D15 Trend", styles["Heading2"]))
        story.append(Spacer(1, 4))
        from io import BytesIO as _BytesIO
        otp_img = Image(_BytesIO(otp_chart_png), width=14 * cm, height=6.2 * cm)
        story.append(otp_img)
        story.append(Spacer(1, 8))

    if include_snapshot:
        snap_df = pd.DataFrame(
            [
                {
                    "Flights": fmt_int(snapshot.get("Flights")),
                    "Total DEP Delay Minutes": fmt_int(snapshot.get("TotalMinutes")),
                    "Avg DEP Delay per Flight (min)": fmt_float(snapshot.get("AvgDelay"), 2),
                    "DEP OTP D15 (%)": fmt_float(snapshot.get("OTP"), 2),
                    "Avg Basis": snapshot.get("AvgDelayBasis"),
                    "OTP Basis": snapshot.get("OTPBasis"),
                }
            ]
        )
        add_table_from_df("B) Snapshot", snap_df)
    if include_ranking:
        worst_pdf = df_for_pdf(worst_df, ["Station", "AvgDelayMin", "Flights", "TotalMinutes"], max_rows=10).copy()
        best_pdf = df_for_pdf(best_df, ["Station", "AvgDelayMin", "Flights", "TotalMinutes"], max_rows=10).copy()
        if not worst_pdf.empty:
            worst_pdf["AvgDelayMin"] = worst_pdf["AvgDelayMin"].map(lambda x: fmt_float(x, 2))
            worst_pdf["Flights"] = worst_pdf["Flights"].map(fmt_int)
            worst_pdf["TotalMinutes"] = worst_pdf["TotalMinutes"].map(fmt_int)
        if not best_pdf.empty:
            best_pdf["AvgDelayMin"] = best_pdf["AvgDelayMin"].map(lambda x: fmt_float(x, 2))
            best_pdf["Flights"] = best_pdf["Flights"].map(fmt_int)
            best_pdf["TotalMinutes"] = best_pdf["TotalMinutes"].map(fmt_int)
        add_table_from_df("C1) Station Ranking - Worst 10", worst_pdf)
        add_table_from_df("C2) Station Ranking - Best 10", best_pdf)
    if include_drivers:
        story.append(Paragraph("D) Drivers Summary", styles["Heading2"]))
        story.append(Paragraph(f"Scope note: {drivers_scope_note}", styles["Normal"]))
        story.append(Spacer(1, 6))
        drv_pdf = df_for_pdf(drivers_df, ["DelayCategory", "Minutes", "SharePct"], max_rows=10).copy()
        if not drv_pdf.empty:
            drv_pdf["Minutes"] = drv_pdf["Minutes"].map(fmt_int)
            drv_pdf["SharePct"] = drv_pdf["SharePct"].map(lambda x: fmt_float(x, 2))
        add_table_from_df("Top 10 Delay Categories", drv_pdf)
    if pareto_chart_png is not None:
        story.append(Spacer(1, 6))
        from io import BytesIO as _BytesIO
        pareto_img = Image(_BytesIO(pareto_chart_png), width=14 * cm, height=7.3 * cm)
        story.append(pareto_img)
        story.append(Spacer(1, 8))
    if include_qa:
        qa_pdf = df_for_pdf(
            qa_excerpt_df,
            ["FileName", "Exists", "SizeBytes", "Rows", "Cols", "MinYearMonth", "MaxYearMonth", "MinYearWeek", "MaxYearWeek"],
            max_rows=20,
        )
        add_table_from_df("E) Coverage / QA Excerpt", qa_pdf)

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
    include_qa: bool = True,
) -> Path:
    if not isinstance(ctx, dict):
        raise ValueError("Invalid ctx: expected dict.")

    run_stamp = ctx.get("run_stamp") if isinstance(ctx.get("run_stamp"), dict) else {}
    region = str(ctx.get("region", "")).strip()
    mode = str(ctx.get("mode", "")).strip()
    pass_stamp = ctx.get("stamp")
    git_commit = ctx.get("git_commit")
    git_branch = ctx.get("git_branch")
    required_files = ensure_list(ctx.get("required_files"))
    insights_dir = Path(ctx.get("insights_dir", default_insights_dir()))
    qa_summary_path = ctx.get("qa_path")
    artifacts_by_name = ctx.get("artifacts_by_name", {}) or {}

    if not region or not mode:
        raise RuntimeError("Run context missing region/mode.")
    if not required_files:
        raise RuntimeError("run_stamp.required_files is missing/empty.")

    station_kpi_name = f"2025_DEP_Monthly_Station_KPIs__{region}.csv" if grain == "Monthly" else f"2025_DEP_Weekly_Station_KPIs__{region}.csv"
    delay_minutes_name = f"2025_DEP_DelayCategory_Minutes__{region}_NORM__MONTHLY.csv" if grain == "Monthly" else f"2025_DEP_DelayCategory_Minutes__{region}_NORM__WEEKLY.csv"
    _, station_kpi_path = resolve_required_file(required_files, station_kpi_name, insights_dir, artifacts_by_name)
    _, delay_minutes_path = resolve_required_file(required_files, delay_minutes_name, insights_dir, artifacts_by_name)

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

    snapshot = build_snapshot(period_slice, station_selected, kpi_mapping)
    worst_df, best_df = build_station_ranking(period_slice, kpi_mapping)
    coverage_flights = compute_coverage_flights_operated(station_kpi_df, grain, period_selected, station_selected)
    if not delay_work.empty:
        drivers_df, drivers_scope_note = build_drivers_summary(delay_work, period_selected, station_selected, delay_mapping)
    else:
        drivers_df, drivers_scope_note = pd.DataFrame(), "delay minutes source unavailable"

    qa_df = load_csv(qa_summary_path) if (qa_summary_path is not None and qa_summary_path.exists()) else None
    qa_excerpt_df = build_qa_excerpt(qa_df, required_files, insights_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_name = (
        f"DecisionPack__{sanitize_token(region)}__{sanitize_token(mode)}__{sanitize_token(pass_stamp, 'nostamp')}"
        f"__{sanitize_token(grain)}__{sanitize_token(period_selected)}__{sanitize_token(station_selected)}.pdf"
    )
    pdf_path = out_dir / pdf_name
    latest_pdf_path = out_dir / "DecisionPack__LATEST.pdf"

    ops_gate_pack_verdict = maybe_get(run_stamp, "ops_gate_pack_verdict", "OpsGatePackVerdict")
    ops_gate_pack_log_path = maybe_get(run_stamp, "ops_gate_pack_log_path", "OpsGatePackLogPath")
    b1_log_path = maybe_get(run_stamp, "b1_refresh_publish_log_path", "B1RefreshPublishLogPath", "b1_log_path")
    filters_meta = ctx.get("mode_a_filters", {}) if isinstance(ctx.get("mode_a_filters"), dict) else {}
    f_grain = str(filters_meta.get("grain", grain))
    f_periods = [str(p) for p in filters_meta.get("periods", []) if str(p).strip()]
    f_stations = [str(s) for s in filters_meta.get("stations", []) if str(s).strip()]
    f_period_summary = f_periods[0] if len(f_periods) == 1 else (f"{f_periods[0]}..{f_periods[-1]} ({len(f_periods)})" if f_periods else str(period_selected))
    f_station_mode = str(filters_meta.get("station_mode", "NETWORK" if f_stations == ["NETWORK"] else "Multi"))
    f_station_count = 1 if f_stations == ["NETWORK"] else len(f_stations)
    provenance_pdf = [
        f"run_stamp stamp: {pass_stamp}",
        f"region/mode: {region}/{mode}",
        f"git_branch: {git_branch}",
        f"git_commit: {git_commit}",
        f"ops_gate_pack_verdict: {ops_gate_pack_verdict if ops_gate_pack_verdict else 'not available'}",
        f"selection: grain={grain} period={period_selected} station={station_selected}",
        f"Coverage: Flights Operated: {coverage_flights:,}",
        f"EvidenceRefs: ops_gate_pack_log_path: {ops_gate_pack_log_path if ops_gate_pack_log_path else 'not available'}",
        f"EvidenceRefs: ops_gate_pack_verdict: {ops_gate_pack_verdict if ops_gate_pack_verdict else 'not available'}",
        f"EvidenceRefs: qa_summary_path: {qa_summary_path if qa_summary_path else 'not available'}",
        f"EvidenceRefs: b1_refresh_publish_log_path: {b1_log_path if b1_log_path else 'not available'}",
        f"GlobalFilters: grain={f_grain}",
        f"GlobalFilters: periods={f_period_summary}",
        f"GlobalFilters: stations_mode={f_station_mode} count={f_station_count}",
    ]
    inputs_pdf = [
        f"station_kpi_file: {station_kpi_path}",
        f"delaycategory_minutes_file: {delay_minutes_path}",
        f"qa_summary_path: {qa_summary_path}",
    ]
    # Build chart PNGs (graceful: None if kaleido unavailable or data insufficient)
    otp_chart_png = build_otp_trend_png(kpi_work, kpi_mapping, grain)
    pareto_chart_png = build_pareto_png(drivers_df) if include_drivers else None

    pdf_bytes = build_pdf_bytes(
        title=pdf_name,
        provenance_lines=provenance_pdf,
        input_lines=inputs_pdf,
        snapshot=snapshot,
        worst_df=worst_df,
        best_df=best_df,
        drivers_df=drivers_df,
        drivers_scope_note=drivers_scope_note,
        qa_excerpt_df=qa_excerpt_df,
        include_snapshot=include_snapshot,
        include_ranking=include_ranking,
        include_drivers=include_drivers,
        include_qa=include_qa,
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
    parser = argparse.ArgumentParser(description="D7 PDF Decision Pack headless exporter")
    parser.add_argument("--region", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--grain", required=True, choices=["Monthly", "Weekly"])
    parser.add_argument("--period", required=True, help="latest or explicit period label")
    parser.add_argument("--station", required=True, help="ALL or station code")
    parser.add_argument("--include_snapshot", default="1")
    parser.add_argument("--include_ranking", default="1")
    parser.add_argument("--include_drivers", default="1")
    parser.add_argument("--include_qa", default="1")
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

        verdict = f"D7_PDF=PASS OutputPdf={pdf_path} OutputLatestPdf={latest_pdf_path} Log={log_path}"
        write_text_log(
            log_path,
            [
                f"Timestamp={datetime.now().isoformat(timespec='seconds')}",
                f"RunContextPath={cli_ctx.get('source_path')}",
                f"RunStamp={cli_ctx.get('stamp')}",
                f"GitCommit={cli_ctx.get('git_commit')}",
                f"Region={region}",
                f"Mode={mode}",
                f"Grain={grain}",
                f"Period={period_arg}",
                f"Station={station_arg}",
                f"OutputPdf={pdf_path}",
                f"OutputLatestPdf={latest_pdf_path}",
                verdict,
            ],
        )
        print(verdict)
        return 0
    except Exception as e:
        err = re.sub(r"\s+", " ", f"{type(e).__name__}: {e}").strip()
        verdict = f"D7_PDF=FAIL Error={err} Log={log_path}"
        try:
            write_text_log(
                log_path,
                [
                    f"Timestamp={datetime.now().isoformat(timespec='seconds')}",
                    f"Region={getattr(args, 'region', '')}",
                    f"Mode={getattr(args, 'mode', '')}",
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
    st.set_page_config(page_title="PDF Decision Pack Export", layout="wide")

    st.title("PDF Decision Pack Export")
    st.caption("PASS-only PDF. Uses run_stamp + published CSVs. No DB calls.")

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

    run_stamp = ctx.get("run_stamp") if isinstance(ctx.get("run_stamp"), dict) else {}
    ctx_source_path = ctx.get("source_path", "")
    region = str(ctx.get("region", "")).strip()
    mode = str(ctx.get("mode", "")).strip()
    pass_stamp = ctx.get("stamp")
    git_commit = ctx.get("git_commit")
    git_branch = ctx.get("git_branch")
    required_files = ensure_list(ctx.get("required_files"))
    ops_gate_pack_log_path = maybe_get(run_stamp, "ops_gate_pack_log_path", "OpsGatePackLogPath")
    ops_gate_pack_stamp = maybe_get(run_stamp, "ops_gate_pack_stamp", "OpsGatePackStamp")
    insights_dir = Path(ctx.get("insights_dir", default_insights_dir()))
    qa_summary_path = ctx.get("qa_path")

    if not region or not mode:
        st.error("Run context is missing region/mode.")
        st.stop()

    with st.expander("📋 Pipeline Parameters", expanded=False):
        prov_lines = [
            f"run_context_path={ctx_source_path}",
            f"stamp={pass_stamp}",
            f"region={region}",
            f"mode={mode}",
            f"git_branch={git_branch}",
            f"git_commit={git_commit}",
            f"insights_dir={insights_dir}",
            f"qa_summary_path={qa_summary_path}",
            f"ops_gate_pack_stamp={ops_gate_pack_stamp}",
            f"ops_gate_pack_log_path={ops_gate_pack_log_path}",
        ]
        st.code("\n".join([str(x) for x in prov_lines]), language="text")

    if not required_files:
        st.warning("run_stamp.required_files is missing/empty; generation may fail contract checks.")

    grain = str(filters.get("grain", "Monthly"))
    periods_from_filters = [str(p) for p in filters.get("periods", []) if str(p).strip()]
    stations_from_filters = [str(s) for s in filters.get("stations", []) if str(s).strip()]
    period_summary = periods_from_filters[0] if len(periods_from_filters) == 1 else (f"{periods_from_filters[0]}..{periods_from_filters[-1]} ({len(periods_from_filters)})" if periods_from_filters else "<auto>")
    station_summary = "NETWORK" if stations_from_filters == ["NETWORK"] else f"{len(stations_from_filters)} selected"
    st.caption(f"Filters: Grain={grain} | Periods={period_summary} | Stations={station_summary}")

    t1, t2, t3, t4 = st.columns(4)
    include_snapshot = bool(t1.toggle("Include Network Snapshot", value=True))
    include_ranking = bool(t2.toggle("Include Station Ranking", value=True))
    include_drivers = bool(t3.toggle("Include Drivers Summary", value=True))
    include_qa = bool(t4.toggle("Include Coverage/QA excerpt", value=True))

    station_kpi_name = (
        f"2025_DEP_Monthly_Station_KPIs__{region}.csv"
        if grain == "Monthly"
        else f"2025_DEP_Weekly_Station_KPIs__{region}.csv"
    )
    artifacts_by_name = ctx.get("artifacts_by_name", {}) or {}
    _, station_kpi_path = resolve_required_file(required_files, station_kpi_name, insights_dir, artifacts_by_name)
    if not station_kpi_path.exists():
        st.error(f"Missing Station KPI file: `{station_kpi_path}`")
        st.stop()

    station_kpi_df = load_csv(station_kpi_path)
    if station_kpi_df is None:
        st.error(f"Could not load Station KPI file: `{station_kpi_path}`")
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
                    f"RunContextPath={ctx_source_path}",
                    f"RunStamp={pass_stamp}",
                    f"GitCommit={git_commit}",
                    f"Region={region}",
                    f"Mode={mode}",
                    f"Grain={grain}",
                    f"Period={period_selected}",
                    f"Station={station_selected}",
                    f"OutputPdf={pdf_path}",
                    f"OutputLatestPdf={latest_pdf_path}",
                ],
            )

            st.session_state["decision_pack_pdf_bytes"] = pdf_bytes
            st.session_state["decision_pack_pdf_name"] = pdf_path.name
            st.success(f"PDF generated: `{pdf_path}`\n\nLATEST updated: `{latest_pdf_path}`\n\nRun log: `{log_path}`")
        except Exception as e:
            st.error(f"PDF generation failed: {type(e).__name__}: {e}")

    if st.session_state.get("decision_pack_pdf_bytes") is not None:
        st.download_button(
            label="Download PDF",
            data=st.session_state["decision_pack_pdf_bytes"],
            file_name=st.session_state.get("decision_pack_pdf_name") or "DecisionPack.pdf",
            mime="application/pdf",
        )


