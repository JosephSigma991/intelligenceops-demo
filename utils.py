"""
utils.py — shared utility functions for IntelligenceOps Mode A.
Import from here; do not re-define in page files.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from kpi_config import PLOTLY_CONFIG

# ── String helpers ──────────────────────────────────────────────────────────

def normalize_name(v: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(v).lower())


def ensure_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if str(x).strip()]
    return [str(v)] if str(v).strip() else []


# ── DataFrame column resolution ─────────────────────────────────────────────

def pick_column(
    df: pd.DataFrame,
    candidates: list[str],
    token_groups: list[list[str]] | None = None,
) -> str | None:
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
    if token_groups:
        for c in cols:
            lc = c.lower()
            for toks in token_groups:
                if all(t in lc for t in toks):
                    return c
    return None


# ── Period parsing ───────────────────────────────────────────────────────────

def parse_period(value: Any, grain: str) -> tuple[str | None, pd.Timestamp]:
    if pd.isna(value):
        return None, pd.NaT
    s = str(value).strip()
    if not s:
        return None, pd.NaT

    if grain == "Weekly":
        m = re.match(r"^(\d{4})-?W?(\d{1,2})$", s, flags=re.IGNORECASE)
        if m:
            y, w = int(m.group(1)), int(m.group(2))
            label = f"{y:04d}-W{w:02d}"
            return label, pd.to_datetime(f"{label}-1", format="%G-W%V-%u", errors="coerce")
        dt = pd.to_datetime(s, errors="coerce")
        if pd.notna(dt):
            iso = dt.isocalendar()
            label = f"{int(iso.year):04d}-W{int(iso.week):02d}"
            return label, pd.to_datetime(f"{label}-1", format="%G-W%V-%u", errors="coerce")
        return None, pd.NaT

    m = re.match(r"^(\d{4})[-/ ]?(\d{1,2})$", s)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        label = f"{y:04d}-{mo:02d}"
        return label, pd.to_datetime(f"{label}-01", format="%Y-%m-%d", errors="coerce")
    dt = pd.to_datetime(s[:10], errors="coerce")
    if pd.notna(dt):
        label = f"{int(dt.year):04d}-{int(dt.month):02d}"
        return label, pd.to_datetime(f"{label}-01", format="%Y-%m-%d", errors="coerce")
    return None, pd.NaT


# ── Weighted aggregation ─────────────────────────────────────────────────────

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


# ── OTP regression model ─────────────────────────────────────────────────────

def fit_otp_model(
    avg_delay: pd.Series,
    otp: pd.Series,
    flights: pd.Series,
    min_obs: int = 6,
    flights_floor: int = 5,
) -> dict[str, float] | None:
    """Fit a weighted linear regression: OTP = intercept + slope × AvgDelay.

    Returns dict with {intercept, slope, r2, n_obs, delay_min, delay_max}
    or None if insufficient data.

    Uses numpy polyfit with sqrt(flights) weighting — no new dependencies.
    """
    import numpy as np

    x = pd.to_numeric(avg_delay, errors="coerce")
    y = pd.to_numeric(otp, errors="coerce")
    w = pd.to_numeric(flights, errors="coerce")

    mask = x.notna() & y.notna() & w.notna() & (w >= flights_floor)
    x_clean = x[mask].values
    y_clean = y[mask].values
    w_clean = w[mask].values

    if len(x_clean) < min_obs:
        return None

    # Weighted OLS via numpy polyfit (degree=1)
    # polyfit accepts weights as 1/sigma; for flight-weighting use sqrt(flights)
    weights = np.sqrt(w_clean)
    coeffs = np.polyfit(x_clean, y_clean, deg=1, w=weights)
    slope, intercept = float(coeffs[0]), float(coeffs[1])

    # Weighted R²
    y_pred = intercept + slope * x_clean
    ss_res = float(np.sum(w_clean * (y_clean - y_pred) ** 2))
    y_wmean = float(np.sum(w_clean * y_clean) / np.sum(w_clean))
    ss_tot = float(np.sum(w_clean * (y_clean - y_wmean) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "intercept": intercept,
        "slope": slope,
        "r2": max(0.0, r2),
        "n_obs": int(len(x_clean)),
        "delay_min": float(x_clean.min()),
        "delay_max": float(x_clean.max()),
    }


# ── Dict helper ──────────────────────────────────────────────────────────────

def maybe_get(d: dict[str, Any], *keys: str) -> Any:
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d:
            return d[k]
    lowered = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        if str(k).lower() in lowered:
            return lowered[str(k).lower()]
    return None


# ── Logging ──────────────────────────────────────────────────────────────────

def runtime_data_log(msg: str) -> None:
    print(f"[RUNTIME_DATA] {msg}", flush=True)


# ── QA path resolution ───────────────────────────────────────────────────────

def resolve_qa_path(run_stamp: dict[str, Any], insights_dir: Path, region: str, mode: str) -> Path | None:
    """Resolve the qa_summary CSV path from run_stamp or by convention."""
    from_json = maybe_get(run_stamp, "qa_summary_path", "QaSummaryPath")
    if isinstance(from_json, str) and from_json.strip():
        p = Path(from_json.strip())
        return p if p.is_absolute() else (insights_dir / p)
    stamp = maybe_get(run_stamp, "stamp", "Stamp")
    if isinstance(stamp, str) and stamp.strip():
        return insights_dir / f"qa_summary__{region}__{mode}__{stamp.strip()}.csv"
    return None


# ── File resolution ──────────────────────────────────────────────────────────

def resolve_required_file(
    required_files: list[str],
    target_name: str,
    insights_dir: Path,
    artifacts_by_name: dict[str, Path] | None = None,
) -> tuple[bool, Path]:
    artifacts = artifacts_by_name or {}
    if target_name in artifacts:
        return True, Path(artifacts[target_name])
    target = target_name.lower()
    for name, path in artifacts.items():
        if str(name).lower() == target:
            return True, Path(path)
    for f in required_files:
        p = Path(str(f))
        if p.name.lower() == target:
            full = p if p.is_absolute() else (insights_dir / p)
            return True, full
    prefix = target_name.split("__")[0].lower()
    for name, path in artifacts.items():
        if str(name).lower().startswith(prefix):
            return True, Path(path)
    return False, insights_dir / target_name


# ── CSV loading ──────────────────────────────────────────────────────────────
# Uses conditional caching: @st.cache_data when Streamlit runtime is active,
# plain function when running headless (CLI / test).

def _has_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        try:
            from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx  # type: ignore
            return get_script_run_ctx() is not None
        except Exception:
            return False


def _read_csv_plain(path_str: str, mtime: float | None) -> pd.DataFrame | None:
    _ = mtime
    try:
        return pd.read_csv(path_str)
    except Exception:
        return None


def _make_cached_reader():
    try:
        import streamlit as st
        @st.cache_data(show_spinner=False)
        def _cached(path_str: str, mtime: float | None) -> pd.DataFrame | None:
            _ = mtime
            try:
                return pd.read_csv(path_str)
            except Exception:
                return None
        return _cached
    except Exception:
        return _read_csv_plain


_cached_reader = _make_cached_reader()


def read_csv_cached(path_str: str, mtime: float | None) -> pd.DataFrame | None:
    return _cached_reader(path_str, mtime)


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = None
    return read_csv_cached(str(path), mtime)


# ── Plotly rendering ─────────────────────────────────────────────────────────

def show_plotly(fig, height: int | None = None, key: str | None = None) -> None:
    """Canonical Plotly chart renderer. key is REQUIRED — raises if missing."""
    import streamlit as st
    if height is not None:
        fig.update_layout(height=height)
    if key is None:
        raise ValueError("show_plotly requires an explicit stable key.")
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=key)


# ── Data freshness ───────────────────────────────────────────────────────────

def parse_stamp_dt(stamp: Any) -> datetime | None:
    """Parse run stamp string 'YYYYMMDD_HHMMSS' → datetime. Returns None on failure."""
    if not stamp:
        return None
    s = str(stamp).strip()
    for fmt in ("%Y%m%d_%H%M%S", "%Y%m%d%H%M%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def render_freshness_banner(ctx: dict, warn_days: int = 7) -> None:
    """Render a data-freshness info/warning banner using the run stamp timestamp.

    Always shows: "Data as of: <date> · <N> days ago · Stamp: <stamp>"
    Escalates to st.warning if age > warn_days.
    Safe to call even if ctx is missing, stamp is unparseable, or no Streamlit runtime.
    """
    # Issue #6: guard — safe to call from any context (CLI, test, Streamlit)
    if not _has_streamlit_runtime():
        return

    if not isinstance(ctx, dict):
        return

    # Issue #2: fix broken ternary — clean explicit fallback
    stamp = ctx.get("stamp")
    if not stamp:
        run_stamp = ctx.get("run_stamp")
        if isinstance(run_stamp, dict):
            stamp = run_stamp.get("stamp")

    stamp_dt = parse_stamp_dt(stamp)

    # Issue #1: use naive local time consistently (both sides same machine)
    now = datetime.now()

    if stamp_dt is None:
        import streamlit as st
        st.info("ℹ️ Data freshness unknown — run stamp timestamp could not be parsed.")
        return

    # Issue #7: full-day rounding via total_seconds
    age_days = int((now - stamp_dt).total_seconds() / 86400)
    age_hours = int((now - stamp_dt).total_seconds() // 3600)
    stamp_label = stamp_dt.strftime("%Y-%m-%d %H:%M")

    if age_days == 0:
        age_str = f"{age_hours}h ago"
    elif age_days == 1:
        age_str = "1 day ago"
    else:
        age_str = f"{age_days} days ago"

    info_line = f"📅 Data as of: **{stamp_label}** · {age_str} · Stamp: `{stamp}`"

    import streamlit as st
    if age_days > warn_days:
        st.warning(
            f"⚠️ **Stale data** — pipeline last ran **{age_str}** ({stamp_label}). "
            f"Refresh the pipeline before presenting to management.\n\n{info_line}"
        )
    else:
        st.info(info_line)


# ── Active filter banner ──────────────────────────────────────────────────────

def render_filter_banner(ctx: dict, filters: dict) -> None:
    """Render a prominent active-filter pill banner.
    Shows: [Region/Mode] [Grain] [Period range] [Station(s)]
    Safe to call even if ctx or filters are missing/incomplete.
    Only renders when Streamlit runtime is active.
    """
    if not _has_streamlit_runtime():
        return
    import streamlit as st

    region = str(ctx.get("region", "?")).strip() if isinstance(ctx, dict) else "?"
    mode = str(ctx.get("mode", "?")).strip() if isinstance(ctx, dict) else "?"

    if not isinstance(filters, dict):
        st.info(f"🔎 Scope: **{region} / {mode}**")
        return

    grain = str(filters.get("grain", "?"))
    periods = [str(p) for p in filters.get("periods", []) if str(p).strip()]
    stations = [str(s) for s in filters.get("stations", []) if str(s).strip()]
    station_mode = str(filters.get("station_mode", "NETWORK"))

    if not periods:
        period_label = "—"
    elif len(periods) == 1:
        period_label = periods[0]
    else:
        period_label = f"{periods[0]} → {periods[-1]} ({len(periods)})"

    if station_mode == "NETWORK" or stations == ["NETWORK"]:
        station_label = "NETWORK"
    elif len(stations) == 1:
        station_label = stations[0]
    else:
        station_label = f"{len(stations)} stations"

    pills_html = (
        "<div style='display:flex;flex-wrap:wrap;gap:0.4rem;margin:0.4rem 0 0.6rem 0;'>"
        f"<span style='background:#4F46E5;color:#E2E8F0;border-radius:999px;padding:0.22rem 0.7rem;"
        f"font-size:0.78rem;font-weight:600;'>🌍 {region} / {mode}</span>"
        f"<span style='background:#4F46E5;color:#E2E8F0;border-radius:999px;padding:0.22rem 0.7rem;"
        f"font-size:0.78rem;font-weight:600;'>📅 {grain}</span>"
        f"<span style='background:#4F46E5;color:#E2E8F0;border-radius:999px;padding:0.22rem 0.7rem;"
        f"font-size:0.78rem;font-weight:600;'>🗓 {period_label}</span>"
        f"<span style='background:#4F46E5;color:#E2E8F0;border-radius:999px;padding:0.22rem 0.7rem;"
        f"font-size:0.78rem;font-weight:600;'>✈ {station_label}</span>"
        "</div>"
    )
    st.markdown(pills_html, unsafe_allow_html=True)


# ── Premium card rendering ──────────────────────────────────────────────────

import numpy as np
import plotly.graph_objects as go


def _enforce_annotation_readability(fig, color: str) -> None:
    for ann in fig.layout.annotations or ():
        current_font = ann.font
        size = 12
        ann_color = color
        if current_font is not None:
            if current_font.size is not None:
                size = max(12, int(current_font.size))
            if current_font.color is not None:
                ann_color = current_font.color
        ann.update(font=dict(size=size, color=ann_color, family="Inter, Arial, sans-serif"))


def apply_plotly_white_card(fig) -> None:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#0B1220", family="Inter, Arial, sans-serif", size=13),
        legend=dict(font=dict(color="#0B1220", size=12, family="Inter, Arial, sans-serif")),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#CBD5E1",
            font=dict(color="#0B1220", size=12, family="Inter, Arial, sans-serif"),
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        gridcolor="#E5E7EB",
        zeroline=False,
        tickfont=dict(color="#0B1220", size=12, family="Inter, Arial, sans-serif"),
        title_font=dict(color="#0B1220", size=12, family="Inter, Arial, sans-serif"),
    )
    fig.update_yaxes(
        showgrid=False,
        gridcolor="#E5E7EB",
        zeroline=False,
        tickfont=dict(color="#0B1220", size=12, family="Inter, Arial, sans-serif"),
        title_font=dict(color="#0B1220", size=12, family="Inter, Arial, sans-serif"),
    )
    red_tokens = {"#e31e24", "rgb(227,30,36)", "red"}
    for trace in fig.data:
        textfont = getattr(trace, "textfont", None)
        trace_size = 12
        trace_color = "#0B1220"
        if textfont is not None and textfont.size is not None:
            trace_size = max(12, int(textfont.size))
        if textfont is not None and textfont.color is not None:
            candidate = str(textfont.color).strip().lower()
            if candidate in red_tokens:
                trace_color = textfont.color
        trace.textfont = dict(color=trace_color, size=trace_size, family="Inter, Arial, sans-serif")
    _enforce_annotation_readability(fig, color="#0B1220")


def style_plotly_card(
    fig,
    title: str | None = None,
    height: int = 260,
    margin: dict | None = None,
    y_grid: bool = False,
) -> None:
    fig.update_layout(
        height=height,
        margin=margin or dict(l=10, r=10, t=48 if title else 10, b=10),
        showlegend=False,
    )
    if title is not None:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left", font=dict(size=16, color="#0B1220")))
    apply_plotly_white_card(fig)
    fig.update_yaxes(showgrid=y_grid, gridcolor="#E5E7EB")


def card_sparkline(y: list | None, n_points: int = 12):
    if y is None or len(y) == 0:
        vals = [1.0] * n_points
    else:
        s = pd.Series(y, dtype="float64").replace([np.inf, -np.inf], np.nan)
        s = s.interpolate(limit_direction="both")
        fill_val = float(s.median()) if s.notna().any() else 1.0
        vals = s.fillna(fill_val).tolist()

    x = list(range(len(vals)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=vals, mode="lines",
        line=dict(color="#94A3B8", width=2),
        hoverinfo="skip",
    ))
    if len(vals) >= 2:
        fig.add_trace(go.Scatter(
            x=[x[0], x[-1]], y=[vals[0], vals[-1]],
            mode="markers+text",
            marker=dict(color=["#94A3B8", "#4F46E5"], size=[5, 7]),
            text=["", f"{vals[-1]:,.0f}"],
            textposition="top center",
            textfont=dict(size=9, color="#4F46E5", family="Inter"),
            hoverinfo="skip",
        ))
    style_plotly_card(fig, height=74, margin=dict(l=0, r=0, t=12, b=0), y_grid=False)
    fig.update_xaxes(showticklabels=False, fixedrange=True)
    fig.update_yaxes(showticklabels=False, fixedrange=True)
    return fig


def otp_rag_class(value) -> str:
    """Return CSS class for OTP RAG status border."""
    if value is None or pd.isna(value):
        return "kpi-rag-neutral"
    v = float(value)
    if v >= 85.0:
        return "kpi-rag-green"
    elif v >= 80.0:
        return "kpi-rag-amber"
    else:
        return "kpi-rag-red"


def delay_rag_class(value, thresholds: tuple = (20.0, 30.0)) -> str:
    """Return CSS class for delay metric RAG (lower is better)."""
    if value is None or pd.isna(value):
        return "kpi-rag-neutral"
    v = float(value)
    if v <= thresholds[0]:
        return "kpi-rag-green"
    elif v <= thresholds[1]:
        return "kpi-rag-amber"
    else:
        return "kpi-rag-red"


def _compute_delta_html(series_list: list | None, metric_key: str, unit: str) -> str:
    """Compute MoM delta arrow HTML from series."""
    if series_list is None or len(series_list) < 2:
        return "<div class='kpi-delta kpi-delta-flat'>vs prev &mdash;</div>"
    cur = series_list[-1]
    prev = series_list[-2]
    if pd.isna(cur) or pd.isna(prev):
        return "<div class='kpi-delta kpi-delta-flat'>vs prev &mdash;</div>"
    delta = cur - prev
    is_higher_good = metric_key in ("OTP", "Flights")
    is_good = (delta > 0) if is_higher_good else (delta < 0)
    if abs(delta) < 0.5:
        css_class = "kpi-delta-flat"
        arrow = "&#8596;"
    elif is_good:
        css_class = "kpi-delta-good"
        arrow = "&#9650;" if delta > 0 else "&#9660;"
    else:
        css_class = "kpi-delta-bad"
        arrow = "&#9650;" if delta > 0 else "&#9660;"
    val_str = f"{abs(delta):,.0f}" if abs(delta) >= 1 else f"{abs(delta):.1f}"
    return f"<div class='kpi-delta {css_class}'>vs prev {arrow} {val_str}{unit}</div>"


def render_kpi_card(
    label: str,
    value_html: str,
    series: list | None,
    chart_key: str,
    rag_class: str = "kpi-rag-neutral",
    delta_html: str = "",
) -> None:
    import streamlit as st
    with st.container(border=True):
        st.markdown(
            f"<div class='{rag_class}' style='padding-left:0.5rem;'>"
            f"<div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{value_html}</div>"
            f"{delta_html}"
            f"</div>",
            unsafe_allow_html=True,
        )
        show_plotly(card_sparkline(series), key=chart_key)


def render_gate_verdict_banner(ctx: dict) -> None:
    """Render a business-friendly 3-state OPS_GATE banner.

    States:
      PASS — all gates green → st.success
      WARN — OPS_GATE=FAIL but Contract=PASS and DbSanity=PASS
             (only pipeline-run gates failing; data is valid) → st.warning
      FAIL — OPS_GATE=FAIL and Contract=FAIL or DbSanity=FAIL
             (data integrity issue) → st.error
    """
    if not _has_streamlit_runtime():
        return
    import streamlit as _st
    if not isinstance(ctx, dict):
        return

    verdict_str = str(ctx.get("ops_gate_pack_verdict", "") or "")
    if not verdict_str:
        run_stamp = ctx.get("run_stamp")
        if isinstance(run_stamp, dict):
            verdict_str = str(run_stamp.get("ops_gate_pack_verdict", "") or "")
    if not verdict_str:
        return

    # Parse individual gate tokens
    gates: dict[str, str] = {}
    for token in verdict_str.split():
        if "=" in token:
            k, v = token.split("=", 1)
            gates[k] = v

    overall    = gates.get("OPS_GATE", "UNKNOWN")
    contract_ok = gates.get("Contract",     "") == "PASS"
    db_ok       = gates.get("DbSanity",     "") == "PASS"
    loader_ok   = gates.get("LoaderGate",   "") == "PASS"
    landing_ok  = gates.get("LandingDelays","") == "PASS"

    # ── PASS ────────────────────────────────────────────────────────────────
    if overall == "PASS":
        _st.success("✅ All data quality checks passed. Dashboard data is fully validated.")
        return

    # ── 3-state classification ───────────────────────────────────────────────
    data_integrity_ok = contract_ok and db_ok   # core data is sound

    if data_integrity_ok:
        # WARN: pipeline-run gates only — data numbers are still reliable
        pipeline_issues: list[str] = []
        if not loader_ok:
            pipeline_issues.append("no new flight data has been loaded since the last pipeline run")
        if not landing_ok:
            pipeline_issues.append("arrival delay cross-checks are pending")
        issue_text = " and ".join(pipeline_issues) if pipeline_issues else "some pipeline checks are pending"
        _st.warning(
            f"⚠️ **Data Note:** {issue_text.capitalize()}. "
            "Core data files and database checks are valid — **dashboard numbers are reliable** for the periods shown."
        )
    else:
        # FAIL: data integrity problem — management should not rely on these numbers
        integrity_issues: list[str] = []
        if not contract_ok:
            integrity_issues.append("one or more required data files are missing or invalid")
        if not db_ok:
            integrity_issues.append("database consistency check failed")
        if not loader_ok:
            integrity_issues.append("no new flight data loaded since last pipeline run")
        if not landing_ok:
            integrity_issues.append("arrival delay cross-checks are pending")
        issue_text = "; ".join(integrity_issues) if integrity_issues else "data quality checks failed"
        _st.error(
            f"🔴 **Data Quality Alert:** {issue_text.capitalize()}. "
            "**Do not share these numbers with management until the pipeline is re-run and validated.**"
        )


def render_situation_summary(text: str, severity: str = "info") -> None:
    """Render a computed one-liner situation summary with colored left border.

    severity: "good" (green), "warn" (amber), "bad" (red), "info" (indigo).
    Safe to call from CLI/test contexts (no-op if no Streamlit runtime).
    """
    if not _has_streamlit_runtime():
        return
    import streamlit as st

    border_colors = {
        "good": "#059669",
        "warn": "#D97706",
        "bad": "#E11D48",
        "info": "#4F46E5",
    }
    border = border_colors.get(severity, "#4F46E5")
    st.markdown(
        f"<div style='border-left:4px solid {border}; padding:0.5rem 0.75rem; "
        f"margin:0.5rem 0 1rem 0; background:#FFFFFF; border-radius:6px; "
        f"font-size:0.9rem; color:#0F172A; line-height:1.5;'>"
        f"{text}</div>",
        unsafe_allow_html=True,
    )
