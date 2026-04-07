from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from utils import normalize_name, pick_column, parse_period


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _resolve_station_kpi_path(ctx: dict[str, Any], grain: str) -> Path:
    region = str(ctx.get("region", "")).strip()
    insights_dir = Path(ctx.get("insights_dir"))
    artifacts_by_name = ctx.get("artifacts_by_name", {}) or {}

    fn = f"2025_DEP_Monthly_Station_KPIs__{region}.csv" if grain == "Monthly" else f"2025_DEP_Weekly_Station_KPIs__{region}.csv"
    if fn in artifacts_by_name:
        return Path(artifacts_by_name[fn])

    for k, v in artifacts_by_name.items():
        if str(k).lower() == fn.lower():
            return Path(v)

    required = [Path(str(x)).name for x in ctx.get("required_files", [])]
    for n in required:
        if n.lower() == fn.lower():
            return insights_dir / n

    return insights_dir / fn


def _active_periods_and_stations(df: pd.DataFrame, grain: str) -> tuple[pd.DataFrame, str | None, str | None, str | None]:
    period_col = pick_column(
        df,
        ["YearMonth", "YearWeek", "Period", "OpMonth", "OpWeek", "Month", "Week"],
        token_groups=[["yearmonth"], ["yearweek"], ["period"]],
    )
    station_col = pick_column(df, ["Station", "DepartureAirport"], token_groups=[["station"], ["airport"]])
    flights_col = pick_column(
        df,
        ["Flights_Operated", "FlightsOperated", "Flights Operated", "Flights"],
        token_groups=[["flights", "operated"], ["flights", "ops"], ["flight", "oper"]],
    )
    if period_col is None or station_col is None or flights_col is None:
        return pd.DataFrame(), period_col, station_col, flights_col

    w = df.copy()
    parsed = w[period_col].map(lambda v: parse_period(v, grain))
    w["_period"] = parsed.map(lambda t: t[0])
    w["_sort"] = parsed.map(lambda t: t[1])
    w = w.dropna(subset=["_period", "_sort"]).copy()
    w[flights_col] = pd.to_numeric(w[flights_col], errors="coerce").fillna(0.0)
    w["_station"] = w[station_col].astype(str).str.strip()
    w["_station_u"] = w["_station"].str.upper()

    rows: list[dict[str, Any]] = []
    for (p, s), g in w.groupby(["_period", "_sort"], dropna=True):
        network_rows = g[g["_station_u"] == "NETWORK"]
        if not network_rows.empty:
            flights = float(network_rows[flights_col].sum())
        else:
            flights = float(g[flights_col].sum())
        rows.append({"Period": str(p), "_sort": s, "Flights": flights})

    periods_df = pd.DataFrame(rows).sort_values("_sort").reset_index(drop=True)
    periods_df = periods_df[periods_df["Flights"] > 0].copy()
    return periods_df, period_col, station_col, flights_col


def _format_period_summary(periods: list[str]) -> str:
    if not periods:
        return "none"
    if len(periods) == 1:
        return periods[0]
    return f"{periods[0]} .. {periods[-1]} ({len(periods)})"


def render_global_filters(st, ctx: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ctx, dict):
        st.error("Run context missing.")
        st.stop()

    prev = st.session_state.get("mode_a_filters", {}) if isinstance(st.session_state.get("mode_a_filters"), dict) else {}

    st.sidebar.header("Global Filters")
    grain_default_idx = 0 if str(prev.get("grain", "Monthly")) == "Monthly" else 1
    grain = st.sidebar.radio("Grain", options=["Monthly", "Weekly"], index=grain_default_idx, key="mode_a_global__grain")

    kpi_path = _resolve_station_kpi_path(ctx, grain)
    if not kpi_path.exists():
        st.sidebar.error(f"Missing Station KPI file: {kpi_path.name}")
        st.stop()
    try:
        df = _read_csv(kpi_path)
    except Exception as e:
        st.sidebar.error(f"Failed to read {kpi_path.name}: {type(e).__name__}: {e}")
        st.stop()

    periods_df, period_col, station_col, flights_col = _active_periods_and_stations(df, grain)
    if period_col is None or station_col is None or flights_col is None:
        st.sidebar.error("Could not infer period/station/flights columns for global filters.")
        st.stop()
    if periods_df.empty:
        st.sidebar.error("No active periods found (Flights_Operated > 0).")
        st.stop()

    all_periods = periods_df["Period"].astype(str).tolist()
    sort_lookup = dict(zip(periods_df["Period"].astype(str), periods_df["_sort"]))
    period_mode_prev = str(prev.get("period_mode", "Range"))
    period_mode_opts = ["Single", "Multi", "Range", "Full Year"]
    period_mode_idx = period_mode_opts.index(period_mode_prev) if period_mode_prev in period_mode_opts else 0
    period_mode = st.sidebar.radio("Period Mode", options=period_mode_opts, index=period_mode_idx, key="mode_a_global__period_mode")

    periods_selected: list[str] = []
    if period_mode == "Single":
        prev_period = str(prev.get("periods", [all_periods[-1]])[-1]) if prev.get("periods") else all_periods[-1]
        if prev_period not in all_periods:
            prev_period = all_periods[-1]
        p = st.sidebar.selectbox("Period", options=all_periods, index=all_periods.index(prev_period), key="mode_a_global__period_single")
        periods_selected = [str(p)]
    elif period_mode == "Multi":
        prev_multi = [str(p) for p in prev.get("periods", []) if str(p) in all_periods]
        if not prev_multi:
            default_n = 6 if grain == "Monthly" else 13
            prev_multi = all_periods[max(0, len(all_periods) - default_n):]
        p_multi = st.sidebar.multiselect("Periods", options=all_periods, default=prev_multi, key="mode_a_global__period_multi")
        periods_selected = [str(p) for p in p_multi if str(p) in all_periods]
    elif period_mode == "Range":
        prev_periods = [str(p) for p in prev.get("periods", []) if str(p) in all_periods]
        default_start = prev_periods[0] if prev_periods else all_periods[max(0, len(all_periods) - (6 if grain == "Monthly" else 13))]
        default_end = prev_periods[-1] if prev_periods else all_periods[-1]
        start_p = st.sidebar.selectbox("Start", options=all_periods, index=all_periods.index(default_start), key="mode_a_global__period_range_start")
        end_p = st.sidebar.selectbox("End", options=all_periods, index=all_periods.index(default_end), key="mode_a_global__period_range_end")
        s1 = sort_lookup.get(str(start_p))
        s2 = sort_lookup.get(str(end_p))
        if s1 is None or s2 is None:
            periods_selected = [all_periods[-1]]
        else:
            lo, hi = (s1, s2) if s1 <= s2 else (s2, s1)
            periods_selected = [p for p in all_periods if (sort_lookup.get(p) is not None and lo <= sort_lookup[p] <= hi)]
    else:
        years = sorted({str(p)[:4] for p in all_periods if re.match(r"^\d{4}", str(p))})
        if not years:
            years = [str(all_periods[-1])[:4]]
        prev_year = str(prev.get("period_year", years[-1]))
        if prev_year not in years:
            prev_year = years[-1]
        year_sel = st.sidebar.selectbox("Year", options=years, index=years.index(prev_year), key="mode_a_global__period_year")
        periods_selected = [p for p in all_periods if str(p).startswith(str(year_sel))]

    if not periods_selected:
        periods_selected = [all_periods[-1]]
    periods_selected = sorted(set(periods_selected), key=lambda p: sort_lookup.get(p, pd.Timestamp.max))

    w = df.copy()
    w[flights_col] = pd.to_numeric(w[flights_col], errors="coerce").fillna(0.0)
    parsed = w[period_col].map(lambda v: parse_period(v, grain))
    w["_period"] = parsed.map(lambda t: t[0])
    w["_station"] = w[station_col].astype(str).str.strip()
    w = w[w["_period"].isin(periods_selected)].copy()
    w = w[w[flights_col] > 0].copy()
    stations_active = sorted([s for s in w["_station"].dropna().astype(str).unique().tolist() if s and s.upper() != "ALL"])

    station_mode_prev = str(prev.get("station_mode", "NETWORK"))
    station_mode_opts = ["NETWORK", "Single", "Multi"]
    station_mode_idx = station_mode_opts.index(station_mode_prev) if station_mode_prev in station_mode_opts else 0
    station_mode = st.sidebar.radio("Station Mode", options=station_mode_opts, index=station_mode_idx, key="mode_a_global__station_mode")

    stations_selected: list[str] = []
    if station_mode == "NETWORK":
        stations_selected = ["NETWORK"]
    elif station_mode == "Single":
        if not stations_active:
            stations_selected = ["NETWORK"]
            station_mode = "NETWORK"
        else:
            prev_station = None
            prev_stations = prev.get("stations", [])
            if isinstance(prev_stations, list):
                for s in prev_stations:
                    if str(s) in stations_active:
                        prev_station = str(s)
                        break
            if prev_station is None:
                prev_station = stations_active[0]
            s = st.sidebar.selectbox("Station", options=stations_active, index=stations_active.index(prev_station), key="mode_a_global__station_single")
            stations_selected = [str(s)]
    else:
        if not stations_active:
            stations_selected = ["NETWORK"]
            station_mode = "NETWORK"
        else:
            prev_multi = [str(s) for s in prev.get("stations", []) if str(s) in stations_active and str(s).upper() != "NETWORK"]
            if not prev_multi:
                prev_multi = stations_active
            sm = st.sidebar.multiselect("Stations", options=stations_active, default=prev_multi, key="mode_a_global__station_multi")
            stations_selected = [str(s) for s in sm if str(s).strip() and str(s).upper() != "NETWORK"]
            if not stations_selected:
                stations_selected = ["NETWORK"]
                station_mode = "NETWORK"

    if any(str(s).upper() == "NETWORK" for s in stations_selected):
        stations_selected = ["NETWORK"]
        station_mode = "NETWORK"

    filters = {
        "grain": grain,
        "period_mode": period_mode,
        "period_year": str(periods_selected[-1])[:4] if periods_selected else None,
        "periods": periods_selected,
        "station_mode": station_mode,
        "stations": stations_selected,
    }
    st.session_state["mode_a_filters"] = filters

    st.sidebar.caption(
        f"Periods: {_format_period_summary(periods_selected)} | Stations: "
        f"{'NETWORK' if stations_selected == ['NETWORK'] else len(stations_selected)}"
    )
    return filters


def apply_filters(df: pd.DataFrame, filters: dict[str, Any], station_col: str | None, period_col: str | None) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy()
    out = df.copy()
    if not isinstance(filters, dict):
        return out

    periods = [str(p) for p in filters.get("periods", []) if str(p).strip()]
    stations = [str(s) for s in filters.get("stations", []) if str(s).strip()]

    if period_col and period_col in out.columns and periods:
        out = out[out[period_col].astype(str).isin(periods)].copy()

    if station_col and station_col in out.columns and stations:
        if len(stations) == 1 and stations[0].upper() == "NETWORK":
            s = out[station_col].astype(str).str.strip().str.upper()
            net_rows = out[s == "NETWORK"].copy()
            if not net_rows.empty:
                out = net_rows
        else:
            keys = {s.strip().upper() for s in stations if s.strip()}
            s = out[station_col].astype(str).str.strip().str.upper()
            out = out[s.isin(keys)].copy()
    return out


def weighted_aggregate(df: pd.DataFrame, flights_col: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if df is None or df.empty:
        return out
    if flights_col not in df.columns:
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                out[c] = float(pd.to_numeric(df[c], errors="coerce").fillna(0).sum())
        return out

    w = pd.to_numeric(df[flights_col], errors="coerce").fillna(0.0)
    out[flights_col] = float(w.sum())

    for c in df.columns:
        if c == flights_col:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().sum() == 0:
            continue
        lc = str(c).lower()
        weighted_mode = lc.endswith("_pct") or ("perflight" in lc) or lc.startswith("avg")
        if weighted_mode:
            mask = vals.notna() & w.notna()
            denom = float(w[mask].sum())
            if denom > 0:
                out[c] = float((vals[mask] * w[mask]).sum() / denom)
        else:
            out[c] = float(vals.fillna(0.0).sum())
    return out
