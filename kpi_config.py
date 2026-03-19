"""
kpi_config.py — Shared KPI configuration for IntelligenceOps Mode A.
Import from here; do not re-define constants in page files.
"""

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "scrollZoom": False,
    "responsive": True,
}

KPI_CANDIDATES = {
    "Flights Operated": ["Flights_Operated", "Flights Operated", "FlightsOperated", "Flights"],
    "Avg DEP Delay per Flight (min)": [
        "AvgDepDelayMin_PerFlight",
        "Avg DEP Delay per Flight (min)",
        "Avg_DEP_Delay_per_Flight_Min",
        "AvgDepDelay",
    ],
    "DEP OTP D15 (%)": [
        "OTP_D15_Pct",
        "DEP OTP D15 (%)",
        "OTP_D15",
        "OTP D15",
        "OTP_D15_Pct_basis_PerfEligible",
    ],
    "Total DEP Delay Minutes": [
        "DepDelayMin_Total",
        "Total DEP Delay Minutes",
        "DepDelayTotal_PosOnly_Min",
        "DepDelayTotal_Minutes",
    ],
    "Controllable Minutes": [
        "Controllable_Min_NORM_Total",
        "Controllable Minutes",
        "Controllable_Minutes",
        "Controllable_Min_Total",
    ],
    "Inherited Minutes (Late Arrival)": [
        "Reactionary_Min_NORM_Total",
        "Inherited Minutes (Late Arrival)",
        "Inherited_Minutes",
        "LateArrival_Minutes",
        "Late Arrival Minutes",
        "Reactionary_Minutes",
    ],
    "Ground Ops Minutes": [
        "GOPS_Min_NORM_Total",
        "Ground Ops Minutes",
        "GroundOps_Minutes",
        "Ground Ops",
    ],
}

WEEKLY_PERIOD_CANDIDATES  = ["YearWeek",  "OpWeek",  "Week",  "Period"]
MONTHLY_PERIOD_CANDIDATES = ["YearMonth", "OpMonth", "Month", "Period"]
STATION_CANDIDATES        = ["Station", "DepartureAirport", "DepAirport", "Airport"]
FLIGHTS_CANDIDATES        = ["Flights_Operated", "Flights Operated", "FlightsOperated", "Flights"]
TOTAL_MIN_CANDIDATES      = [
    "Total_DEP_Delay_Minutes",
    "DepDelayMin_Total",
    "Total DEP Delay Minutes",
    "DepDelayTotal_PosOnly_Min",
    "DepDelayTotal_Minutes",
]
AVG_DELAY_CANDIDATES = [
    "AvgDepDelayMin_PerFlight",
    "Avg DEP Delay per Flight (min)",
    "Avg_DEP_Delay_per_Flight_Min",
    "AvgDepDelay",
]
OTP_CANDIDATES = [
    "OTP_D15_Pct",
    "DEP OTP D15 (%)",
    "DEP_OTP_D15",
    "OTP_D15",
    "OTP D15",
    "OTP_D15_Pct_basis_PerfEligible",
]
CATEGORY_CANDIDATES = [
    "DelayCategory",
    "DelayCategory_Basis_DepDelayMin",
    "Owner_Basis_DepDelayMin",
    "Owner",
    "Category",
]
MINUTES_CANDIDATES = [
    "Minutes_NORM_to_DepDelayMin",
    "Minutes_NORM_Total",
    "Minutes_NORM",
    "DelayMinutes",
    "Minutes",
    "DelayMin",
]
CONTROLLABLE_CANDIDATES = [
    "Controllable_Min_NORM_Total",
    "Controllable Minutes",
    "Controllable_Minutes",
    "Controllable_Min_Total",
]
INHERITED_CANDIDATES = [
    "Reactionary_Min_NORM_Total",
    "Inherited Minutes (Late Arrival)",
    "Inherited_Minutes",
    "LateArrival_Minutes",
    "Reactionary_Minutes",
]
GROUND_CANDIDATES = [
    "GOPS_Min_NORM_Total",
    "Ground Ops Minutes",
    "GroundOps_Minutes",
    "Ground Ops",
]

# ── Chart theme ───────────────────────────────────────────────────────────────

OTP_TARGET   = 85.0
OTP_WARN_LOW = 80.0

# Aliases kept for backward-compat with page imports — values updated to premium palette.
# New code should import ACCENT_PRIMARY / RAG_* directly.
COLOR_PRIMARY    = "#4F46E5"   # was #4C78A8 (dark theme) → Indigo 600
COLOR_SECONDARY  = "#F59E0B"   # was #F58518 → Amber 500
COLOR_GREEN      = "#059669"   # was #22c55e → Emerald 600
COLOR_AMBER      = "#D97706"   # was #f59e0b → Amber 600
COLOR_RED        = "#E11D48"   # was #ef4444 → Rose 600
COLOR_TARGET_LINE = "#E11D48"  # was #ef4444 → Rose 600

# ── OTP regression model ─────────────────────────────────────────────────────
OTP_MODEL_MIN_OBS = 6          # Minimum station-month observations to fit model
OTP_MODEL_MIN_R2_WARN = 0.40   # Below this R², show low-confidence warning
OTP_MODEL_FLIGHTS_FLOOR = 5    # Exclude station-months with fewer flights
OTP_MODEL_DELAY_DOMAIN_MAX = 45.0  # Beyond this avg delay, flag as extrapolation

# ── Premium Light Theme Palette ──────────────────────────────────────────────
# Page & card
BG_PAGE       = "#F8FAFC"
BG_CARD       = "#FFFFFF"
BORDER_CARD   = "#E2E8F0"
SHADOW_CARD   = "0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04)"

# Text hierarchy
TEXT_PRIMARY   = "#0F172A"
TEXT_SECONDARY = "#64748B"
TEXT_LABEL     = "#94A3B8"

# Chart accent
ACCENT_PRIMARY   = "#4F46E5"   # Indigo 600
ACCENT_SECONDARY = "#F59E0B"   # Amber 500

# RAG status (for KPI conditional coloring)
RAG_GREEN  = "#059669"   # Emerald 600 — good
RAG_AMBER  = "#D97706"   # Amber 600 — warning
RAG_RED    = "#E11D48"    # Rose 600 — critical
COLOR_MUTED    = "#94A3B8"   # Slate 400 — muted/baseline bars

# OTP-specific thresholds (reuse existing OTP_TARGET=85, OTP_WARN_LOW=80)
# ≥85% → RAG_GREEN, 80-85% → RAG_AMBER, <80% → RAG_RED

# Sidebar
BG_SIDEBAR     = "#F1F5F9"
TEXT_SIDEBAR    = "#334155"

# Divider
BORDER_DIVIDER = "#E2E8F0"
