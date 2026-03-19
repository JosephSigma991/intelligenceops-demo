# IntelligenceOps

**Production-grade departure performance analytics platform for airline operations.**

> **This is a public demo deployment.** All data is synthetic — station codes, delay minutes, and OTP figures are computer-generated. No real operational data is present. Architecture, design, and all application code are production-identical.

IntelligenceOps processes raw flight and delay data from SQL Server through a validated ETL pipeline, publishes contract-verified CSV artifacts, and renders them as an interactive multi-page dashboard with executive PDF export capability. The system covers OTP D15 tracking, delay root cause analysis, station ranking, what-if scenario modeling, and data quality monitoring, all built against IATA delay accountability standards.

Currently deployed for a regional airline's East Africa and Turkey network (6 stations), with architecture designed for multi-region, multi-mode extensibility.

---

## Architecture

```
SQL Server (FlightOpsDB)
    |
    v
ETL Pipeline (PowerShell + Python)
    |
    v
CSV Contract (13 files, 3 time grains)
    |
    v
Run Stamp JSON (Single Source of Truth)
    |
    v
Streamlit Dashboard (8 pages)          PDF Decision Pack (ReportLab + Kaleido)
```

**Two-layer design.** Layer 1 is a SQL Server ETL pipeline that ingests XLS flight exports, cleans data through hygiene views, and publishes validated CSVs. Layer 2 is the Streamlit application that consumes those artifacts read-only. The layers are decoupled through a JSON run stamp manifest that declares which files were published, when, and by which pipeline run.

**Contract boundary.** The dashboard never renders data it hasn't validated. A CI pipeline checks all 13 required files across Annual, Monthly, and Weekly grains on every commit. Defensive column detection handles upstream schema drift without crashing.

---

## Pages

| # | Page | Audience | Description |
|---|------|----------|-------------|
| 0 | Homepage | Management | 4 KPI cards with RAG status, delay breakdown donut, station ranking, concentration panel |
| 1 | Network Overview | Exec / Ops Director | OTP D15 trend with 85% target, avg delay trend with MoM annotation |
| 2 | Station Ranking | Ops Manager | Sortable station table, KPI heatmap, single-station trend preview |
| 3 | Drivers / RCA | Analyst | Delay category Pareto, category trend, top delay codes drilldown, computed insights |
| 4 | Scenarios | Management | What-if simulation, OTP regression model (weighted OLS), before/after visualization |
| 5 | Dictionary | Everyone | KPI definitions (management tab), artifact registry (analyst tab) |
| 6 | Data Quality | Pipeline Engineer | Provenance, gate evidence, flight data quality metrics, coverage trends |
| 7 | PDF Decision Pack | Exec (via email) | Toggle sections, preview, generate + download PDF with embedded charts |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Runtime | Python 3.11 |
| Web UI | Streamlit 1.36+ |
| Charts | Plotly 5.18+ |
| Data | pandas 2.0+, NumPy 1.24+ |
| Database | SQL Server 2022 (ETL layer only) |
| PDF Export | ReportLab + Kaleido |
| Linter | Ruff 0.15.0 |
| CI | GitHub Actions (ubuntu-latest) |
| Pipeline | PowerShell (.ps1) |

---

## Key Design Decisions

**Run stamp as SSOT.** Every page resolves artifact paths through a JSON manifest, not filesystem assumptions. The stamp carries git branch, git commit, and timestamp for full traceability.

**Defensive column detection.** `pick_column()` tries exact match, case-insensitive match, then normalized match against a candidate list. The app handles upstream schema drift without silent failures.

**Owner = Delay Category.** A deliberate simplification for an operations accountability dashboard. Delay categories map to IATA owner groups. A separate Owner dimension was evaluated and rejected because it would confuse accountability conversations.

**POSONLY mode.** Only positive delays are analyzed. This is the industry-standard approach for departure delay reporting, aligned with how IATA OTP D15 works.

**Frozen KPI definitions.** Four core calculations are locked and cannot be changed without explicit sign-off: AvgDepDelayMin_PerFlight, OTP_D15_Pct, Controllable_Min_NORM_Total, Minutes_NORM_to_DepDelayMin.

**OTP regression model.** Weighted linear regression fitted at runtime from published Station KPI data. Includes R-squared confidence threshold (0.40), extrapolation detection using actual model bounds, and explicit "[estimated]" labeling to distinguish from reported KPIs.

---

## Data Contract

7 required CSV files in the demo deployment (full production contract: 13 files):

- 2x Station KPIs (Monthly + Weekly)
- 3x Delay Category Minutes (Annual + Monthly + Weekly)
- 1x Owner Minutes (Annual)
- 1x Top Delay Codes (Annual)

All validated by `validate_mode_a_contract.py` across 3 time grains in CI.

---

## Module Architecture

```
Main_App.py              Entry point, homepage, CSS injection
kpi_config.py            38 shared constants (candidates, colors, thresholds)
utils.py                 17 shared functions (single source of truth)
mode_a_run_context.py    Run stamp discovery, scope selector
mode_a_filters.py        Global sidebar filters (Grain / Period / Station)
pages/
  10_Network_Overview.py
  20_Station_Ranking.py
  30_Drivers_RCA.py
  40_Scenarios_Levers.py
  50_Dictionary_Policy.py
  60_Data_Quality_Coverage.py
  70_PDF_Decision_Pack.py
```

---

## Quick Start

```bash
# Prerequisites: Python 3.11, pip
pip install -r requirements.txt

# Run the dashboard
python -m streamlit run Main_App.py
```

Demo data is pre-loaded in `demo_data/insight_out/`. The app reads from there automatically.

---

## CI Pipeline

```
py_compile (all .py files)
  -> validate_mode_a_contract.py --grain Annual
  -> validate_mode_a_contract.py --grain Monthly
  -> validate_mode_a_contract.py --grain Weekly
```

Runs on every push to `dev`/`main` and every PR targeting `main`. All 3 grain validations must pass.

---

## Visual Design

**Theme: "Slate Clarity" (Light Executive)**

Light background (#F8FAFC), white cards, Inter font. RAG-colored KPI card borders (green/amber/red). Plotly charts with consistent branding (Indigo 600 accent, 85% OTP target line). Management-facing titles, technical metadata behind expanders.

Design principle: every page must survive a screenshot. The person who needs this dashboard most will screenshot it on their phone and send it in a WhatsApp group at 7 AM.

---

## Development Workflow

1. Branch from `main` using `feat/<scope>-<name>` pattern
2. Make changes (UI-only preferred; data/contract changes require sign-off)
3. Run: `python -m compileall .`
4. PR, CI green, merge to `main`, delete branch

---

## AI Development Workflow

This project uses a 3-tier AI development model:

| Tier | Purpose | Tool |
|------|---------|------|
| Strategic | Architecture review, audit, roadmap | Claude (high-context sessions) |
| Planning | Spec writing, prompt crafting, code review | Claude Desktop |
| Execution | Multi-file code changes, refactors, validation | Claude Code (VS Code) |

Domain decisions are made by the developer. AI handles execution, review, and refactoring under human oversight. Every AI-generated change is validated through the CI pipeline before merge.

---

## Engineering Maturity

Rated **7/10** in an independent architecture review. Strengths: contract-based architecture, defensive column detection, centralized rendering, documented guardrails, CI-validated data contract. Areas for growth: automated unit tests beyond contract validation, structured logging, error recovery in page rendering.

For a system built entirely by one person alongside a full-time operational role, this represents genuine engineering discipline and real domain expertise.

---

## Author

**Youssef Hamdaoui** (Mr BI)
Airport Duty Manager & Operations Analyst
Casablanca, Morocco

- LinkedIn: [link]
- Portfolio: [link]

---

## License

This repository contains proprietary analysis code. Contact the author for licensing inquiries.
