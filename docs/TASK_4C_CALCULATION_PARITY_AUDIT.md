# Task 4C Calculation Parity Audit

## Executive Verdict

**PARTIAL.**

The public demo preserves the main Mode A KPI contract shape, weighted aggregation rules, periodized DelayCategory RCA files, and scenario-model pattern. The homepage **Controllable Delays - Category Split** is not calculation-parity safe in the default multi-period view. It can show **100% Ground Ops / 0% Other Categories** even when the synthetic data contains other controllable categories.

No real CSVs, real station codes, real rankings, real dates, or internal operational values were used in this audit.

## Specific Cause Of The 100% Ground Ops Anomaly

The anomaly is a **public demo chart logic bug**, not a synthetic data limitation.

In `Main_App.py`, the category split uses two different scopes:

- `cat_df` is filtered to all selected periods.
- `cur_controllable` is pulled from `period_sel`, the last selected period only.
- `ground_ops_controllable` is then clamped to `controllable_total_for_split`.

When the default filter is a range, range-wide Ground Ops minutes can exceed last-period controllable minutes. The clamp then makes Ground Ops equal to the denominator and `Other Categories` becomes zero.

The synthetic demo data is sufficient to show a realistic split:

- Periodized DelayCategory files contain multiple categories, including non-Ground Ops controllable categories.
- Station KPI rows have `GOPS_Min_NORM_Total` below `Controllable_Min_NORM_Total` for single-period slices.
- The 100% display appears when the chart mixes a range numerator with a single-period denominator.

## Public Demo Files And Functions Responsible

- `Main_App.py`
  - `build_network_series()` builds period-level network KPI rows.
  - `val_for()` extracts the last selected period from `net`.
  - `cur_controllable = val_for(period_sel, "Controllable")` makes the denominator single-period.
  - `cat_df = filter_to_selected_period(...)` makes the category numerator multi-period.
  - `ground_ops_controllable`, `controllable_total_for_split`, and `other_controllable_minutes` build the chart values.
  - `own_labels = ["Ground Ops", "Other Categories"]` renders the affected chart.

- `mode_a_filters.py`
  - `render_global_filters()` defaults the homepage to a multi-period range, making the scope mismatch visible by default.

- Synthetic artifacts
  - `demo_data/insight_out/2025_DEP_Monthly_Station_KPIs__DEMO.csv`
  - `demo_data/insight_out/2025_DEP_Weekly_Station_KPIs__DEMO.csv`
  - `demo_data/insight_out/2025_DEP_DelayCategory_Minutes__DEMO_NORM__MONTHLY.csv`
  - `demo_data/insight_out/2025_DEP_DelayCategory_Minutes__DEMO_NORM__WEEKLY.csv`

These files are not the root cause; they already contain the fields needed for a non-100% split.

## Real Mode A Methodology References

The real project source-of-truth methodology is in code, not in public values:

- `scripts/task2_export_delaycategory_station_period.py`
  - `build_delay_components()` classifies reactionary, Ground Ops, uncontrollable, and controllable components from DelayCategory evidence.
  - `Controllable_Min_NORM_Total` is computed as total delay-category minutes minus uncontrollable minutes.
  - `GOPS_Min_NORM_Total` is a Ground Ops category subset, not all controllable delay.
  - `Owner_Basis_DepDelayMin = DelayCategory`, preserving the single accountability dimension.

- `validate_mode_a_contract.py`
  - Validates required files and periodized DelayCategory artifacts.
  - Checks owner/category share sanity and negative-minute guardrails.

- `mode_a_filters.py`
  - `weighted_aggregate()` sums additive minute fields and weights percentage/per-flight metrics by operated flights.

- `utils.py`
  - `weighted_avg()` supports weighted KPI aggregation.
  - `fit_otp_model()` supports the scenario page regression workflow.

- `pages/70_PDF_Decision_Pack.py` in the real project
  - Contains `compute_controllable_split()` and explicit category classification sets for inherited vs controllable split in decision-pack output.
  - The public demo PDF page is older and does not yet carry this split helper.

## Audit By Area

1. **Controllable / inherited / reactionary delay logic: PARTIAL**
   - Export methodology is sound: controllable excludes uncontrollable/reactionary categories, and reactionary is tracked separately.
   - Public homepage breakdown depends on current chart logic and can mix selected-range and last-period scope.

2. **Ground Ops minutes logic: PARTIAL**
   - Correct methodology treats Ground Ops as a category subset.
   - Public chart can visually imply Ground Ops equals all controllable delay because of the mixed denominator.

3. **DelayCategory / Owner mapping logic: PASS**
   - Public demo keeps Owner as DelayCategory.
   - Owner and DelayCategory artifacts use category-based accountability, not a duplicate owner dimension.

4. **Category split chart logic: FAIL**
   - The affected homepage chart is the only confirmed parity failure.
   - It should compute Ground Ops and other controllable categories over the same selected scope.

5. **Pareto logic: PASS WITH CAVEAT**
   - `pages/30_Drivers_RCA.py` groups selected DelayCategory minutes and computes shares against the selected slice.
   - Top delay codes remain annual-scope in the public contract; the page discloses this limitation.

6. **KPI denominator rules: PASS**
   - Additive minute fields are summed.
   - OTP and average/per-flight metrics use operated-flight weighting where aggregation is needed.
   - Scenario average delay uses total delay divided by operated flights for simulation.

7. **Station/network aggregation logic: PASS**
   - Station filters normalize station labels.
   - Network views sum additive fields across selected stations and periods.
   - Percentage and per-flight metrics use weighted aggregation.

8. **Scenario model logic: PASS**
   - `pages/40_Scenarios_Levers.py` applies filters before aggregation.
   - Total, controllable, inherited, and Ground Ops buckets use the same filtered scope.
   - OTP regression is weighted by operated flights and includes minimum observation and confidence guardrails.

9. **PDF decision pack logic: PARTIAL**
   - Public PDF can generate snapshot, ranking, drivers summary, and QA excerpt from synthetic files.
   - It is behind the real project: the real PDF has richer executive summary and controllable-split logic.
   - Public PDF provenance/input lines may expose local-looking paths if rendered, so a later confidentiality pass should sanitize display strings before public use.

10. **Data contract fields needed by the public demo: PARTIAL**
    - Current 7-file synthetic contract is enough for homepage, RCA, scenarios, and basic PDF.
    - It lacks some full-system artifacts used by the real project, such as periodized delay-code Pareto and rotation-chain evidence.
    - Those omissions are acceptable for a public demo if clearly positioned as an architecture subset.

## Safe Fix Plan

1. Fix only the homepage chart calculation in `Main_App.py`.
2. Build the category split from one consistent scope:
   - Use selected-period, selected-station DelayCategory rows for both numerator and denominator; or
   - Use selected-period, selected-station Station KPI rows for both numerator and denominator.
3. Prefer the DelayCategory route when periodized category artifacts are available:
   - Ground Ops = selected-scope category minutes where DelayCategory is Ground Ops.
   - Other controllable = selected-scope category minutes where category is not inherited/reactionary/uncontrollable and not Ground Ops.
   - Denominator = Ground Ops + other controllable.
4. If periodized category artifacts are unavailable, fall back to Station KPI fields using the same selected scope:
   - Ground Ops = sum `GOPS_Min_NORM_Total` over selected rows.
   - Controllable = sum `Controllable_Min_NORM_Total` over selected rows.
   - Other controllable = max(controllable - Ground Ops, 0).
5. Remove or avoid the clamp that hides scope mismatches. If Ground Ops exceeds controllable after consistent-scope calculation, show a data-quality warning instead of forcing 100%.
6. Add a lightweight public-demo assertion or diagnostic that the default synthetic range does not render Ground Ops as all controllable unless the data truly supports it.

## Data Changes Needed

No required synthetic data change is needed to fix the anomaly.

Optional synthetic-data hardening:

- Add a small QA note or fixture check confirming that default selected ranges include non-Ground Ops controllable categories.
- Keep all values synthetic and avoid any real station, region, or operational fingerprints.

## Code Changes Needed

Recommended implementation scope for the next task:

- `Main_App.py`
  - Introduce a small helper such as `compute_controllable_category_split(...)`.
  - Reuse existing category masks, but add a broader non-controllable mask matching the real methodology tokens.
  - Use selected-scope rows for numerator and denominator.
  - Preserve chart layout and labels.

- Optional later parity upgrade:
  - Bring the public `pages/70_PDF_Decision_Pack.py` closer to the real PDF split methodology without copying confidential text, paths, or values.
  - Sanitize public PDF provenance path rendering.

## Confidentiality Risk Check

This audit did not copy real exports, real station codes, real metrics, real rankings, real dates, internal manager names, or internal labels into the public repo.

Remaining public-demo risks observed during audit:

- Some PDF and pipeline metadata views can display local-looking artifact paths. These should be sanitized in a future public-demo hardening task.
- Public PDF logic is older than the real project and should not be presented as full parity until the split methodology is updated.

## Recommended Next Task Prompt

```text
Task 4D - Fix public demo controllable category split

Repo: JosephSigma991/intelligenceops-demo
Branch: master

Goal:
Fix the homepage "Controllable Delays - Category Split" chart so it uses one consistent selected scope for both Ground Ops and other controllable categories. Do not change KPI definitions, synthetic values, chart design, page layout, or data files unless a small synthetic QA fixture/check is necessary.

Requirements:
- Use selected-period and selected-station scope consistently.
- Prefer periodized DelayCategory artifacts when available.
- Ground Ops = selected-scope DelayCategory minutes for Ground Ops.
- Other controllable = selected-scope controllable categories excluding Ground Ops, inherited/reactionary, and uncontrollable categories.
- If falling back to Station KPI fields, sum GOPS and Controllable over the same selected Station KPI rows.
- Do not clamp Ground Ops to controllable in a way that hides a scope mismatch; warn if a consistency violation is detected.
- Preserve existing chart title, labels, layout, and public confidentiality wording.

Verification:
- Default Monthly range synthetic demo must not render 100% Ground Ops unless selected-scope data genuinely has no other controllable categories.
- Default Weekly range synthetic demo must not render 100% Ground Ops unless selected-scope data genuinely has no other controllable categories.
- Single-period views must still render correctly.
- python -m py_compile Main_App.py
- Do not modify real data or copy real project exports.

Commit message:
Fix public demo controllable split scope
```
