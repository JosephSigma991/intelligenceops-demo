# Task 4E Public PDF Decision Pack Audit

## 1. Executive Verdict

**PARTIAL.**

The public PDF Decision Pack page is useful as technical proof that the demo can read validated synthetic CSV artifacts, generate ReportLab PDFs, embed Plotly chart images when available, and export a downloadable decision pack. It is not yet public-portfolio ready as a decision-grade output.

The main gaps are confidentiality posture and executive structure:

- It exposes run metadata, QA filenames, artifact paths, and pipeline-style labels too directly.
- It renders provenance before business conclusions.
- It does not yet include the stronger real Mode A executive summary, controlled technical appendix pattern, controllable/inherited/reactionary split, or recommendation structure.
- It uses synthetic data through the Streamlit app context, but the CLI/default path behavior is too close to the private project shape and should be made demo-local before public promotion.

## 2. Current Public PDF Page Weaknesses

- The generated PDF starts with provenance and inputs instead of an executive decision page.
- Public page copy says `PASS-only PDF`, `run_stamp`, `published CSVs`, and `Pipeline Parameters`, which is technically true but reads like internal machinery.
- The Streamlit UI exposes detailed parameters such as run context path, stamp, region/mode, commit, insights directory, QA summary path, and gate log path.
- The PDF includes raw provenance lines and input file paths before the reader sees the business snapshot.
- The generated filename includes scope/mode/stamp tokens. These are sanitized after Task 4B, but still feel like internal release metadata.
- The output directory and run log path are displayed after generation.
- The page has tables and charts, but limited decision narrative.
- The public PDF driver section is a generic top-category table and Pareto chart. It does not yet state a clear action route, evidence basis, or caveat in public-safe language.
- The public page does not include the real V2 executive layer: OTP target gap, prior-period movement, top findings, controllable/inherited/reactionary split, and action lanes.
- The page can still generate PDFs with a technical QA excerpt visible by default.

## 3. Confidentiality Risks

- **Local path exposure:** Streamlit UI and PDF provenance can show local-looking paths for run context, insights directory, QA summary, output PDF, and run log.
- **Internal metadata exposure:** Build stamp, commit hash, QA filenames, and run context terminology are visible to public visitors.
- **Scope/mode labeling:** `DEMO / SYNTH` is safe enough, but presenting it as `region/mode` still hints at internal operating dimensions.
- **Artifact-name exposure:** Required CSV names are useful for architecture proof, but full filenames in the PDF can distract and look like internal contract evidence.
- **Operational scale leakage:** PDF tables can show synthetic values. That is acceptable only if the UI/PDF clearly labels the output as synthetic and does not imply real operational performance.
- **Manager-facing wording risk:** The real project has stronger leadership/action wording, but some of it is role-specific and should not be copied verbatim into the public demo.
- **Default data path risk:** The public page contains a default insights-directory function that should not point toward a private-project style location for CLI/headless use.

## 4. Real PDF Methodology / Design Elements Worth Porting

Port the methodology shape, not real values or private wording:

- **Business-first cover page:** Show public-safe scope, period, synthetic data label, and generation timestamp without raw pipeline paths.
- **Executive Summary:** Add a concise A-section with KPI status, target gap, prior-period movement, and top findings.
- **Decision-grade findings:** Convert raw tables into statements that explain what changed, why it matters, and what review path follows.
- **Controllable / inherited / reactionary split:** Use the real methodology pattern that separates controllable, uncontrollable, and reactionary buckets from validated period-compatible evidence.
- **Period compatibility rule:** Keep the real guardrail that monthly/weekly packs must use periodized evidence and must not fall back to annual-only driver/category data.
- **Station ranking with action status:** Keep a public-safe recommendation column or status band, but avoid private thresholds or manager-specific labels.
- **Drivers summary:** Keep top DelayCategory, Pareto chart, and category share, but add caveats and public-safe action logic.
- **Technical Appendix:** Move provenance, inputs, QA excerpt, and file registry behind an optional appendix, not the cover page.
- **Graceful chart behavior:** Keep the current Kaleido fallback pattern so PDF generation works even when PNG export is unavailable.

## 5. Elements That Must Not Be Ported

- Real data, real station names, real rankings, real dates, real OTP values, real delay minutes, real flight volumes, or real operational scale.
- Private region/scope names, private mode tokens, private run stamp names, or private QA filenames.
- Internal manager names, role-specific directives, or wording that implies employer endorsement.
- Local machine paths, private export directories, private log paths, or private source file references.
- HTML cockpit work, cockpit renderer logic, or cockpit-specific copy. That is outside Task 4E/4F scope.
- Any changes to KPI definitions or calculation formulas.
- Any public claim that the demo PDF is a live production output.

## 6. Recommended Public PDF Structure

Recommended public-facing PDF order:

1. **Cover**
   - Title: `IntelligenceOps Synthetic Decision Pack`
   - Subtitle: `Public demo using synthetic data`
   - Scope label: `Synthetic Demo`
   - Period and station selection, without internal path or run labels

2. **A) Executive Summary**
   - KPI status snapshot
   - Target gap or status label
   - Prior-period movement where available
   - Top 3 public-safe findings

3. **B) KPI Snapshot**
   - Flights, delay minutes, average delay, OTP
   - Clear basis labels such as `computed from synthetic selected scope`

4. **C) Delay Accountability**
   - DelayCategory Pareto chart/table
   - Controllable / inherited / reactionary split
   - Public-safe caveat: `Owner basis is DelayCategory; no separate owner axis`

5. **D) Station Review**
   - Top stations by selected KPI or delay minutes
   - Public-safe status band or review priority
   - No private action owner wording

6. **E) Scenario / Watch Next**
   - Optional section using synthetic scenario model outputs if available
   - Label OTP estimates as estimates

7. **Technical Appendix**
   - Optional and off by default for public visitors
   - Sanitized artifact list, contract status, and QA status
   - No local paths, commit hashes, private-looking log names, or raw run context paths

## 7. Required Implementation Tasks For Task 4F

1. Update `pages/70_PDF_Decision_Pack.py` only unless a small helper is clearly justified.
2. Change public UI wording from internal pipeline terms to public-demo terms:
   - `Pipeline Parameters` -> `Demo Data Contract`
   - `region/mode` -> `Scope`
   - `run_stamp` -> `Demo build`
3. Hide raw provenance by default and remove local path display from the main page and generated PDF.
4. Make `include_qa` or technical appendix off by default for public visitors.
5. Add a business-first cover page and executive summary before technical evidence.
6. Port public-safe versions of:
   - executive summary
   - controllable/inherited/reactionary split
   - station ranking recommendation/status
   - prior-period comparison if available from synthetic selected scope
7. Sanitize output filenames and success messages:
   - Use a simple download name such as `IntelligenceOps_Synthetic_DecisionPack.pdf`.
   - Do not print output directory or run log path in Streamlit.
8. Keep all data reads resolved through the existing synthetic run context.
9. Adjust CLI/default behavior so public code does not default toward a private-project style data directory.
10. Preserve current ReportLab/Kaleido graceful fallback behavior.

## 8. Verification Checklist

- `python -m py_compile pages/70_PDF_Decision_Pack.py`
- Generate a PDF from the public Streamlit page using synthetic demo artifacts.
- Generated PDF contains `synthetic` or `Synthetic Demo`.
- Generated PDF does not contain local filesystem paths.
- Generated PDF does not contain raw run context paths.
- Generated PDF does not contain private scope/mode labels.
- Generated PDF does not contain raw QA filenames unless in an explicitly enabled sanitized appendix.
- Streamlit success message does not print output paths or log paths.
- Public UI does not expose local paths in the default view.
- PDF includes an executive summary before technical appendix.
- PDF includes controllable/inherited/reactionary split using selected-scope synthetic evidence.
- PDF does not copy real values, real station names, real rankings, real dates, manager-specific wording, or employer-identifying text.
- No changes to synthetic data values or KPI definitions.

## 9. Next Codex Prompt For Implementation

```text
Task 4F - Public-safe PDF Decision Pack upgrade

Repo: JosephSigma991/intelligenceops-demo
Branch: master

Goal:
Upgrade the public Streamlit PDF Decision Pack page so the generated PDF is business-first, synthetic-data clearly labeled, and safe for public portfolio visitors.

Do not modify:
- synthetic data values
- KPI definitions
- chart visual design unless needed for PDF ordering
- README
- real IntelligenceOps repo
- HTML cockpit work

Required changes in pages/70_PDF_Decision_Pack.py:
- Replace internal-facing UI wording with public demo wording.
- Do not show local paths, raw run context paths, QA paths, output paths, or run log paths in the default UI or PDF.
- Make technical appendix / QA details optional and off by default.
- Add public-safe PDF structure:
  1. Cover: IntelligenceOps Synthetic Decision Pack
  2. Executive Summary
  3. KPI Snapshot
  4. Delay Accountability with Pareto and controllable/inherited/reactionary split
  5. Station Review
  6. Optional sanitized Technical Appendix
- Use selected-scope synthetic artifacts only.
- Keep ReportLab and Kaleido fallback behavior.
- Use a simple public-safe download filename.

Verification:
- python -m py_compile pages/70_PDF_Decision_Pack.py
- Generate a PDF using demo_data/insight_out.
- PDF must contain "Synthetic Demo" or "synthetic".
- PDF must not contain local filesystem paths.
- PDF must not contain raw run context paths.
- PDF must not contain private scope/mode labels.
- UI success message must not print output or log paths.
- Only pages/70_PDF_Decision_Pack.py should change unless a tiny helper is justified.

Commit message:
Make public PDF decision pack portfolio-safe
```
