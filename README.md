# AIMS Predictive Evaluation Competition

Clean publication repository for the Codabench Predictive Evaluation Competition.

## Final Result

- Selected submission: `submission/submission_anchor_fam_b60_debias.zip`
- Hosted negative log-loss: `-0.57`
- Approximate log-loss: `0.57`
- Competition-reported AUC-ROC: `0.73`
- Method family: revealed-label anchored base-rate model

The final method uses the platform's random revealed labels to estimate a hidden-run base rate, then applies a moderate family-aware subject correction. It falls back to the compact domain prior when useful labels are unavailable. It does not use exact public item memory, tree/forecast ensembles, or an active adaptive-labeling policy.

## Top Hosted Submissions

Ranks use competition-style ties. The three `-0.60` rows are tied for rank 3, giving five top hosted rows total.

| Rank | Submission | Hosted negative log-loss | Approximate log-loss | Method family |
| ---: | --- | ---: | ---: | --- |
| 1 | `submission_anchor_fam_b60_debias.zip` | `-0.57` | `0.57` | Label-anchored, family-aware, debiased |
| 2 | `submission_anchor_g60.zip` | `-0.59` | `0.59` | Label-anchored, global subject offset |
| T-3 | `submission_anchor_g60_debias.zip` | `-0.60` | `0.60` | Label-anchored, global subject offset, debiased |
| T-3 | `submission_anchor_fam_b50_debias.zip` | `-0.60` | `0.60` | Label-anchored, family-aware, lower subject weight |
| T-3 | `submission_anchor_fam_b60.zip` | `-0.60` | `0.60` | Label-anchored, family-aware |

## Repository Contents

- `submission/`
  - `submission_anchor_fam_b60_debias.zip`: uploadable final competition ZIP.
  - `source/`: extracted final submission source for inspection.
- `unused_submissions/`
  - Alternate submission ZIPs archived for auditability, excluding the selected final ZIP.
- `report/`
  - `main.pdf`: technical report.
  - `main.tex`, `references.bib`: report source.
  - `figures/`, `tables/`: generated report artifacts.
- `results/`
  - `eval_finals.txt`: hosted submission results used by the report.
  - `hosted_results_clean.csv`, `family_summary.csv`: parsed result tables.
- `tools/`
  - Minimal starter-kit validation tools for ZIP contract and smoke tests.
- `sample_data/`
  - Tiny starter-kit sample files required by the smoke test.

## Validation

From the repository root:

```powershell
python tools/check_submission_zip.py submission/submission_anchor_fam_b60_debias.zip
python tools/run_smoke_test.py submission/source
```

To rebuild the report PDF from source:

```powershell
cd report
tectonic main.tex
```

## Publication Scope

This clean repository intentionally excludes exploratory submission ZIPs, validation extraction folders, Python caches, raw downloaded benchmark data, virtual environments, and iterative analysis artifacts. The report explains the broader method search without requiring those working files.
