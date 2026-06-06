# AIMS Predictive Evaluation Competition

Clean publication repository for the Codabench Predictive Evaluation Competition.

## Final Result

- Selected submission: `submission/submission_domain_live_mid600.zip`
- Hosted negative log-loss: `-0.62`
- Approximate log-loss: `0.62`
- Competition-reported AUC-ROC: `0.72`
- Method family: compact domain prior

The final method uses a conservative global center, a shrunk subject prior, and a coarse item-domain prior. It does not use exact public item memory, public benchmark lookup, tree/forecast ensembles, or an active adaptive-labeling policy.

## Repository Contents

- `submission/`
  - `submission_domain_live_mid600.zip`: uploadable final competition ZIP.
  - `source/`: extracted final submission source for inspection.
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
python tools/check_submission_zip.py submission/submission_domain_live_mid600.zip
python tools/run_smoke_test.py submission/source
```

To rebuild the report PDF from source:

```powershell
cd report
tectonic main.tex
```

## Publication Scope

This clean repository intentionally excludes exploratory submission ZIPs, validation extraction folders, Python caches, raw downloaded benchmark data, virtual environments, and iterative analysis artifacts. The report explains the broader method search without requiring those working files.
