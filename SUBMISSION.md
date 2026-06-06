# Final Submission

## Selected ZIP

`submission/submission_domain_live_mid600.zip`

This is the final selected competition submission. The ZIP contains:

```text
model.py
artifacts/domain_prior.json
```

It does not include an adaptive-labeling policy.

## Method Summary

The prediction has the form:

```text
p = clip(c + w_s * subject_prior + w_d * domain_prior, epsilon, 1 - epsilon)
```

where:

- `c` is a conservative global center near `0.600`.
- `subject_prior` is a shrunk ability estimate from public subject observations and subject aliases.
- `domain_prior` is a coarse item-domain adjustment inferred from item text.
- The final probability is clipped away from exactly `0` and `1`.

The hosted evidence favored this compact, calibrated prior family over high-AUC local stacks, item-memory variants, tree/forecast ensembles, and hierarchical OOD variants.

## Contract Checks

```powershell
python tools/check_submission_zip.py submission/submission_domain_live_mid600.zip
python tools/run_smoke_test.py submission/source
```

Expected result: both checks pass and predictions are finite probabilities in `[0, 1]`.
