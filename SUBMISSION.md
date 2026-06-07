# Final Submission

## Selected ZIP

`submission/submission_anchor_fam_b60_debias.zip`

This is the final selected competition submission. The ZIP contains:

```text
model.py
artifacts/domain_prior.json
```

It does not include an adaptive-labeling policy.

## Method Summary

The prediction has the form:

```text
p = sigmoid(anchor_logit - beta * revealed_subject_offset_mean + beta * target_subject_family_offset)
```

where:

- `anchor_logit` is a shrunk logit-space base-rate estimate from platform-random revealed labels.
- `beta` is `0.60` in the selected family-aware debiased variant.
- `revealed_subject_offset_mean` removes strong/weak subject-composition bias from the revealed sample.
- `target_subject_family_offset` is a capped public-data subject ability offset within the target item family.
- If useful revealed labels are unavailable, the runtime falls back to the compact domain-prior formula.

The hosted evidence favored compact calibrated priors over high-AUC local stacks, item-memory variants, tree/forecast ensembles, and hierarchical OOD variants. The final improvement came from anchoring to the hidden-run revealed labels rather than sharpening public-data signals.

## Contract Checks

```powershell
python tools/check_submission_zip.py submission/submission_anchor_fam_b60_debias.zip
python tools/run_smoke_test.py submission/source
```

Expected result: both checks pass and predictions are finite probabilities in `[0, 1]`.
