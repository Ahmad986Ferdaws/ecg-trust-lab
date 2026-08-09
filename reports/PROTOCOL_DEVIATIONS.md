# Protocol deviations

This log is part of the research record. It must be cited by the final report
and model card; entries are never deleted or rewritten to improve the apparent
cleanliness of the experiment.

## DEV-001 — accidental raw fold-10 label-row exposure during a metadata search

**Stage:** pre-fold-9 release audit, before calibration export or model
evaluation

**Date:** August 8, 2026; exact event timestamp unavailable

**Recorded at:** 2026-08-08T23:32:05Z

**Status:** disclosed immediately; contained; no row-level values used

During a code/documentation audit, a read-only search intended to establish the
meaning of the PTB-XL `sex` field was run with this exact command:

```powershell
rg -n -uuu "male.*0|female.*1|0.*male|1.*female|sex" data README.md docs `
  -g "*.md" -g "*.txt" -g "*.html" -g "*.csv" | Select-Object -First 100
```

Because `-uuu` included ignored raw data and the pattern matched full CSV
lines, the bounded output contained 12 documentation matches, the raw CSV
header, and at most approximately 87 source rows. Some visible rows belonged
to fold 10. Full row fields were consequently visible, including identities,
demographics, acquisition metadata, reports, SCP codes, fold assignments, and
file paths. The exact number of raw or fold-10 rows was not reconstructed,
because doing so would require repeating or inspecting the prohibited output.

The auditor stopped immediately, disclosed the incident, did not repeat the
command, and did not copy, summarize, compute on, or use any row-level value in
a recommendation. The age-`300` censored-sentinel finding was independently
obtained earlier from the checked data card, not from the exposed rows.

### Scientific impact

- Strict operator-level fold-10 outcome-label blindness was breached. The
  project must not claim that no person or agent ever saw any fold-10 label.
- No fold-10 waveform was loaded, no model prediction was generated or opened,
  and no fold-10 metric, aggregate label distribution, model ranking, threshold,
  calibration value, or subgroup result was observed.
- The formal one-time fold-10 **model evaluation** remains unopened. Its ledger
  will still govern the first and only prediction/report batch, but the final
  paper and model card must disclose this earlier raw-label-row exposure.
- Because no exposed value informed model, calibration, coverage, subgroup, or
  reporting choices, the benchmark remains usable with this stated limitation;
  the deviation weakens the strongest possible confirmatory-blinding claim.

### Containment and remediation

1. Remaining scientific and reporting choices are frozen in committed code and
   artifacts before fold-9 or fold-10 inference.
2. Prediction export now loads identities for all folds but materializes target
   columns only for folds 1–7 plus the currently authorized role fold.
3. The subgroup builder reads only identity, fold, age, and sex columns and
   produces a self-hashed pre-opening artifact.
4. No further ad hoc raw-data searches are permitted. Fold-10 execution and
   reporting will use only the automated, ledgered exact-six pipeline.
5. This deviation will be carried into the final model card, limitations, and
   reproducibility record rather than being described as a clean blind test.
