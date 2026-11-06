# REVIEW.md: hostile review of v1 and disposition in v2

Paper: "Scaling ML Observability from Pilot to Portfolio" (v1, Zenodo 10.5281/zenodo.18445588).
Sources reviewed: enhanced markdown (4117 lines), TKDE LaTeX v4, Zenodo metadata, v1 code project.

## Round 1: Frontier (hidden assumptions, prerequisites, demanded baselines)

- **Data.** v1 reports "10M fraud transactions", "500K clinical assessments", "5M recommendations",
  "118 models over 6 months", "47 models over 9 months". No dataset, access path, or log exists in the
  repo or code project. Assumption: these were never measured. Prerequisite for v2: public data only
  (OpenML adult, bank-marketing, electricity), cached with sha256.
- **Ground truth.** v1 "sensitivity/specificity" needs labelled drift/no-drift windows. Real logs do not
  provide this. v2 injects drift into real feature distributions (labelled as simulated) and uses one
  real temporal shift (Elec2) without a drift label.
- **Thresholds.** v1 compares detectors at fixed thresholds (PSI 0.2, KS 0.1, W 0.25). Different
  detectors at arbitrary thresholds are not comparable. Demanded baseline: TPR at a calibrated FPR.
- **Binning.** PSI/KL depend on bin count, bin placement, and zero-bin smoothing. v1 states none of these
  in its "benchmark". v2 fixes B=10 reference-quantile bins, 0.5 additive smoothing, and ablates B.
- **Sample size.** PSI rule-of-thumb thresholds ignore the window size. Expected null PSI grows like
  (B-1)(1/n + 1/m). A reviewer will ask for false alarm rates per window size.
- **Multiplicity.** v1's scaling story never addresses that 1000 models x 20 features = 20000 tests/day.
  Demanded baselines: per-test alpha, Bonferroni, Benjamini-Hochberg, hierarchical.
- **Labels.** v1 Tier 2 relies on delayed labels; the claim that input drift predicts degradation is
  untested. Demanded: agreement of input-drift and performance alarms on a real stream.
- **Libraries/hardware.** v1 gives timings ("<1ms", "50-200ms") without hardware or n. v2 reports
  median wall time on a stated machine and sizes.
- **Sequential baselines.** ADWIN, DDM, Page-Hinkley, CUSUM: cited but not run in v1; not run in v2
  either (recorded as threat).

## Round 2: Breakpoints

| Breakpoint | v2 handling |
|---|---|
| Scale x100 (10 -> 1000 models) | Tested: portfolio N in {10, 100, 1000} |
| Missing data / timeouts of a detector | Tested: 10% missing p-values; policies "exclude" vs "p=1" |
| Window too small | Implemented: abstain below N_MIN=30 (status insufficient_data); unit-tested |
| Unseen categories | Implemented: out-of-vocabulary bin; unit-tested |
| Discrete / zero-inflated / tied features | Tested: exact tie-aware KS; results split discrete vs continuous |
| Correlated features within a model | Tested: equicorrelated rho=0.6 |
| Window size | Tested: n in {100, 500, 2000} (benchmark) and {100, 200, 500} (portfolio) |
| Gradual drift | Tested: streaming ramp, delay in batches |
| Label feed missing / delayed | Implemented: performance test returns unknown below 30 labels; delay not simulated (threat) |
| Real temporal shift | Tested: Elec2, 5 training windows |
| Adversarial inputs, multivariate-only shift | Not tested (threat) |
| Seasonality / autocorrelated windows | Not tested in simulation (threat); Elec2 partly covers it |

## Round 3: Defensibility

- **Is the primary metric right?** False alarms/day and 3-day event recall are operational proxies for
  on-call cost and harm from late detection. They ignore severity and business cost (threat).
- **What falsifies H1?** BH keeping < 80% of per-test 3-day recall, or removing < 90% of false alarms,
  with the CI fully below the bound (preregistration.json).
- **Measurement bias.** Calibrated thresholds use separate null windows (calibration vs evaluation split).
  Simulated drift is Gaussian mean shift in the portfolio: favours KS-type tests; stated as limitation.
  Elec2 "performance alarm" uses a z-test that assumes independent errors within a week (threat).
- A runtime pilot (seed 0, after registration) was used only to size the job; registration was not edited.

## v1 claims: unsupported, fabricated, or mis-cited

| v1 claim | Status | v2 action |
|---|---|---|
| "91% of production ML models experience degradation" | No source; not in any cited abstract | Removed |
| "47 days average detection", "67% lack monitoring", "$2.3M average cost", "15% automate retraining" | Unsourced | Removed |
| Three "real-world failure" cases (bank, lender, retailer) with $ figures | Unverifiable anecdotes | Removed |
| Case studies: fraud (15M customers, 50M tx/day), healthcare (10 hospitals), e-commerce | No data, logs, or code | Removed |
| "Reduces MTTD by 60-80%", "alert fatigue reduced by 70%", p<0.001, p=0.003, kappa 0.81 | No experiment exists | Removed; replaced by measured portfolio simulation |
| Detector table: PSI 78/85%, KL 75/82%, W 88/91%, KS 72/94%, chi2 80/88% | No benchmark exists; thresholds arbitrary | Removed; measured at calibrated FPR |
| "Wasserstein highest accuracy", "PSI best balance" | Unsupported | Reframed as measured, condition-dependent results |
| Compute times "<1ms", "50-200ms", "5ms" | No hardware/n | Measured with n and machine stated |
| "Sub-5-minute detection latency at scale", "38x vs 200x compute" | Unmeasured | Removed |
| Tier scoring function weights, "46%/21% MTTD reduction" ablation | Unmeasured | Removed |
| Interpretability scores 1-5 "from practitioners" | No survey | Removed |
| PSI interpretation bands 0.1/0.25 as universal | Folk rule | Reframed: measured as n-dependent (H2) |
| "Production-ready", "comprehensive framework" | Hype | Removed |
| Refs to blogs (Monte Carlo, Evidently blog, DataCamp, Aliac) | Not verifiable sources for numbers | Removed; tools cited as software only |
| Sculley 2022 "IEEE Softw. 39(2)" "ML systems: hidden technical debt and observability gaps" | Could not be verified | Removed |
| Kaur 2023 "IEEE Trans. AI 4(3)" MLOps maturity survey | Could not be verified | Removed |
| van der Linden 2022 NannyML NeurIPS workshop paper | Could not be verified | Replaced by software citation |
| Kubernetes manifests, Grafana JSON, SQL schemas, retraining orchestrator | Untested engineering artefacts | Kept only as an architecture description (no claims of performance) |
| Regulatory mapping (SR 11-7, EU AI Act, GDPR) | Legal claims outside evidence | Removed |

## v1 scores (C1-C11)

| Criterion | v1 | Reason |
|---|---|---|
| C1 filler/hype/em-dash | Fail | "comprehensive", "production-ready", many em-dashes |
| C2 citation rigor | Fail | blogs; unverifiable entries; numbers attributed to nobody |
| C3 algorithmic soundness | Partial | PSI/KL/W/KS formulas, no system model/objective/constraints |
| C4 outcome handling | Fail | no missing-label, timeout, or abstention semantics |
| C5 substantiated measurements | Fail | percentages without numerators/denominators/seeds |
| C6 internal consistency | Fail | abstract 61-78% vs body 23-78%; thresholds differ across sections |
| C7 labels referenced | Partial | |
| C8 claims within evidence | Fail | anecdotes presented as validation |
| C9 numeric grounding | Fail | nearly all numbers ungrounded |
| C10 threats to validity | Fail | none |
| C11 preregistration / honest verdict | Fail | none |

Dimensions (1-10): Soundness 2, Novelty 3, Clarity 5, Reproducibility 1. Weighted 0.6x2 + 0.4x3 = 2.4 (reject).

## v2 re-score

### Process notes (disclosed)
- preregistration.json was written before any experiment ran. A single-seed runtime pilot followed; the registration was not edited.
- After the first full run, a tolerance bug in `lattice_step` made Elec2 `period` (48 levels stored to 6 decimals) count as continuous. Fixed (grid detection plus snapping to stored levels) and everything was re-run. The fix changes only the detector-benchmark rows of that feature; H1 and H3 results were identical and H2 unchanged at reported precision.
- KS p-values in the portfolio now come from a lookup table of `kstwo.sf` on the 1/lcm(n,m) lattice (identical values, unit-tested), which cut the run from ~20 min (loaded machine) to ~3.5 min.

### Outcomes
- H1 (primary): SUPPORTED. R_FA 0.9956 [0.9952, 0.9961]; RR 0.866 [0.850, 0.883] (N=1000, n=200, 5 seeds).
  Breakpoint: at n=100 BH 3-day recall 0.580 vs per-test 0.964 (retention fails). Reported in abstract and conclusion.
- H2: SUPPORTED. PSI>0.1 null FPR gap n=100 vs n=2000 = 0.221 [0.214, 0.229].
- H3: SUPPORTED by rule but degenerate: input alarms fired in 200/200 weeks, so kappa = 0 (undefined in one run). Reported as such.

### Criteria (v2)
| Criterion | v2 | Evidence |
|---|---|---|
| C1 | Pass | desk check: no filler/hype/em-dash |
| C2 | Pass | 43 bib entries; 38 verified via Crossref/arXiv; 5 software/dataset @misc with URL only, no numbers attributed |
| C3 | Pass | 10 labelled Methodology equations (system model, detectors, policies, objective, endpoints) |
| C4 | Pass | Outcome Handling subsection + Algorithm 1; abstention, error, unknown p-values, OOV, label feed; missing scenario measured |
| C5 | Pass | 5 seeds / 5 training windows, mean+-std and t-CIs; numerator/denominator tables; ablations: bins, discrete vs continuous, policies, window size, correlation, missing |
| C6 | Pass | all tables/figures generated from results/ by code |
| C7 | Pass | every fig/tab/eq/alg label referenced |
| C8 | Pass | "simulated" stated in setup, captions, abstract; Elec2 is the only real shift |
| C9 | Pass | desk check numeric grounding (257 numbers checked) |
| C10 | Pass | Threats lists untested breakpoints, omitted baselines, abstentions |
| C11 | Pass | prereg + verdict.json; no proves/guarantees/always/eliminates |

Dimensions (1-10, self-assessed): Soundness 7, Novelty 6, Clarity 8, Reproducibility 9. Weighted 0.6x7 + 0.4x6 = 6.6.
Novelty is modest (known statistics, new measurement of their portfolio-scale trade-off); soundness is capped by simulated drift
and one real stream. This is below the 7.0 accept line of the rubric; the honest framing is preferred over inflated claims.

### Final desk check
```
Desk check: .../paper/main.tex  sections=['Introduction', 'Related Work', 'Methodology', 'Evaluation', 'Threats to Validity', 'Conclusion', 'Disposition of Version-1 Claims', 'Acknowledgment']  abstract_words=229  method_eqs=10  numbers_checked=257
PASS
```
