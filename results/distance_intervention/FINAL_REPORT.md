# Controlled distance-intervention results

Validated 1008 exact conditions and 2520 timed requests.

| Checkpoint | 16K info KL | Gain over one | Far evidence CE | Two-call seconds |
|---|---:|---:|---:|---:|
| original | 14.6671 | -11.8678 | 6.4299 | 1.3026 |
| near_seed20271011 | 6.1875 | -3.4148 | 2.1547 | 1.3014 |
| balanced_seed20271011 | 0.0002 | 2.7724 | 0.0000 | 1.3016 |
| near_seed20271012 | 4.8595 | -2.0869 | 2.4997 | 1.3072 |
| balanced_seed20271012 | 0.0000 | 2.7726 | 0.0000 | 1.3060 |
| near_seed20271013 | 1.8482 | 0.9244 | 0.6931 | 1.3008 |
| balanced_seed20271013 | 0.0003 | 2.7724 | 0.0000 | 1.3019 |

Primary effect: near-only minus distance-balanced information-set conditional error at16K far.
- Seed 20271011: 8.3036, paired problem95%CI [7.0562, 9.6582].
- Seed 20271012: 7.0757, paired problem95%CI [6.0192, 8.1509].
- Seed 20271013: 2.7723, paired problem95%CI [2.7721, 2.7725].

Seed-averaged effect: 6.0506, conditional problem95%CI [5.3916, 6.7353].

Paired problem resampling, conditional on these three adaptation seeds and selected starting lineage; not population training-seed uncertainty

- All adaptation runs share a selected starting checkpoint.
- Fresh conditions use a previously observed held-out structure catalog.
- Distance manipulation moves instructions together with equations.
- Counterfactual responses use one fixed visible history; endpoint KL integrates all histories.
- Same-call dependence differences are not isolated unless measured estimation errors are controlled.
- A restricted task and implementation comparison does not establish general language-model superiority.
