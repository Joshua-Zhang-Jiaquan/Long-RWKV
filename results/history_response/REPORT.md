# Exhaustive evidence-response results

Post-study exact-history diagnostic on reused problems, conditional on observed adaptation seeds and selected starting lineage; not independent replication.

| Checkpoint | Far paired CE | Floor | Centering penalty | Both correct | Mean unaffected change |
|---|---:|---:|---:|---:|---:|
| original | 5.256043 | 0.867215 | 4.388828 | 0.06738281 | 0.09876315 |
| near_seed20271011 | 1.990361 | 0.6838045 | 1.306557 | 0.04833984 | 0.05938094 |
| balanced_seed20271011 | 2.780001e-05 | 4.123187e-11 | 2.779997e-05 | 1 | 0.0001221228 |
| near_seed20271012 | 1.750135 | 0.6635025 | 1.086633 | 0.03955078 | 0.010086 |
| balanced_seed20271012 | 1.213607e-06 | 1.762135e-11 | 1.213589e-06 | 1 | 8.276648e-06 |
| near_seed20271013 | 0.6931181 | 0.6931133 | 4.86735e-06 | 0.008789062 | 6.490543e-05 |
| balanced_seed20271013 | 2.972982e-05 | 1.420554e-07 | 2.958777e-05 | 1 | 7.577869e-05 |

Far near-minus-balanced paired CE: 1.47785 nats, conditional 95% CI [1.3274506371527812, 1.6370838443961722].
All three seed directions positive: True.
Archived metric maximum discrepancy: 8.29697e-05.

All positions, individual seed intervals and worst-case summaries are in report.json; all 43,008 individual interventions are in exact_*.json. Symmetrized paired CE is not the original endpoint E.
