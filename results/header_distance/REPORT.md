# Independent header/task distance results

Fresh prompts from a previously observed structure catalog, conditional on six fixed adaptation checkpoints and one selected starting lineage; inference layout intervention does not separate components of the earlier training treatment.

| Checkpoint | Header far/task far | Header far/task near | Header near/task far | Both near | Legacy far | Legacy near |
|---|---:|---:|---:|---:|---:|---:|
| near_seed20271011 | 13.63419 | 0.0001340236 | 12.47181 | 6.819601e-05 | 9.544303 | 6.703504e-05 |
| balanced_seed20271011 | 1.235206e-05 | 7.123395e-06 | 1.344242e-05 | 8.588033e-06 | 4.080732e-05 | 8.634884e-06 |
| near_seed20271012 | 9.620678 | 1.705931e-06 | 7.635036 | 1.017307e-06 | 7.282155 | 1.029651e-06 |
| balanced_seed20271012 | 3.11368e-06 | 3.535085e-06 | 3.845228e-06 | 4.145081e-06 | 4.248013e-06 | 4.131214e-06 |
| near_seed20271013 | 2.773426 | 8.743945e-06 | 2.77315 | 7.057104e-06 | 2.772659 | 7.170452e-06 |
| balanced_seed20271013 | 9.140663e-05 | 6.023527e-05 | 0.000152435 | 0.0001064319 | 8.012036e-05 | 0.0001062537 |

Primary error reduction: 7.626609 nats, conditional95%CI [6.908102286434451, 8.301452193489855].
All three primary directions positive: True.
Balanced primary and nearby-control quality conditions: {'20271011': True, '20271012': True, '20271013': True}.
Predeclared stronger interpretation supported: True.

Every layout and legacy control is retained; full factorial contrasts, first/second errors and one-call comparisons are in report.json.
