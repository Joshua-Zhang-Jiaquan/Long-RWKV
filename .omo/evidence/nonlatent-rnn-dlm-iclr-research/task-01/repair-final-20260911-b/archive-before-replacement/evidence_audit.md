# Nonlatent historical evidence audit

## Source roots

- snapshot: `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/DiffRWKV-RELAY/DAN/v7_arch_round`
- external: `/inspire/hdd/global_user/zhangjiaquan-253108540222`
- live: `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/DiffRWKV-RELAY/scale`
- staged: `/inspire/hdd/global_user/zhangjiaquan-253108540222/qz_stage_traj4096_v7/scale`

## Artifacts

| Source | Origin | Path | Kind | Identity | SHA-256 | Bytes | Files | Reason |
|---|---|---|---|---|---|---:|---:|---|
| R1 | snapshot | README.md | document | sha256 | d5c8562fb2cc2f1033d9c433d71baf6ed76c49dd737d8343560fe25c8b8621b2 | 5055 | 1 |  |
| R1 | snapshot | ARCHITECTURE.md | document | sha256 | 651c1abbca41345b0d55d404c5699de1ce1ba6fb60cbd94552152acf1e332427 | 5474 | 1 |  |
| R1 | snapshot | TRAINING_HISTORY.md | document | sha256 | acce643d69eb03ee805f13c70f80b28da16c891a0b714c27ab3314f54d685078 | 5238 | 1 |  |
| R1 | snapshot | report.md | document | sha256 | ed2901cae9d4fc748c0ff688c3119e360038dfc4ed70951ddc6c9494787351e3 | 12499 | 1 |  |
| R2 | snapshot | results/gate_analysis_s4750_20260910.txt | raw_record | sha256 | 470fc49d752544cdac61f4ed1501f2ff9946ff094554f2d3147c040db7e3b280 | 16476 | 1 |  |
| R2 | snapshot | results/ability_curve_m4loop.csv | raw_record | sha256 | 6cf28727e4e4112feef428393fd35b2bfd45abc462294a97ffa98f23fdd1b88f | 338 | 1 |  |
| R2 | snapshot | results/ability_curve_m4mhc.csv | raw_record | sha256 | 814abdef6476817586459e35873bd9065c07acbadc3ee5b9cb38213735dead68 | 466 | 1 |  |
| R3 | snapshot | scripts/wl_gate_analysis.py | code | sha256 | ef056c461ae5c8002159e14d9bca77a92efc9cd1ce4f893c145be30e3a85428a | 5155 | 1 |  |
| R4 | snapshot | code/eval/sampler_eval_offline.py | code | sha256 | 6ad36b9250593e22f52dcb723043af4d354b747aa0cf3f094975946d1912811a | 12599 | 1 |  |
| R5 | snapshot | code/models/birwkv7_diffusion.py | code | sha256 | 2591f58b1dfb7b567b979fb1f05cbb8c1e9a60490fc9fba293fcb2da48e0154a | 62145 | 1 |  |
| R5 | snapshot | code/models/residual_streams.py | code | sha256 | 5ba645d5924b1782c60fccd8a17dec26bc904e3675f2f1834d9e948cac7e9b68 | 9184 | 1 |  |
| R5 | snapshot | code/train/test_backbone_loop.py | code | sha256 | 748fb431f0492ce0f8447e2a172e2c775c4bafd101edd83f6845a4c7051351d2 | 6371 | 1 |  |
| R6 | snapshot | code/eval/lm_eval.py | code | sha256 | 49772946d6e0bd1a2f27f9c0bbb9b152662e475badb6f5af440b381be53f84c0 | 22118 | 1 |  |
| R6 | snapshot | code/train/train_birwkv_diffusion.py | code | sha256 | 0bee15b5af05a785b70ddfeffa3064161f04beccea36b44e9bfd01029e286b60 | 95892 | 1 |  |
| R6 | snapshot | specs/m4_loop_32gpu_half.json | manifest | sha256 | 9d8d1951d80871440567f4653b2b903022fc5128240798347466db949ff0105f | 2397 | 1 |  |
| R6 | snapshot | JOBS.md | document | sha256 | 0e32669d31dfde17222657601f59b05692531111c0e38e0ee1e4ea52d5d47afb | 2842 | 1 |  |
| R3 | external | cap_lm_m4loop_s4750 | raw_record | sha256 | 6cfca24579b181c9aa848ce2b25a4af70f4a2027aa2e4ce496d9b8c0d40fa998 | 172238760 | 46 | bounded manifest of 46 files |
| R3 | external | cap_lm_n2_s4000 | raw_record | sha256 | f9ff54fac6cae4aef7afee2bc05e9c6c36ecbc2891633261125cb32d426845d9 | 172238354 | 46 | bounded manifest of 46 files |
| R3 | external | cap_lm_n2_s6000 | raw_record | sha256 | e538c74187172b40e40e7b435add1f8f00e23c340f9b77e3acf7c05351e5d555 | 172232880 | 46 | bounded manifest of 46 files |
| R3 | external | cap_lm_m4loop_s4750_reps2 | raw_record | sha256 | eb636a02b887ed763f3c757b08211ebece84189e6369098f794255d490df449c | 172237084 | 46 | bounded manifest of 46 files |
| R3 | external | cap_lm_m4loop_s4750_reps3 | raw_record | sha256 | 19b745b0c0667a292925439dc2fb3c16a60de7aef9400cb15dcc904d89b5b464 | 172237204 | 46 | bounded manifest of 46 files |
| R3 | external | cap_lm_m4loop_s4750_reps4 | raw_record | sha256 | dbfaef6ee58b1efcdd6d0a15e1da4fc469c663e696a4bdaa88fe8ab7ccb23317 | 172238294 | 46 | bounded manifest of 46 files |
| R3 | external | sampler_gate_m4_loop_s4750 | raw_record | sha256 | 92211848900ef3f481b0c38f036ef6f77221993c4c28a1f237dff794867562d0 | 3363571 | 9 | bounded manifest of 9 files |
| R3 | external | sampler_gate_m4_n2_s4000 | raw_record | sha256 | e16c55aa37c9da1b1ded5e0469fd861db549aa141d3d3a52ce4eaec43702f715 | 3365107 | 9 | bounded manifest of 9 files |
| R3 | external | sampler_gate_m4_n2_s6000 | raw_record | sha256 | f6fd1abf678c5531bf787acff3f226cdf4be08249754d58741a10e25cec42685 | 3366648 | 9 | bounded manifest of 9 files |
| R3 | external | sampler_gate_m4_loop_s4750_reps2 | raw_record | sha256 | cd9b1f512a6a2ec8a036ed7b2895967eb41d51c2b8d22b5f97d18e7f217a7e6e | 3364524 | 9 | bounded manifest of 9 files |
| R3 | external | sampler_gate_m4_loop_s4750_reps3 | raw_record | sha256 | 565b4d1c0f3e9c3a587ff229fca86e37e70c1a733c728c1ee4c34476f74877e5 | 3364366 | 9 | bounded manifest of 9 files |
| R3 | external | sampler_gate_m4_loop_s4750_reps4 | raw_record | sha256 | 572eaf15035728823a3fe14b0d44ea4086f378d5d3c03f5f5bb21baed081aa42 | 3364215 | 9 | bounded manifest of 9 files |
| R3 | external | m2_baseline_triangle/ability_curve_m4loop.csv | raw_record | sha256 | 6cf28727e4e4112feef428393fd35b2bfd45abc462294a97ffa98f23fdd1b88f | 338 | 1 | bounded raw file |
| R6 | external | outputs_birwkv_diffusion/m4-loop-2p9b/step_00004750/meta.json | metadata | sha256 | 321ce0465b2cedb13c870c1ad4ec2c7c36d14cd37670e9caa018b9e63954e7ef | 43 | 1 | metadata identity; payload inference blocked |
| R6 | external | outputs_birwkv_diffusion/n2-knowpt-2p9b/step_00006000_probe_copy/meta.json | metadata | sha256 | ffd02c77be71b72b28413e3ab58cff63fbc158239e91b0a1c820b304192b24e2 | 43 | 1 | metadata identity; payload inference blocked |
| R6 | external | models/RWKV7-Goose-World3-2.9B-HF/tokenizer_config.json | metadata | sha256 | 4e03aa0f5d6b1a4006a0d9e9f070f01418e734ab17c7c48f28333596d6bd5e29 | 1101 | 1 | metadata identity; payload inference blocked |

## Historical claims

| Source | State | Detail | Exact records | Fields |
|---|---|---|---|---|
| R2 | claimed | LM1B historical loss/PPL assertion; interpretation requires task2 | cap_lm_m4loop_s4750/lm1b/merged/merged_m4loop_endpoint_ckpt_nll-ppl-bits.json, results/gate_analysis_s4750_20260910.txt | record_count, records[].metrics.nll, records[].metrics.ppl |
| R2 | claimed | WikiText103 historical loss/PPL assertion; interpretation requires task2 | cap_lm_m4loop_s4750/wikitext103/merged/merged_m4loop_endpoint_ckpt_nll-ppl-bits.json, results/gate_analysis_s4750_20260910.txt | record_count, records[].metrics.nll, records[].metrics.ppl |
| R2 | claimed | causal path historical comparison; interpretation requires task2 | cap_lm_m4loop_s4750/lm1b/merged/merged_m4loop_endpoint_ckpt_nll-ppl-bits.json, cap_lm_m4loop_s4750/wikitext103/merged/merged_m4loop_endpoint_ckpt_nll-ppl-bits.json | records[].arm, records[].document_id, records[].metrics.nll |
| R2 | claimed | 4.98B capacity accounting assertion; interpretation requires task2 | outputs_birwkv_diffusion/m4-loop-2p9b/step_00004750/meta.json, m2_baseline_triangle/ability_curve_m4loop.csv | step, tokens_seen, csv.tokens_seen, csv.score |
| R2 | claimed | max_run_frac direction assertion; interpretation requires task2 | sampler_gate_m4_loop_s4750/merged/merged_m4loop_endpoint_ckpt_em-tau-residue-max_run_frac-distinct_frac.json, results/gate_analysis_s4750_20260910.txt | records[].arm, records[].metrics.max_run_frac, delta, ci95 |
| R2 | claimed | masked-token accuracy assertion; interpretation requires task2 | sampler_gate_m4_loop_s4750/merged/merged_m4loop_endpoint_ckpt_em-tau-residue-max_run_frac-distinct_frac.json, results/gate_analysis_s4750_20260910.txt | records[].arm, records[].metrics.em, delta, ci95 |
| R2 | claimed | tau commit-order assertion; interpretation requires task2 | sampler_gate_m4_loop_s4750/merged/merged_m4loop_endpoint_ckpt_em-tau-residue-max_run_frac-distinct_frac.json, results/gate_analysis_s4750_20260910.txt | records[].arm, records[].metrics.tau, delta, ci95 |
| R2 | claimed | r100 fully-masked assertion; interpretation requires task2 | sampler_gate_m4_loop_s4750/merged/merged_m4loop_endpoint_ckpt_em-tau-residue-max_run_frac-distinct_frac.json, results/gate_analysis_s4750_20260910.txt | records[].arm, records[].metrics, delta, ci95 |

## Snapshot/live/staged views

| View | State | Detail |
|---|---|---|
| snapshot | observed | present directory: /inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/DiffRWKV-RELAY/DAN/v7_arch_round |
| live | observed | present directory: /inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/DiffRWKV-RELAY/scale |
| staged | observed | present directory: /inspire/hdd/global_user/zhangjiaquan-253108540222/qz_stage_traj4096_v7/scale |

## Accounting

Packed capacity is never counted as unique tokens.

Historical interpretation remains blocked pending task2; this audit does not recompute science.
