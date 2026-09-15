# Independent payload-final2 command record

Working directory unless otherwise noted:

`/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/DiffRWKV-RELAY`

No qz command, network request, install, git command, CUDA operation, model construction, or checkpoint load was executed.

## Current manifest and 35-file streaming verification

```bash
sha256sum scale/experiments/nonlatent_iclr/qualification/runtime_manifest.json
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python \
  -m scale.experiments.nonlatent_iclr.qualification.cli validate \
  --output-dir /tmp/nonlatent-final2-review-readonly \
  --nonce 00000000000000000000000000000000 \
  --run-id qualification-20260912-01-review000 \
  --controller-receipt /inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification/controller-receipts/qualification-20260912-01-review000.json \
  --controller-receipt-timeout-seconds 60 \
  --expected-world-size 8 --timeout-seconds 1200 \
  --nccl-timeout-seconds 120 --memory-fraction 0.90 \
  --staged-root /inspire/hdd/global_user/zhangjiaquan-253108540222/qz_stage_traj4096_v7/scale \
  --checkpoint-dir /inspire/hdd/global_user/zhangjiaquan-253108540222/outputs_birwkv_diffusion/m4-loop-2p9b/step_00004750 \
  --model-dir /inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-Goose-World3-2.9B-HF \
  --manifest scale/experiments/nonlatent_iclr/qualification/runtime_manifest.json \
  --manifest-sha256 92fbf9afe7d0c7b47829337639dae54debddc8e9409c0903f30358aa75aa656c \
  --preflight-receipt /tmp/nonlatent-final2-review-readonly/preflight.json
```

Exit 0. Observed manifest SHA matched the requested value; receipt reported `PREFLIGHT_VERIFIED`, 35 files, and all six expected package versions. Validate did not read or create the future controller receipt.

## Frozen payload and evidence receipts

```bash
sha256sum -c .omo/evidence/nonlatent-rnn-dlm-iclr-research/qualification-20260912-01/payload-final2/payload-sha256.txt
```

All 25 entries reported `OK`.

From the payload-final2 directory:

```bash
sha256sum -c evidence-sha256.txt
```

All 14 entries reported `OK`.

## Full and targeted CPU tests

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python -m pytest -q -p no:cacheprovider \
  scale/tests/nonlatent_iclr
```

Result: 143 passed in 17.77 seconds.

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python -m pytest -q -p no:cacheprovider \
  scale/tests/nonlatent_iclr/test_qualification_runtime_contracts.py
```

Result: 4 passed in 0.24 seconds.

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python -m pytest -q -p no:cacheprovider \
  scale/tests/nonlatent_iclr/test_qualification_controller_receipt.py \
  scale/tests/nonlatent_iclr/test_qualification_controls.py \
  -k 'controller or stale or missing_rank or malformed or partial_scope'
```

Result: 16 passed, 13 deselected in 0.33 seconds.

## Deadline arithmetic

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python -c \
  'from scale.experiments.nonlatent_iclr.qualification.runtime_config import QualificationLimits; q=QualificationLimits(); phases=q.preflight_timeout_seconds+q.runtime_timeout_seconds+q.aggregation_timeout_seconds; assert phases==1590<=q.workflow_timeout_seconds==1650; assert q.workflow_timeout_seconds+q.kill_grace_seconds==1680<1800; assert q.controller_receipt_timeout_seconds==60; assert q.nccl_timeout_seconds==120; print(phases)'
```

Exit 0; phase total was 1,590 seconds.

## Shell, help, and bad-input boundaries

Commands ran with `PYTHONDONTWRITEBYTECODE=1` where Python imported modules and with an EXIT-trapped `/tmp/nonlatent-final2-boundaries.XXXXXX` directory:

```bash
/usr/bin/bash -n scale/experiments/nonlatent_iclr/qualification/run_qualification.sh
/usr/bin/bash scale/experiments/nonlatent_iclr/qualification/run_qualification.sh --help
/usr/bin/python -m scale.experiments.nonlatent_iclr.qualification.cli --help
/usr/bin/python -m scale.experiments.nonlatent_iclr.qualification.probe --help

/usr/bin/python -m scale.experiments.nonlatent_iclr.qualification.cli validate \
  --output-dir "$tmp_dir/runtime" --nonce 00000000000000000000000000000000 \
  --run-id qualification-20260912-01-review000 \
  --controller-receipt "$tmp_dir/receipt.json" --timeout-seconds 1201 \
  --manifest "$tmp_dir/missing.json" --manifest-sha256 "<64-zero-digest>"

/usr/bin/python -m scale.experiments.nonlatent_iclr.qualification.cli validate \
  --output-dir "$tmp_dir/nccl" --nonce 00000000000000000000000000000000 \
  --run-id qualification-20260912-01-review000 \
  --controller-receipt "$tmp_dir/receipt.json" --nccl-timeout-seconds 121 \
  --manifest "$tmp_dir/missing.json" --manifest-sha256 "<64-zero-digest>"

/usr/bin/python -m scale.experiments.nonlatent_iclr.qualification.cli validate \
  --output-dir "$tmp_dir/controller" --nonce 00000000000000000000000000000000 \
  --run-id qualification-20260912-01-review000 \
  --controller-receipt "$tmp_dir/receipt.json" \
  --controller-receipt-timeout-seconds 61 \
  --manifest "$tmp_dir/missing.json" --manifest-sha256 "<64-zero-digest>"

env -u QUALIFICATION_WORKFLOW_INNER -u QUALIFICATION_RUN_ID \
  -u QUALIFICATION_NONCE -u QUALIFICATION_MANIFEST_SHA256 \
  -u QUALIFICATION_JOB_RECEIPT \
  /usr/bin/bash scale/experiments/nonlatent_iclr/qualification/run_qualification.sh
```

Syntax/help exits were 0. Invalid runtime, NCCL, controller-wait, and launcher inputs exited 2 with their exact structured rejection. No bad-input artifact or child process remained.

## Checkpoint representation and unsafe-global scan without loading

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python -c \
  'import pickletools, torch, zipfile; p="/inspire/hdd/global_user/zhangjiaquan-253108540222/outputs_birwkv_diffusion/m4-loop-2p9b/step_00004750/model.pt"; z=zipfile.ZipFile(p); n=next(x for x in z.namelist() if x.endswith("/data.pkl") or x=="data.pkl"); print(list(pickletools.genops(z.read(n)))[:5]); print(torch.serialization.get_unsafe_globals_in_checkpoint(p)); print(torch.cuda.is_initialized()); print("torch_load_called=false")'
```

The first object is `collections OrderedDict`; static unsafe globals were `[]`; CUDA was uninitialized; `torch.load` was not called.

## HF safetensors headers only

A standard-library reader consumed only each shard's eight-byte header length and JSON header. Result: 1,059 tensors, 2,947,735,040 elements, all `BF16`; no tensor payload was loaded.

## Cleanup check

```bash
test ! -e /tmp/nonlatent-final2-review-readonly
! compgen -G '/tmp/nonlatent-final2-boundaries.*'
```

Both checks passed.
