# Independent recheck command record

Working directory for every command:

`/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/DiffRWKV-RELAY`

No command below contacted qz or another network service, initialized CUDA, constructed a model, called `torch.load`, installed software, or changed product/state files.

## Runtime manifest and all 32 bound files

```bash
sha256sum scale/experiments/nonlatent_iclr/qualification/runtime_manifest.json
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python -m scale.experiments.nonlatent_iclr.qualification.cli validate \
  --output-dir /tmp/qualification-preflight-recheck-readonly \
  --nonce abababababababababababababababab \
  --expected-world-size 8 --timeout-seconds 1500 --memory-fraction 0.90 \
  --staged-root /inspire/hdd/global_user/zhangjiaquan-253108540222/qz_stage_traj4096_v7/scale \
  --checkpoint-dir /inspire/hdd/global_user/zhangjiaquan-253108540222/outputs_birwkv_diffusion/m4-loop-2p9b/step_00004750 \
  --model-dir /inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-Goose-World3-2.9B-HF \
  --manifest scale/experiments/nonlatent_iclr/qualification/runtime_manifest.json \
  --manifest-sha256 364010efe2d9313c35e017a44620d90df16bdc0767a8912f3d518ad4dde1fdee \
  --preflight-receipt /tmp/qualification-preflight-recheck-readonly/preflight.json
```

Exit 0. Receipt: `PREFLIGHT_VERIFIED`, 32 files, expected six package versions. The output-dir argument was not created by the read-only validate command.

## Frozen payload receipt

```bash
sha256sum -c .omo/evidence/nonlatent-rnn-dlm-iclr-research/qualification-20260912-01/payload-final/payload-sha256.txt
```

Exit 0; all 18 entries reported `OK`.

## Designated CPU tests

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python -m pytest -p no:cacheprovider \
  scale/tests/nonlatent_iclr/test_qualification_controls.py \
  scale/tests/nonlatent_iclr/test_qualification_manifest.py \
  scale/tests/nonlatent_iclr/test_qualification_cli.py \
  scale/tests/nonlatent_iclr/test_qualification_probe.py
```

Exit 0: 25 collected, 25 passed in 1.29 seconds.

## Shell/help/bad-input boundaries

Executed with a fresh `/tmp/nonlatent-preflight-recheck.XXXXXX` directory and an EXIT cleanup trap:

```bash
bash -n scale/experiments/nonlatent_iclr/qualification/run_qualification.sh
/usr/bin/python -m scale.experiments.nonlatent_iclr.qualification.cli --help
/usr/bin/python -m scale.experiments.nonlatent_iclr.qualification.probe --help
bash scale/experiments/nonlatent_iclr/qualification/run_qualification.sh --help
/usr/bin/python -m scale.experiments.nonlatent_iclr.qualification.cli validate \
  --output-dir "$tmp_dir/world" --nonce "00000000000000000000000000000000" \
  --expected-world-size 7 --manifest "$tmp_dir/missing.json" --manifest-sha256 "$(printf '0%.0s' {1..64})"
/usr/bin/python -m scale.experiments.nonlatent_iclr.qualification.cli validate \
  --output-dir "$tmp_dir/memory" --nonce "00000000000000000000000000000000" \
  --memory-fraction nan --manifest "$tmp_dir/missing.json" --manifest-sha256 "$(printf '0%.0s' {1..64})"
/usr/bin/python -m scale.experiments.nonlatent_iclr.qualification.probe --nonce bad
  -u QUALIFICATION_IMAGE_DIGEST -u QUALIFICATION_NONCE \
  bash scale/experiments/nonlatent_iclr/qualification/run_qualification.sh
```

Help exits were 0. Invalid-input exits were 2 with structured errors. The temporary directory remained empty and the shell had no live child jobs.

## Static checkpoint safety and representation inspection

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python -c \
  'import inspect, torch; p="/inspire/hdd/global_user/zhangjiaquan-253108540222/outputs_birwkv_diffusion/m4-loop-2p9b/step_00004750/model.pt"; print(inspect.signature(torch.load)); print(torch.cuda.is_initialized()); print(torch.serialization.get_unsafe_globals_in_checkpoint(p)); print(torch.cuda.is_initialized()); print("torch_load_called=false")'
/usr/bin/python -c \
  'import pickletools, zipfile; p="/inspire/hdd/global_user/zhangjiaquan-253108540222/outputs_birwkv_diffusion/m4-loop-2p9b/step_00004750/model.pt"; z=zipfile.ZipFile(p); n=next(x for x in z.namelist() if x.endswith("/data.pkl") or x=="data.pkl"); print(list(pickletools.genops(z.read(n)))[:20])'
```

The static unsafe-global scan returned `[]`; CUDA stayed uninitialized; no load/unpickle occurred. The pickle opcode stream begins `GLOBAL 'collections OrderedDict'`, `EMPTY_TUPLE`, `REDUCE`, proving the top-level reconstructed type that the payload's exact `dict` test rejects.

## Raw safetensors-header dtype check

A standard-library-only reader consumed each shard's eight-byte header length and JSON header, never tensor payloads. Result: 1,059 tensors, 2,947,735,040 elements, all dtype `BF16`.

## Result protocol and hardware predicate checks

```bash
# In a trapped temporary directory containing only an empty rank-8.json:
/usr/bin/python -m scale.experiments.nonlatent_iclr.qualification.cli aggregate \
  --output-dir "$tmp_dir" --nonce 33333333333333333333333333333333 \
  --manifest "$tmp_dir/unused-manifest.json" \
  --manifest-sha256 4444444444444444444444444444444444444444444444444444444444444444
/usr/bin/python -c \
  'from scale.experiments.nonlatent_iclr.qualification.runtime_identity import is_authorized_h100; print(is_authorized_h100("NVIDIA H100 80GB HBM3", 81559)); print(is_authorized_h100("NVIDIA H100 NVL", 95830))'
```

Aggregation exited 1 with `extra_rank_results:rank-8.json`. Both hardware predicates returned `True`, exposing the unauthorized 94-GB-class false acceptance.

## Current-node binary facts and deadline arithmetic

```bash
/usr/bin/python --version
# /usr/local/bin/torchrun exists and its first line is #!/usr/bin/python
# 1500 TERM seconds + 30 KILL-grace seconds = 1530; 1800 - 1530 = 270.
# init_process_group timeout = 1500 - 60 = 1440 seconds.
```

These are current-node facts, not proof about the eventual worker image.

## Attempted dynamic staged import on the CPU node

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/inspire/hdd/global_user/zhangjiaquan-253108540222/qz_stage_traj4096_v7/scale \
/usr/bin/python -c 'from models.birwkv7_diffusion import BiRWKV7ForMaskedDiffusion'
```

Exit 1 at FLA/Triton driver initialization: `RuntimeError: 0 active drivers ([]).` This is expected for the current CPU node and is recorded only as runtime uncertainty; it is not a GPU qualification result.

## Cleanup check

```bash
test ! -e /tmp/qualification-preflight-recheck-readonly
! compgen -G '/tmp/nonlatent-preflight-recheck.*'
! compgen -G '/tmp/nonlatent-result-protocol.*'
```

All checks passed.
