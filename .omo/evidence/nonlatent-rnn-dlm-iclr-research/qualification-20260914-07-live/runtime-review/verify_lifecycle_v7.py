import json, subprocess, sys
from pathlib import Path
from hashlib import sha256
from collections import Counter
from struct import pack
from datetime import datetime, timezone
ROOT=Path.cwd()
RUN="qualification-20260912-01-462d8c9d-eb9b-41fd-9992-d1d44528bc3a"
NONCE="859969188464fd48e9c81beddca875b9"
JOB="job-5b99b0c6-bb92-44e2-924b-761de4ad276b"
REQ="a744ff2fdd62dcfb1aec8d0b9614757d74f7a7fa91155e2578ed9202623b58e1"
MAN="a38e0558a721ab9a1e6c37711c7e1d5599b5cdc8778fa28f4a878500ddd499ed"
BASE=Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification")
WORKER=BASE/RUN
RELEASE=BASE/"payloads/qualification-20260914-07-global-ffaa464d-lifecycle-v7"
EVIDENCE=ROOT/".omo/evidence/nonlatent-rnn-dlm-iclr-research/qualification-20260914-07-live"
Q=Path("scale/experiments/nonlatent_iclr/qualification")
input_hashes={}
def read(p):
    data=p.read_bytes(); digest=sha256(data).hexdigest(); key=str(p)
    assert key not in input_hashes or input_hashes[key]==digest, ("changed_during_review",key)
    input_hashes[key]=digest
    return data
request_bytes=read(EVIDENCE/"createjob-spec.json")
assert sha256(request_bytes).hexdigest()==REQ
request=json.loads(request_bytes); created=json.loads(read(EVIDENCE/"createjob.response.json"))["Result"]
scheduler_command=["qz","train","GetJob","--data",json.dumps({"job_id":JOB}),"-o","json"]
queried=subprocess.run(scheduler_command,capture_output=True,text=True,check=True,timeout=120)
scheduler_stdout=queried.stdout; scheduler_stderr=queried.stderr; scheduler=json.loads(scheduler_stdout)["Result"]
assert scheduler["status"]=="job_succeeded"
assert scheduler["job_id"]==created["job_id"]==JOB
assert scheduler["name"]==created["name"]==request["name"]==RUN
assert created["envs"]==request["envs"]
for k in ("command","project_id","workspace_id","logic_compute_group_id","max_running_time_ms","auto_fault_tolerance","fault_tolerance_max_retry"):
    assert scheduler[k]==created[k]==request[k], k
launcher_path=RELEASE/Q/"run_qualification.sh"
assert request["command"]=="/usr/bin/bash "+str(launcher_path)
assert scheduler["gpu_count"]==8 and scheduler["framework_config"][0]["gpu_count"]==8
assert scheduler["framework_config"][0]["instance_count"]==request["framework_config"][0]["instance_count"]==1
assert scheduler["framework_config"][0]["image"]==request["framework_config"][0]["image"]
assert request["max_running_time_ms"]=="1800000" and request["auto_fault_tolerance"] is False and request["fault_tolerance_max_retry"]==0
records={}; ledger=read(EVIDENCE/"worker-artifacts.sha256")
for line in ledger.decode().splitlines():
    digest,name=line.split(); assert Path(name).name==name and name not in records
    data=read(WORKER/name); assert sha256(data).hexdigest()==digest,name; records[name]=data
expected={"launcher.json","aggregate.json","preflight.json","runtime_environment.json","runtime-manifest.json"}|{f"rank-{i}.json" for i in range(8)}|{f"lifecycle-{i}.json" for i in range(8)}
assert set(records)==expected
for prefix in ("rank","lifecycle"):
    assert {p.name for p in WORKER.glob(prefix+"-*.json")}=={f"{prefix}-{i}.json" for i in range(8)}
for name in ("launcher.json","aggregate.json","preflight.json",*(f"lifecycle-{i}.json" for i in range(8))):
    assert read(EVIDENCE/name)==records[name],("copied_bytes_mismatch",name)
manifest_bytes=read(RELEASE/Q/"runtime_manifest.json")
assert manifest_bytes==records["runtime-manifest.json"] and sha256(manifest_bytes).hexdigest()==MAN
md=json.loads(manifest_bytes); entries={e["path"]:e for e in md["files"]}
assert len(md["files"])==len(entries)==118
lifecycle_names={"lifecycle.py","lifecycle_checks.py","lifecycle_evidence.py","lifecycle_binding.py","lifecycle_settings.py","lifecycle_sidecar.py","lifecycle_runtime.py"}
assert {Path(e["path"]).name for e in md["files"] if Path(e["path"]).name.startswith("lifecycle")}==lifecycle_names
skipped=[]; verified=[]
for e in md["files"]:
    if e["role"] in ("checkpoint_model","hf_shard"):
        skipped.append(e); continue
    p=Path(e["path"]); data=read(p)
    assert len(data)==e["size_bytes"] and sha256(data).hexdigest()==e["sha256"],str(p)
    verified.append(str(p))
assert len(skipped)==3 and len(verified)==115 and str(launcher_path) in entries
for name in lifecycle_names:
    assert read(ROOT/Q/name)==read(RELEASE/Q/name),("approved_lifecycle_source_drift",name)
assert read(ROOT/"scale/experiments/nonlatent_iclr/full_canvas.py")==read(RELEASE/"scale/experiments/nonlatent_iclr/full_canvas.py")
# Parse with the exact manifested release, not mutable editable configuration modules.
sys.path.insert(0,str(RELEASE))
from scale.experiments.nonlatent_iclr.qualification.controller_receipt import ControllerReceipt
from scale.experiments.nonlatent_iclr.qualification.contracts import RankResult, AggregateResult
from scale.experiments.nonlatent_iclr.qualification.manifest import RuntimeManifest, PreflightReceipt
from scale.experiments.nonlatent_iclr.qualification.lifecycle_evidence import LifecycleEvidence
from scale.experiments.nonlatent_iclr.qualification.lifecycle_sidecar import LifecycleSidecar
from packaging.version import Version
import torch
assert not torch.cuda.is_initialized()
for cls in (ControllerReceipt,RankResult,AggregateResult,RuntimeManifest,PreflightReceipt,LifecycleEvidence,LifecycleSidecar):
    assert Path(sys.modules[cls.__module__].__file__).is_relative_to(RELEASE),cls.__name__
manifest=RuntimeManifest.model_validate_json(manifest_bytes)
checkpoint,=[e for e in manifest.files if e.role=="checkpoint_model"]
launcher=json.loads(records["launcher.json"])
assert launcher["status"]=="PARTIAL_QUALIFICATION"
assert (launcher["run_id"],launcher["nonce"],launcher["manifest_sha256"])==(RUN,NONCE,MAN)
assert all(launcher[k]==0 for k in ("preflight_status","launcher_status","aggregate_status"))
receipt_path=Path(launcher["controller_receipt"])
assert receipt_path==BASE/"controller-receipts"/(RUN+".json")
receipt_bytes=read(receipt_path); receipt=ControllerReceipt.model_validate_json(receipt_bytes); controller=receipt.to_binding_evidence()
assert (receipt.job_id,receipt.run_id,receipt.nonce,receipt.source_manifest_sha256,receipt.submitted_request_sha256)==(JOB,RUN,NONCE,MAN,REQ)
assert controller.controller_receipt_sha256==receipt.canonical_sha256()
for k in ("project_id","workspace_id","logic_compute_group_id"):
    assert getattr(receipt,k)==request[k]
assert receipt.spec_id==request["framework_config"][0]["spec_id"] and receipt.observed_image_digest is None
env={e["name"]:e["value"] for e in request["envs"]}
assert (env["QUALIFICATION_RUN_ID"],env["QUALIFICATION_NONCE"],env["QUALIFICATION_MANIFEST_SHA256"],env["QUALIFICATION_JOB_RECEIPT"])==(RUN,NONCE,MAN,str(receipt_path))
preflight=PreflightReceipt.model_validate_json(records["preflight.json"])
assert (preflight.nonce,preflight.manifest_sha256,preflight.verified_source_files)==(NONCE,MAN,118)
assert preflight.expected_image==receipt.requested_image==manifest.expected_image
environment=json.loads(records["runtime_environment.json"])
assert environment["status"]=="MATCH" and environment["manifest_sha256"]==MAN
assert environment["cwd"]==str(RELEASE) and environment["runtime_modules_loaded"]==[]
assert len(environment["packages"])==6 and len(environment["package_sources"])==8
for e in environment["packages"]:
    assert e["status"]=="MATCH" and e["observed_version"]==e["expected_version"]
for e in environment["package_sources"]:
    assert e["status"]=="MATCH" and e["expected_sha256"]==e["observed_sha256"]==entries[e["requested_path"]]["sha256"]
    assert e["expected_size_bytes"]==e["observed_size_bytes"]==entries[e["requested_path"]]["size_bytes"]
aggregate=AggregateResult.model_validate_json(records["aggregate.json"])
assert aggregate.status=="PARTIAL_QUALIFICATION" and len(aggregate.rank_results)==8
rows=tuple(tuple((index+row)%3 if not 8<=index<12 else 65535 for index in range(48)) for row in range(2))
expected_inputs=dict(zip(("a","b","edited_a"),(sha256(pack("<48q",*row)).hexdigest() for row in (*rows,(2,*rows[0][1:]))),strict=True))
rank_summary=[]; gpu_uuids=None
for i in range(8):
    rank=RankResult.model_validate_json(records[f"rank-{i}.json"])
    sidecar=LifecycleSidecar.model_validate_json(records[f"lifecycle-{i}.json"])
    observations=LifecycleEvidence.model_validate_json(json.dumps(json.loads(records[f"lifecycle-{i}.json"])["observations"]))
    assert rank==aggregate.rank_results[i] and rank.status=="PASSED" and rank.evidence.kind=="success"
    core=rank.evidence; binding=sidecar.binding
    assert rank.rank==core.rank==core.local_rank==binding.rank==binding.local_rank==i
    assert rank.nonce==NONCE and core.world_size==binding.world_size==8
    assert core.controller_binding==binding.controller==controller
    assert (binding.run_id,binding.nonce,binding.source_manifest_sha256)==(RUN,NONCE,MAN)
    assert core.source_manifest_sha256==MAN and core.verified_source_files==118
    assert binding.checkpoint==checkpoint and core.checkpoint_sha256==checkpoint.sha256 and core.checkpoint_size_bytes==checkpoint.size_bytes
    assert core.checkpoint_step==binding.checkpoint_step==4750 and core.checkpoint_state_tensors==binding.checkpoint_state_tensors==1955
    assert core.parameters.total_numel==4091581441 and core.parameters.total_tensors==1953 and core.geometry==binding.geometry
    assert binding.device==observations.device==f"cuda:{i}"
    assert sidecar.passed and observations.passed and sidecar.observations==observations
    assert not sidecar.runtime_promoted and not observations.runtime_promoted
    assert observations.model_calls==observations.synchronizations==12 and observations.failure_code is None
    assert Counter(s.kind for s in observations.steps)=={"comparison":9,"rejection":8,"injection":1}
    for s in observations.steps:
        assert s.passed
        if s.kind=="comparison":
            assert s.max_abs_error==0.0 and s.actual_dtype==s.reference_dtype=="torch.bfloat16"
        elif s.kind=="injection":
            assert not s.cuda_device_fault_recovery and (s.calls_before,s.calls_after,s.synchronizations_before,s.synchronizations_after)==(10,11,10,11)
    assert observations.atol==observations.rtol==0.0 and {x.name:x.sha256 for x in observations.inputs}==expected_inputs
    assert sidecar.before==sidecar.after and sidecar.before is not None
    assert sidecar.before.parameter_dtypes==("torch.bfloat16",) and sidecar.before.parameter_devices==(f"cuda:{i}",)
    assert not sidecar.before.training and sidecar.before.inference_mode and not sidecar.before.grad_enabled
    assert sidecar.before.torch_version==core.torch_version
    assert Version(core.torch_version)==Version(next(e["observed_version"] for e in environment["packages"] if e["distribution"]=="torch"))
    assert core.fullmodel_cachedprefix=="NOT_QUALIFIED_FFN_TOKEN_SHIFT_STATE_UNWIRED" and core.loop_cachedprefix=="NOT_QUALIFIED_TIED_LAYER_STATE_ALIASES_PASSES"
    uuids=tuple(g.uuid for g in core.gpu_inventory)
    assert len(set(uuids))==8 and (gpu_uuids is None or gpu_uuids==uuids); gpu_uuids=uuids
    rank_summary.append({"rank":i,"device":binding.device,"rank_status":rank.status,"sidecar_passed":sidecar.passed,"lifecycle_passed":observations.passed,"calls":12,"synchronizations":12,"comparisons":9,"zero_call_rejections":8,"caller_boundary_injections":1,"max_abs_comparison_error":0.0,"settings_unchanged":True,"elapsed_monotonic_seconds":core.elapsed_monotonic_seconds,"peak_allocated_bytes":core.memory.peak_allocated_bytes,"peak_reserved_bytes":core.memory.peak_reserved_bytes,"standalone_attention_cache_delta":core.fla_causal_cache.max_abs_delta})
summary=json.loads(read(EVIDENCE/"lifecycle-reparsed-summary.json"))
assert summary["all_passed"] and (summary["job_id"],summary["run_id"],summary["nonce"],summary["manifest_sha256"],summary["request_sha256"])==(JOB,RUN,NONCE,MAN,REQ)
assert set(summary["reparsed_lifecycle"])=={str(i) for i in range(8)}
for i in range(8):
    s=summary["reparsed_lifecycle"][str(i)]; assert s["passed"] and s["completed"] and s["failure"] is None and s["calls"]==s["sync"]==12 and s["device"]==f"cuda:{i}"
assert not torch.cuda.is_initialized()
report={"schema_version":1,"review_type":"independent_actual_lifecycle_v7_gpu_runtime_review","reviewed_at_utc":datetime.now(timezone.utc).isoformat(),"verdict":"PASS","promotion_target":"BOUNDED_FULL_CANVAS_GPU_LIFECYCLE_MILESTONE_ONLY","task_3_complete":False,"scheduler_status":scheduler["status"],"scientific_status":aggregate.status,"source_binding":{"job_id":JOB,"run_id":RUN,"nonce":NONCE,"request_sha256":REQ,"manifest_sha256":MAN,"manifest_entries":118,"controller_receipt_file_sha256":sha256(receipt_bytes).hexdigest(),"controller_binding_canonical_sha256":controller.controller_receipt_sha256,"worker_ledger_sha256":sha256(ledger).hexdigest(),"createjob_response_sha256":input_hashes[str(EVIDENCE/"createjob.response.json")],"v7_launcher_sha256":input_hashes[str(launcher_path)]},"verification":{"worker_records_hash_verified":21,"nonweight_manifest_entries_hash_verified":115,"weight_entries_not_rehashed":skipped,"preflight_verified_files":118,"lifecycle_modules":sorted(lifecycle_names),"exact_deployed_validators_used":True,"sidecars_json_reparsed":8,"core_ranks_json_reparsed":8,"aggregate_matches_individual_ranks":True,"coordinator_summary_independently_confirmed":True,"cuda_initialized_by_review":False},"checkpoint":{"sha256":checkpoint.sha256,"size_bytes":checkpoint.size_bytes,"step":4750,"state_tensors":1955,"parameter_tensors":1953,"parameter_numel":4091581441},"rank_results":rank_summary,"accepted_claim":"Bounded GPU full-canvas no-cache/nonlatent forwarding, exact recomputation comparisons, session/foreign-request isolation, edit invalidation, close/reset/reopen retirement, and recovery after a synchronized injected caller-boundary exception on the tested inputs and eight ranks only.","preserved_limitations":{"full_model_prefix_cache":"NOT_QUALIFIED_FFN_TOKEN_SHIFT_STATE_UNWIRED","loop_prefix_cache":"NOT_QUALIFIED_TIED_LAYER_STATE_ALIASES_PASSES","cuda_fault_recovery":False,"long_context_claim":False,"performance_or_goodput_claim":False,"observed_image_digest":None,"arbitrary_torch_binary_identity_proven":False,"optimizer_or_training_qualification":False,"mask_loss_semantics_qualification":False,"universal_determinism_proven":False,"source_ledgers_authenticated_by_signature":False},"provenance_notes":["Direct qz submission was explicitly requested; schema authorization labels do not establish a separate controller-permit ceremony.","Controller raw JSON file hash and canonical typed binding digest differ because serialization differs; both are recorded and the latter is reconciled on all ranks and sidecars.","GetJob omits environment entries; exact request and CreateJob response retain matching environment identities, corroborated by worker records.","Exact equality was observed with BF16 parameters, TF32 matmul enabled and deterministic_algorithms false; this is not universal determinism.","Torch nv25.6 versus nv25.06 is metadata normalization, not binary or container identity.","Checkpoint and HF shard bytes were not opened or rehashed; manifest, preflight and in-run checks bind their historical observed identities.","Initial comparison of editable runtime_config with v7 failed as expected for release-specific defaults/typing. Final parsing uses the exact manifested deployed validators; launcher explicitly passes v7 staged paths."],"exact_next_task_3_gap":"Reconcile this accepted lifecycle milestone and separately approved source-bound CPU trainer/initialization evidence into the canonical architecture contract. Close remaining active/model-bound optimizer membership, gradient policy and input/output mask-loss semantics acceptance without inferring them from inference-only evidence. Unsupported full-model/loop caches need not be implemented unless those modes are enabled or claimed."}
print(json.dumps(report,indent=2,sort_keys=True))
