import json, subprocess, sys
from pathlib import Path
from hashlib import sha256
from collections import Counter
from struct import pack
from datetime import datetime, timezone
ROOT=Path.cwd()
RUN="qualification-20260912-01-3257f249-eace-4a74-841f-0d7d8e0e32e2"
NONCE="a9750fd783cc98d70435b185135ebd38"
JOB="job-cd3c6850-91cc-4a1e-8f10-a3ce6c5baec9"
REQ="ffcb99d8b34fdf29a6c8d52aa342d589abc063080143ca34a6860d58ad703ffb"
MAN="bae0fa5a71aca185d1718757d56de2d8f01a2606575d2d907b7a0d88e82ab68a"
BASE=Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification")
WORKER=BASE/RUN
RELEASE=BASE/"payloads/qualification-20260914-08-global-ffaa464d-semantics-v8"
EVIDENCE=ROOT/".omo/evidence/nonlatent-rnn-dlm-iclr-research/qualification-20260914-08-live"
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
expected |= {f"model-semantics-{i}.json" for i in range(8)}
assert set(records)==expected
for prefix in ("rank","lifecycle","model-semantics"):
    assert {p.name for p in WORKER.glob(prefix+"-*.json")}=={f"{prefix}-{i}.json" for i in range(8)}
for name in ("launcher.json","aggregate.json","preflight.json",*(f"lifecycle-{i}.json" for i in range(8))):
    assert read(EVIDENCE/name)==records[name],("copied_bytes_mismatch",name)
manifest_bytes=read(RELEASE/Q/"runtime_manifest.json")
assert manifest_bytes==records["runtime-manifest.json"] and sha256(manifest_bytes).hexdigest()==MAN
md=json.loads(manifest_bytes); entries={e["path"]:e for e in md["files"]}
assert len(md["files"])==len(entries)==126
lifecycle_names={"lifecycle.py","lifecycle_checks.py","lifecycle_evidence.py","lifecycle_binding.py","lifecycle_settings.py","lifecycle_sidecar.py","lifecycle_runtime.py"}
assert {Path(e["path"]).name for e in md["files"] if Path(e["path"]).name.startswith("lifecycle")}==lifecycle_names
skipped=[]; verified=[]
for e in md["files"]:
    if e["role"] in ("checkpoint_model","hf_shard"):
        skipped.append(e); continue
    p=Path(e["path"]); data=read(p)
    assert len(data)==e["size_bytes"] and sha256(data).hexdigest()==e["sha256"],str(p)
    verified.append(str(p))
assert len(skipped)==3 and len(verified)==123 and str(launcher_path) in entries
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
assert (preflight.nonce,preflight.manifest_sha256,preflight.verified_source_files)==(NONCE,MAN,126)
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
    assert core.source_manifest_sha256==MAN and core.verified_source_files==126
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

import math
TRAINER_SHA="0bee15b5af05a785b70ddfeffa3064161f04beccea36b44e9bfd01029e286b60"
TRAINER_PATH=Path("DAN/v7_arch_round/code/train/train_birwkv_diffusion.py")
assert sha256(read(RELEASE/TRAINER_PATH)).hexdigest()==TRAINER_SHA
assert read(ROOT/TRAINER_PATH)==read(RELEASE/TRAINER_PATH)
semantics_names={"semantics_evidence.py","semantics_optimizer.py","semantics_probe.py","semantics_sidecar.py","semantics_runtime.py"}
assert {Path(e["path"]).name for e in md["files"] if Path(e["path"]).name.startswith("semantics_")}==semantics_names
for name in semantics_names|{"runtime_checks.py"}:
    assert read(ROOT/Q/name)==read(RELEASE/Q/name),("semantics_source_difference",name)
for name in ("trainer_semantics.py","trainer_semantics_contract.py"):
    assert read(ROOT/Q.parent/name)==read(RELEASE/Q.parent/name)
from scale.experiments.nonlatent_iclr.qualification.semantics_sidecar import SemanticsSidecar
assert Path(sys.modules[SemanticsSidecar.__module__].__file__).is_relative_to(RELEASE)
# Verify hash-bound AST regions without executing trainer functions or loading a model.
import ast
from scale.experiments.nonlatent_iclr.trainer_semantics_contract import SEGMENTS, SOURCE_SHA256
assert SOURCE_SHA256==TRAINER_SHA
trainer_tree=ast.parse(read(RELEASE/TRAINER_PATH))
for segment,(start,end,digest) in SEGMENTS.items():
    selected=[]; pending=[trainer_tree]
    while pending:
        node=pending.pop()
        if isinstance(node,ast.stmt) and start<=node.lineno<=end:
            assert node.end_lineno is not None and node.end_lineno<=end
            selected.append(node)
        else:
            pending.extend(ast.iter_child_nodes(node))
    body=ast.Module(body=sorted(selected,key=lambda n:n.lineno),type_ignores=[])
    assert sha256(ast.dump(body,include_attributes=False).encode()).hexdigest()==digest,segment
semantic_results=[]
name_inventory=None
for i in range(8):
    name=f"model-semantics-{i}.json"
    assert read(EVIDENCE/name)==records[name]
    sem=SemanticsSidecar.model_validate_json(records[name])
    rank=RankResult.model_validate_json(records[f"rank-{i}.json"])
    life=LifecycleSidecar.model_validate_json(records[f"lifecycle-{i}.json"])
    assert rank.evidence.kind=="success"
    core=rank.evidence
    assert sem.passed and not sem.runtime_promoted and sem.issue is None
    assert sem.binding==life.binding and sem.binding.controller==core.controller_binding==controller
    assert sem.binding.rank==sem.binding.local_rank==i and sem.binding.device==f"cuda:{i}"
    assert (sem.binding.run_id,sem.binding.nonce,sem.binding.source_manifest_sha256)==(RUN,NONCE,MAN)
    assert sem.binding.checkpoint==checkpoint and sem.trainer_source_sha256==TRAINER_SHA
    assert sem.before==sem.after and sem.before is not None
    assert not sem.before.training and not sem.before.inference_mode and sem.before.grad_enabled
    assert sem.before.parameter_devices==(f"cuda:{i}",) and sem.before.parameter_dtypes==("torch.bfloat16",)
    assert sem.before.torch_version==core.torch_version
    assert sem.mask is not None and sem.gradients is not None
    base,frozen=sem.membership
    assert not base.freeze_backbone and frozen.freeze_backbone
    for membership in sem.membership:
        assert membership.passed and not membership.latent_on and membership.optimizer_state_entries==0
        assert membership.total_tensors==1953 and membership.total_numel==4091581441
        assert len(membership.groups)==2
        assert membership.each_trainable_once and membership.frozen_absent and membership.cross_group_disjoint and membership.freeze_rule_exact
    assert base.trainable_tensors==1953 and base.trainable_numel==4091581441
    assert frozen.trainable_tensors==frozen.trainable_numel==0
    assert all(g.tensors==g.numel==0 and not g.names for g in frozen.groups)
    rows=sem.gradients.parameters
    assert len(rows)==1953 and sum(r.numel for r in rows)==4091581441
    assert all(r.requires_grad for r in rows)
    by_name={r.name:r for r in rows}; assert len(by_name)==1953
    assert not any("latent_cond" in name for name in by_name)
    identities=tuple((r.name,r.numel,r.requires_grad) for r in rows)
    assert name_inventory is None or identities==name_inventory
    name_inventory=identities
    group_names=[tuple(g.names) for g in base.groups]
    assert not set(group_names[0]).intersection(group_names[1])
    assert set(group_names[0])|set(group_names[1])==set(by_name)
    for group in base.groups:
        assert len(set(group.names))==group.tensors and sum(by_name[n].numel for n in group.names)==group.numel
    assert [g.tensors for g in base.groups]==[1310,643]
    assert [g.numel for g in base.groups]==[4089937920,1643521]
    grad=sem.gradients
    assert (grad.step,grad.steps,grad.stage_a_frac,grad.stage_b_frac,grad.n_layers)==(0,100,0.2,0.6,32)
    histogram=Counter()
    for row in rows:
        expected_gated=not ("attn_bwd" in row.name or "fuse_" in row.name)
        expected_class="backward_attention" if "attn_bwd" in row.name else "forward_attention" if "attn_fwd" in row.name else "fusion" if "fuse_" in row.name else "loop" if row.name.startswith("loop.") else "shared"
        assert row.gated==expected_gated and row.name_class==expected_class
        assert row.before!="nonfinite" and row.after!="nonfinite" and row.passed
        assert row.after==("zero" if expected_gated and row.before!="none" else row.before)
        if not expected_gated:
            assert row.unchanged_when_ungated
        histogram[(row.name_class,row.gated,row.before,row.after)]+=1
    gated_nonzero=sum(row.gated and row.before=="nonzero" for row in rows)
    ungated_nonzero=sum(not row.gated and row.after=="nonzero" for row in rows)
    assert gated_nonzero>0 and ungated_nonzero>0 and grad.passed
    for category,prefix in (("forward_attention","forward_attention"),("backward_attention","backward_attention"),("fusion","fusion"),("loop","loop"),("shared","shared")):
        category_rows=[r for r in rows if r.name_class==category]
        assert len(category_rows)==getattr(core.parameters,prefix+"_tensors")
        assert sum(r.numel for r in category_rows)==getattr(core.parameters,prefix+"_numel")
    mask=sem.mask
    assert mask.passed and mask.device==f"cuda:{i}" and mask.shape==(1,48,65536)
    assert mask.reduction=="historical_rank_local" and mask.eligible_tokens==46 and mask.selected_tokens==22
    assert mask.eligibility_exact and mask.corruption_exact and mask.logits_finite and mask.unselected_logit_grad_zero
    assert all(math.isfinite(v) for v in (mask.loss,mask.selected_mean,mask.unselected_loss,mask.empty_loss))
    assert math.isclose(mask.loss,mask.selected_mean,rel_tol=1e-6,abs_tol=2e-6)
    assert math.isclose(mask.loss,mask.unselected_loss,rel_tol=1e-6,abs_tol=2e-6)
    assert mask.empty_loss==0.0
    semantic_results.append({"rank":i,"device":mask.device,"sidecar_passed":True,"membership_passed":[base.passed,frozen.passed],"baseline_trainable_tensors":base.trainable_tensors,"baseline_trainable_numel":base.trainable_numel,"baseline_groups":[{"tensors":g.tensors,"numel":g.numel} for g in base.groups],"frozen_trainable_tensors":0,"frozen_trainable_numel":0,"optimizer_state_entries":0,"gated_nonzero_before":gated_nonzero,"ungated_nonzero_after":ungated_nonzero,"gradient_histogram":[{"name_class":key[0],"gated":key[1],"before":key[2],"after":key[3],"tensors":value} for key,value in sorted(histogram.items())],"mask":mask.model_dump(mode="json"),"settings_unchanged":True})
summary=json.loads(read(EVIDENCE/"v8-reparsed-summary.json"))
assert summary["all_passed"] and (summary["job_id"],summary["run_id"],summary["nonce"],summary["manifest_sha256"])==(JOB,RUN,NONCE,MAN)
assert set(summary["reparsed"])=={str(i) for i in range(8)}
for i in range(8):
    item=summary["reparsed"][str(i)]
    assert item["device"]==f"cuda:{i}" and item["semantics_passed"] and item["lifecycle_passed"] and item["mask_passed"] and item["gradients_passed"] and item["membership"]==[True,True]
    assert item["lifecycle_calls"]==item["lifecycle_sync"]==12
plan_path=ROOT/".omo/plans/nonlatent-rnn-dlm-iclr-research.md"
contract_path=ROOT/"DAN/nonlatent_iclr/architecture_contract.json"
plan=read(plan_path).decode(); contract=json.loads(read(contract_path))
read(ROOT/"scale/experiments/nonlatent_iclr/architecture_contract.py")
read(ROOT/"DAN/nonlatent_iclr/architecture_contract.receipt.json")
assert "48 block evaluations are distinct from outer NFEs" in plan
assert contract["task3_acceptance"]["active_optimizer_membership_and_gradient_policy"]=="open_source_bound_cpu_only"
assert contract["task3_acceptance"]["active_input_output_mask_loss_semantics"]=="open_cpu_source_and_finite_gpu_forward_only"
assert contract["readiness"]["whole_architecture_ready"] is False
assert not torch.cuda.is_initialized()
report={
 "schema_version":1,"review_type":"independent_actual_semantics_v8_gpu_runtime_review","reviewed_at_utc":datetime.now(timezone.utc).isoformat(),
 "verdict":"PASS","promotion_target":"BOUNDED_MODEL_SEMANTICS_AND_LIFECYCLE_RUNTIME_MILESTONE",
 "scheduler_status":scheduler["status"],"scientific_status":aggregate.status,
 "source_binding":{"job_id":JOB,"run_id":RUN,"nonce":NONCE,"request_sha256":REQ,"manifest_sha256":MAN,"manifest_entries":126,"controller_receipt_file_sha256":sha256(receipt_bytes).hexdigest(),"controller_binding_canonical_sha256":controller.controller_receipt_sha256,"worker_ledger_sha256":sha256(ledger).hexdigest(),"trainer_source_sha256":TRAINER_SHA,"v8_launcher_sha256":input_hashes[str(launcher_path)]},
 "verification":{"worker_records_hash_verified":29,"nonweight_manifest_members_hash_verified":123,"weight_members_not_rehashed":skipped,"preflight_verified_files":126,"exact_deployed_validators_used":True,"rank_records_reparsed":8,"lifecycle_sidecars_reparsed":8,"semantics_sidecars_reparsed":8,"coordinator_summary_independently_confirmed":True,"trainer_ast_regions_verified":len(SEGMENTS),"cuda_initialized_by_review":False},
 "checkpoint":{"sha256":checkpoint.sha256,"size_bytes":checkpoint.size_bytes,"step":4750,"state_tensors":1955,"parameter_tensors":1953,"parameter_numel":4091581441},
 "lifecycle_rank_results":rank_summary,"semantics_rank_results":semantic_results,
 "task3_assessment":{"four_stated_acceptance_criteria":{"nonlatent_path_explicit":"SATISFIED_BOUNDED_GPU","48_block_evaluations_distinct_from_outer_nfe":"SATISFIED_PRIOR_HASH_BOUND_CPU_CLOCK_EVIDENCE","state_replay_cannot_cross_unrelated_samples":"SATISFIED_BOUNDED_GPU_LIFECYCLE","full_canvas_and_streaming_distinct":"SATISFIED_STREAMING_EXPLICITLY_UNSUPPORTED"},"previously_open_optimizer_membership_and_gradient_policy":"CLOSED_FOR_LOADED_MODEL_TWO_FREEZE_CONFIGS_AND_STAGE_A_POLICY","previously_open_input_output_mask_loss_semantics":"CLOSED_FOR_TESTED_CORRUPTION_AND_RANK_LOCAL_LOSS_BACKWARD","remaining_empirical_gaps_at_documented_scope":[],"contract_work_item":"NOT_YET_COMPLETE_CURRENT_CANONICAL_CONTRACT_STILL_MARKS_BOTH_ITEMS_OPEN_AND_DOES_NOT_INGEST_V8","task_3_complete_as_published":False,"remaining_work":["Ingest this immutable v8 review and all eight model-semantics sidecars plus matching core/lifecycle records into the canonical architecture contract; update the two OPEN fields with exact tested scopes and preserve limitations.","Publish and verify the updated contract/receipt and scoped Task3 happy/failure gates before changing the plan completion marker; no further GPU experiment is demanded by this review."]},
 "preserved_limitations":{"full_model_prefix_cache":"NOT_QUALIFIED_FFN_TOKEN_SHIFT_STATE_UNWIRED","loop_prefix_cache":"NOT_QUALIFIED_TIED_LAYER_STATE_ALIASES_PASSES","cache_implementation_required_while_disabled":False,"long_context_claim":False,"performance_or_goodput_claim":False,"observed_image_digest":None,"arbitrary_torch_binary_identity_proven":False,"cuda_fault_recovery":False,"later_gradient_stages_gpu_qualified":False,"distributed_loss_reduction_or_fsdp_qualified":False,"optimizer_updates_or_state_serialization_qualified":False,"general_training_or_convergence_claim":False},
 "interpretation_notes":["Membership and gradient policy are observed on actual loaded parameter objects; no optimizer step or update-state allocation occurs. Frozen-backbone nonlatent configuration has two empty optimizer groups, not useful training.","Stage-A zeroing preserves None gradients; a zero gradient does not guarantee weight immobility under AdamW weight decay or prior momentum. No GPU optimizer step was claimed.","Unselected loss inputs and their logit gradients are excluded; visible context may still influence selected bidirectional logits.","Mask loss checks use rel_tol=1e-6, abs_tol=2e-6, distinct from lifecycle exact equality at atol=rtol=0.","Direct qz submission is accepted as instructed; receipt schema labels do not prove a separate controller-permit ceremony.","Controller raw-file SHA differs from canonical controller-binding SHA due to serialization. Canonical binding is reconciled on all 24 rank/sidecar records.","Worker wall-clock timestamps and scheduler timeline are separately recorded; no exact billing/runtime reconciliation claim follows.","Torch version metadata normalization is not binary/container identity."]
}
print(json.dumps(report,indent=2,sort_keys=True))
