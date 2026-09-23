import json
import math
import pytest
from lrwkv_evidence.long_context_eval.short_gate import collect
from lrwkv_evidence.long_context_eval.exact import audit,partitions
from lrwkv_evidence.long_context_mvp import tasks as T


def fixture(path,oracle):
    for rank in range(8):
        ex=T.problem('dev',20270923,rank//2,T.FAMILIES[rank%2])
        def predict(visible):
            probabilities=T.oracle_marginals(ex['matrix_rows'],ex['syndrome'],visible,8) if oracle else [.5]*8
            ps=[min(1-1e-12,max(1e-12,p)) for p in probabilities]
            return [[math.log1p(-p),math.log(p)] for p in ps]
        rows=[dict(kind='provenance',rank=rank,checkpoint_sha256='checkpoint',contract_sha256='contract',
                   split='dev',data_seed=20270923,task_version=T.VERSION,
                   within_repeat_max_logprob_diff=0.,cross_rank_max_logprob_diff=0.),
              dict(kind='condition',family=ex['family'],index=ex['index'],requested_context_tokens=None,
                   context_tokens=95,position='middle',evidence_span=[20,30],records=1,filler_tokens=0,
                   audit_seconds=1.,condition_wall_seconds=10.,exact=audit(ex,predict),
                   costs=[dict(method=m,calls=len(g),seconds=[.1]*3,peak_allocated_bytes=1000,sample_outputs=[0]*3) for m,g in partitions(ex).items()]),
              dict(kind='gate',local_pass=oracle,all_eight_conditions_pass=oracle)]
        (path/f'rank{rank}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows)+'{"unfinished_long_record":')
    return dict(out=str(path),checkpoint_sha256='checkpoint')


@pytest.mark.parametrize('oracle',[True,False])
def test_gate_can_finish_before_long_records_without_claiming_full_completion(tmp_path,oracle):
    result=collect(fixture(tmp_path,oracle))
    assert result['short_gate_complete'] and not result['execution_complete']
    assert result['competence_gate_passed']==oracle
    assert len(result['prefix_sha256'])==8


@pytest.mark.parametrize('damage',['global','local','checkpoint','mode','missing'])
def test_unvalidated_prefix_cannot_start_comparators(tmp_path,damage):
    plan=fixture(tmp_path,True);p=tmp_path/'rank0.jsonl';lines=p.read_text().splitlines()[:3];rows=[json.loads(s) for s in lines]
    if damage=='global':rows[2]['all_eight_conditions_pass']=False
    if damage=='local':rows[2]['local_pass']=False
    if damage=='checkpoint':rows[0]['checkpoint_sha256']='other'
    if damage=='mode':rows[0]['cost_only']=True
    if damage=='missing':rows.pop()
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):collect(plan)
