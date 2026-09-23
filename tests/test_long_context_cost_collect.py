import copy
import json
import pytest
from lrwkv_evidence.long_context_eval.cost_collect import collect
from lrwkv_evidence.long_context_eval.exact import partitions
from lrwkv_evidence.long_context_eval.numerical import specifications
from lrwkv_evidence.long_context_mvp import tasks as T


def fixture(path,kind='rwkv'):
    for rank in range(8):
        ex=T.problem('dev',20270923,rank//2,T.FAMILIES[rank%2])
        p=dict(kind='provenance',cost_only=True,comparator=kind,rank=rank,timing_repeats=10,
               checkpoint_sha256='checkpoint',checkpoint_step=600,contract_sha256='contract',
               split='dev',data_seed=20270923,task_version=T.VERSION,
               within_repeat_max_logprob_diff=0.,cross_rank_max_logprob_diff=0.,
               runtime=dict(gpu_name='fixture',gpu_total_memory_bytes=80000))
        rows=[p]
        for i,(h,pos) in enumerate([(None,'middle')]+[(h,pos) for h in (1024,4096,16384) for pos in ('far','middle','near')]):
            policies={'cached_causal':[[]]*8} if kind=='causal_rwkv' else partitions(ex)
            rows.append(dict(kind='cost_condition',family=ex['family'],index=ex['index'],exact=None,
                             requested_context_tokens=h,context_tokens=h or 95,position=pos,
                             evidence_span=[20,30],records=1,filler_tokens=0,condition_wall_seconds=20.,
                             costs=[dict(method=m,calls=len(groups),seconds=[.1]*10,peak_allocated_bytes=1000,sample_outputs=[0]*10) for m,groups in policies.items()]))
            if i==0:rows.append(dict(kind='numerical_qualification',checks=[dict(s,within_repeat_max_logprob_diff=0.,cross_rank_max_logprob_diff=0.) for s in specifications(kind)]))
        rows.append(dict(kind='complete',cost_only=True))
        (path/f'rank{rank}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return dict(out=str(path),cost_only=True,comparator=kind,checkpoint_sha256='checkpoint',checkpoint_step=600,contract_sha256='contract')


@pytest.mark.parametrize('kind',['rwkv','attention','causal_rwkv'])
def test_all_costs_retained_without_a_quality_claim(tmp_path,kind):
    result=collect(fixture(tmp_path,kind))
    assert result['cost_only'] and result['condition_count']==80
    assert all(r['exact'] is None for r in result['conditions'])
    assert all(r['mean_request_seconds']==pytest.approx(.1) for r in result['aggregates'])


@pytest.mark.parametrize('damage',['timing','checkpoint','quality','duplicate','mode','calls'])
def test_invalid_cost_evidence_is_rejected(tmp_path,damage):
    plan=fixture(tmp_path);p=tmp_path/'rank0.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()]
    if damage=='timing':rows[1]['costs'][0]['seconds'][0]=-1.
    if damage=='checkpoint':rows[0]['checkpoint_step']=20
    if damage=='quality':rows[1]['exact']={'joint_kl_nats':0.}
    if damage=='duplicate':rows.insert(1,copy.deepcopy(rows[1]))
    if damage=='mode':rows[0]['cost_only']=False
    if damage=='calls':rows[1]['costs'][0]['calls']=0
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):collect(plan)
