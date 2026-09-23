import copy
import json
import math
import pytest
from lrwkv_evidence.long_context_eval.collect import collect
from lrwkv_evidence.long_context_eval.exact import METHODS, audit, partitions
from lrwkv_evidence.long_context_mvp import tasks as T


def fixture(tmp_path):
    for rank in range(8):
        ex = T.problem('dev', 20270923, rank // 2, T.FAMILIES[rank % 2])
        exact = audit(ex, lambda visible: [[-math.log(2)] * 2 for _ in range(8)])
        rows = [dict(kind='provenance', rank=rank, checkpoint_sha256='checkpoint', contract_sha256='contract',
                     split='dev', data_seed=20270923, task_version=T.VERSION,
                     runtime=dict(gpu_name='fixture',gpu_total_memory_bytes=80000,packages=dict(torch='fixture',triton='fixture')),
                     within_repeat_max_logprob_diff=0., cross_rank_max_logprob_diff=0.),
                dict(kind='condition', family=ex['family'], index=rank // 2, exact=exact,
                     requested_context_tokens=None, context_tokens=95, position='middle',
                     evidence_span=[20, 30], records=1, filler_tokens=0,
                     audit_seconds=.5,condition_wall_seconds=10.,
                     costs=[dict(method=k, calls=len(v), seconds=[.1, .2, .3], peak_allocated_bytes=1000,
                                 sample_outputs=[0, 1, 2]) for k, v in partitions(ex).items()]),
                dict(kind='gate', local_pass=False, all_eight_conditions_pass=False),
                dict(kind='complete', long_context_gate_passed=False)]
        (tmp_path / f'rank{rank}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return dict(out=str(tmp_path), checkpoint_sha256='checkpoint')


def test_complete_failed_gate_is_honest_result(tmp_path):
    report = collect(fixture(tmp_path))
    assert report['execution_complete'] and not report['competence_gate_passed']
    assert report['condition_count'] == 8
    assert len(report['aggregates']) == 2*len(METHODS)


def test_vendored_package_version_may_be_unknown_without_erasing_provenance(tmp_path):
    plan=fixture(tmp_path)
    path=tmp_path/'rank0.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()]
    rows[0]['runtime']['packages']['triton']=None
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=collect(plan)
    assert report['provenances'][0]['runtime']['packages']['triton'] is None


@pytest.mark.parametrize('competent',[True,False])
def test_diagnostic_milestone_never_expands_or_becomes_terminal(tmp_path,competent):
    plan=fixture(tmp_path);plan['diagnostic_step200']=True
    for rank in range(8):
        path=tmp_path/f'rank{rank}.jsonl'
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        rows[0].update(diagnostic_step200=True,checkpoint_step=200)
        if competent:
            ex=T.problem('dev',20270923,rank//2,T.FAMILIES[rank%2])
            def predict(visible):
                ps=T.oracle_marginals(ex['matrix_rows'],ex['syndrome'],visible,8)
                ps=[min(1-1e-12,max(1e-12,p)) for p in ps]
                return [[math.log1p(-p),math.log(p)] for p in ps]
            rows[1]['exact']=audit(ex,predict)
            rows[2].update(local_pass=True,all_eight_conditions_pass=True)
            rows[-1]['long_context_gate_passed']=True
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=collect(plan)
    assert result['checkpoint_step']==200 and result['diagnostic_step200']
    assert result['condition_count']==8 and result['competence_gate_passed']==competent
    plan['diagnostic_step200']=False
    with pytest.raises(ValueError,match='milestone'):collect(plan)


def test_passed_gate_requires_all_long_conditions_and_numerical_screen(tmp_path):
    from lrwkv_evidence.long_context_eval.numerical import specifications
    plan=fixture(tmp_path)
    for rank in range(8):
        path=tmp_path/f'rank{rank}.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        ex=T.problem('dev',20270923,rank//2,T.FAMILIES[rank%2])
        def predict(visible):
            ps=T.oracle_marginals(ex['matrix_rows'],ex['syndrome'],visible,8)
            ps=[min(1-1e-12,max(1e-12,p)) for p in ps]
            return [[math.log1p(-p),math.log(p)] for p in ps]
        rows[1]['exact']=audit(ex,predict)
        rows[2].update(local_pass=True,all_eight_conditions_pass=True)
        rows[-1]['long_context_gate_passed']=True
        expanded=[]
        for n in (1024,4096,16384):
            for pos in ('far','middle','near'):
                r=copy.deepcopy(rows[1]);r.update(requested_context_tokens=n,context_tokens=n,position=pos)
                expanded.append(r)
        numerical=dict(kind='numerical_qualification',checks=[dict(s,within_repeat_max_logprob_diff=0.,cross_rank_max_logprob_diff=0.) for s in specifications()])
        rows=rows[:3]+[numerical]+expanded+[rows[-1]]
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=collect(plan)
    assert result['competence_gate_passed'] and result['condition_count']==80
    path=tmp_path/'rank0.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
    rows=[r for r in rows if r['kind']!='numerical_qualification']
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='numerical qualification required'):collect(plan)


@pytest.mark.parametrize('comparator', ['attention', 'causal_rwkv'])
def test_comparator_shards_keep_model_and_policy_identity(tmp_path, comparator):
    from lrwkv_evidence.long_context_baseline_eval.exact import causal_audit
    plan = fixture(tmp_path)
    plan['comparator'] = comparator
    for rank in range(8):
        path = tmp_path / f'rank{rank}.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['comparator'] = comparator
        if comparator == 'causal_rwkv':
            ex = T.problem('dev', 20270923, rank // 2, T.FAMILIES[rank % 2])
            rows[1]['exact'] = causal_audit(ex, lambda y: [[-math.log(2)]*2 for _ in range(8)])
            rows[1]['costs'] = [dict(method='cached_causal', calls=8, seconds=[.1, .2, .3],
                                     peak_allocated_bytes=1000, sample_outputs=[0, 1, 2])]
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    result = collect(plan)
    assert result['comparator'] == comparator and not result['competence_gate_passed']
    assert len(result['aggregates']) == (2 if comparator == 'causal_rwkv' else 2*len(METHODS))
    plan['comparator'] = 'rwkv'
    with pytest.raises(ValueError, match='comparator'): collect(plan)


@pytest.mark.parametrize('damage', ['incomplete', 'duplicate', 'metric', 'support', 'gate', 'checkpoint', 'timing', 'runtime', 'hardware'])
def test_corrupt_or_incomplete_evidence_rejected(tmp_path, damage):
    plan = fixture(tmp_path)
    path = tmp_path / 'rank0.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if damage == 'incomplete': rows.pop()
    if damage == 'duplicate': rows.insert(2, copy.deepcopy(rows[1]))
    if damage == 'metric': rows[1]['exact']['results'][0]['joint_kl_nats'] = 0.
    if damage == 'support': rows[1]['exact']['support'] = [255]
    if damage == 'gate': rows[2]['local_pass'] = True
    if damage == 'checkpoint': rows[0]['checkpoint_sha256'] = 'wrong'
    if damage == 'timing': rows[1]['costs'][0]['seconds'][0] = -1.
    if damage == 'runtime': rows[1]['audit_seconds'] = rows[1]['condition_wall_seconds'] + 1
    if damage == 'hardware': rows[0]['runtime']['gpu_name'] = ''
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(ValueError): collect(plan)
