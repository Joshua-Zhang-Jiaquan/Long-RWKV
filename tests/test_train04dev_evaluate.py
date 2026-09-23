import math
from lrwkv_evidence.train04dev import core as C
from lrwkv_evidence.train04dev.evaluate import endpoint_metrics
from lrwkv_evidence.train04 import tasks


def test_exact_accounting_across_curriculum_dimensions():
    for n in (2,4,8):
        for index in (0,1):
            ex=C.example('dev',51917,index,n)
            def oracle(ex,visible,stage):
                return [[math.log(x) if x else -math.inf for x in (1-p,p)] for p in tasks.oracle_marginals(ex['matrix_rows'],ex['syndrome'],visible,n)]
            def fair(ex,visible,stage):return [[-math.log(2)]*2 for _ in range(n)]
            for method in ('information_set','sequential'):
                exact=endpoint_metrics(ex,method,oracle)
                assert abs(exact['joint_kl_nats'])<1e-12
                assert abs(exact['exact_valid_mass']-1)<1e-12
            for method in ('one','information_set','random_halves','sequential'):
                row=endpoint_metrics(ex,method,fair)
                assert abs(row['joint_kl_nats']-n/2*math.log(2))<1e-12
                assert abs(row['exact_valid_mass']-2**(-n/2))<1e-12
                assert abs(row['decomposition_residual'])<1e-12


def test_collector_requires_complete_panel_and_recomputes_metrics(tmp_path):
    import json
    import pytest
    from lrwkv_evidence.train04dev.collect_evaluation import collect
    from lrwkv_evidence.train04dev.history_response import history_response
    for rank in range(8):
        rows=[dict(kind='provenance',rank=rank,world=8,checkpoint_sha256='fixture',step=700,objective='hard',split='dev',seed=51917,conditions_per_dimension=16,dimensions=[2,4,8])]
        for n in (2,4,8):
            for index in range(rank,16,8):
                ex=C.example('dev',51917,index,n)
                def fair(ex,visible,stage):return [[-math.log(2)]*2 for _ in range(n)]
                cache={}
                for method in ('one','information_set','random_halves'):
                    rows.append(dict(kind='endpoint',n=n,index=index,family=ex['family'],**endpoint_metrics(ex,method,fair,cache)))
                rows.append(dict(kind='history_response',n=n,index=index,family=ex['family'],**history_response(ex,cache)))
        rows.append(dict(kind='complete'))
        (tmp_path/f'rank{rank}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    result=collect(tmp_path)
    assert len(result['summary'])==18
    assert len(result['history_response_summary'])==6
    assert all(r['paired_signed_response'] in (None,0.) for r in result['history_response_summary'])
    assert all(abs(r['random_minus_information_set_kl'])<1e-12 for r in result['paired_condition_contrasts'])
    path=tmp_path/'rank0.jsonl';lines=path.read_text().splitlines();del lines[1];path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(AssertionError):collect(tmp_path)
