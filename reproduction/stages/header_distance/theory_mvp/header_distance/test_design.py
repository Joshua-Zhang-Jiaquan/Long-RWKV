import math
from collections import Counter
from pathlib import Path
import pytest
from lrwkv_evidence.header_distance import tasks as T
from lrwkv_evidence.distance_intervention import tasks as OLD
from lrwkv_evidence.train04.worker import load_tokenizer


@pytest.fixture(scope='module')
def tok():
    return load_tokenizer(Path('/inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B'))[0]


@pytest.mark.parametrize('index',range(32))
def test_exact_slots_and_whole_records(tok,index):
    ex=T.example(index);layouts={c:T.serialize(ex,tok,c) for c in T.CELLS}
    reference=layouts['legacy_near']['ids']
    assert len(reference)==16383
    for cell,r in layouts.items():
        assert len(r['ids'])==len(reference)
        assert Counter(r['ids'])==Counter(reference)
        if cell.startswith('legacy_'):continue
        a,b=r['filler_header_records'],r['filler_task_records']
        assert not set(a)&set(b)
        assert sum(len(tok.encode('\narchive_'+str(i)+' = unused;')) for i in a)==24
        assert sum(len(tok.encode('\narchive_'+str(i)+' = unused;')) for i in b)==88
    for position in ('far','near'):
        assert layouts[f'header_{position}_task_far']['header_span']==layouts[f'header_{position}_task_near']['header_span']
        assert layouts[f'header_far_task_{position}']['task_span']==layouts[f'header_near_task_{position}']['task_span']


def test_fresh_prompts():
    old={OLD.example(i)['prompt'] for i in range(32)}
    new={T.example(i)['prompt'] for i in range(32)}
    assert len(new)==32 and not old&new


def test_exact_endpoint_oracle_and_independent():
    ex=T.example(0);initial=[[math.log(.5)]*2 for _ in range(8)];eps=1e-7
    conditional=[]
    for visible in T.histories(ex):
        y=next(y for y in ex['support'] if all((y>>i)&1==b for i,b in visible.items()))
        conditional.append([[math.log(1-eps if b==(y>>i)&1 else eps) for b in (0,1)] for i in range(8)])
    result=T.endpoint(ex,initial,conditional)
    assert result['one_kl']==pytest.approx(4*math.log(2))
    assert result['info_kl']==pytest.approx(-4*math.log(1-eps))
    assert result['valid_mass_info']==pytest.approx((1-eps)**4)
    independent=T.endpoint(ex,initial,[initial]*16)
    assert independent['info_kl']==pytest.approx(4*math.log(2))
    assert independent['gain_over_one']==pytest.approx(0)


def test_reject_missing_histories():
    with pytest.raises(ValueError):T.endpoint(T.example(0),[],[])
