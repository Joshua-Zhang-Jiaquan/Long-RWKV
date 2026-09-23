import copy
import math
import pytest
from lrwkv_evidence.long_context_mvp import tasks as T

class Tokenizer:
    binary_ids = [1001,1002]
    mask_id = 1003
    def encode(self, text): return list(text.encode())

@pytest.mark.parametrize('family', T.FAMILIES)
def test_paired_contexts_and_gold_isolation(family):
    tok=Tokenizer();ex=T.problem('dev',17,3,family)
    evidence=tok.encode(T.record_text(ex['key'],ex['value']))
    for length in (1024,4096):
        for position in ('near','middle','far'):
            serial=T.serialize(ex,tok,length,position)
            assert len(serial['prefix_ids'])==length
            lo,hi=serial['evidence_span'];assert serial['prefix_ids'][lo:hi]==evidence
            prompt=bytes(serial['prefix_ids']).decode()
            assert prompt.count(T.record_text(ex['key'],ex['value']))==1
            altered=copy.deepcopy(ex);altered['bits']=[1-b for b in ex['bits']];altered['support']=[]
            assert T.serialize(altered,tok,length,position)==serial
            assert T.canvas(ex,serial,tok)['ids']==T.canvas(altered,serial,tok)['ids']


def test_split_keys_disjoint_and_length_pairing():
    banks=[set(T.key_bank(s)) for s in T.SPLITS]
    assert all(not banks[i]&banks[j] for i in range(3) for j in range(i))
    a=T.problem('dev',17,3,'retrieval');b=T.problem('dev',17,3,'dependent')
    assert (a['key'],a['value'])==(b['key'],b['value'])


def test_oracle_schedule_mechanism():
    tok=Tokenizer();ex=T.problem('dev',17,4,'dependent');s=T.serialize(ex,tok)
    assert len(ex['support'])==16 and T.canvas(ex,s,tok)['oracle']==[.5]*8
    for y in ex['support']:
        visible={i:(y>>i)&1 for i in range(4)}
        oracle=T.canvas(ex,s,tok,visible)['oracle']
        assert oracle==[(y>>i)&1 for i in range(8)]
    assert math.isclose(len(ex['support'])/256,1/16)
    control=T.problem('dev',17,4,'retrieval')
    assert len(control['support'])==1
    assert T.canvas(control,T.serialize(control,tok),tok)['oracle']==control['bits']


def test_bad_budget_and_inconsistent_history_refused():
    tok=Tokenizer();ex=T.problem('dev',17,4,'dependent')
    with pytest.raises(ValueError):T.serialize(ex,tok,1)
    with pytest.raises(ValueError):T.canvas(ex,T.serialize(ex,tok),tok,{0:0,4:1-ex['value'][0]})


def test_inference_does_not_solve_or_read_hidden_labels():
    tok=Tokenizer();ex=T.problem('dev',17,4,'dependent');serial=T.serialize(ex,tok)
    bad={0:0,4:1-ex['value'][0]}
    public={k:v for k,v in ex.items() if k not in ('bits','support','matrix_rows','syndrome')}
    result=T.inference_canvas(public,serial,tok,bad)
    assert result['ids'][-8]==tok.binary_ids[0]
    assert result['ids'][-4]==tok.binary_ids[bad[4]]
    assert result['gold']==[0]*8 and result['oracle']==[.5]*8
