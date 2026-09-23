"""Control dispatch reuses qualified kernels and preserves transformed targets."""
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lrwkv_evidence.e3 import control_runner as X


def fixture():
    return dict(status='ok',kind='evidence_only',gold_answers=['v1911'],
        decoder_input=dict(input_ids=[1,2,65535,65535,65535],target_span=[2,5]),
        transformation_sha256='control-hash',input_ids_sha256='input-hash',
        provenance=dict(source_input_ids_sha256='source-hash',cell_id='associative_recall_L16384_p50_b8_similar'))


def receipt(control, model='f2'):
    return dict(status='ok',split='dev',item_id='item',kind=control['kind'],model=model,
        provenance_sha256='provenance',transformation_sha256=control['transformation_sha256'],
        source_input_ids_sha256='source-hash',input_ids_sha256='input-hash',
        target_span=[2,5],gold=['v1911'],prompt_hash_verified=True,
        policy=X.CONFIG if model=='f2' else X.A.POLICY,
        text='v1911',prediction='v1911',parse_ok=True,correct=True,output_ids=[1,2,3],actual_nfe=3,
        wall_seconds=.2,logit_reuse_parity_sha256='parity',tokenizer_check={'native_decode_equal':True})


def test_f2_dispatch_uses_frozen_policy_cache_and_original_span():
    model=SimpleNamespace(mask_token_id=65535,pad_token_id=0)
    ids=object(); span=[2,5]; generator=object()
    with patch.object(X.U,'sample_target',return_value=([1,2,3],2,[0,3])) as sample:
        assert X.dispatch('f2',model,ids,span,generator=generator)==([1,2,3],2,[0,3])
        sample.assert_called_once_with(model,ids,span,X.CONFIG,mask_id=65535,pad_id=0,
                                       generator=generator,reuse_unchanged_logits=True)
        assert sample.call_args.args[2] is span


def test_r0_dispatch_uses_qualified_greedy_and_original_span():
    ids=object(); model=object(); span=[2,5]; legal=[1,2,3]
    with patch.object(X.A,'greedy_suffix',return_value=([1,2,3],3)) as decode:
        assert X.dispatch('r0',model,ids,span,legal_ids=legal)==([1,2,3],3,None)
        decode.assert_called_once_with(model,ids,span,legal)
        assert decode.call_args.args[2] is span


@pytest.mark.parametrize('field,value', [('target_span',[1,4]),('gold',['v9999']),
    ('transformation_sha256','changed'),('provenance_sha256','changed'),
    ('correct',False),('actual_nfe',9),('text','v1234'),('output_ids',[1]),
    ('wall_seconds',float('nan'))])
def test_stale_or_corrupted_receipts_fail_closed(field,value):
    control=fixture(); row=receipt(control)
    X.validate_receipt(row,control,item='item',provenance_hash='provenance',model_kind='f2')
    row[field]=value
    with pytest.raises(ValueError):
        X.validate_receipt(row,control,item='item',provenance_hash='provenance',model_kind='f2')


def test_source_pin_survives_staging_path_but_not_code_change():
    X.verify_source_subset({'/old/gpu_runner.py':'abc'},{'/new/gpu_runner.py':'abc'})
    with pytest.raises(ValueError,match='dependency changed'):
        X.verify_source_subset({'/old/gpu_runner.py':'abc'},{'/new/gpu_runner.py':'def'})


def test_real_banked_control_rank_priority_and_target_preservation():
    panel=Path(__file__).resolve().parents[1]/'results/lc_dev_panel'
    rows=list(X.control_items(panel,rank=0,world=8))
    assert len(rows)==36
    assert [r[3]['kind'] for r in rows]==[kind for kind in X.KINDS for _ in range(9)]
    assert [r[0] for r in rows[:9]]==[r[0] for r in rows[9:18]]
    for item,cell,inst,control,encoder in rows:
        data=control['decoder_input']; lo,hi=data['target_span']
        assert data['input_ids'][lo:hi]==[65535]*(hi-lo)
        assert hi-lo==inst.target_span[1]-inst.target_span[0]
        assert control['gold_answers']==list(inst.answers)
        assert control['queries']==list(inst.queries)
        assert set(data)=={'input_ids','target_span'}
