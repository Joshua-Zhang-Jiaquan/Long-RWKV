import copy
import pytest
from theory_mvp.train04confirm.training_audit import validate_records
from lrwkv_evidence.train04confirm import contract as K


def fixture(phase='independent'):
    start, stop = (0, 1500) if phase == 'parent' else (1500, 2500)
    parent = None if phase == 'parent' else 'parent-sha'
    contract = dict(version=6, phase=phase, seed=53, initial_seed=0,
                    manifest_sha256='manifest', qualification=False, stop_step=stop,
                    parent_checkpoint_sha256=parent, objective='rao_blackwell', recipe=K.RECIPE)
    provenance = dict(contract=contract, start_step=start, stop_step=stop)
    completion = dict(execution_complete=True, qualification=False, seed=53,
                      phase=phase, step=stop, checkpoint_sha256='terminal-sha', elapsed_seconds=stop+1.)
    rows = [dict(step=s, seed=53,
                 phase='ordinary_n2' if s <= 300 else 'label_mixture_n4' if s <= 700 else 'label_mixture_n8' if s <= 1500 else 'paired_n8',
                 global_input_tokens=1776 if s <= 300 else 2944 if s <= 700 else 4768,
                 lr=3e-5*min(s/50, 1.), mean_loss=.5, grad_norm_rank0=1., elapsed_seconds=float(s))
            for s in range(start+1, stop+1)]
    return [53, phase, 'manifest', parent, provenance, completion, rows]


@pytest.mark.parametrize('phase,tokens', [('parent',5524800),('independent',4768000),('complementary',4768000)])
def test_complete_interval(phase, tokens):
    assert validate_records(*fixture(phase))['input_tokens'] == tokens


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'parent', 'nan', 'tokens', 'lr', 'clock', 'incomplete'])
def test_corrupt_training_evidence_rejected(fault):
    args = copy.deepcopy(fixture())
    if fault == 'missing': args[-1].pop(50)
    if fault == 'duplicate': args[-1][50] = args[-1][49].copy()
    if fault == 'parent': args[4]['contract']['parent_checkpoint_sha256'] = 'different-seed-parent'
    if fault == 'nan': args[-1][50]['mean_loss'] = float('nan')
    if fault == 'tokens': args[-1][50]['global_input_tokens'] += 1
    if fault == 'lr': args[-1][50]['lr'] *= 2
    if fault == 'clock': args[-1][50]['elapsed_seconds'] = 0.
    if fault == 'incomplete': args[5]['execution_complete'] = False
    with pytest.raises(ValueError): validate_records(*args)
