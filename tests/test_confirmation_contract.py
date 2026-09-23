import json
import pytest
from lrwkv_evidence.train04confirm import contract as K


def setup(root):
    model=root/'model_source';(model/'longrwkv').mkdir(parents=True)
    relative=[*K.REQUIRED,'model_source/longrwkv/model.py']
    for name in relative:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# fixture\n')
    manifest=dict(status='frozen',recipe=K.RECIPE,training=K.TRAINING,inference_triton_precision='ieee',
                  selection_evidence='fixed development report',sources={p:K.sha(root/p) for p in relative})
    path=root/'manifest.json';path.write_text(json.dumps(manifest));return path,model,manifest


@pytest.mark.parametrize('mutation',['unfrozen','wrong_seed','missing_source','changed_source','precision','qualification'])
def test_rejects_unfrozen_or_mismatched_execution(tmp_path,mutation):
    path,model,manifest=setup(tmp_path)
    assert K.validate(path,tmp_path,model)[0]['status']=='frozen'
    if mutation=='unfrozen':manifest['status']='draft'
    if mutation=='wrong_seed':manifest['training']={**K.TRAINING,'data_seeds':[17,53,71]}
    if mutation=='missing_source':manifest['sources'].pop(K.REQUIRED[0])
    if mutation=='changed_source':(tmp_path/K.REQUIRED[0]).write_text('# altered\n')
    if mutation=='precision':manifest['inference_triton_precision']='default'
    if mutation=='qualification':manifest['status']='qualification_only'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):K.validate(path,tmp_path,model)


def test_qualification_cannot_accept_a_frozen_research_manifest(tmp_path):
    path,model,manifest=setup(tmp_path)
    with pytest.raises(ValueError):K.validate(path,tmp_path,model,qualification=True)
    manifest['status']='qualification_only';path.write_text(json.dumps(manifest))
    assert K.validate(path,tmp_path,model,qualification=True)[0]['status']=='qualification_only'
