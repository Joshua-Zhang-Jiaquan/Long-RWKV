"""Portable exports preserve accepted counts and reject incomplete evidence."""
import gzip
import json
from pathlib import Path

import pytest

from lrwkv_evidence.e3 import export_receipts as E


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    panel=root/'panel';cell='associative_recall_L16384_p50_b8_similar'
    instances=[dict(cell_id=cell,data_seed=101,instance_index=i,input_ids_sha256=f'prompt{i}') for i in range(2)]
    write(panel/'grid_manifest.json',dict(split='iclr2027_grid'))
    write(panel/f'{cell}.json',dict(cell_id=cell,status='ok',history_length=16384,total_instances=2,instances=instances))
    jobs=[];parts=[]
    for model in E.MODELS:
        out=root/model
        provenance={'model':model,'mode':'confirm'}
        write(out/'provenance_rank0.json',provenance)
        receipts=out/'decoder' if model=='F2' else out
        for inst in instances:
            row=dict(inst,status='ok',split='iclr2027_grid',prompt_hash_verified=True,
                     correct=model=='F2' and inst['instance_index']==0,parse_ok=True,actual_nfe=2,
                     wall_seconds=.5,peak_allocated_bytes=100,peak_reserved_bytes=200,
                     provenance_sha256=E.digest(provenance))
            write(receipts/f'{cell}__101__{inst["instance_index"]}.json',row)
        # Malformed extras outside accepted assignment must never be parsed.
        (receipts/'outside_cell__101__0.json').write_text('invalid and excluded')
        jobs.append(dict(model=model.lower(),length=16384,out=str(out),partitions=[dict(out=str(out),cell_ids=[cell])]))
        parts.append(dict(model=model,length=16384,out=str(out),assigned_cells=1,used_receipts=2))
    plan=root/'plan.json';summary=root/'summary.json'
    write(plan,dict(panel=str(panel),jobs=jobs))
    write(summary,dict(complete=True,report_kind='complete_conditional_comparison',
        observed_items={'F2':2,'R0':2},supported_cells=1,excluded_cells=0,partitions=parts,
        cell_scores=[dict(cell_id=cell,n=2,F2=.5,R0=0,F2_only=1,R0_only=0,both_correct=0)]))
    return plan,summary,cell


def test_accepted_export_counts_hashes_and_deterministic_gzip(tmp_path):
    plan,summary,_=fixture(tmp_path)
    result=E.export(plan,summary,tmp_path/'export',expected_total=4)
    assert result['total_records']==4
    assert result['record_count_per_model']=={'F2':2,'R0':2}
    assert result['summary_cell_counts_verified'] and result['csv_roundtrip_counts_verified']
    for model in E.MODELS:
        output=Path(result['csv'][model]['path'])
        assert E.sha(output)==result['csv'][model]['sha256']
        with gzip.open(output,'rt') as stream:
            lines=stream.readlines()
        assert len(lines)==3
        assert 'source_file_sha256' in lines[0]
        assert not any('outside_cell' in line for line in lines)
    rerun=E.export(plan,summary,tmp_path/'export2',expected_total=4)
    assert {m:r['sha256'] for m,r in result['csv'].items()}=={m:r['sha256'] for m,r in rerun['csv'].items()}


@pytest.mark.parametrize('problem',['missing','duplicate','wrong_summary','incomplete_summary'])
def test_incomplete_duplicate_or_disagreeing_export_is_refused(tmp_path,problem):
    plan,summary,cell=fixture(tmp_path)
    source=tmp_path/'F2/decoder'/f'{cell}__101__0.json'
    if problem=='missing':source.unlink()
    elif problem=='duplicate':
        duplicate=tmp_path/'F2/other'/source.name
        duplicate.parent.mkdir();duplicate.write_bytes(source.read_bytes())
    else:
        data=E.read(summary)
        if problem=='wrong_summary':data['cell_scores'][0]['F2']=0
        else:data['complete']=False
        write(summary,data)
    with pytest.raises(ValueError):
        E.export(plan,summary,tmp_path/'export',expected_total=4)
    assert not list((tmp_path/'export').glob('*.csv.gz'))
