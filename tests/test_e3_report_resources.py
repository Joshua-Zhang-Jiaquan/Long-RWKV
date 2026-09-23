"""Resource quantiles are computed from portable records and checked against totals."""
import copy

import pytest

from lrwkv_evidence.e3 import report_resources as R


def fixture():
    cells=[];records=[];resources={m:{} for m in R.MODELS}
    for length in R.LENGTHS:
        cell=f'associative_recall_L{length}_p50_b8_similar'
        cells.append(dict(cell_id=cell,length=length,family='associative_recall',n=2,F2=.5,R0=.5))
        for model in R.MODELS:
            resources[model][str(length)]=dict(n=2,nfe_mean=2,nfe_median=2,
                wall_seconds_mean=.5,wall_seconds_median=.5,wall_seconds_p95=.77,
                wall_seconds_sum=1,peak_allocated_bytes_max=2*2**30,peak_reserved_bytes_max=3*2**30)
            for i in range(2):
                records.append(dict(model=model,cell_id=cell,data_seed=101,instance_index=i,
                    correct=int(i==0),parse_ok=int(i==0),actual_nfe=1+2*i,wall_seconds=.2+.6*i,
                    peak_allocated_bytes=(i+1)*2**30,peak_reserved_bytes=(i+2)*2**30))
    summary=dict(complete=True,report_kind='complete_conditional_comparison',cell_scores=cells,
                 observed_items={m:6 for m in R.MODELS},resources=resources)
    return records,summary


def test_exact_quantiles_memory_units_and_parse_denominators():
    records,summary=fixture();report=R.summarize(records,summary)
    assert len(report['resources'])==6
    for row in report['resources']:
        assert row['nfe_p95']==pytest.approx(2.9)
        assert row['wall_seconds_p95']==pytest.approx(.77)
        assert row['peak_allocated_gib']==2 and row['peak_reserved_gib']==3
    assert all(r['parsed']==1 and r['n']==2 and r['parse_rate']==.5 for r in report['parsing'])
    assert 'JIT' in R.latex(report) and 'no KV cache' in R.latex(report)


@pytest.mark.parametrize('problem',['missing','duplicate','resource_mismatch','incomplete'])
def test_incomplete_or_disagreeing_sources_rejected(problem):
    records,summary=fixture()
    if problem=='missing':records.pop()
    elif problem=='duplicate':records.append(copy.deepcopy(records[0]))
    elif problem=='resource_mismatch':records[0]['wall_seconds']=5
    else:summary['complete']=False
    with pytest.raises(ValueError):R.summarize(records,summary)
