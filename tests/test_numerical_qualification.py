import json
import math
import pytest
from lrwkv_evidence.train04dev.numerical_qualification import collect


def fixture(root):
    for rank in range(8):
        rows=[dict(n=n,index=i,stage=s,log_probabilities=[[-math.log(2)]*2 for _ in range(n)],
                   wrapper_max_log_probability_difference=0.,repeat_max_log_probability_difference=0.)
              for n in (2,4,8) for i in (0,1) for s in (4,8)]
        (root/f'numerical_rank{rank}.json').write_text(json.dumps(dict(rank=rank,world=8,execution_complete=True,
            checkpoint_sha256='same',triton_f32_default='ieee',torch_tf32=False,rows=rows,scope='test')))


def change(root,fn):
    path=root/'numerical_rank7.json';report=json.loads(path.read_text());fn(report);path.write_text(json.dumps(report))


def test_detects_cross_process_disagreement(tmp_path):
    fixture(tmp_path);assert collect(tmp_path)['passed']
    change(tmp_path,lambda r:r['rows'][0]['log_probabilities'].__setitem__(0,[math.log(.51),math.log(.49)]))
    assert not collect(tmp_path)['passed']


@pytest.mark.parametrize('failure',['nonfinite','missing','checkpoint','duplicate'])
def test_rejects_invalid_evidence(tmp_path,failure):
    fixture(tmp_path)
    def corrupt(r):
        if failure=='nonfinite':r['rows'][0]['repeat_max_log_probability_difference']=float('nan')
        if failure=='missing':r['rows'].pop()
        if failure=='checkpoint':r['checkpoint_sha256']='different'
        if failure=='duplicate':r['rows'].append(r['rows'][0])
    change(tmp_path,corrupt)
    with pytest.raises(ValueError):collect(tmp_path)
