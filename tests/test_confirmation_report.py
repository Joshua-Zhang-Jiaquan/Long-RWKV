import copy
import math
import pytest
from theory_mvp.train04confirm.report import aggregate,SEEDS,ARMS,FAMILIES,METHODS


def reports():
    result=[]
    effects={53:-.2,71:.1,89:.3}
    for seed in SEEDS:
        for arm in ARMS:
            rows=[];contrasts=[]
            for family in FAMILIES:
                count=16 if family=='paired_parity' else 11
                kl=2+(effects[seed] if arm=='independent' else 0)
                for method in METHODS:
                    rows.append(dict(family=family,method=method,structures=count,conditions=2*count,
                        joint_kl_nats=kl,exact_valid_mass=.5,dependence_cost_nats=.25,
                        estimation_error_nats=kl-.25,conditional_valid_kl_nats=kl-math.log(2)))
                for i in range(count):
                    contrasts.append(dict(family=family,structure=[i],random_minus_information_set_kl=0.,information_set_minus_random_valid_mass=0.))
            result.append(dict(provenance=dict(seed=seed,phase=arm,manifest_sha256='manifest',panel_sha256='panel',partitions_sha256='partitions',step=2500,triton_f32_default='ieee',torch_tf32=False),
                               family_summary=rows,structure_contrasts=contrasts))
    return result


def test_keeps_adverse_seed_in_paired_training_comparison():
    result=aggregate(reports())
    assert len(result['per_seed'])==48 and len(result['training_contrasts_by_seed'])==24
    for row in result['training_contrasts']:
        summary=row['independent_minus_complementary_kl']
        assert summary['seeds']==[53,71,89]
        assert summary['values']==pytest.approx([-.2,.1,.3])
        assert summary['minimum']==pytest.approx(-.2)
        assert summary['mean']==pytest.approx(.2/3)


@pytest.mark.parametrize('mutation',['missing','duplicate','precision','family','structure','nonfinite'])
def test_rejects_incomplete_or_incomparable_results(mutation):
    values=copy.deepcopy(reports())
    if mutation=='missing':values.pop()
    if mutation=='duplicate':values[-1]=copy.deepcopy(values[0])
    if mutation=='precision':values[-1]['provenance']['triton_f32_default']='default'
    if mutation=='family':values[-1]['family_summary'].pop()
    if mutation=='structure':values[-1]['structure_contrasts'].pop()
    if mutation=='nonfinite':values[-1]['family_summary'][0]['joint_kl_nats']=float('nan')
    with pytest.raises(ValueError):aggregate(values)
