"""Complete, paired control tables cannot silently change their denominator."""
import copy

import pytest

from lrwkv_evidence.e3 import report_controls as R


def rows():
    return {model:[dict(model=model,kind=kind,
                       item_id=f'{family}_L16384_p50_b8_similar__201__{i}',
                       status='ok',correct=(family==R.FAMILIES[0]),
                       parse_ok=(family!=R.FAMILIES[2]))
                  for kind in R.C.KINDS for family in R.FAMILIES for i in range(24)]
            for model in R.MODELS}


def test_exact_counts_family_macro_and_parse_rates():
    report=R.aggregate(rows())
    assert len(report['rows'])==24
    assert len(report['macros'])==8
    for macro in report['macros']:
        assert macro['n']==72 and macro['correct']==24 and macro['parsed']==48
        assert macro['family_macro_accuracy']==pytest.approx(1/3)
        assert macro['family_macro_parse_rate']==pytest.approx(2/3)
    assert 'independent confirmation' in report['interpretation']
    assert '24/24' in R.latex(report)


@pytest.mark.parametrize('mutation', ['remove','duplicate','replace_task'])
def test_partial_duplicate_and_unpaired_panels_refused(mutation):
    data=rows()
    if mutation=='remove':
        data['f2'].pop()
    elif mutation=='duplicate':
        data['f2'][-1]=copy.deepcopy(data['f2'][-2])
    else:
        data['f2'][-1]['item_id']+='-different-task'
    with pytest.raises(ValueError):
        R.aggregate(data)


def test_unsupported_not_converted_to_wrong_answer_or_smaller_denominator():
    data=rows()
    data['r0'][0]=dict(data['r0'][0],status='unsupported',unsupported_reason='canvas too small')
    with pytest.raises(ValueError,match='unsupported control'):
        R.aggregate(data)
