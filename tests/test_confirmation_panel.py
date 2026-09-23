from lrwkv_evidence.train04.confirmation_panel import build,condition_key,SEED
from lrwkv_evidence.train04 import tasks


def test_confirmation_panel_excludes_old_conditions_and_covers_catalogs():
    panel=build();assert panel['counts']=={'systematic':22,'paired_parity':32}
    old={condition_key(tasks.make_example('test',s,i)) for s,count in ((73191,32),(73192,128)) for i in range(count)}
    fresh=[tasks.make_example('test',SEED,r['index']) for r in panel['records']]
    assert len({condition_key(e) for e in fresh})==54
    assert not old.intersection(condition_key(e) for e in fresh)
    for family in tasks.FAMILIES:
        assert {tuple(sorted(e['matrix_rows'])) for e in fresh if e['family']==family}==set(tasks.split_catalog('test',family))
        for split in ('train','dev'):
            assert not set(tasks.split_catalog(split,family)).intersection(tuple(sorted(e['matrix_rows'])) for e in fresh if e['family']==family)
