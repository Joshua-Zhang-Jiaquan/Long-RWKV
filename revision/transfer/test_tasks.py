"""Independent exact checks for the prospective experiment, before GPU use."""
import itertools
from lrwkv_evidence.predictive_transfer import tasks as T
from lrwkv_evidence.predictive_transfer import training as R
from lrwkv_evidence.train04.worker import load_tokenizer


def test_splits_public_bases_and_oracles():
    cat=T.catalog()
    assert len(cat)==25 and sum(len(v['matrices']) for v in cat.values())==20160
    assert len({tuple(v['weight_enumerator']) for v in cat.values()})==25
    for c in range(25):
        ex=T.make_example('calibration' if c<15 else 'heldout',42,c,c)
        ys=T.support(ex);assert len(ys)==16
        assert T.public_bases(ex)==ex['bases']
        for p in (0,1):
            first=T.public_bases(ex)[p]
            assert len({tuple(y[i] for i in first) for y in ys})==16
            assert all(sum(y[i] for y in ys)==8 for i in range(8))
            features=T.features(ex,p,'far')
            for y in ys:
                visible={i:y[i] for i in first}
                probs=T.oracle_marginals(ex['matrix_rows'],ex['syndrome'],visible,8)
                assert probs==list(map(float,y))
            # Independently reconstruct each predicted parity relation's size.
            for i,(_,fanin,eqs,_) in features[4:]:
                found=[]
                for count in range(1,5):
                    for subset in itertools.combinations(ex['matrix_rows'],count):
                        row=0
                        for v in subset:row^=v
                        if row>>i&1 and set(j for j in range(8) if row>>j&1)<={i,*first}:
                            found.append((row.bit_count()-1,count))
                assert found==[(fanin,eqs)]


def test_native_training_records():
    tok,_,_=load_tokenizer('/inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B')
    for step in (1,301,701,2501,2502,2503):
        rows=R.records(step,0,tok,R.SEEDS[0])
        assert len(rows)==4
        for row in rows:
            assert len(row['ids'])==len(row['position_codes'])==len(row['position_times'])
            assert len(row['target_positions'])==len(row['gold'])
            for i,pos in enumerate(row['target_positions']):
                assert row['ids'][pos]==(tok.mask_id if row['masked'][i] else tok.binary_ids[row['gold'][i]])
            if step>2500:assert 16370 <= row['prefix'] <= 16384
            if step>700:
                assert sum(row['loss_mask'])==4
                expected=.5 if row['round_index']==0 else None
                for i,selected in enumerate(row['loss_mask']):
                    if selected:assert row['oracle'][i]==(expected if expected is not None else row['gold'][i])
