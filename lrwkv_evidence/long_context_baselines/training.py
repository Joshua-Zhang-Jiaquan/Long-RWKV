"""Same public task/curriculum for practical attention and causal comparators."""
from lrwkv_evidence.long_context_mvp import training as R, tasks as T


def recipe(kind):
    if kind not in ('attention', 'causal_rwkv'):
        raise ValueError('unknown comparator')
    return dict(R.RECIPE, comparator=kind,
                objective=R.RECIPE['objective'] if kind == 'attention' else 'teacher-forced exact conditional soft cross entropy',
                comparability='Same examples, global batch, updates and optimizer. Native pretrained weights/tokenizers differ for Pythia; not an architecture-causal ablation.')


def records(kind, step, rank, tokenizer):
    if kind == 'attention':
        return R.records(step, rank, tokenizer)
    if kind != 'causal_rwkv' or not 1 <= step <= 600 or not 0 <= rank < 8:
        raise ValueError('outside comparator development recipe')
    result = []
    for j in range(4):
        index = (step - 1) * 32 + rank * 4 + j
        ex = T.problem('train', 17, index, T.FAMILIES[index % 2])
        serial = T.serialize(ex, tokenizer, None if step <= 200 else 1024, ('near', 'middle', 'far')[index % 3])
        oracle = []
        for i in range(8):
            visible = {k: ex['bits'][k] for k in range(i)}
            oracle.append(T.oracle_marginals(ex['matrix_rows'], ex['syndrome'], visible, 8)[i])
        result.append(dict(ids=serial['prefix_ids'] + [tokenizer.binary_ids[b] for b in ex['bits']],
                           prefix=serial['context_tokens'], gold=ex['bits'], oracle=oracle, masked=[False]*8,
                           stage=8, family=ex['family'], instance_id=ex['instance_id'],
                           loss_mask=[True]*8, loss_scale=1/8))
    return result


batch = R.batch
loss = R.loss
