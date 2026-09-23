"""Fresh public conditions and length-preserving syndrome interventions."""
import hashlib
from lrwkv_evidence.train04 import tasks as T
from lrwkv_evidence.train04.evaluate import support
from lrwkv_evidence.posterior_context_probe.tasks import serialize, partitions

SEED = 20271021


def example(index):
    if not 0 <= index < 32: raise ValueError('outside frozen test panel')
    ex = T.make_example('test', SEED, index, family='paired_parity')
    ex['support'] = [sum(b << i for i,b in enumerate(y)) for y in support(ex)]
    return ex


def counterfactual(ex):
    row_index = ex['index'] % 4
    syndrome = list(ex['syndrome']); syndrome[row_index] ^= 1
    names = ex['coordinate_names']
    equations = [' XOR '.join(names[j] for j in range(8) if row >> j & 1) + f' = {v}'
                 for row,v in zip(ex['matrix_rows'], syndrome)]
    prompt = (T.INSTRUCTION+'\nBits: '+', '.join(names)+'\nConstraints:\n'+'\n'.join(equations)
              +'\nOutput order: '+', '.join(names)+'\nAnswer:')
    changed = [i for i in range(8) if ex['matrix_rows'][row_index] >> i & 1 and i not in ex['information_set']]
    if len(changed) != 1: raise ValueError('counterfactual must affect one remaining target')
    cf = dict(ex, prompt=prompt, syndrome=syndrome,
              instance_id=hashlib.sha256((ex['instance_id']+'/syndrome_flip/'+str(row_index)).encode()).hexdigest())
    # Bits are metadata only; keep them feasible even though inference never uses them.
    cf['bits'] = list(ex['bits']); cf['bits'][changed[0]] ^= 1
    cf['support'] = [sum(b << i for i,b in enumerate(y)) for y in support(cf)]
    visible = {i:0 for i in ex['information_set']}
    original_oracle = T.oracle_marginals(ex['matrix_rows'], ex['syndrome'], visible, 8)
    flipped_oracle = T.oracle_marginals(cf['matrix_rows'], cf['syndrome'], visible, 8)
    if [i for i in range(8) if original_oracle[i] != flipped_oracle[i]] != changed:
        raise ValueError('unexpected counterfactual support')
    return cf, dict(row_index=row_index, target_coordinate=changed[0], visible=visible,
                   original_target=int(original_oracle[changed[0]]), flipped_target=int(flipped_oracle[changed[0]]),
                   unaffected_targets=[i for i in range(8) if i not in visible and i != changed[0]])


def validate_layout(encoded, changed):
    if len(encoded['ids']) != len(changed['ids']) or encoded['public_block_span'] != changed['public_block_span']:
        raise ValueError('counterfactual changes native context layout')
    indices = [i for i,(a,b) in enumerate(zip(encoded['ids'], changed['ids'])) if a != b]
    lo,hi = encoded['public_block_span']
    if len(indices) != 1 or not lo <= indices[0] < hi:
        raise ValueError('counterfactual must change exactly one public-block token')
    return indices[0]
