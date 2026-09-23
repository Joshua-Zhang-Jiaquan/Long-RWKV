"""Locked same-checkpoint sampling comparison on held-out labeled GF(2) codes."""
import argparse
import copy
from collections import Counter
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import time
import torch
from . import tasks
from . import worker as W

METHODS = ('one', 'information_set', 'random_halves', 'bernoulli2', 'bernoulli4', 'bernoulli8', 'sequential')
CONTRACT = dict(version=1, split='test', data_seed=73191, conditions_per_family=16,
                draws_per_condition=32, calibration_conditions_per_family=64,
                exact_joint_conditions_per_family=2, temperature=1,
                methods=list(METHODS), random_halves='fixed independent partition per condition',
                checkpoint='terminal500', calibration='8 stages; independent forward masks; masked-position oracle-to-model KL',
                joint_kl='exact endpoint KL only for fixed schedules; sum over all16 valid vectors',
                uncertainty='paired conditions conditional on each trained checkpoint; three training seeds reported separately')


def rng_for(*key):
    return random.Random(int(hashlib.sha256(json.dumps(key).encode()).hexdigest(), 16))


def fixed_groups(example, method):
    n = example['n']
    if method == 'one': return [list(range(n))]
    if method == 'sequential': return [[i] for i in range(n)]
    if method == 'information_set': first = example['information_set']
    elif method == 'random_halves':
        first = list(range(n)); rng_for('partition', example['instance_id']).shuffle(first); first = first[:n//2]
    else: raise ValueError(method)
    return [list(first), [i for i in range(n) if i not in first]]


def valid(example, bits):
    y = sum(b << i for i, b in enumerate(bits))
    return all((y & row).bit_count() % 2 == c for row, c in zip(example['matrix_rows'], example['syndrome']))


def support(example):
    particular, basis, _ = tasks.solve_affine(example['matrix_rows'], example['syndrome'], example['n'])
    result = []
    for k in range(1 << len(basis)):
        y = particular
        for j, b in enumerate(basis):
            if k >> j & 1: y ^= b
        result.append(tuple(y >> i & 1 for i in range(example['n'])))
    return result


class Predictor:
    def __init__(self, model, tokenizer, binary_ids, mask_id, device):
        self.model, self.tokenizer, self.binary_ids, self.mask_id, self.device = model, tokenizer, binary_ids, mask_id, device
        self.calls = 0

    @torch.inference_mode()
    def log_probs(self, example, visible, stage):
        prefix = self.tokenizer.encode(example['prompt'])
        target = [self.binary_ids[visible[i]] if i in visible else self.mask_id for i in range(example['n'])]
        ids = torch.tensor([prefix + target], dtype=torch.long, device=self.device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = self.model(ids, (len(prefix), len(prefix)+example['n']), stage)
        logp = logits.double().log_softmax(-1).cpu().tolist()
        if any(not math.isfinite(p) for row in logp for p in row): raise ValueError('nonfinite log probabilities')
        self.calls += 1
        return logp

    def __call__(self, example, visible, stage):
        return [math.exp(row[1]) for row in self.log_probs(example, visible, stage)]


def log_probabilities(predict, example, visible, stage):
    if hasattr(predict, 'log_probs'): return predict.log_probs(example, visible, stage)
    # Exact oracle seam for CPU checks.
    return [[math.log(q) if q else -math.inf for q in (1-p,p)] for p in predict(example,visible,stage)]


def sample(example, method, predict, draw):
    visible = {}; n = example['n']; calls_before = predict.calls
    # Position/stage-indexed uniforms couple draws across policies without relying on RNG consumption order.
    uniform = [[rng_for('sample', example['instance_id'], draw, t, i).random() for i in range(n)] for t in range(8)]
    if method.startswith('bernoulli'):
        total = int(method.removeprefix('bernoulli'))
        for step in range(total):
            remaining = total-step
            if len(visible) == n: break
            stage = remaining * 8 // total
            probs = predict(example, visible, stage)
            for i in range(n):
                if i not in visible and rng_for('reveal', example['instance_id'], draw, total, step, i).random() < 1/remaining:
                    visible[i] = int(uniform[step][i] < probs[i])
    else:
        for step, group in enumerate(fixed_groups(example, method)):
            stage = math.ceil(8*(n-len(visible))/n)
            probs = predict(example, visible, stage)
            for i in group: visible[i] = int(uniform[step][i] < probs[i])
    if len(visible) != n: raise AssertionError('unfinished canvas')
    return [visible[i] for i in range(n)], predict.calls-calls_before


def exact_joint_kl(example, method, predict):
    """Fixed partition only: probability of each valid endpoint via its unique reveal path."""
    targets = support(example); terms = []
    for bits in targets:
        visible = {}; logq = 0.
        for group in fixed_groups(example, method):
            logp = log_probabilities(predict, example, visible, math.ceil(8*(example['n']-len(visible))/example['n']))
            for i in group:
                logq += logp[i][bits[i]]; visible[i] = bits[i]
        terms.append(-math.log(len(targets))-logq)
    return sum(terms)/len(terms)


def calibration(example, predict):
    rows = []
    for stage in range(1, 9):
        randomizer = rng_for('calibration', example['instance_id'], stage)
        visible = {i:b for i,b in enumerate(example['bits']) if randomizer.random() >= stage/8}
        logp = log_probabilities(predict, example, visible, stage)
        oracle = tasks.oracle_marginals(example['matrix_rows'], example['syndrome'], visible, example['n'])
        masked = [i for i in range(example['n']) if i not in visible]
        kl = 0.; correct = deterministic = 0; fair_deviation = []
        for i in masked:
            q = math.exp(logp[i][1]); p = oracle[i]
            if p > 0: kl += p*(math.log(p)-logp[i][1])
            if p < 1: kl += (1-p)*(math.log(1-p)-logp[i][0])
            if p in (0.,1.):
                deterministic += 1; correct += int((q >= .5) == bool(p))
            else: fair_deviation.append(abs(q-.5))
        rows.append(dict(stage=stage, masked=len(masked), marginal_kl_sum_nats=kl,
                         elbo_excess_per_token=8*kl/(stage*example['n']),
                         deterministic=deterministic, deterministic_correct=correct,
                         fair_abs_deviation_sum=sum(fair_deviation), fair_count=len(fair_deviation)))
    return rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--base',type=Path,required=True); ap.add_argument('--model-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    rank=int(os.environ.get('RANK',0)); world=int(os.environ.get('WORLD_SIZE',1)); local=int(os.environ.get('LOCAL_RANK',0))
    torch.set_num_threads(4); torch.cuda.set_device(local)
    tokenizer, binary_ids, mask_id = W.load_tokenizer(args.base)
    model, identity, config = W.build_model(args.model_root,args.base,binary_ids,mask_id,0)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if payload['step'] != 500: raise ValueError('locked terminal500 checkpoint required')
    training = payload['provenance']; contract = training['contract']
    if training['mode'] != 'train' or training['seed'] not in (17,29,43): raise ValueError('wrong training run')
    if W.canonical_sha(contract) != training['contract_sha256']: raise ValueError('training contract corrupted')
    expected_sources = sorted(W.file_sha(p) for p in [Path(W.__file__),Path(tasks.__file__),*sorted((args.model_root/'longrwkv').glob('*.py'))])
    if contract['source_content_sha256'] != expected_sources: raise ValueError('training sources differ from evaluator sources')
    if contract['base_sha256'] != W.file_sha(args.base/'model.safetensors') or list(contract['binary_ids']) != list(binary_ids) or contract['mask_id'] != mask_id: raise ValueError('base/token contract differs')
    if contract['vocab_sha256'] != W.file_sha(tokenizer.vocab_path) or contract['tokenizer_source_sha256'] != W.file_sha(tokenizer.implementation_path): raise ValueError('tokenizer differs')
    model.load_state_dict(payload['model'],strict=True); del payload
    model=model.to('cuda').eval()
    # Qualify the locally copied attention forward against the installed native FLA class.
    # This checks implementation drift, not equivalence to the official BlinkDL kernel.
    import fla
    from fla.layers.rwkv7 import RWKV7Attention
    from longrwkv.fla_cond_attention import parity_with_native_mixer
    native_parity=[]
    for layer_index in (0,len(model.backbone.layers)-1):
        conditioned=model.backbone.layers[layer_index].attn
        native=copy.deepcopy(conditioned); native.__class__=RWKV7Attention
        torch.manual_seed(20260922+layer_index)
        hidden=torch.randn(1,128,config['hidden_size'],device='cuda',dtype=torch.bfloat16)
        value=torch.randn_like(hidden)
        cu=torch.tensor([0,128],dtype=torch.int32,device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            check=parity_with_native_mixer(native,conditioned,hidden,value,cu_seqlens=cu,tolerance=0.)
        native_parity.append(dict(layer=layer_index,**check));del native,hidden,value,cu
    predict=Predictor(model,tokenizer,binary_ids,mask_id,'cuda')
    args.out.mkdir(parents=True,exist_ok=True); output=args.out/f'rank{rank}.jsonl'
    if output.exists(): raise ValueError('evaluation output exists')
    start=time.perf_counter()
    with output.open('w') as stream:
        def emit(row): stream.write(json.dumps(row,allow_nan=False)+'\n'); stream.flush()
        emit(dict(kind='provenance',rank=rank,world=world,contract=CONTRACT,
                  checkpoint_sha256=W.file_sha(args.checkpoint),identity=identity,training_provenance=training,native_fla_mixer_parity=native_parity,
                  fla_version=getattr(fla,'__version__','unknown'),native_attention_source_sha256=W.file_sha(inspect.getfile(RWKV7Attention)),
                  sources={p.name:W.file_sha(p) for p in [Path(__file__),Path(tasks.__file__),Path(W.__file__)]}))
        for index in range(rank,2*CONTRACT['calibration_conditions_per_family'],world):
            example=tasks.make_example('test',CONTRACT['data_seed']+1,index)
            emit(dict(kind='calibration',index=index,family=example['family'],instance_id=example['instance_id'],rows=calibration(example,predict)))
        for index in range(rank,2*CONTRACT['conditions_per_family'],world):
            example=tasks.make_example('test',CONTRACT['data_seed'],index)
            emit(dict(kind='condition',index=index,example=example))
            for method in METHODS:
                samples=[]; calls=0; timer=time.perf_counter()
                for draw in range(CONTRACT['draws_per_condition']):
                    bits,nfe=sample(example,method,predict,draw); calls+=nfe; samples.append(bits)
                sampling_seconds=time.perf_counter()-timer
                counts=Counter(tuple(b) for b in samples); valid_counts={b:c for b,c in counts.items() if valid(example,b)}
                exact = exact_joint_kl(example,method,predict) if index < 2*CONTRACT['exact_joint_conditions_per_family'] and not method.startswith('bernoulli') else None
                emit(dict(kind='sampling',index=index,family=example['family'],method=method,
                          samples=samples,valid=sum(valid_counts.values()),draws=len(samples),
                          valid_support_coverage=len(valid_counts)/16,unique_outputs=len(counts),
                          max_output_fraction=max(counts.values())/len(samples),
                          collision_probability=sum(c*(c-1) for c in counts.values())/(len(samples)*(len(samples)-1)),
                          actual_sampling_nfe=calls,sampling_seconds=sampling_seconds,seconds_including_joint_audit=time.perf_counter()-timer,
                          exact_joint_kl_nats=exact))
            print(json.dumps(dict(rank=rank,condition=index,elapsed=time.perf_counter()-start)),flush=True)
        emit(dict(kind='complete',rank=rank,total_nfe=predict.calls,elapsed_seconds=time.perf_counter()-start))

if __name__=='__main__': main()
