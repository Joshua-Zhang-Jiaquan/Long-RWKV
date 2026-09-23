"""Development-only competence controls using qualified, unchanged decode kernels.

Run one model per torchrun job; all evidence-only items precede no-evidence items.
CPU collection imports no torch. These post-dev diagnostics do not select decoders.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

from . import controls as C, gpu_runner as U, causal_runner as A
from . import dev_grid as D, grid324 as G, lane as L, runner as R

KINDS = ('evidence_only', 'no_evidence', 'paired_anchor4096', 'paired_anchor8192')
CONFIG = dict(sampler='matched_bernoulli', steps=8, tau=.7, commit_order='confidence')


def sources(scale):
    hop, tp, registry = G._import_generator()
    return [Path(m.__file__) for m in (C, U, A, D, G, L, R, hop, tp, registry)] + [
        Path(__file__), scale / 'eval/capability/birwkv_diffusion_model.py',
        scale / 'models/birwkv7_diffusion.py', scale / 'eval/capability/hf_causal_model.py']


def verify_source_subset(qualified, current):
    """Compare qualified dependencies by basename across immutable staging roots."""
    for old, expected in qualified.items():
        matches = [value for path, value in current.items() if Path(path).name == Path(old).name]
        if not matches or any(value != expected for value in matches):
            raise ValueError('qualified dependency changed or missing: ' + old)


def control_items(panel, rank=0, world=1):
    if world not in (1, 8) or not 0 <= rank < world:
        raise ValueError('one or eight valid ranks required')
    manifest = json.loads((panel / 'dev_manifest.json').read_text())
    if manifest.get('split') != 'dev' or manifest.get('instances') != 72 or manifest.get('partial_run', False):
        raise ValueError('controls require the complete 72-item dev panel')
    hop, tp, registry = G._import_generator()
    encoder = registry.TrieEncoder.from_vocab(str(G.VOCAB))
    pools = G._PermutingPoolFactory(hop._Pool)
    specs = []
    for cell in D.panel_cells():
        path = panel / f'{cell.cell_id}.json'
        rec = L.load_cell_record(path)
        for inst in L.instances_of(rec):
            if inst.data_seed not in D.dev_seeds():
                raise ValueError('non-dev instance')
            specs.append((cell, inst, path, U.file_sha(path)))
    keys = [(c.cell_id, i.data_seed, i.instance_index) for c, i, _, _ in specs]
    if len(keys) != 72 or len(set(keys)) != 72:
        raise ValueError('missing or duplicate dev items')
    for kind in KINDS:
        for ordinal, (cell, inst, path, source_hash) in enumerate(specs):
            if ordinal % world != rank:
                continue
            task, canvas = U.verified_instance(hop, tp, encoder, pools, cell, inst, 'dev')
            provenance = dict(source_path=str(path), source_sha256=source_hash,
                              source_input_ids_sha256=inst.input_ids_sha256,
                              cell_id=cell.cell_id, data_seed=inst.data_seed,
                              instance_index=inst.instance_index, split='dev')
            if kind.startswith('paired_anchor'):
                length = int(kind.removeprefix('paired_anchor'))
                control = C.transform_task(task, canvas, tp, encoder, position=cell.position,
                                           provenance=provenance, anchors=(length,))[-1]
                control['kind'] = kind
                control.pop('transformation_sha256', None)
                control['transformation_sha256'] = U.digest(control)
            else:
                transformed = C.evidence_only(canvas) if kind == 'evidence_only' else C.no_evidence(canvas)
                control = C._record(kind, transformed, task, provenance,
                                    operation='delete filler_mask' if kind == 'evidence_only' else 'replace evidence_span with filler',
                                    filler_id=canvas.pad_id)
            item = f'{cell.cell_id}__{inst.data_seed}__{inst.instance_index}'
            yield item, cell, inst, control, encoder


def dispatch(model_kind, model, ids, span, *, generator=None, legal_ids=None):
    """Use exactly the qualified kernels; target spans are forwarded unchanged."""
    if model_kind == 'f2':
        return U.sample_target(model, ids, span, CONFIG, mask_id=model.mask_token_id,
                               pad_id=model.pad_token_id, generator=generator,
                               reuse_unchanged_logits=True)
    if model_kind == 'r0':
        tokens, calls = A.greedy_suffix(model, ids, span, legal_ids)
        return tokens, calls, None
    raise ValueError('unknown model')


def validate_receipt(row, control, *, item, provenance_hash, model_kind, encoder=None):
    if control['status'] == 'unsupported':
        if row != dict(control, item_id=item, model=model_kind, provenance_sha256=provenance_hash):
            raise ValueError('stale unsupported-control receipt')
        return
    gold = control['gold_answers']
    lo, hi = control['decoder_input']['target_span']
    prediction = R.extract_answer(row.get('text', ''), G.FAMILY_MAP[control['provenance']['cell_id'].split('_L')[0]])
    expected = len(gold) == 1 and prediction == gold[0]
    if (row.get('status') != 'ok' or row.get('split') != 'dev'
        or row.get('item_id') != item or row.get('kind') != control['kind']
        or row.get('model') != model_kind or row.get('provenance_sha256') != provenance_hash
        or row.get('transformation_sha256') != control['transformation_sha256']
        or row.get('source_input_ids_sha256') != control['provenance']['source_input_ids_sha256']
        or row.get('input_ids_sha256') != control['input_ids_sha256']
        or row.get('target_span') != [lo, hi] or row.get('gold') != gold
        or row.get('prompt_hash_verified') is not True
        or row.get('policy') != (CONFIG if model_kind == 'f2' else A.POLICY)
        or type(row.get('correct')) is not bool or row['correct'] != expected
        or row.get('prediction') != prediction or row.get('parse_ok') != (prediction is not None)
        or not isinstance(row.get('output_ids'), list) or len(row['output_ids']) != hi-lo
        or any(type(token) is not int or not 0 <= token < 65536 for token in row['output_ids'])
        or type(row.get('actual_nfe')) is not int
        or not (1 <= row['actual_nfe'] <= 8 if model_kind == 'f2' else row['actual_nfe'] == hi-lo)
        or not math.isfinite(row.get('wall_seconds', float('nan'))) or row['wall_seconds'] < 0):
        raise ValueError('stale or invalid control receipt')
    if encoder is not None and encoder.decode_sample_ids(row['output_ids']) != row['text']:
        raise ValueError('receipt text does not match output IDs')
    if model_kind == 'f2' and not row.get('logit_reuse_parity_sha256'):
        raise ValueError('missing cache parity binding')
    if model_kind == 'r0' and row.get('tokenizer_check', {}).get('native_decode_equal') is not True:
        raise ValueError('missing tokenizer qualification')


def validate_parity(parity, *, provenance_hash, rank, kind, receipt_hash=None):
    cached = parity.get('runs', {}).get('cached', {})
    uncached = parity.get('runs', {}).get('uncached', {})
    if (parity.get('qualified') is not True or parity.get('provenance_sha256') != provenance_hash
        or parity.get('rank') != rank or parity.get('kind') != kind or parity.get('config') != CONFIG
        or not cached.get('output_ids') or cached.get('output_ids') != uncached.get('output_ids')
        or not cached.get('commits_per_step') or cached.get('commits_per_step') != uncached.get('commits_per_step')
        or not 0 < cached.get('actual_nfe', 0) <= uncached.get('actual_nfe', 0) <= CONFIG['steps']
        or (receipt_hash is not None and U.digest(parity) != receipt_hash)):
        raise ValueError('stale or invalid cache parity receipt')


def collect(out, panel, model_kind):
    manifests = [json.loads(p.read_text()) for p in sorted(out.glob('provenance_rank*.json'))]
    if not manifests or any(p != manifests[0] for p in manifests):
        raise ValueError('missing or mixed provenance')
    provenance = manifests[0]
    if provenance['model'] != model_kind or provenance['panel_sha256'] != U.digest({p.name: U.file_sha(p) for p in sorted(panel.glob('*.json'))}):
        raise ValueError('model or panel provenance mismatch')
    phash = U.digest(provenance)
    rows = []
    for item, cell, inst, control, encoder in control_items(panel):
        path = out / control['kind'] / f'{item}.json'
        row = json.loads(path.read_text())
        validate_receipt(row, control, item=item, provenance_hash=phash, model_kind=model_kind, encoder=encoder)
        if model_kind == 'f2' and row['status'] == 'ok':
            parity = json.loads((out / 'cache_parity' / f"rank{row['rank']}_{control['kind']}.json").read_text())
            validate_parity(parity,provenance_hash=phash,rank=row['rank'],kind=control['kind'],
                            receipt_hash=row['logit_reuse_parity_sha256'])
        rows.append(row)
    panels = {}
    for kind in KINDS:
        selected = [r for r in rows if r['kind'] == kind]
        scored = [r for r in selected if r['status'] == 'ok']
        panels[kind] = dict(expected_n=72, n=len(scored), unsupported=len(selected)-len(scored),
                            correct=sum(r['correct'] for r in scored),
                            accuracy=sum(r['correct'] for r in scored)/len(scored) if scored else None)
    summary = dict(schema='lrwkv_e3_controls_summary_v1', model=model_kind, provenance_sha256=phash,
                   interpretation='Post-development competence diagnostics on reused dev tasks; no independent confirmation or decoder selection.',
                   panels=panels)
    U.atomic_json(out / 'control_summary.json', summary)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=['f2', 'r0'], required=True)
    ap.add_argument('--panel', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model-dir', type=Path)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--protocol', type=Path)
    ap.add_argument('--qualification', type=Path)
    ap.add_argument('--scale', type=Path, default=U.DEFAULT_SCALE)
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--cpu-threads', type=int, default=4)
    ap.add_argument('--collect', action='store_true')
    args = ap.parse_args(argv)
    if args.collect:
        print(json.dumps(collect(args.out, args.panel, args.model), indent=2)); return
    if not args.model_dir:
        ap.error('--model-dir required')
    if args.model == 'f2' and (not args.protocol or not args.checkpoint):
        ap.error('F2 requires --protocol and --checkpoint')
    if args.model == 'r0' and not args.qualification:
        ap.error('R0 requires --qualification')
    source_hashes = {str(p): U.file_sha(p) for p in sources(args.scale)}
    policy = CONFIG if args.model == 'f2' else A.POLICY
    provenance = dict(schema=1, model=args.model, split='dev', purpose='competence_controls',
        policy=policy, kinds=list(KINDS), seed=args.seed if args.model == 'f2' else 0,
        panel_sha256=U.digest({p.name: U.file_sha(p) for p in sorted(args.panel.glob('*.json'))}),
        sources=source_hashes, vocab_sha256=U.file_sha(G.VOCAB), cpu_threads=args.cpu_threads,
        execution_env={k: os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS',
            'LOOP_REPS_OVERRIDE','BIRWKV_META_CONSTRUCT','BIRWKV_DIRECT_LOAD','BIRWKV_META_LOAD')})
    if args.model == 'f2':
        decoder = U.verify_confirmation_freeze(json.loads(args.protocol.read_text()))
        if any(decoder[k] != CONFIG[k] for k in CONFIG):
            raise ValueError('unexpected frozen F2 decoder')
        old = json.loads((Path(decoder['dev_measurements_path']).parent / 'provenance_rank0.json').read_text())
        if U.digest(old) != decoder['dev_provenance_sha256']:
            raise ValueError('calibration provenance mismatch')
        verify_source_subset(old['sources'], source_hashes)
        identity = dict(checkpoint_sha256=U.file_sha(args.checkpoint/'model.pt'),
            model_config_sha256=U.file_sha(args.model_dir/'config.json'),
            tokenizer_files={p.name: U.file_sha(p) for p in sorted(args.model_dir.iterdir()) if p.is_file()
                and (p.name.startswith('tokeniz') or p.name.startswith('rwkv_vocab') or p.name=='special_tokens_map.json')})
        if identity['checkpoint_sha256'] != decoder['checkpoint_sha256'] or any(identity[k] != old[k] for k in identity):
            raise ValueError('F2 model/tokenizer identity differs from qualification')
        if provenance['vocab_sha256'] != old['vocab_sha256'] or provenance['panel_sha256'] != old['panel_sha256']:
            raise ValueError('F2 panel/vocabulary changed since dev')
        provenance.update(identity=identity, qualification_sha256=U.file_sha(args.protocol))
    else:
        qualification = json.loads(args.qualification.read_text())
        if not qualification.get('qualified') or qualification.get('n') != 72 or qualification.get('policy') != A.POLICY:
            raise ValueError('R0 lacks qualified greedy policy')
        if qualification['runner_sha256'] != U.file_sha(Path(A.__file__)):
            raise ValueError('qualified R0 runner changed')
        verify_source_subset(qualification['dependencies'], source_hashes)
        files = sorted(set(args.model_dir.glob('*.safetensors')) | set(args.model_dir.glob('*.json')) | set(args.model_dir.glob('*.py')))
        identity = dict(model_dir=str(args.model_dir.resolve()), files={p.name:U.file_sha(p) for p in files})
        if any(identity[k] != qualification['model_identity'][k] for k in identity):
            raise ValueError('R0 model identity changed')
        provenance.update(identity=identity, qualification_sha256=U.file_sha(args.qualification))
    import torch
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(0)
    rank, world, local = (int(os.environ.get(k,d)) for k,d in [('RANK','0'),('WORLD_SIZE','1'),('LOCAL_RANK','0')])
    if world not in (1,8) or not 0 <= rank < world:
        raise ValueError('one or eight workers required')
    torch.cuda.set_device(local)
    device=f'cuda:{local}'
    sys.path.insert(0,str(args.scale))
    if args.model=='f2':
        from eval.capability.birwkv_diffusion_model import load_birwkv_diffusion
        loaded=load_birwkv_diffusion(str(args.checkpoint),str(args.model_dir),device=device)
        legal=native=None
    else:
        from eval.capability.hf_causal_model import load_hf_causal
        loaded=load_hf_causal(str(args.model_dir),device=device)
        _,_,registry=G._import_generator()
        encoder=registry.TrieEncoder.from_vocab(str(G.VOCAB))
        native,legal,token_manifest=A.qualify_tokenizer(encoder,loaded.tokenizer)
        if token_manifest != qualification['model_identity']['tokenizer']:
            raise ValueError('R0 tokenizer qualification changed')
        provenance['identity']['tokenizer']=token_manifest
    provenance.update(torch=torch.__version__,cuda=torch.version.cuda,device_name=torch.cuda.get_device_name())
    phash=U.digest(provenance)
    U.pin_provenance(args.out/f'provenance_rank{rank}.json',provenance)
    parities={}
    for item,cell,inst,control,encoder in control_items(args.panel,rank,world):
        kind=control['kind']; dest=args.out/kind/f'{item}.json'
        if dest.exists():
            previous=json.loads(dest.read_text())
            validate_receipt(previous,control,item=item,provenance_hash=phash,model_kind=args.model,encoder=encoder)
            if args.model=='f2' and previous['status']=='ok':
                parity=json.loads((args.out/'cache_parity'/f"rank{previous['rank']}_{kind}.json").read_text())
                validate_parity(parity,provenance_hash=phash,rank=previous['rank'],kind=kind,
                                receipt_hash=previous['logit_reuse_parity_sha256'])
            continue
        if control['status'] == 'unsupported':
            U.atomic_json(dest,dict(control,item_id=item,model=args.model,provenance_sha256=phash))
            continue
        data=control['decoder_input']; span=data['target_span']
        ids=torch.tensor([data['input_ids']],dtype=torch.long,device=device)
        seed=U.item_seed('dev_controls',cell.cell_id,inst.data_seed,inst.instance_index,args.seed)
        if args.model=='f2' and kind not in parities:
            parity_path=args.out/'cache_parity'/f'rank{rank}_{kind}.json'
            if parity_path.exists():
                parity=json.loads(parity_path.read_text())
                validate_parity(parity,provenance_hash=phash,rank=rank,kind=kind)
            else:
                parity=U.qualify_logit_reuse(loaded.model,ids,span,CONFIG,mask_id=loaded.model.mask_token_id,pad_id=loaded.model.pad_token_id,seed=seed)
                parity.update(provenance_sha256=phash,rank=rank,kind=kind,transformation_sha256=control['transformation_sha256'])
                U.atomic_json(parity_path,parity)
            parities[kind]=parity
        check=A.verify_prefix(data['input_ids'][:span[0]],encoder,native,loaded.tokenizer) if args.model=='r0' else None
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
        tokens,calls,trace=dispatch(args.model,loaded.model,ids,span,generator=torch.Generator(device=device).manual_seed(seed),legal_ids=legal)
        torch.cuda.synchronize(); elapsed=time.perf_counter()-start
        text=encoder.decode_sample_ids(tokens); prediction=R.extract_answer(text,cell.generator_family)
        row=dict(status='ok',split='dev',model=args.model,kind=kind,item_id=item,policy=policy,
            provenance_sha256=phash,transformation_sha256=control['transformation_sha256'],
            source_input_ids_sha256=inst.input_ids_sha256,input_ids_sha256=control['input_ids_sha256'],
            target_span=span,prompt_hash_verified=True,gold=control['gold_answers'],
            prediction=prediction,parse_ok=prediction is not None,correct=prediction==control['gold_answers'][0],
            text=text,output_ids=tokens,actual_nfe=calls,commits_per_step=trace,wall_seconds=elapsed,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            rank=rank,rng_seed=seed,tokenizer_check=check,
            logit_reuse_parity_sha256=U.digest(parities[kind]) if args.model=='f2' else None)
        validate_receipt(row,control,item=item,provenance_hash=phash,model_kind=args.model,encoder=encoder)
        U.atomic_json(dest,row)
        print(f'{kind} {item} correct={row["correct"]} seconds={elapsed:.3f}',flush=True)


if __name__=='__main__':
    main()
