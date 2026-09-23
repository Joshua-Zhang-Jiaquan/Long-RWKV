"""Validate raw calls, reconcile the prior study, and report paired problem effects."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lrwkv_evidence.history_response import design as H
from lrwkv_evidence.distance_intervention import tasks as T
from lrwkv_evidence.long_context_eval.numerical import validate
from lrwkv_evidence.train04.worker import load_tokenizer

FOLDER = ROOT/'results/history_response'
MANIFEST = ROOT/'theory_mvp/history_response/FROZEN.json'
POSITIONS = ('far', 'middle', 'near')
SEEDS = (20271011, 20271012, 20271013)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def collect(role):
    manifest = json.loads(MANIFEST.read_text())
    plan = json.loads((FOLDER/f'plan_{role}.json').read_text())
    if plan['manifest_sha256'] != sha(MANIFEST):
        raise ValueError('manifest changed')
    for rel, digest in plan['sources'].items():
        if sha(Path(plan['stage'])/rel) != digest:
            raise ValueError('staged source changed: '+rel)
    if sha(plan['checkpoint']) != manifest['checkpoints'][role]['sha256']:
        raise ValueError('checkpoint changed')
    prior_path = ROOT/manifest['prior_outputs'][role]['path']
    if sha(prior_path) != manifest['prior_outputs'][role]['sha256']:
        raise ValueError('prior report changed')
    prior = {(r['index'], r['serial']['position']): r for r in json.loads(prior_path.read_text())['conditions'] if r['primary']}
    tok, _, _ = load_tokenizer(Path('/inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B'))
    output = []; hashes = {}; maximum_discrepancy = 0.; provenances = []
    for rank in range(8):
        path = Path(plan['out'])/f'rank{rank}.jsonl'
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        hashes[str(path)] = sha(path)
        if rows[-1] != dict(kind='complete', conditions=12, all_declared_cells_retained=True):
            raise ValueError('incomplete rank')
        provenance = [r for r in rows if r['kind'] == 'provenance']
        if len(provenance) != 1:
            raise ValueError('bad provenance count')
        pr = provenance[0]
        for key, value in dict(rank=rank, world=8, role=role, seed=T.SEED,
                               checkpoint_sha256=plan['checkpoint_sha256'],
                               manifest_sha256=sha(MANIFEST), followup='history_response_v1').items():
            if pr[key] != value:
                raise ValueError('bad provenance '+key)
        provenances.append(pr)
        screens = [r for r in rows if r['kind'] == 'numerical_screen']
        if len(screens) != 1:
            raise ValueError('missing numerical screen')
        validate(screens[0]['rows'])
        conditions = [r for r in rows if r['kind'] == 'condition']
        keys = [(r['index'], r['position']) for r in conditions]
        if len(keys) != 12 or set(keys) != {(i,p) for i in range(rank,32,8) for p in POSITIONS}:
            raise ValueError('missing or duplicated cells')
        for r in conditions:
            ex = T.example(r['index']); encoded = T.serialize(ex,tok,16384,r['position'])
            if r['serial'] != {k:v for k,v in encoded.items() if k != 'ids'}:
                raise ValueError('original layout mismatch')
            for j in range(4):
                encoded_cf = T.serialize(H.flip(ex,j),tok,16384,r['position'])
                if r['changed_native_token_indices'][j] != T.validate_layout(encoded,encoded_cf):
                    raise ValueError('intervention changed layout')
                if r['flipped_token_sha256'][j] != encoded_cf['token_sha256']:
                    raise ValueError('flipped input mismatch')
            result = H.summarize(ex,r['initial'],r['original'],r['flipped'])
            old = prior[r['index'],r['position']]
            differences = [abs(result['first_error']-old['stages']['first_round_estimation_nats']),
                           abs(result['second_error']-old['stages']['second_round_estimation_nats'])]
            fixed = next(v for v in result['pairs'] if v['history'] == 0 and v['constraint'] == r['index']%4)
            old_fixed = old['counterfactual']['response']
            differences += [abs(fixed[k]-old_fixed[k]) for k in result['response']]
            maximum_discrepancy = max(maximum_discrepancy, *differences)
            if max(differences) > 1e-4:
                raise ValueError('archived predictions not reproduced')
            output.append(dict(index=r['index'],position=r['position'],**result))
    return dict(role=role, validated=True, manifest_sha256=sha(MANIFEST),
                checkpoint_sha256=plan['checkpoint_sha256'], conditions=output,
                raw_file_sha256=hashes, provenance=provenances,
                prior_max_discrepancy=maximum_discrepancy)


def estimate(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (32,) or not np.isfinite(values).all():
        raise ValueError('32 finite problem means required')
    indices = np.random.default_rng(20260923).integers(0,32,size=(10000,32))
    ci = np.quantile(values[indices].mean(axis=1),[.025,.975])
    return dict(mean=float(values.mean()), ci95=ci.tolist(), problems=32)


def metric(data, position, key):
    rows = {r['index']:r for r in data['conditions'] if r['position'] == position}
    if set(rows) != set(range(32)):
        raise ValueError('missing paired problems')
    return np.array([rows[i]['response'][key] if key in rows[i]['response'] else rows[i][key] for i in range(32)])


def report():
    manifest = json.loads(MANIFEST.read_text())
    data = {role:json.loads((FOLDER/f'exact_{role}.json').read_text()) for role in manifest['roles']}
    if not all(r['validated'] and r['manifest_sha256'] == sha(MANIFEST) for r in data.values()):
        raise ValueError('unvalidated results')
    keys = ('mean_correct_conditional_ce','response_ce_lower_bound','centering_penalty',
            'signed_logit_contrast','both_variants_correct','unaffected_mean_abs_probability_change',
            'information_set_error','second_error','worst_history_second_error','worst_pair_ce','max_unaffected_change')
    checkpoints = {role:{p:{k:estimate(metric(d,p,k)) for k in keys} for p in POSITIONS} for role,d in data.items()}
    effects = {}; pooled = {}
    for seed in SEEDS:
        n,b = data[f'near_seed{seed}'],data[f'balanced_seed{seed}']
        values = {}
        for p in POSITIONS:
            for k in ('mean_correct_conditional_ce','response_ce_lower_bound','centering_penalty','second_error'):
                values[p+'_'+k+'_reduction'] = metric(n,p,k)-metric(b,p,k)
        values['far_minus_near_ce_reduction'] = values['far_mean_correct_conditional_ce_reduction']-values['near_mean_correct_conditional_ce_reduction']
        effects[str(seed)] = {k:estimate(v) for k,v in values.items()}
        for k,v in values.items():
            pooled.setdefault(k,[]).append(v)
    pooled = {k:estimate(np.mean(v,axis=0)) for k,v in pooled.items()}
    primary = pooled['far_mean_correct_conditional_ce_reduction']
    complete = dict(manifest_sha256=sha(MANIFEST),conditions=672,paired_interventions=43008,
                    checkpoints=checkpoints,per_seed=effects,seed_averaged=pooled,
                    primary_positive_ci=primary['ci95'][0]>0,
                    same_direction_all_seeds=all(v['far_mean_correct_conditional_ce_reduction']['mean']>0 for v in effects.values()),
                    prior_max_discrepancy=max(d['prior_max_discrepancy'] for d in data.values()),
                    inference_scope='Post-study exact-history diagnostic on reused problems, conditional on observed adaptation seeds and selected starting lineage; not independent replication.')
    (FOLDER/'report.json').write_text(json.dumps(complete,indent=2)+'\n')
    lines=['# Exhaustive evidence-response results','',complete['inference_scope'],'',
           '| Checkpoint | Far paired CE | Floor | Centering penalty | Both correct | Mean unaffected change |',
           '|---|---:|---:|---:|---:|---:|']
    for role, pp in checkpoints.items():
        p=pp['far']; numbers=[p[k]['mean'] for k in ('mean_correct_conditional_ce','response_ce_lower_bound','centering_penalty','both_variants_correct','unaffected_mean_abs_probability_change')]
        lines.append('| '+role+' | '+' | '.join(f'{x:.7g}' for x in numbers)+' |')
    lines += ['', f"Far near-minus-balanced paired CE: {primary['mean']:.6g} nats, conditional 95% CI {primary['ci95']}.",
              f"All three seed directions positive: {complete['same_direction_all_seeds']}.",
              f"Archived metric maximum discrepancy: {complete['prior_max_discrepancy']:.6g}.",
              '', 'All positions, individual seed intervals and worst-case summaries are in report.json; all 43,008 individual interventions are in exact_*.json. Symmetrized paired CE is not the original endpoint E.']
    (FOLDER/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in complete.items() if k not in ('checkpoints','per_seed','seed_averaged')},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--role');args=parser.parse_args()
    if args.role:
        result=collect(args.role)
        (FOLDER/f'exact_{args.role}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(role=args.role,validated=True,conditions=len(result['conditions']))))
    else:
        report()
