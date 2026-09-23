"""Audit all nine completed training jobs without changing the frozen experiment.

Run as python -m theory_mvp.train04confirm.training_audit after training finishes.
Elapsed GPU hours exclude initialization, queue time, and job teardown.
"""
import json
import math
from pathlib import Path
from lrwkv_evidence.train04confirm import contract as K

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (53, 71, 89)
PHASES = ('parent', 'independent', 'complementary')


def validate_records(seed, phase, digest, parent_sha, provenance, completion, rows):
    start, stop = (0, 1500) if phase == 'parent' else (1500, 2500)
    contract = provenance['contract']
    expected = dict(version=6, phase=phase, seed=seed, initial_seed=0,
                    manifest_sha256=digest, qualification=False, stop_step=stop,
                    parent_checkpoint_sha256=parent_sha, objective='rao_blackwell',
                    recipe=K.RECIPE)
    if any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError('training contract or parent lineage mismatch')
    if provenance['start_step'] != start or provenance['stop_step'] != stop:
        raise ValueError('unexpected training interval')
    if not completion['execution_complete'] or completion['qualification']:
        raise ValueError('incomplete or qualification run')
    if (completion['seed'], completion['phase'], completion['step']) != (seed, phase, stop):
        raise ValueError('completion identity mismatch')
    if [r['step'] for r in rows] != list(range(start + 1, stop + 1)):
        raise ValueError('missing, duplicate, or out-of-order updates')
    last_elapsed = -1.
    for row in rows:
        step = row['step']
        tokens = 1776 if step <= 300 else 2944 if step <= 700 else 4768
        label = ('ordinary_n2' if step <= 300 else 'label_mixture_n4' if step <= 700
                 else 'label_mixture_n8' if step <= 1500 else 'paired_n8')
        if row['seed'] != seed or row['phase'] != label or row['global_input_tokens'] != tokens:
            raise ValueError('training seed, phase, or token count mismatch')
        if not all(math.isfinite(row[k]) for k in ('mean_loss', 'grad_norm_rank0', 'lr', 'elapsed_seconds')):
            raise ValueError('nonfinite training record')
        if not math.isclose(row['lr'], 3e-5 * min(step / 50, 1.), rel_tol=1e-12):
            raise ValueError('learning rate mismatch')
        if row['elapsed_seconds'] < last_elapsed:
            raise ValueError('nonmonotonic training clock')
        last_elapsed = row['elapsed_seconds']
    elapsed = completion['elapsed_seconds']
    if not math.isfinite(elapsed) or elapsed < last_elapsed:
        raise ValueError('invalid completion elapsed time')
    return dict(seed=seed, phase=phase, updates=len(rows),
                input_tokens=sum(r['global_input_tokens'] for r in rows),
                checkpoint_sha256=completion['checkpoint_sha256'],
                parent_checkpoint_sha256=parent_sha,
                training_elapsed_seconds=elapsed, training_elapsed_gpu_hours=8 * elapsed / 3600)


def main():
    folder = ROOT / 'results/train04confirm'
    manifest_path = ROOT / 'theory_mvp/train04confirm/FROZEN_RECIPE.json'
    manifest = json.loads(manifest_path.read_text()); digest = K.sha(manifest_path)
    jobs = []
    for seed in SEEDS:
        parent_sha = None
        for phase in PHASES:
            plan_path = folder / f'plan_{digest[:16]}_confirm_{phase}_seed{seed}.json'
            plan = json.loads(plan_path.read_text()); out = Path(plan['out'])
            if (plan['seed'], plan['phase'], plan['manifest_sha256']) != (seed, phase, digest):
                raise ValueError('plan identity mismatch')
            provenance = json.loads((out / 'provenance.json').read_text())
            completion = json.loads((out / 'completion.json').read_text())
            rows = [json.loads(s) for s in (out / 'train.jsonl').read_text().splitlines()]
            if provenance['contract']['base_sha256'] != manifest['base_sha256']:
                raise ValueError('base checkpoint mismatch')
            if K.sha(out / 'resume.pt') != completion['checkpoint_sha256']:
                raise ValueError('checkpoint checksum mismatch')
            result = validate_records(seed, phase, digest, parent_sha, provenance, completion, rows)
            result['plan'] = str(plan_path.relative_to(ROOT)); result['job_id'] = plan['submission']['job_id']
            jobs.append(result)
            if phase == 'parent':
                parent_sha = completion['checkpoint_sha256']
    total = sum(j['input_tokens'] for j in jobs)
    if total != 45182400:
        raise ValueError('full campaign token budget mismatch')
    result = dict(manifest_sha256=digest, jobs=jobs, total_training_input_tokens=total,
                  training_elapsed_gpu_hours=sum(j['training_elapsed_gpu_hours'] for j in jobs),
                  scope='Completed training intervals and same-seed parent lineage; checkpoint hashes verified. GPU hours exclude initialization, queue time, teardown, qualification, development, and evaluation. Not a scheduler billing total or a model-competence test.')
    dest = folder / 'training_completion.json'
    dest.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(report=str(dest), jobs=len(jobs), input_tokens=total)))


if __name__ == '__main__':
    main()
