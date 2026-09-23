"""Validate all development shards before publishing any aggregate endpoints."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from .exact import METHODS, audit, logsumexp, partitions
from .numerical import validate as validate_numerical_screen
from lrwkv_evidence.long_context_mvp import tasks as T


def close(a, b):
    if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a, b, abs_tol=1e-7, rel_tol=1e-7):
        raise ValueError(f'inconsistent metric: {a} versus {b}')


def validate_condition(row, rank, comparator='rwkv'):
    if any(not math.isfinite(row[k]) or row[k] <= 0 for k in ('audit_seconds', 'condition_wall_seconds')) or row['audit_seconds'] > row['condition_wall_seconds']:
        raise ValueError('invalid audit runtime accounting')
    family = T.FAMILIES[rank % 2]
    index = rank // 2
    ex = T.problem('dev', 20270923, index, family)
    exact = row['exact']
    if (row['family'], row['index'], exact['family'], exact['instance_id'], exact['support']) != (
            family, index, family, ex['instance_id'], ex['support']):
        raise ValueError('unexpected example or support')
    length = row['requested_context_tokens']
    if length is not None and row['context_tokens'] != length:
        raise ValueError('incorrect context length')
    lo, hi = row['evidence_span']
    if not 0 <= lo < hi <= row['context_tokens'] or row['records'] < 1 or row['filler_tokens'] < 0:
        raise ValueError('invalid serialization metadata')
    methods = ['cached_causal'] if comparator == 'causal_rwkv' else list(METHODS)
    if [r['method'] for r in exact['results']] != methods:
        raise ValueError('missing or duplicate endpoint policy')
    if [r['method'] for r in row['costs']] != methods:
        raise ValueError('missing or duplicate timing policy')
    reference = {r['method']: r['dependence_nats'] for r in audit(
        ex, lambda visible: [[-math.log(2)] * 2 for _ in range(8)])['results']}
    reference['cached_causal'] = 0.
    for result, cost in zip(exact['results'], row['costs']):
        lp = result['support_log_probabilities']
        if len(lp) != len(ex['support']) or any(not math.isfinite(v) or v > 1e-8 for v in lp):
            raise ValueError('invalid support probabilities')
        kl = -math.log(len(lp)) - statistics.mean(lp)
        logvalid = logsumexp(lp)
        if logvalid > 1e-8:
            raise ValueError('valid mass exceeds one')
        close(result['joint_kl_nats'], kl)
        close(result['log_valid_mass'], logvalid)
        close(result['exact_valid_mass'], math.exp(logvalid))
        close(result['within_valid_kl_nats'], kl + logvalid)
        close(kl, result['dependence_nats'] + result['estimation_nats'])
        close(result['dependence_nats'], reference[result['method']])
        if min(kl, result['within_valid_kl_nats'], result['dependence_nats'], result['estimation_nats']) < -1e-7:
            raise ValueError('negative divergence')
        calls = 8 if cost['method'] == 'cached_causal' else len(partitions(ex)[cost['method']])
        if cost['calls'] != calls:
            raise ValueError('incorrect request call count')
        if len(cost['seconds']) != 3 or any(not math.isfinite(v) or v <= 0 for v in cost['seconds']):
            raise ValueError('invalid measured timing')
        if cost['peak_allocated_bytes'] <= 0 or len(cost['sample_outputs']) != 3 or any(type(v) is not int or not 0 <= v < 256 for v in cost['sample_outputs']):
            raise ValueError('invalid cost metadata')
    return ex


def collect(plan):
    diagnostic = plan.get('diagnostic_step200', False)
    comparator = plan.get('comparator', 'rwkv')
    if comparator not in ('rwkv', 'attention', 'causal_rwkv'):
        raise ValueError('unknown comparator')
    if diagnostic and comparator != 'rwkv':
        raise ValueError('milestone diagnostic requires main RWKV')
    out = Path(plan['out'])
    all_rows = []
    provenances = []
    gates = []
    hashes = {}
    contract = None
    for rank in range(8):
        path = out / f'rank{rank}.jsonl'
        raw = path.read_bytes()
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        rows = [json.loads(line) for line in raw.splitlines()]
        if not rows or rows[0]['kind'] != 'provenance' or rows[-1]['kind'] != 'complete':
            raise ValueError('incomplete shard')
        if any(r['kind'] not in ('provenance', 'condition', 'gate', 'numerical_qualification', 'complete') for r in rows):
            raise ValueError('unknown record')
        p = rows[0]
        if p.get('cost_only'):raise ValueError('cost-only study is not an endpoint audit')
        if p.get('diagnostic_step200', False) != diagnostic or p.get('checkpoint_step', 600) != (200 if diagnostic else 600):
            raise ValueError('wrong diagnostic or terminal milestone')
        runtime = p['runtime']
        if not runtime['gpu_name'] or runtime['gpu_total_memory_bytes'] <= 0 or not runtime['packages']['torch']:
            raise ValueError('missing runtime provenance')
        if p.get('comparator', 'rwkv') != comparator:
            raise ValueError('comparator provenance mismatch')
        if p['rank'] != rank or p['checkpoint_sha256'] != plan['checkpoint_sha256'] or p['split'] != 'dev' or p['data_seed'] != 20270923 or p['task_version'] != T.VERSION:
            raise ValueError('provenance mismatch')
        differences = [p['within_repeat_max_logprob_diff'], p['cross_rank_max_logprob_diff']]
        if any(not math.isfinite(v) or v < 0 for v in differences) or differences[0] > 1e-5 or differences[1] > 1e-4:
            raise ValueError('numerical qualification failure')
        if contract is None:
            contract = p['contract_sha256']
        if p['contract_sha256'] != contract:
            raise ValueError('mixed contracts')
        if [r['kind'] for r in rows].count('provenance') != 1 or [r['kind'] for r in rows].count('complete') != 1:
            raise ValueError('duplicate terminal/provenance record')
        gate_rows = [r for r in rows if r['kind'] == 'gate']
        if len(gate_rows) != 1:
            raise ValueError('missing or duplicate gate')
        gate = gate_rows[0]
        numerical = [r for r in rows if r['kind'] == 'numerical_qualification']
        if gate['all_eight_conditions_pass'] and not diagnostic:
            if len(numerical) != 1:
                raise ValueError('long-context numerical qualification required')
            validate_numerical_screen(numerical[0]['checks'], comparator)
        elif numerical:
            raise ValueError('unexpected long-context qualification')
        conditions = [r for r in rows if r['kind'] == 'condition']
        expected = [(None, 'middle')]
        if gate['all_eight_conditions_pass'] and not diagnostic:
            expected += [(length, pos) for length in (1024, 4096, 16384) for pos in ('far', 'middle', 'near')]
        if [(r['requested_context_tokens'], r['position']) for r in conditions] != expected:
            raise ValueError('missing, duplicated, or unexpected conditions')
        for row in conditions:
            validate_condition(row, rank, comparator)
        info = conditions[0]['exact']['results'][0 if comparator == 'causal_rwkv' else 1]
        local_pass = info['joint_kl_nats'] < .1 and info['exact_valid_mass'] > .9
        if gate['local_pass'] != local_pass or rows[-1]['long_context_gate_passed'] != gate['all_eight_conditions_pass']:
            raise ValueError('incorrect competence gate')
        all_rows.extend(conditions)
        provenances.append(p)
        gates.append(gate)
    passed = all(g['local_pass'] for g in gates)
    if any(g['all_eight_conditions_pass'] != passed for g in gates):
        raise ValueError('inconsistent global competence gate')
    groups = {}
    for row in all_rows:
        for result, cost in zip(row['exact']['results'], row['costs']):
            key = (row['family'], row['requested_context_tokens'], row['position'], result['method'])
            groups.setdefault(key, []).append((result, cost))
    aggregates = []
    for (family, length, position, method), values in groups.items():
        if len(values) != 4:
            raise ValueError('incomplete paired development panel')
        entry = dict(family=family, context_tokens=length, position=position, method=method, conditions=4)
        for metric in ('joint_kl_nats', 'dependence_nats', 'estimation_nats', 'exact_valid_mass', 'within_valid_kl_nats'):
            entry[metric] = statistics.mean(v[0][metric] for v in values)
        entry['mean_request_seconds'] = statistics.mean(statistics.mean(v[1]['seconds']) for v in values)
        entry['max_peak_allocated_bytes'] = max(v[1]['peak_allocated_bytes'] for v in values)
        aggregates.append(entry)
    return dict(execution_complete=True, development_only=True, comparator=comparator, checkpoint_sha256=plan['checkpoint_sha256'],
                diagnostic_step200=diagnostic,checkpoint_step=200 if diagnostic else 600,
                contract_sha256=contract, shard_sha256=hashes, competence_gate_passed=passed,
                summed_condition_gpu_seconds=sum(r['condition_wall_seconds'] for r in all_rows),
                condition_count=len(all_rows), aggregates=aggregates, provenances=provenances, conditions=all_rows,
                limitation='Four conditions per family; development screen only. No final uncertainty, architecture comparison, or long-context gain follows from passing the gate.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    from lrwkv_evidence.train04.worker import file_sha
    for relative, expected in plan['sources'].items():
        if file_sha(Path(plan['stage']) / relative) != expected:
            raise ValueError('changed evaluation source: ' + relative)
    training = json.loads(Path(plan['training_plan']).read_text())
    if plan.get('diagnostic_step200'):
        checkpoint = Path(plan['checkpoint'])
        import torch
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        if payload['step'] != 200 or payload['contract']['recipe']['terminal_step'] != 600:
            raise ValueError('wrong diagnostic checkpoint')
        completion = dict(contract_sha256=payload['contract_sha256'])
        del payload
    else:
        completion = json.loads((Path(training['out']) / 'completion.json').read_text())
        if completion['step'] != 600 or completion['checkpoint_sha256'] != plan['checkpoint_sha256']:
            raise ValueError('wrong training completion')
        checkpoint = Path(training['out']) / 'resume.pt'
    if file_sha(checkpoint) != plan['checkpoint_sha256']:
        raise ValueError('checkpoint content changed')
    report = collect(plan)
    report['collector_source_sha256'] = file_sha(Path(__file__))
    report['evaluation_plan'] = str(args.plan.resolve())
    if report['contract_sha256'] != completion['contract_sha256']:
        raise ValueError('evaluation/training contract mismatch')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.out)
    print(json.dumps({k: v for k, v in report.items() if k not in ('conditions', 'provenances', 'aggregates', 'shard_sha256')}, indent=2))


if __name__ == '__main__':
    main()
