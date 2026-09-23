"""Report paired development mechanisms and measured, provisional budget bands.

No p-values or test-set claims are made from the four-condition development panel.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(reports):
    models = {}
    for report in reports:
        if not report['execution_complete'] or not report['development_only'] or report.get('diagnostic_step200'):
            raise ValueError('requires complete development audits')
        kind = report['comparator']
        if kind in models:
            raise ValueError('ambiguous comparator checkpoints')
        rows = {}
        for row in report['conditions']:
            key = (row['family'], row['index'], row['requested_context_tokens'], row['position'])
            if key in rows:
                raise ValueError('duplicate condition')
            rows[key] = row
        models[kind] = (report, rows)
    if 'rwkv' not in models:
        raise ValueError('main RWKV audit required')
    main, rows = models['rwkv']
    comparisons = []
    for key, row in rows.items():
        metrics = {r['method']: r for r in row['exact']['results']}
        costs = {r['method']: statistics.mean(r['seconds']) for r in row['costs']}
        one = metrics['one']; info = metrics['information_set']; pair = metrics['pair_preserving_halves']
        result = dict(family=row['family'], index=row['index'], context_tokens=row['requested_context_tokens'],
                      position=row['position'], instance_id=row['exact']['instance_id'],
                      one_minus_information_set_kl=one['joint_kl_nats']-info['joint_kl_nats'],
                      pair_preserving_minus_information_set_kl=pair['joint_kl_nats']-info['joint_kl_nats'],
                      information_set_minus_one_valid_mass=info['exact_valid_mass']-one['exact_valid_mass'],
                      information_set_minus_pair_preserving_valid_mass=info['exact_valid_mass']-pair['exact_valid_mass'],
                      estimation_gap_info_minus_one=info['estimation_nats']-one['estimation_nats'],
                      dependence_reduction_one_minus_info=one['dependence_nats']-info['dependence_nats'],
                      product_kl_floor_certificate_margin=one['dependence_nats']-info['joint_kl_nats'],
                      information_set_seconds=costs['information_set'], one_call_seconds=costs['one'])
        for kind in ('attention', 'causal_rwkv'):
            comparator = models.get(kind)
            other = comparator[1].get(key) if comparator else None
            if other is None:
                result[kind] = dict(status='not_measured_for_this_condition')
                continue
            if other['exact']['instance_id'] != row['exact']['instance_id'] or other['exact']['support'] != row['exact']['support']:
                raise ValueError('unpaired comparison')
            om = {r['method']: r for r in other['exact']['results']}
            oc = {r['method']: statistics.mean(r['seconds']) for r in other['costs']}
            if kind == 'attention':
                lower = max(costs['information_set'], oc['one'])
                upper = min(v for k, v in oc.items() if k != 'one')
                result[kind] = dict(status='paired_development_measurement',
                    native_context_tokens=other['context_tokens'], records=other['records'],
                    recurrent_native_context_tokens=row['context_tokens'], recurrent_records=row['records'],
                    one_call_seconds=oc['one'], two_call_seconds=oc['information_set'],
                    mean_timing_budget_lower_inclusive_seconds=lower,
                    mean_timing_budget_upper_exclusive_seconds=upper,
                    nonempty_provisional_budget_band=lower<upper,
                    joint_kl_certificate_and_budget_band=lower<upper and result['product_kl_floor_certificate_margin']>0,
                    one_call_kl_minus_rwkv_two_call_kl=om['one']['joint_kl_nats']-info['joint_kl_nats'])
            else:
                result[kind] = dict(status='paired_development_measurement',
                    cached_request_seconds=oc['cached_causal'],
                    cached_kl_minus_rwkv_two_call_kl=om['cached_causal']['joint_kl_nats']-info['joint_kl_nats'],
                    rwkv_two_call_minus_cached_valid_mass=info['exact_valid_mass']-om['cached_causal']['exact_valid_mass'],
                    rwkv_two_call_to_cached_cost_ratio=costs['information_set']/oc['cached_causal'])
        comparisons.append(result)
    return dict(development_only=True, main_competence_gate_passed=main['competence_gate_passed'],
                checkpoints={k:v[0]['checkpoint_sha256'] for k,v in models.items()}, comparisons=comparisons,
                limitations=['Four development conditions per family, no final inferential claims.',
                             'Timing bands use three-repeat means and are provisional; final budgets must be fixed before test.',
                             'Pretraining and native tokenization differ for attention; comparisons are practical, not architecture-causal.',
                             'Absent comparator or failed-gate long conditions remain missing, never zero-filled.'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main', type=Path, required=True)
    ap.add_argument('--attention', type=Path)
    ap.add_argument('--causal', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    files = [p for p in (args.main, args.attention, args.causal) if p is not None]
    from .collect import collect
    audits = []
    for path in files:
        original = json.loads(path.read_text())
        recomputed = collect(json.loads(Path(original['evaluation_plan']).read_text()))
        if any(original[k] != recomputed[k] for k in ('checkpoint_sha256', 'contract_sha256', 'shard_sha256', 'conditions', 'competence_gate_passed', 'comparator')):
            raise ValueError('changed audit or raw evidence: ' + str(path))
        audits.append(original)
    result = summarize(audits)
    result['input_reports'] = [str(p.resolve()) for p in files]
    result['input_report_sha256'] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# Development mechanism audit', '',
             'This is a development screen, not the final paper result.', '',
             f'Main short-context competence gate passed: {result["main_competence_gate_passed"]}.', '',
             '| Family | Context | Position | Index | One − two KL | Pair-control − two KL | Product-floor margin | Two-call seconds |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in result['comparisons']:
        h = r['context_tokens'] if r['context_tokens'] is not None else 'evidence-only'
        lines.append(f'| {r["family"]} | {h} | {r["position"]} | {r["index"]} | '
                     f'{r["one_minus_information_set_kl"]:.6f} | {r["pair_preserving_minus_information_set_kl"]:.6f} | '
                     f'{r["product_kl_floor_certificate_margin"]:.6f} | {r["information_set_seconds"]:.6f} |')
    lines += ['', 'A positive product-floor margin certifies KL below every one-call product law for that condition; it is not a latency or generalization certificate.', '']
    lines += ['- '+s for s in result['limitations']]
    args.out.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(args.out), conditions=len(result['comparisons']), models=list(result['checkpoints']))))


if __name__ == '__main__':main()
