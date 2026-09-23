"""Read-only source audit and local paper tables for complete competence controls.

The original collector writes a summary. We run it on a temporary directory of
read-only source links, so its validation is reused without writing GPU outputs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile

from . import control_runner as C, grid324 as G, gpu_runner as U

MODELS = ('f2', 'r0')
FAMILIES = G.PAPER_FAMILIES
KIND_NAMES = dict(evidence_only='Evidence only', no_evidence='No evidence (16K)',
                  paired_anchor4096='Paired 4K', paired_anchor8192='Paired 8K')
FAMILY_NAMES = dict(associative_recall='Recall', overwrite_delayed_query='Overwrite', code_dataflow='Dataflow')


def family_of(row):
    family = row['item_id'].split('_L')[0]
    if family not in FAMILIES:
        raise ValueError('unknown control family')
    return family


def aggregate(rows_by_model):
    """Require identical 72 tasks in every model/control; unsupported cannot be zero."""
    if set(rows_by_model) != set(MODELS):
        raise ValueError('both model panels required')
    reference = None
    table = []
    macros = []
    for model in MODELS:
        rows = rows_by_model[model]
        if len(rows) != 72 * len(C.KINDS):
            raise ValueError('partial control panel: require 4 x 72 receipts per model')
        pairs = [(r['kind'], r['item_id']) for r in rows]
        if len(set(pairs)) != len(pairs) or any(r['kind'] not in C.KINDS for r in rows):
            raise ValueError('duplicate or unknown control records')
        for kind in C.KINDS:
            selected = [r for r in rows if r['kind'] == kind]
            keys = {r['item_id'] for r in selected}
            if len(keys) != 72:
                raise ValueError('partial control kind')
            if reference is None:
                reference = keys
            if keys != reference:
                raise ValueError('control panels do not pair the same 72 tasks')
            if any(r['status'] != 'ok' for r in selected):
                raise ValueError('unsupported control: paper table requires all 72 scored tasks')
            if any(type(r.get('correct')) is not bool or type(r.get('parse_ok')) is not bool for r in selected):
                raise ValueError('invalid Boolean outcome')
            family_rows = []
            for family in FAMILIES:
                group = [r for r in selected if family_of(r) == family]
                if len(group) != 24:
                    raise ValueError('partial or imbalanced family: require 24 tasks')
                row = dict(model=model, kind=kind, family=family, n=24,
                           correct=sum(r['correct'] for r in group), parsed=sum(r['parse_ok'] for r in group))
                row.update(accuracy=row['correct']/24, parse_rate=row['parsed']/24)
                table.append(row); family_rows.append(row)
            macros.append(dict(model=model, kind=kind, n=72,
                               correct=sum(r['correct'] for r in selected),
                               parsed=sum(r['parse_ok'] for r in selected),
                               family_macro_accuracy=sum(r['accuracy'] for r in family_rows)/3,
                               family_macro_parse_rate=sum(r['parse_rate'] for r in family_rows)/3))
    return dict(schema='lrwkv_e3_controls_paper_v1', rows=table, macros=macros,
                interpretation='Post-development diagnostics reusing the same 72 dev tasks (24 per family), not independent confirmation. Paired 4K/8K anchors rerender the same logical task; these do not estimate full-grid degradation Gamma. Supplied gold target length and frozen qualified decoders are retained.')


def coverage(plan):
    return dict(schema='lrwkv_e3_controls_coverage_v1', status_only=True,
        models={job['model']: {kind: dict(expected=72,
                receipts=len(list((Path(job['out'])/kind).glob('*.json')))) for kind in C.KINDS}
                for job in plan['jobs']})


def validated_rows(out, panel, model):
    """Reuse complete collector, isolating its sole output write in /tmp."""
    with tempfile.TemporaryDirectory(prefix='e3-controls-report-') as temporary:
        mirror = Path(temporary)
        for kind in (*C.KINDS, 'cache_parity'):
            source = out/kind
            if source.is_dir():
                (mirror/kind).symlink_to(source.resolve(), target_is_directory=True)
        for source in out.glob('provenance_rank*.json'):
            (mirror/source.name).symlink_to(source.resolve())
        summary = C.collect(mirror, panel, model)
    source_summary = out/'control_summary.json'
    if source_summary.exists() and json.loads(source_summary.read_text()) != summary:
        raise ValueError('existing control summary disagrees with validated receipts')
    rows = []
    hashes = {}
    for kind in C.KINDS:
        for source in sorted((out/kind).glob('*.json')):
            data = source.read_bytes()
            row = json.loads(data)
            rows.append(row)
            import hashlib
            hashes[str(source)] = hashlib.sha256(data).hexdigest()
    return rows, dict(validated_summary=summary, receipt_sha256=hashes,
                     provenance_files={str(p): U.file_sha(p) for p in out.glob('provenance_rank*.json')},
                     source_summary_sha256=U.file_sha(source_summary) if source_summary.exists() else None)


def report(plan):
    if len(plan['jobs']) != 2 or {r['model'] for r in plan['jobs']} != set(MODELS):
        raise ValueError('plan must identify exactly F2 and R0')
    panel = Path(plan['stage'])/'results/lc_dev_panel'
    rows, evidence = {}, {}
    # Require expected file counts before expensive prompt reconstruction.
    status = coverage(plan)
    if any(v['receipts'] != 72 for kinds in status['models'].values() for v in kinds.values()):
        raise ValueError('partial controls: use --status for coverage without paper scores')
    for job in plan['jobs']:
        rows[job['model']], evidence[job['model']] = validated_rows(Path(job['out']), panel, job['model'])
    result = aggregate(rows)
    result.update(evidence=evidence, source_panel=str(panel),
                  source_panel_sha256=U.digest({p.name:U.file_sha(p) for p in panel.glob('*.json')}))
    return result


def latex(result):
    lookup = {(r['model'],r['kind'],r['family']):r for r in result['rows']}
    macro = {(r['model'],r['kind']):r for r in result['macros']}
    lines = [r'\begin{table}[t]',r'\centering',r'\small',r'\begin{tabular}{llrrrr}',r'\toprule',
             r'Control & Family & F2 correct/$n$ & R0 correct/$n$ & F2 parse (\%) & R0 parse (\%) \\',r'\midrule']
    for kind in C.KINDS:
        for family in FAMILIES:
            f,r = (lookup[m,kind,family] for m in MODELS)
            lines.append(f"{KIND_NAMES[kind]} & {FAMILY_NAMES[family]} & {f['correct']}/24 & {r['correct']}/24 & {100*f['parse_rate']:.1f} & {100*r['parse_rate']:.1f}"+r' \\')
        f,r=(macro[m,kind] for m in MODELS)
        lines.append(f" & Family macro (\\%) & {100*f['family_macro_accuracy']:.2f} & {100*r['family_macro_accuracy']:.2f} & {100*f['family_macro_parse_rate']:.1f} & {100*r['family_macro_parse_rate']:.1f}"+r' \\')
        lines.append(r'\midrule')
    lines[-1]=r'\bottomrule'
    lines += [r'\end{tabular}',
        r'\caption{Post-development competence diagnostics using the same 72 development tasks (24 per family) for both models and all controls. R0 is released RWKV-7 2.9B; F2 uses its frozen 8-step, temperature-0.7 matched-Bernoulli decoder. Evidence-only removes marked filler; no-evidence replaces the evidence payload with neutral filler at 16K. Paired 4K/8K anchors rerender the identical logical task without redrawing its bindings. Exact-match counts, family-macro accuracy and parsing rates retain the supplied target length. These reused-development diagnostics are not independent confirmation and do not estimate full-grid $\Gamma$.}',
        r'\label{tab:e3-controls}',r'\end{table}','']
    return '\n'.join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=Path('results/e3_controls_plan.json'))
    parser.add_argument('--out-dir',type=Path,default=Path('results'))
    parser.add_argument('--status',action='store_true')
    args=parser.parse_args(argv)
    plan=json.loads(args.plan.read_text())
    if args.status:
        print(json.dumps(coverage(plan),indent=2)); return
    result=report(plan)
    result['plan_sha256']=U.file_sha(args.plan)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    U.atomic_json(args.out_dir/'e3_controls_summary.json',result)
    (args.out_dir/'e3_controls_summary.tex').write_text(latex(result))
    print(json.dumps(result['macros'],indent=2))


if __name__=='__main__':
    main()
