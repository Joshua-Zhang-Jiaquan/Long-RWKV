"""Exact analytical references on the already-frozen confirmation panel.

No neural model is loaded or evaluated. These results do not establish trained
model competence and cannot change the frozen recipe, inputs, or reveal groups.
"""
import hashlib
import json
import math
from pathlib import Path
from lrwkv_evidence.train04 import tasks
from lrwkv_evidence.train04dev.evaluate import endpoint_metrics
from lrwkv_evidence.train04confirm.evaluate import load_panel,METHODS
from lrwkv_evidence.train04confirm.collect import summarize
from lrwkv_evidence.train04confirm import contract as K

ROOT=Path(__file__).resolve().parents[2]


def main():
    manifest_path=ROOT/'theory_mvp/train04confirm/FROZEN_RECIPE.json'
    manifest=json.loads(manifest_path.read_text())
    digest=K.sha(manifest_path)
    plan=json.loads((ROOT/f'results/train04confirm/plan_{digest[:16]}_confirm_parent_seed53.json').read_text())
    stage=Path(plan['stage']);frozen,stage_sha=K.validate(stage/'manifest.json',stage,stage/'model_source')
    if stage_sha!=digest:raise ValueError('frozen stage differs')
    panel=load_panel(stage,frozen);results={}
    for name in ('oracle_conditionals','initial_marginal_product','always_fair'):
        def predict(ex,visible,time):
            if name=='always_fair':probs=[.5]*8
            else:probs=tasks.oracle_marginals(ex['matrix_rows'],ex['syndrome'],visible if name=='oracle_conditionals' else {},8)
            return [[math.log(1-p) if p<1 else -math.inf,math.log(p) if p>0 else -math.inf] for p in probs]
        rows=[]
        for record,ex in panel:
            cache={}
            for method in METHODS:
                metric=endpoint_metrics(ex,method,predict,cache)
                if name=='oracle_conditionals':
                    assert abs(metric['estimation_error_nats'])<1e-10
                    assert abs(metric['joint_kl_nats']-metric['dependence_cost_nats'])<1e-10
                    assert abs(metric['exact_valid_mass']-math.exp(-metric['dependence_cost_nats']))<1e-10
                    if method in ('information_set','sequential'):assert abs(metric['joint_kl_nats'])<1e-10
                rows.append(dict(n=8,index=record['index'],family=record['family'],structure=record['structure'],instance_id=record['instance_id'],**metric))
        results[name]=summarize(rows,panel)
    for row in results['initial_marginal_product']['family_summary']:
        target=math.log(16) if row['family']=='paired_parity' else 0.
        assert abs(row['joint_kl_nats']-target)<1e-10
    for row in results['initial_marginal_product']['condition_contrasts']:
        assert abs(row['random_minus_information_set_kl'])<1e-10
    result=dict(scope='Analytical reference laws on fixed confirmation conditions after recipe freeze; no neural outputs and no outcome-based selection',
                manifest_sha256=digest,panel_sha256=frozen['panel_sha256'],partitions_sha256=frozen['partitions_sha256'],
                source_sha256=K.sha(Path(__file__)),references=results)
    dest=ROOT/'results/train04confirm/oracle_reference.json'
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name,data in results.items():print(name,json.dumps(data['family_summary']))


if __name__=='__main__':main()
