"""Freeze all scientific choices before starting the six main lineages."""
import json
from pathlib import Path
from lrwkv_evidence.predictive_transfer import tasks as T,training as R,prediction as P
from lrwkv_evidence.train04.worker import file_sha,canonical_sha
ROOT=Path(__file__).resolve().parents[2]

def main():
    target=ROOT/'revision/transfer/FROZEN.json'
    if target.exists():raise ValueError('refuse refreezing')
    plan=json.loads((ROOT/'results/predictive_transfer/plan_qualification.json').read_text())
    done=json.loads((Path(plan['out'])/'completion.json').read_text())
    if not done.get('execution_complete') or not done.get('qualification') or done['source_sha256']!=plan['source_sha256']:
        raise ValueError('matching qualification required')
    protocol=dict(status='frozen',version=1,contribution='prospective conditional-error prediction on unseen dependency classes',
      training=R.RECIPE,training_seeds=list(R.SEEDS),training_source_sha256=plan['source_sha256'],
      qualification=dict(plan_sha256=file_sha(ROOT/'results/predictive_transfer/plan_qualification.json'),completion_sha256=file_sha(Path(plan['out'])/'completion.json')),
      catalog_sha256=canonical_sha(T.catalog()),
      classes={c:dict(weight_enumerator=v['weight_enumerator'],split=v['split'],matrices=len(v['matrices'])) for c,v in T.catalog().items()},
      calibration_panel=P.panel('calibration'),heldout_panel=P.panel('heldout'),
      predictor=dict(method='source-only per-output bin means with pooled-round-and-position shrinkage',
                     features=['fair/deterministic','parity_fanin_or_zero','equations_needed_or_column_degree','position'],
                     minimum_distinct_prompts=P.MIN_SUPPORT,shrinkage_outputs=P.SHRINKAGE,
                     fallback='if any feature in either policy unsupported, choose dependence-only content-hash tie',
                     commitment='all six model prediction artifacts plus hashes sealed before any held-out neural evaluation'),
      estimands=dict(primary='mean KL(dependence-only hash tie) minus KL(predicted choice), all 6 lineages x10 classes x8 prompts x3 positions',
                      secondary=['regret to posterior-best policy','random expected KL','fixed A','fixed B','source unconditional preference','fanin heuristic',
                                 'prediction RMSE','per-lineage and per-class primary effect','absolute selected KL','supported decision fraction'],
                      uncertainty='10000 percentile bootstrap replicates; sample 6 lineages and10 classes with replacement independently; within each sampled class sample8 prompt indices, retaining all3 positions; shared prompt resample across lineages',
                      bootstrap_seed=202709239,interval=[.025,.975]),
      success_gates=dict(mean_primary_at_least_nats=.1,primary_interval_lower_above=0.,supported_fraction_at_least=.8,
                         complete_lineages=6,competent_lineages_at_least=5,
                         competence='mean selected-policy held-out KL below4log2, averaged over all declared classes/prompts/positions',
                         calibration_added_value='report gain over fanin heuristic and source-unconditional baseline; positive transfer alone does not prove calibration needed'),
      retention=dict(checkpoint='step3100 for every seed, no quality selection or retries for poor learning',
                     operational_failure='retain failed attempt and cost; manual identical scientific-source repair only, never automatic restart',
                     missing_lineage='report incomplete result, no full-study success claim; no imputed finite KL',
                     negative_result='retain failed gates and all policy scores, do not redefine task, panel or predictor'),
      cost=dict(max_live_including_queued=48,primary=32,extra=16,gpus_per_job=8,training_wall_limit_hours=24,
                preference='fastest completion; report real utilization without padding'),
      interpretation=dict(both_policies_D=0,calls_each=2,one_call_product_floor_nats=4*__import__('math').log(2),
                           transfer='empirical source-to-target hypothesis, no finite context-density bound across disjoint classes',
                           lineage='fresh task-training lineages sharing public pretrained backbone; no seed71 descendants'))
    sources=['lrwkv_evidence/predictive_transfer/'+p for p in ('tasks.py','algebra.py','training.py','worker.py','prediction.py','evaluate.py')]
    protocol['scientific_sources']={p:file_sha(ROOT/p) for p in sources}
    protocol['analysis_sources']={p:file_sha(ROOT/p) for p in ('revision/transfer/analyze.py',)}
    target.write_text(json.dumps(protocol,indent=2,sort_keys=True)+'\n')
    print(file_sha(target))
if __name__=='__main__':main()
