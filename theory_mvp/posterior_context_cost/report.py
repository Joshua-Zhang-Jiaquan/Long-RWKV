"""Restricted empirical mean-budget certificate, never an attention quality estimate."""
import json,math,statistics
from pathlib import Path
from lrwkv_evidence.train04.worker import file_sha
ROOT=Path(__file__).resolve().parents[2]
def main():
 folder=ROOT/'results/posterior_context_cost';cost={k:json.loads((folder/f'exact_{k}.json').read_text()) for k in ('rwkv','attention')}
 quality=json.loads((ROOT/'results/posterior_context_final/final_report.json').read_text())
 for r in cost.values():
  if not r['execution_complete'] or r['final_design_sha256']!=quality['design_sha256']:raise ValueError('unmatched panel')
  for p,sha in r['raw_sha256'].items():
   if file_sha(Path(p))!=sha:raise ValueError('changed timing evidence')
 if cost['rwkv']['checkpoint_sha256']!=quality['checkpoint_sha256']['adapted200']:raise ValueError('unmatched quality checkpoint')
 # Verify exact recurrent native inputs match all48 accuracy conditions.
 inputs={}
 for shard in range(4):
  r=json.loads((ROOT/f'results/posterior_context_final/exact_adapted200_shard{shard}.json').read_text())
  for row in r['conditions']:
   if row['family']=='paired_parity' and row['serial']['requested_context_tokens']==16384:inputs[row['index'],row['serial']['position']]=row['serial']
 if len(inputs)!=48:raise ValueError('missing accuracy inputs')
 if any(inputs[r['index'],r['serial']['position']]!=r['serial'] for r in cost['rwkv']['conditions']):raise ValueError('different recurrent timing inputs')
 stats={k:{r['method']:r for r in v['summary']} for k,v in cost.items()};budget=1.5
 kl=statistics.mean(r['mean_kl'] for r in quality['summary'] if r['role']=='adapted200' and r['family']=='paired_parity' and r['context_tokens']==16384 and r['method']=='information_set')
 checks=dict(recurrent_two_call_fits=stats['rwkv']['information_set']['mean_request_seconds']<=budget,attention_one_call_fits=stats['attention']['one']['mean_request_seconds']<=budget,attention_information_set_excluded=stats['attention']['information_set']['mean_request_seconds']>budget,attention_pair_preserving_excluded=stats['attention']['pair_preserving_halves']['mean_request_seconds']>budget,recurrent_kl_below_one_call_product_floor=kl<4*math.log(2))
 result=dict(execution_complete=True,empirical_panel_certificate=all(checks.values()),checks=checks,budget_seconds=budget,recurrent_mean_kl=kl,one_call_product_kl_lower_bound=4*math.log(2),cost_summaries=stats,quality_design_sha256=quality['design_sha256'],checkpoint_sha256={k:v['checkpoint_sha256'] for k,v in cost.items()},scope='Restricted declared attention policy set on this empirical paired panel, using mean complete-request latency. No per-request deadline, broad population, causal-sampler, correlated-head, batching, or untested-implementation guarantee. Attention accuracy was not measured; its one-call lower bound is analytic.')
 (folder/'budget_certificate.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# Matched-task mean-budget check','',result['scope'],'',f"Predeclared budget:1.5s; empirical-panel certificate:{result['empirical_panel_certificate']}",'','| Model | Policy | Mean seconds | Pointwise95% interval | Peak GiB |','|---|---|---:|---|---:|']
 for k,v in stats.items():
  for method,r in v.items():lines.append(f"|{k}|{method}|{r['mean_request_seconds']:.6f}|{r['pointwise_95_percentile_interval']}|{r['peak_allocated_bytes']/2**30:.4f}|")
 lines+=['',f"Recurrent mean KL:{kl:.9g}; analytic one-call floor:{4*math.log(2):.9g}."]
 (folder/'BUDGET_REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
