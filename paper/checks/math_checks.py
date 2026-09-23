"""Deterministic checks of manuscript identities, not language-model experiments.

Run: python math_checks.py --output math_checks.json
Requires NumPy. These checks do not load RWKV weights, run FLA kernels,
measure benchmark performance, or certify assumptions of a trained model.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def run_checks() -> dict:
    rng = np.random.default_rng(20260917)
    results = {}
    # 1. Exact normalized delta-rule residual identity.
    max_residual = 0.0
    for _ in range(100):
        S = rng.normal(size=(5, 7)); k = rng.normal(size=7); k /= np.linalg.norm(k)
        v = rng.normal(size=5); beta = rng.uniform(0.01, 1.99)
        updated = S + beta * np.outer(v - S @ k, k)
        error = np.linalg.norm((updated @ k - v) - (1-beta)*(S @ k-v))
        max_residual = max(max_residual, float(error))
    assert max_residual < 1e-11
    results['delta_residual_identity'] = {'trials': 100, 'max_absolute_residual': max_residual}

    # 2. Latest-write retrieval versus a recency-aware softmax comparator.
    keys = np.eye(3); indices = [0, 1, 0, 2, 0]
    values = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.], [0., 1.]])
    S = np.zeros((2,3)); additive = np.zeros_like(S); counts = np.zeros(3)
    for i, v in zip(indices, values):
        k = keys[i]; S += np.outer(v-S@k, k)
        additive += np.outer(v, k); counts[i] += 1
    latest = values[-1]; y_delta = S@keys[0]; y_add = additive@keys[0]/counts[0]
    margin = 8.0; logits = np.full(len(indices), -margin); logits[-1] = 0.
    p = np.exp(logits); p /= p.sum(); y_attention = p@values
    delta_bound = 2*(len(indices)-1)*np.exp(-margin)
    err_delta = float(np.linalg.norm(y_delta-y_attention))
    err_add = float(np.linalg.norm(y_add-y_attention))
    D = float(np.linalg.norm(y_add-latest))
    assert np.allclose(y_delta, latest) and D > 2*delta_bound
    assert err_delta <= delta_bound+1e-12 and err_add >= D-delta_bound-1e-12
    results['overwrite_attention_comparison'] = {
        'records': len(indices), 'logit_margin': margin,
        'delta_attention_error': err_delta, 'additive_attention_error': err_add,
        'theorem_upper_bound': float(delta_bound), 'separation_D': D}

    # 3. Variable-metric perturbation bound for native row-state RWKV form.
    dk, dv, N = 6, 4, 400
    a_prev = np.full(dk, 0.4)
    E = np.zeros((dv,dk)); scalar_bound = 0.0
    max_q = 0.0; max_excess = -np.inf; symmetry_residual = 0.0
    epsilon = 1e-4
    for _ in range(N):
        a = rng.uniform(0.398, 0.402, dk)
        w = rng.uniform(0.62, 0.86, dk)
        u = rng.normal(size=dk); u /= np.linalg.norm(u)
        A = np.diag(w) - np.outer(u, a*u)
        P = np.diag(np.sqrt(a)); Pinv = np.diag(1/np.sqrt(a))
        B = P@A@Pinv
        symmetry_residual = max(symmetry_residual, float(np.linalg.norm(B-B.T)))
        r = float(np.max(np.abs(np.linalg.eigvalsh(B))))
        m = float(np.max(np.sqrt(a_prev/a))); q = r*m
        Xi = rng.normal(size=(dv,dk)); Xi *= epsilon/np.linalg.norm(Xi)
        E = E@A+Xi
        scalar_bound = q*scalar_bound+float(np.linalg.norm(Xi@Pinv))
        actual = float(np.linalg.norm(E@Pinv))
        max_excess = max(max_excess, actual-scalar_bound)
        max_q = max(max_q,q); a_prev = a
    assert symmetry_residual < 1e-12 and max_excess < 1e-12 and max_q < 1
    results['variable_metric_scan_bound'] = {
        'positions': N, 'max_effective_contraction': max_q,
        'max_symmetry_residual': symmetry_residual,
        'max_actual_minus_bound': float(max_excess),
        'final_weighted_error': actual, 'final_weighted_bound': scalar_bound}

    # 4. Finite-state, finite-horizon Markov perturbation inequality.
    def tv(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(a-b).sum()/2)
    nstates, steps = 5, 30
    mu = rng.dirichlet(np.ones(nstates)); nu = rng.dirichlet(np.ones(nstates))
    bound = tv(mu,nu); max_excess = -np.inf
    max_kappa = 0.0
    for _ in range(steps):
        base = rng.dirichlet(np.ones(nstates), size=nstates)
        reference = 0.7*base+0.3*np.ones((nstates,nstates))/nstates
        alternative = 0.98*reference+0.02*rng.dirichlet(np.ones(nstates),size=nstates)
        kappa = max(tv(reference[i], reference[j]) for i in range(nstates) for j in range(nstates))
        epsilon_k = max(tv(alternative[i],reference[i]) for i in range(nstates))
        bound = kappa*bound+epsilon_k
        mu = mu@alternative; nu = nu@reference
        max_excess = max(max_excess,tv(mu,nu)-bound); max_kappa=max(max_kappa,kappa)
    assert max_excess < 1e-12
    results['markov_perturbation_bound'] = {
        'steps': steps, 'states': nstates, 'max_dobrushin_coefficient': max_kappa,
        'final_total_variation': tv(mu,nu), 'final_bound': bound,
        'max_actual_minus_bound': float(max_excess)}

    # 5. A projection-to-attention optional design, not deployed tied-depth RWKV.
    Q = rng.normal(size=(80,8)); teacher = rng.normal(size=(80,4))
    M = rng.normal(size=(8,4)); opt=np.linalg.lstsq(Q,teacher,rcond=None)[0]
    C=Q.T@Q/len(Q); eig=np.linalg.eigvalsh(C); eta=1/eig[-1]
    gamma=float(np.max(np.abs(1-eta*eig)))
    def gap(M):
        residual = Q@(M-opt)
        return float(np.square(residual).sum()/(2*len(Q)))
    init_gap=gap(M); max_excess=-np.inf
    for _ in range(20):
        before=gap(M); M -= eta*(Q.T@(Q@M-teacher))/len(Q)
        max_excess=max(max_excess,gap(M)-gamma**2*before)
    assert max_excess < 1e-10
    results['optional_attention_projection_descent'] = {
        'iterations':20,'contraction':gamma,'initial_loss_gap':init_gap,
        'final_loss_gap':gap(M),'max_actual_minus_bound':float(max_excess)}

    # 6. Finite compression cannot preserve an omitted fair bit.
    # H=(b1,b2), C=b1, Y=b2: H(Y|C)=ln(2), H(Y|H)=0.
    results['compression_counterexample'] = {
        'log_loss_gap_nats': float(np.log(2)), 'best_answer_accuracy':0.5,
        'extra_history_bit_log_loss_reduction_nats':float(np.log(2))}
    results['memory_arithmetic'] = {
        'audited_parameters':4091581441,
        'bf16_weight_payload_GiB':4091581441*2/2**30,
        'ideal_4bit_weight_payload_GiB':4091581441/2/2**30,
        '64K_by_65536_bf16_logits_GiB':65536*65536*2/2**30,
        '64K_by_65536_fp32_logits_GiB':65536*65536*4/2**30}
    return {'scope':'Synthetic identity and inequality checks; NOT model benchmarks.',
            'seed':20260917,'all_checks_passed':True,'checks':results}

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('math_checks.json'))
    args=parser.parse_args()
    result=run_checks(); args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
