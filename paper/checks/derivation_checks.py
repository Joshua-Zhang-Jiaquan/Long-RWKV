"""Finite, deterministic checks of Long-RWKV's analytic derivations.

No trained weights, benchmark datasets, GPUs, or inferred empirical results.
Run: python derivation_checks.py --output derivation_checks.json
Requires Python >=3.10 and NumPy. All logarithms are natural.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL for normalized discrete probabilities, allowing zeros in p."""
    active = p > 0
    if np.any(q[active] <= 0):
        return float('inf')
    return float(np.sum(p[active] * np.log(p[active] / q[active])))


def softmax(x: np.ndarray) -> np.ndarray:
    out = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return out / out.sum(axis=-1, keepdims=True)


def categorical_example(steps: int, student_shift: float) -> dict:
    """Exhaustively enumerate a correlated binary pair and all masked states."""
    N, V, mask = 2, 2, 2
    clean = list(itertools.product(range(V), repeat=N))
    states = list(itertools.product(range(V + 1), repeat=N))
    index = {x: i for i, x in enumerate(states)}
    prior = np.array([0.45, 0.05, 0.05, 0.45], dtype=float)
    q0 = np.zeros(len(states))
    for y, py in zip(clean, prior):
        q0[index[y]] = py
    rho = np.linspace(1., 0., steps + 1)
    marginals = [q0]
    reverse_student = []
    total_path = excess = dependence = vlb = 0.0
    max_decomp_residual = 0.0
    for t in range(1, steps + 1):
        survival = rho[t] / rho[t - 1]
        trans = np.zeros((len(states), len(states)))
        for a, s in enumerate(states):
            for b, x in enumerate(states):
                prob = 1.0
                for si, xi in zip(s, x):
                    prob *= (float(xi == mask) if si == mask else
                             survival if xi == si else 1-survival if xi == mask else 0.)
                trans[a, b] = prob
        qt = marginals[-1] @ trans
        omega = (rho[t-1] - rho[t]) / (1-rho[t])
        R = np.zeros_like(trans)
        P = np.zeros_like(trans)
        for b, x in enumerate(states):
            if qt[b] <= 1e-16:
                P[b, b] = R[b, b] = 1.
                continue
            R[b] = marginals[-1] * trans[:, b] / qt[b]
            post = prior.copy()
            for a, y in enumerate(clean):
                for xi, yi in zip(x, y):
                    post[a] *= (1-rho[t]) if xi == mask else rho[t] if xi == yi else 0.
            post /= post.sum()
            single = np.array([[sum(post[j] for j,y in enumerate(clean) if y[i] == v)
                                for v in range(V)] for i in range(N)])
            pred = single.copy()
            U = [i for i, xi in enumerate(x) if xi == mask]
            for i in U:
                pred[i] = softmax(np.log(single[i]) + student_shift * np.array([1.,-1.]) * (i+1))
            for a, s in enumerate(states):
                prob = 1.
                for i in range(N):
                    if x[i] != mask:
                        prob *= float(s[i] == x[i])
                    elif s[i] == mask:
                        prob *= 1-omega
                    else:
                        prob *= omega * pred[i, s[i]]
                P[b, a] = prob
            local_excess = omega * sum(kl(single[i], pred[i]) for i in U)
            local_dep = 0.0
            for flags in itertools.product([False, True], repeat=len(U)):
                J = [i for i, flag in zip(U, flags) if flag]
                weight = omega**len(J) * (1-omega)**(len(U)-len(J))
                if len(J) < 2 or weight == 0:
                    continue
                tuples = list(itertools.product(range(V), repeat=len(J)))
                pj = np.array([sum(post[a] for a,y in enumerate(clean)
                                   if tuple(y[i] for i in J) == vals) for vals in tuples])
                prod = np.array([np.prod([single[i,v] for i,v in zip(J, vals)]) for vals in tuples])
                local_dep += weight * kl(pj, prod)
            local_path = kl(R[b], P[b])
            max_decomp_residual = max(max_decomp_residual, abs(local_path-local_excess-local_dep))
            total_path += qt[b] * local_path
            excess += qt[b] * local_excess
            dependence += qt[b] * local_dep
            # Conditional expectation of exact weighted clean-token CE.
            for a, y in enumerate(clean):
                vlb += qt[b] * post[a] * omega * sum(-np.log(pred[i,y[i]]) for i in U)
        assert np.max(np.abs(P.sum(axis=1)-1.)) < 1e-12
        reverse_student.append(P)
        marginals.append(qt)
    output = marginals[-1].copy()
    for P in reversed(reverse_student):
        output = output @ P
    entropy = -float(np.sum(prior * np.log(prior)))
    endpoint = kl(q0, output)
    mesh_bound = math.comb(N,2)*math.log(V)*np.sum(np.diff(rho)**2)
    assert max_decomp_residual < 1e-11
    assert abs(total_path - (excess + dependence)) < 1e-11
    assert abs(vlb - entropy - total_path) < 1e-11
    assert endpoint <= total_path + 1e-11
    assert dependence <= mesh_bound + 1e-11
    return dict(steps=steps, shift=student_shift, path_kl=total_path,
                endpoint_kl=endpoint, posterior_excess=excess, dependence_gap=dependence,
                dependence_bound=float(mesh_bound), nelbo=vlb, data_entropy=entropy,
                nelbo_identity_residual=abs(vlb-entropy-total_path),
                max_one_step_identity_residual=max_decomp_residual)


def run_checks() -> dict:
    rng = np.random.default_rng(20260920)
    result = {}
    tol = 1e-10
    # Gaussian posterior via independent precision calculation and parameterization.
    worst = 0.
    for _ in range(100):
        abar_prev, beta = rng.uniform(.05,.95), rng.uniform(.01,.4)
        a = 1-beta; abar = a*abar_prev; b = np.sqrt(1-abar)
        x0, noise, pred = rng.normal(size=(3,7))
        xt = np.sqrt(abar)*x0+b*noise
        bt = beta*(1-abar_prev)/(1-abar)
        A = beta*np.sqrt(abar_prev)/(1-abar)
        B = np.sqrt(a)*(1-abar_prev)/(1-abar)
        mu = A*x0+B*xt
        precision = a/beta+1/(1-abar_prev)
        mu_direct = (np.sqrt(a)*xt/beta + np.sqrt(abar_prev)*x0/(1-abar_prev))/precision
        mu_pred = (xt-beta*pred/b)/np.sqrt(a)
        sigma2 = rng.uniform(.02,.5)
        lam = beta**2/(2*sigma2*a*(1-abar))
        left = np.linalg.norm(mu-mu_pred)**2/(2*sigma2)
        right = lam*np.linalg.norm(noise-pred)**2
        velocity = np.sqrt(abar)*noise-b*x0
        recovered = np.sqrt(abar)*xt-b*velocity
        worst = max(worst, abs(1/precision-bt), np.max(abs(mu-mu_direct)),
                    abs(left-right), np.max(abs(recovered-x0)))
    assert worst < tol
    result['ddpm_posterior_noise_weight_and_velocity'] = dict(trials=100, max_residual=float(worst))

    # General categorical bridge, and non-equivalence of joint substitution.
    Q1, Q2 = rng.dirichlet(np.ones(4), size=4), rng.dirichlet(np.ones(4), size=4)
    Qbar = Q1@Q2; b = 2
    bridge = Q1 * Q2[:, b][None, :] / Qbar[:, b][:, None]
    pi = rng.dirichlet(np.ones(4))
    mix = pi @ bridge
    joint = (pi@Q1)*Q2[:,b]; joint /= joint.sum()
    tilted = pi*Qbar[:,b]; tilted /= tilted.sum()
    assert np.max(abs(bridge.sum(axis=1)-1)) < tol
    assert np.max(abs(joint-tilted@bridge)) < tol
    assert np.linalg.norm(mix-joint) > 1e-5
    result['general_d3pm_bridge_and_parameterization'] = dict(
        posterior_row_sum_residual=float(np.max(abs(bridge.sum(axis=1)-1))),
        mixture_vs_joint_l1_gap=float(np.sum(abs(mix-joint))))

    # General mixture-head logit derivative and bridge-channel KL contraction.
    logits=rng.normal(size=4); pi=softmax(logits); target=bridge[1].copy()
    modeled=pi@bridge
    grad=pi-np.sum(target[None,:]*(pi[:,None]*bridge)/modeled[None,:],axis=1)
    finite=np.zeros(4); h=1e-6
    for i in range(4):
        plus=logits.copy();plus[i]+=h;minus=logits.copy();minus[i]-=h
        finite[i]=(-np.sum(target*np.log(softmax(plus)@bridge))+
                   np.sum(target*np.log(softmax(minus)@bridge)))/(2*h)
    pstar=rng.dirichlet(np.ones(4))
    assert np.max(abs(grad-finite))<1e-8
    assert kl(pstar@bridge,pi@bridge)<=kl(pstar,pi)+tol
    result['general_categorical_gradient_and_bridge_contraction'] = dict(
        max_gradient_residual=float(np.max(abs(grad-finite))),
        input_kl=kl(pstar,pi),output_kl=kl(pstar@bridge,pi@bridge))

    # Absorbing posterior KL is exactly reveal-probability times CE.
    worst = 0.
    for _ in range(100):
        pi = rng.dirichlet(np.ones(5)); y = int(rng.integers(5)); omega = rng.uniform(.01,.99)
        true = np.zeros(6); true[y] = omega; true[-1] = 1-omega
        pred = np.r_[omega*pi,1-omega]
        worst = max(worst, abs(kl(true,pred)+omega*np.log(pi[y])))
    assert worst < tol
    result['absorbing_bridge_kl_equals_weighted_ce'] = dict(trials=100, max_residual=float(worst))

    # Exact selected-mask scaling; naive expected-count substitution differs.
    p, N, omega, r = .37, 3, .42, .2
    target = corrected = naive = 0.
    for mask in itertools.product([False,True], repeat=N):
        m = sum(mask); prob = p**m*(1-p)**(N-m)
        losses = np.array([.1+(i+1)*.3+m*.4 for i in range(N)])
        selected = losses[np.array(mask)].sum()
        target += prob*omega/(N*r)*selected
        mean = selected/m if m else 0.
        corrected += prob*omega*m/(N*r)*mean
        naive += prob*omega*p/r*mean
    assert abs(target-corrected) < tol and abs(target-naive)>1e-3
    result['selected_mask_normalization'] = dict(exact_expected_estimator=target,
        corrected_selected_mean=corrected, naive_expected_count_value=naive)

    # Finite positive kernel attention: batch, prefix state, and exact bidirectional state.
    phi_q = np.exp(rng.normal(size=(11,6))); phi_k = np.exp(rng.normal(size=(11,6)))
    values = rng.normal(size=(11,4)); K = phi_q@phi_k.T
    batch = (K@values)/K.sum(axis=1,keepdims=True)
    S = values.T@phi_k; z=phi_k.sum(axis=0)
    state = np.array([(S@q)/(z@q) for q in phi_q])
    bidir=[]
    for i,q in enumerate(phi_q):
        Sl=values[:i+1].T@phi_k[:i+1]; Sr=values[i+1:].T@phi_k[i+1:]
        zl=phi_k[:i+1].sum(axis=0); zr=phi_k[i+1:].sum(axis=0)
        bidir.append(((Sl+Sr)@q)/((zl+zr)@q))
    resid = max(np.max(abs(batch-state)),np.max(abs(batch-np.array(bidir))))
    assert resid<tol
    result['exact_kernel_state_and_bidirectional_identity'] = dict(max_residual=float(resid))

    # Normalized kernel error and constructive tensor truncation.
    max_ratio=0.
    for _ in range(100):
        weights=np.exp(rng.normal(size=13)); vals=rng.normal(size=(13,5))
        eta=.15; changed=weights*(1+rng.uniform(-eta,eta,13))
        err=np.linalg.norm(weights@vals/weights.sum()-changed@vals/changed.sum())
        bound=2*np.linalg.norm(vals,axis=1).max()*eta/(1-eta)
        assert err <= bound+tol
        max_ratio=max(max_ratio,err/bound)
    result['normalized_kernel_perturbation_bound'] = dict(trials=100,max_error_bound_ratio=float(max_ratio))
    pdegree,R=6,.7; errors=[]
    for _ in range(100):
        q,k=rng.normal(size=(2,3)); q*=R/np.linalg.norm(q); k*=R/np.linalg.norm(k)
        dot=q@k
        trunc=sum(dot**j/math.factorial(j) for j in range(pdegree+1))
        bound=math.exp(R*R)*(R*R)**(pdegree+1)/math.factorial(pdegree+1)
        errors.append(abs(math.exp(dot)-trunc))
        assert errors[-1]<=bound+tol
    result['tensor_feature_truncation_bound'] = dict(trials=100,max_absolute_error=float(max(errors)),bound=bound)

    # Residual update and exact RLS recursion.
    Pinv=np.eye(5)/.8; S=np.zeros((3,5)); C=.8*np.eye(5); M=np.zeros((3,5)); worst=0.
    for _ in range(40):
        k=rng.normal(size=5); v=rng.normal(size=3)
        gain=Pinv@k/(1+k@Pinv@k)
        S=S+np.outer(v-S@k,gain)
        Pinv=Pinv-np.outer(Pinv@k,k@Pinv)/(1+k@Pinv@k)
        C+=np.outer(k,k); M+=np.outer(v,k)
        worst=max(worst,np.max(abs(S-M@np.linalg.inv(C))),np.max(abs(Pinv-np.linalg.inv(C))))
    assert worst<tol
    result['rls_state_equals_batch_ridge_regression'] = dict(writes=40,max_residual=float(worst))

    # Exact additive-versus-generalized-state decomposition and its norm bound.
    n,dv,dk=9,3,4
    A=rng.normal(scale=.22,size=(n,dk,dk)); BK=rng.normal(size=(n,dv,dk))
    BR=BK+rng.normal(scale=.1,size=BK.shape); initial=rng.normal(size=(dv,dk))
    S=initial.copy()
    for i in range(n): S=S@A[i]+BR[i]
    K=BK.sum(axis=0)
    suffix=[np.eye(dk) for _ in range(n+1)]
    for i in reversed(range(n)): suffix[i]=A[i]@suffix[i+1]
    decomposition=initial@suffix[0]
    for j in range(n):
        decomposition+=(BR[j]-BK[j])@suffix[j+1]+BK[j]@(suffix[j+1]-np.eye(dk))
    norms=np.linalg.norm(A,ord=2,axis=(1,2))
    upper=np.linalg.norm(initial)*np.prod(norms)
    for j in range(n):
        upper+=np.linalg.norm(BR[j]-BK[j])*np.prod(norms[j+1:])
        upper+=np.linalg.norm(BK[j])*sum(np.linalg.norm(A[r]-np.eye(dk),2)*np.prod(norms[r+1:])
                                          for r in range(j+1,n))
    residual=np.max(abs(S-K-decomposition))
    assert residual<tol and np.linalg.norm(S-K)<=upper+tol
    result['explicit_additive_to_rwkv_state_bound'] = dict(
        max_identity_residual=float(residual),actual_state_difference=float(np.linalg.norm(S-K)),
        upper_bound=float(upper))

    # Output KL bridge and quadratic teacher/student softmax bound.
    max_ratio=0.; bridge_margin=1e9
    for _ in range(100):
        a,b=rng.normal(size=(2,9)); pstar=rng.dirichlet(np.ones(9)); tau=.8
        pa,pb=softmax(a/tau),softmax(b/tau)
        quad=np.sum((a-b)**2)/(4*tau*tau)
        max_ratio=max(max_ratio,kl(pa,pb)/quad)
        rhs=kl(pstar,pa)+2*np.max(abs(a-b))/tau
        bridge_margin=min(bridge_margin,rhs-kl(pstar,pb))
        assert kl(pa,pb)<=quad+tol and kl(pstar,pb)<=rhs+tol
    result['logit_to_posterior_kl_bounds'] = dict(trials=100,max_quadratic_ratio=float(max_ratio),
                                               minimum_linear_bound_margin=float(bridge_margin))

    # Full dependency and risk chain, with exact and imperfect denoisers.
    result['enumerated_categorical_path_and_loss'] = [categorical_example(s, shift)
         for shift in [0.,.15] for s in [1,4,16,64]]

    # History compression chain rule: H=(a,b), C=a, noisy Y=b.
    joint=np.zeros((2,2,2))
    for a,b,y in itertools.product(range(2),repeat=3):
        joint[a,b,y]=.25*(.9 if y==b else .1)
    compressed=np.array([[.35,.65],[.7,.3]])
    lhs=info=residual=0.
    for a,b in itertools.product(range(2),repeat=2):
        truth=joint[a,b]/joint[a,b].sum(); marg=joint[a].sum(axis=0)/joint[a].sum()
        mass=joint[a,b].sum()
        lhs+=mass*kl(truth,compressed[a]); info+=mass*kl(truth,marg)
        residual+=mass*kl(marg,compressed[a])
    assert abs(lhs-info-residual)<tol
    result['history_compression_kl_chain_rule'] = dict(total_kl=lhs,information_lost=info,
                               compressed_predictor_kl=residual,residual=abs(lhs-info-residual))

    # Affine-state adjoint checked by finite differences.
    n,dv,dk=4,2,3
    As=rng.normal(scale=.2,size=(n,dk,dk)); Bs=rng.normal(size=(n,dv,dk))
    rs=rng.normal(size=(n,dk)); target=rng.normal(size=(n,dv))
    def loss(A,B,ret=False):
        states=[np.zeros((dv,dk))]; ys=[]
        for i in range(n):
            states.append(states[-1]@A[i]+B[i]); ys.append(states[-1]@rs[i])
        return (states,np.array(ys)-target) if ret else .5*np.sum((np.array(ys)-target)**2)
    states,g=loss(As,Bs,True); J=np.zeros((dv,dk)); dA=np.zeros_like(As);dB=np.zeros_like(Bs)
    for i in reversed(range(n)):
        J=np.outer(g[i],rs[i])+(J@As[i+1].T if i<n-1 else 0)
        dA[i]=states[i].T@J;dB[i]=J
    maxerr=0.;h=1e-6
    for array,grad,which in [(As,dA,0),(Bs,dB,1)]:
        for idx in np.ndindex(array.shape):
            old=array[idx];array[idx]=old+h;plus=loss(As,Bs)
            array[idx]=old-h;minus=loss(As,Bs);array[idx]=old
            maxerr=max(maxerr,abs((plus-minus)/(2*h)-grad[idx]))
    assert maxerr<1e-7
    result['affine_state_adjoint_finite_difference'] = dict(max_absolute_residual=float(maxerr))
    return dict(scope='Synthetic algebraic checks only; no language-model or hardware experiment',
                seed=20260920,status='PASS',checks=result)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('derivation_checks.json'))
    args=parser.parse_args()
    results=run_checks()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(results,indent=2))
