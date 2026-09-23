"""Exact schedule enumeration for the repetition-law mesh sharpness example."""
import itertools
import json
import math
from pathlib import Path


def quantities(n, stages):
    b = (n / stages * sum((1-j/stages)**(n-1) for j in range(stages))-1)*math.log(2)
    terms = []
    for j in range(stages):
        hi = n*math.log((j+.5)/stages)
        ratio = 0 if j == 0 else math.exp(n*math.log(j/(j+.5)))
        terms.append(hi+math.log1p(-ratio))
    pivot = max(terms)
    k = -(math.log(2)+pivot+math.log(sum(math.exp(x-pivot) for x in terms)))
    return b, k


def main():
    finite = []
    for n in (2, 3, 4, 6, 8):
        for stages in (1, 2, 3, 4):
            count = stages**n
            er = mass = 0.
            for times in itertools.product(range(stages), repeat=n):
                r = times.count(min(times))
                er += r/count
                mass += 2**(1-r)/count
            b, k = quantities(n, stages)
            assert abs(b-(er-1)*math.log(2)) < 1e-10
            assert abs(math.exp(-k)-mass) < 1e-12
            assert -1e-12 <= k <= b+1e-12 <= n/stages*math.log(2)+2e-12
            finite.append(dict(n=n, stages=stages, path_dependence=b, endpoint_kl=k,
                               valid_mass=mass, linear_mesh_bound=n/stages*math.log(2)))
    large = []
    for n, stages in ((100,10),(1000,10),(10000,100),(1000000,1000)):
        b,k = quantities(n,stages)
        large.append(dict(n=n,stages=stages,path_dependence=b,endpoint_kl=k,
                          path_ratio_to_N_over_T=b/(n/stages),endpoint_ratio_to_N_over_T=k/(n/stages)))
    small = []
    for n in (2,4,8):
        for stages in (128,1024,8192):
            b,k = quantities(n,stages)
            small.append(dict(n=n,stages=stages,path_ratio_to_N_over_T=b*stages/n,
                              endpoint_ratio_to_N_over_T=k*stages/n))
        assert abs(b*stages/n-math.log(2)/2)<1e-4
        assert abs(k*stages/n-.25)<1e-4
    proportional = []
    for ratio in (.125,.5,1.,2.,8.):
        limit_b=(ratio/(-math.expm1(-ratio))-1)*math.log(2)
        limit_k=math.log1p(math.exp(ratio/2))-math.log(2)
        for stages in (128,1024,8192):
            n=int(ratio*stages);b,k=quantities(n,stages)
            proportional.append(dict(n=n,stages=stages,ratio=ratio,path_dependence=b,endpoint_kl=k,
                                     limiting_path_dependence=limit_b,limiting_endpoint_kl=limit_k))
        assert abs(b-limit_b)<1e-4 and abs(k-limit_k)<2e-4
    dest=Path(__file__).resolve().parents[1]/'results/theory_mvp/repetition_mesh_sharpness.json'
    dest.write_text(json.dumps(dict(scope='Analytical repetition-law sharpness example; no neural experiment and no novelty claim',finite_enumerations=finite,asymptotic_examples=large,small_error_examples=small,proportional_limit_examples=proportional),indent=2)+'\n')
    print(f'Passed {len(finite)} exhaustive schedule checks; wrote {dest}')


if __name__ == '__main__':
    main()
