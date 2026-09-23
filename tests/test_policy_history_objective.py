"""Finite-support checks of the proposed policy-history loss identity."""
import math
from lrwkv_evidence.train04 import tasks
from lrwkv_evidence.train04.evaluate import support, fixed_groups


def test_policy_excess_equals_endpoint_estimation_error():
    for family in tasks.FAMILIES:
        ex=tasks.make_example('dev',91,0,family=family)
        points=support(ex);groups=fixed_groups(ex,'information_set');n=ex['n']
        endpoint_kl=objective=oracle_entropy=0.
        for y in points:
            visible={};logq=0.
            for k,group in enumerate(groups):
                p=tasks.oracle_marginals(ex['matrix_rows'],ex['syndrome'],visible,n)
                # Unequal round sampling probabilities must cancel with importance weights.
                rho=(.3,.7)[k]
                for i in group:
                    q=.13+.07*((i+sum(visible.values())+len(visible))%10)
                    ce=-(p[i]*math.log(q)+(1-p[i])*math.log1p(-q))
                    h=0. if p[i] in (0.,1.) else math.log(2)
                    objective+=rho*ce/(n*rho*len(points))
                    oracle_entropy+=rho*h/(n*rho*len(points))
                    logq+=math.log(q if y[i] else 1-q)
                visible.update({i:y[i] for i in group})
            endpoint_kl+=(-math.log(len(points))-logq)/len(points)
        assert abs(n*(objective-oracle_entropy)-endpoint_kl)<1e-12


def test_corruption_coverage_coefficients():
    n=8
    second_coefficient=(1/8)*(.5**n)*8/(4*n)
    first_coefficient=(1/8)*1*8/(8*n)
    assert second_coefficient==1/8192
    assert first_coefficient==1/64
    # Nonnegative remaining terms can only increase corruption excess.
    for first,second,remainder in ((0.,1.,0.),(2.,0.,0.),(.2,3.,.7)):
        corruption=first_coefficient*first+second_coefficient*second+remainder
        assert first+second<=8192*corruption
