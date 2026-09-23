"""Scientific checks against exact affine-uniform oracle, independent of neural runtime."""
import math
from lrwkv_evidence.train04 import evaluate as E, tasks

class Oracle:
    def __init__(self): self.calls=0
    def __call__(self,ex,visible,stage):
        self.calls+=1
        probabilities=tasks.oracle_marginals(ex['matrix_rows'],ex['syndrome'],visible,ex['n'])
        # After an irreversible inconsistent reveal the true conditional is undefined.
        # Fair bits fill remaining positions; no extension can restore validity.
        return probabilities if probabilities is not None else [.5]*ex['n']

def test_exact_joint_oracle_and_matched_calls():
    for family in tasks.FAMILIES:
        ex=tasks.make_example('test',73191,0,family=family)
        assert len(E.support(ex))==16
        assert all(E.valid(ex,bits) for bits in E.support(ex))
        for method in ('information_set','sequential'):
            assert abs(E.exact_joint_kl(ex,method,Oracle()))<1e-12
        expected=0 if family=='systematic' else 4*math.log(2)
        assert abs(E.exact_joint_kl(ex,'one',Oracle())-expected)<1e-12
        first=set(E.fixed_groups(ex,'random_halves')[0])
        within=sum(all(i in first for i in range(8) if row>>i&1) or all(i not in first for i in range(8) if row>>i&1) for row in ex['matrix_rows']) if family=='paired_parity' else 0
        assert abs(E.exact_joint_kl(ex,'random_halves',Oracle())-within*math.log(2))<1e-12
        for method in ('information_set','random_halves'):
            _,calls=E.sample(ex,method,Oracle(),0)
            assert calls==2

def test_oracle_sampling_validity_and_calibration():
    ex=tasks.make_example('test',73191,1,family='paired_parity')
    for method,expectation in [('one',1/16),('information_set',1.),('sequential',1.),('bernoulli2',(1-1/4)**4),('bernoulli8',(1-1/16)**4)]:
        valid=sum(E.valid(ex,E.sample(ex,method,Oracle(),i)[0]) for i in range(512))/512
        assert abs(valid-expectation)<.06,(method,valid,expectation)
    for row in E.calibration(ex,Oracle()):
        assert row['marginal_kl_sum_nats']<1e-5
        assert row['deterministic_correct']==row['deterministic']

def test_confident_wrong_logits_are_not_censored():
    class ConfidentWrong:
        def log_probs(self,ex,visible,stage):
            return [[-2000.,0.] if b==0 else [0.,-2000.] for b in ex['bits']]
    ex=tasks.make_example('test',73191,0,family='systematic')
    rows=E.calibration(ex,ConfidentWrong())
    assert rows[-1]['marginal_kl_sum_nats']>10000

def test_no_competence_means_no_schedule_improvement():
    class Fair:
        calls=0
        def __call__(self,ex,visible,stage): self.calls+=1; return [.5]*ex['n']
    ex=tasks.make_example('test',73191,1,family='paired_parity')
    for method in ('one','information_set','random_halves','sequential'):
        assert abs(E.exact_joint_kl(ex,method,Fair())-4*math.log(2))<1e-12
