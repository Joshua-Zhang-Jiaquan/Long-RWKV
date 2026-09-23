from lrwkv_evidence.posterior_context_probe.tasks import serialize,partitions
from lrwkv_evidence.train04 import tasks as T
from lrwkv_evidence.train04.evaluate import support
SEED=20271001

def example(index):
    if not 0<=index<32:raise ValueError('outside frozen panel')
    ex=T.make_example('test',SEED,index)
    ex['support']=[sum(b<<i for i,b in enumerate(y)) for y in support(ex)]
    return ex
