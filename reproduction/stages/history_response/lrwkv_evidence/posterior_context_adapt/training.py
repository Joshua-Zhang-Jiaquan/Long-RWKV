from lrwkv_evidence.train04 import tasks
from lrwkv_evidence.train04mix import core as M
from lrwkv_evidence.train04paired import core as P
from lrwkv_evidence.train04binding import core as B
from lrwkv_evidence.posterior_context_probe.tasks import serialize
RECIPE=dict(version=1,seed=20270930,terminal_step=200,world_size=8,global_batch=32,
            context_mixture='25percent evidence-only;75percent1K; balanced far/middle/near',
            objective='original independent paired information-set histories, exact soft CE',
            learning_rate=1e-5,warmup=20,weight_decay=.01,clip=1.,
            precision='FP32;torch TF32 off;Triton IEEE',
            scope='one bounded development context adaptation;preserve known task and target serialization')

def records(step,rank,tok):
    if not 1<=step<=200 or not 0<=rank<8:raise ValueError('outside recipe')
    result=[];length=None if (step-1+rank)%4==0 else 1024
    position=('far','middle','near')[((step-1)//4+rank)%3]
    for j in range(2):
        index=1000000+(step-1)*16+rank*2+j
        ex=tasks.make_example('train',RECIPE['seed'],index)
        encoded=serialize(ex,tok,length,'evidence_only' if length is None else position)
        round_index=P.randomizer('context_adapt_round',RECIPE['seed'],ex['instance_id']).randrange(2)
        for member in P.history_pair(ex,RECIPE['seed'],'independent'):
            c=M.canvas(member,tok,RECIPE['seed'],branch='policy',round_index=round_index)
            c['ids']=encoded['ids']+c['ids'][c['prefix']:];c['prefix']=len(encoded['ids'])
            result.append(B.decorate(member,c,tok))
    return result

def batch(records,device):return B.pack(records,device)
def loss(logits,batch):return M.loss(logits,batch,'rao_blackwell')
