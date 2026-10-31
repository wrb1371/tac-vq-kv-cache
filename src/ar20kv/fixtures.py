"""Load bounded original native layer fixtures for GPU implementation research."""
import json
import pathlib
import torch
from ar19kv.stable import HybridStablePacked
from ar19kv.packed_sparse import HybridPackedSparse
ROOT=pathlib.Path(__file__).resolve().parents[2]


def load_tables():
    cp=ROOT/'artifacts/ar17/packages/PSRVQ_d16/completion.json';meta=json.loads(cp.read_text());tables={}
    for rec in meta['records']:tables.update({k:v.cuda() for k,v in torch.load(cp.parent/rec['file'],map_location='cpu',weights_only=True).items()})
    cfg=json.loads((ROOT/'assets/round13/models/Qwen2.5-Omni-3B/config.json').read_text());theta=cfg['thinker_config']['text_config']['rope_theta']
    inv=(1/(theta**(torch.arange(0,128,2,dtype=torch.float32)/128))).cuda()
    return tables,meta,inv


def load_fixture(file,tables,meta):
    x=torch.load(file,map_location='cpu',weights_only=True)
    o=object.__new__(HybridStablePacked if x['policy']['gop']==2 else HybridPackedSparse);o.tables=[tables];o.package=meta;o.policy=x['policy'];o.plan=x['plan'];o._hybrid_dense=False
    for attr,key in [('_end','end'),('_nt','nt'),('_h','height'),('_w','width'),('_video_list','video_list'),('_text_list','text_list'),('_anchor_ord','anchor_ord')]:setattr(o,attr,x[key])
    o._vid_gpu=torch.tensor(x['video_list'],device='cuda',dtype=torch.int32);o._text_gpu=torch.tensor(x['text_list'],device='cuda',dtype=torch.int32)
    state={k:v.cuda() for k,v in x['state'].items()};state['n']=x['n'];state['text_filled']=len(x['text_list'])
    return x,o,state,x['positions'][:,:x['n']].cuda()
