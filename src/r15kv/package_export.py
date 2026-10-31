"""Export independent deployment tensors on CPU before touching the GPU."""
import json,pathlib,time,torch
from r14kv.common import sha,json_save
def export_nova(source,dest,label):
    source=pathlib.Path(source);dest=pathlib.Path(dest);dest.mkdir(parents=True,exist_ok=False)
    parent=json.loads((source/'completion.json').read_text());assert parent['completed'];records=[]
    for l in range(36):
        heads=[]
        for h in range(2):
            p=source/f'layer{l:02}_head{h}.pt';r=next(x for x in parent['records'] if x['layer']==l and x['head']==h);assert sha(p)==r['SHA256'];heads.append(torch.load(p,map_location='cpu',weights_only=True))
        row={k:torch.stack([h[k] for h in heads]).contiguous().clone() for k in heads[0]};path=dest/f'layer{l:02}.pt';torch.save(row,path)
        records.append({'layer':l,'file':path.name,'SHA256':sha(path),'tensors':{k:{'shape':list(t.shape),'dtype':str(t.dtype),'storage_bytes':t.untyped_storage().nbytes()} for k,t in row.items()}})
    record={'completed':True,'family':'NOVA','label':label,'CPU_export':True,'parent_completion_SHA256':sha(source/'completion.json'),'source':str(source),'records':records,'static_tensor_bytes':sum(t['storage_bytes'] for r in records for t in r['tensors'].values()),'epoch':time.time()};json_save(dest/'completion.json',record);return record
def export_gsrq(source,dest,rk,rv,aware=False):
    source=pathlib.Path(source);dest=pathlib.Path(dest);dest.mkdir(parents=True,exist_ok=False)
    parent=json.loads((source/'completion.json').read_text());assert parent['completed'];records=[]
    for l in range(36):
        row={}
        for kind,r in [('K',rk),('V',rv)]:
            heads=[]
            for h in range(2):
                p=source/(f'layer{l:02}_{kind}_head{h}.pt' if aware else f'layer{l:02}_{kind}_head{h}_R16.pt')
                rec=next(x for x in parent['records'] if x['layer']==l and x['kind']==(kind=='V') and x['head']==h)
                assert sha(p)==rec['SHA256' if aware else 'R16_SHA256']
                t=torch.load(p,map_location='cpu',weights_only=True);head=t[:r].to(torch.float8_e4m3fn).contiguous().clone()
                assert head.untyped_storage().nbytes()==r*256*128 and torch.isfinite(head.float()).all();heads.append(head)
            row[kind]=torch.stack(heads).contiguous().clone()
        path=dest/f'layer{l:02}.pt';torch.save(row,path);records.append({'layer':l,'file':path.name,'SHA256':sha(path),'tensors':{k:{'shape':list(t.shape),'dtype':str(t.dtype),'storage_bytes':t.untyped_storage().nbytes()} for k,t in row.items()}})
    record={'completed':True,'family':'GSRQ','CPU_export':True,'source':str(source),'parent_completion_SHA256':sha(source/'completion.json'),'Rmax_K':rk,'Rmax_V':rv,'FP8_aware_fit':aware,'records':records,'static_tensor_bytes':sum(t['storage_bytes'] for r in records for t in r['tensors'].values()),'no_parent_storage_or_high_precision_mirror':True,'epoch':time.time()};json_save(dest/'completion.json',record);return record
