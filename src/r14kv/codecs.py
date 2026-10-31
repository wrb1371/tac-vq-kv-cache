"""Complete parent payloads with an explicit unfused original-space reader.

NOVA's inverse K synthesis equals its query inverse-transpose in FP32. The
materialized BF16 K/V interface is disclosed and tested separately from the
author SGLang fused path. No R13 whole-vector unit-normalization is used.
"""
import pathlib,torch
from .common import ROOT,sha,apply_key_rope
from .fitting import nearest

def pack2(x):
    a=x.to(torch.uint8).reshape(*x.shape[:-1],32,4)
    return a[...,0]|(a[...,1]<<2)|(a[...,2]<<4)|(a[...,3]<<6)
def unpack2(x):
    return torch.stack([(x>>s)&3 for s in [0,2,4,6]],-1).flatten(-2)

class NOVA:
    method='NOVA-Omni-M0';read_backend='packed-reference-original-space-BF16-FlashAttention2'
    def __init__(self,directory,device='cuda'):
        import json
        self.directory=str(directory);meta=json.loads((pathlib.Path(directory)/'completion.json').read_text())
        assert meta['completed'] and meta['method']=='nova' and meta['full36_layer_artifact']
        self.tables=[]
        for l in range(36):
            rows=[]
            for h in range(2):
                p=pathlib.Path(directory)/f'layer{l:02}_head{h}.pt'
                rec=next(x for x in meta['records'] if x['layer']==l and x['head']==h);assert sha(p)==rec['SHA256']
                rows.append(torch.load(p,map_location='cpu',weights_only=True))
            self.tables.append({k:torch.stack([x[k] for x in rows]).to(device) for k in rows[0]})
        self.metadata={'calibration':meta['calibration'],'quantization_timing':'postfill-mcq','read_backend':self.read_backend,
            'default_FP_band':[64,256],'V_absolute_clip_ratio':.92,'codebook_dtype':'float8_e5m2','transform_dtype':'bfloat16',
            'rho_and_V_scale_zero_dtype':'float32, source serve_method default SCALE_DTYPE','FP16_codebook_mirror_persistent':False,
            'V_statistics':'true pre-W_O attention output second moment; differs from repository compute_sst proxy',
            'deployment':'source default scaled-e5m2 trainer roundtrip then direct runtime-e5m2 resnap',
            'basis_inverse_materialization':'F32 synthesis then BF16 native attention; not claimed SGLang fused numerical/throughput reproduction'}
    @torch.no_grad()
    def encode_low(self,l,k,v):
        t=self.tables[l];h,n,d=k.shape
        # Matches unfused author's BF16 centered transform before RMS.
        r=torch.bmm((k-t['mean'][:,None]).bfloat16(),t['F']).float()
        rho=r.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8)
        z=(r/rho).reshape(h,n,32,4)
        groups=z.permute(0,2,1,3).reshape(h*32,n,4)
        cb=t['centroids'].float().reshape(h*32,256,4)
        idx=nearest(groups,cb).reshape(h,32,n).transpose(1,2).contiguous().to(torch.uint8)
        vr=torch.bmm(v,t['V_rotation']).float()
        threshold=vr.abs().sort(-1).values[...,int(.92*128):int(.92*128)+1]
        clipped=torch.maximum(torch.minimum(vr,threshold),-threshold)
        low=clipped.amin(-1,keepdim=True);high=clipped.amax(-1,keepdim=True)
        scale=(high-low).clamp_min(1e-8)/3;zero=-low/scale
        q=(clipped/scale+zero+.5).to(torch.int32).clamp(0,3).to(torch.uint8)
        return {'K_index':idx,'rho':rho,'V_index':pack2(q),'V_scale':scale,'V_zero':zero}
    @torch.no_grad()
    def synthesize_low(self,l,low):
        t=self.tables[l];ix=low['K_index'].long();h,n,g=ix.shape
        rows=[]
        for head in range(h):
            cb=t['centroids'][head].float();group=torch.arange(g,device=cb.device)[None,:]
            rows.append(cb[group,ix[head]])
        r=torch.stack(rows).reshape(h,n,128)*low['rho']
        k=torch.bmm(r,t['A'].float().transpose(1,2))+t['mean'].float()[:,None]
        vr=(unpack2(low['V_index']).float()-low['V_zero'])*low['V_scale']
        v=torch.bmm(vr,t['V_rotation'].float().transpose(1,2))
        return k.bfloat16(),v.bfloat16()
    @torch.no_grad()
    def append(self,l,state,kpost,v,pre=None):
        # First64 and recent256 have unique BF16 storage, including overlaps.
        kpost=kpost[0];v=v[0];h,_,d=kpost.shape
        if state is None:state={'n':0,'firstK':kpost[:,:0].clone(),'firstV':v[:,:0].clone(),'recentK':kpost[:,:0].clone(),'recentV':v[:,:0].clone(),'low':None}
        nnew=kpost.shape[1];take=min(max(0,64-state['firstK'].shape[1]),nnew)
        if take:
            state['firstK']=torch.cat([state['firstK'],kpost[:,:take]],1);state['firstV']=torch.cat([state['firstV'],v[:,:take]],1)
        rk=torch.cat([state['recentK'],kpost[:,take:]],1);rv=torch.cat([state['recentV'],v[:,take:]],1)
        age=max(0,rk.shape[1]-256)
        if age:
            low=self.encode_low(l,rk[:,:age],rv[:,:age])
            if state['low'] is not None:low={key:torch.cat([state['low'][key],value],1) for key,value in low.items()}
            state['low']=low
        state['recentK']=rk[:,age:].contiguous().clone();state['recentV']=rv[:,age:].contiguous().clone();state['n']+=nnew
        return state
    @torch.no_grad()
    def decode(self,l,state,positions=None,inv_freq=None,attention_scaling=1.):
        ks=[state['firstK']];vs=[state['firstV']]
        if state['low'] is not None:
            k,v=self.synthesize_low(l,state['low']);ks.append(k);vs.append(v)
        ks.append(state['recentK']);vs.append(state['recentV'])
        return torch.cat(ks,1)[None],torch.cat(vs,1)[None]

class GSRQ:
    read_backend='packed-reference-original-space-BF16-FlashAttention2'
    def __init__(self,directory,stages=32,device='cuda'):
        import json
        self.stages=stages;self.method=f'GSRQ-Omni-M0-{2 if stages==32 else 1}'
        self.directory=str(directory);meta=json.loads((pathlib.Path(directory)/'completion.json').read_text())
        assert meta['completed'] and meta['method']=='gsrq' and meta['full36_layer_artifact'] and stages in [16,32]
        self.tables=[]
        for l in range(36):
            row={}
            for kind,key in enumerate(['K','V']):
                heads=[]
                for h in range(2):
                    p=pathlib.Path(directory)/f'layer{l:02}_{key}_head{h}_R{stages}.pt'
                    rec=next(x for x in meta['records'] if x['layer']==l and x['kind']==kind and x['head']==h)
                    assert sha(p)==rec[f'R{stages}_SHA256'];book=torch.load(p,map_location='cpu',weights_only=True)
                    assert book.shape==(stages,256,128) and book.dtype==torch.float16
                    heads.append(book)
                row[key]=torch.stack(heads).to(device)
            self.tables.append(row)
        self.metadata={'calibration':meta['calibration'],'quantization_timing':'postfill-mcq','read_backend':self.read_backend,
            'K_coordinates':'pre-native-MRoPE','index_BPA':stages*8/128,'stages':stages,'centroids':256,'dimension':128,
            'codebook_dtype':'float16','gradient_weighting':'log1p(u/(per-layer-head-kind median+1e-12)), fixed all stages',
            'token_norm_or_stage_gains_stored':False,'decode':'raw additive centroid sum, then native BF16 K MRoPE; BF16 materialization for FA2 V',
            'R16_independent_deployed_storage':True}
    @torch.no_grad()
    def encode(self,x,book):
        residual=x[0].float().clone();codes=[]
        for stage in range(self.stages):
            cb=book[:,stage].float();ix=nearest(residual,cb);codes.append(ix.to(torch.uint8))
            residual-=torch.gather(cb,1,ix[...,None].expand(-1,-1,128))
        return torch.stack(codes,-1).contiguous()
    @torch.no_grad()
    def raw_decode(self,index,book):
        h,n,r=index.shape;out=torch.zeros(h,n,128,device=index.device)
        for stage in range(r):
            ix=index[...,stage].long();cb=book[:,stage].float();out+=torch.gather(cb,1,ix[...,None].expand(-1,-1,128))
        return out
    @torch.no_grad()
    def append(self,l,state,kpost,v,pre=None):
        assert pre is not None and pre.shape==kpost.shape
        new={'K_index':self.encode(pre,self.tables[l]['K']),'V_index':self.encode(v,self.tables[l]['V'])}
        if state is not None:new={k:torch.cat([state[k],x],1) for k,x in new.items()}
        new['n']=new['K_index'].shape[1];return new
    @torch.no_grad()
    def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
        k=self.raw_decode(state['K_index'],self.tables[l]['K'])[None]
        v=self.raw_decode(state['V_index'],self.tables[l]['V'])[None]
        assert positions.shape==(3,state['n'])
        pos=positions.long();phase=pos[:,:,None].float()*inv_freq[None,None,:]
        cos=(phase.cos()*attention_scaling).bfloat16();sin=(phase.sin()*attention_scaling).bfloat16()
        widths=[16,24,24];sc=torch.cat([cos[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1)
        ss=torch.cat([sin[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1)
        sc=torch.cat([sc,sc],-1)[None,None];ss=torch.cat([ss,ss],-1)[None,None]
        return apply_key_rope(k.bfloat16(),sc,ss),v.bfloat16()
