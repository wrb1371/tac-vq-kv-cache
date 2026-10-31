"""Decode-only residual tail interventions, with full R16 indices still paid.

This is diagnostic, not a compressed variable-stage production candidate.
All policies keep TEXT at16. AV tail restoration isolates modality from simple
text protection, and hash/norm0 controls match each128-token block's audio count.
"""
import hashlib,json,random
import torch
from .engineering_codecs import GSRQCompactFP8
from .common import apply_key_rope

class GSRQStageIntervention(GSRQCompactFP8):
    def __init__(self,directory,side='K',policy='none',low_stages=12,device='cuda'):
        super().__init__(directory,stages=16,device=device)
        assert side in ['K','V'] and policy in ['all','none','audio','video','hash','norm0'] and low_stages in [8,12]
        self.side=side;self.policy=policy;self.low=low_stages;self.tags=None;self.keep=None;self.request_index=-1
        self.method=f'GSRQ-D2-{side}-{low_stages}to16-{policy}'
        self.metadata={**self.metadata,'intervention_only':True,'production_candidate':False,'no_storage_saving_claim':True,
            'changed_side':side,'decode_stages_AV_base':low_stages,'decode_stages_selected_tail':16,'policy':policy,
            'text_stages_always16':True,'stored_indices':'all16 K and V codes retained, ignored tail still stored and charged',
            'routing':'one shared uint8 mask plus CPU input tags charged through state0; norm0 uses only first-layer raw K norm',
            'hash_norm_matching':'same selected AV count as audio in every128 original sequence positions; text fixed',
            'scope':'postfill diagnostic; full AV prefill raw K available at initial write; current final prompt token is text'}
    def set_context(self,tags):
        self.tags=tags.detach().to(device='cpu',dtype=torch.uint8).contiguous();self.keep=None;self.request_index+=1
    @torch.no_grad()
    def make_mask(self,pre):
        tags=self.tags;N=len(tags);text=tags==0;mask=text.clone()
        if self.policy=='all':mask[:]=True
        elif self.policy=='audio':mask|=tags==1
        elif self.policy=='video':mask|=tags==2
        elif self.policy in ['hash','norm0']:
            scores=None
            if self.policy=='norm0':
                assert pre is not None and pre.shape[-2]>=N-1 and tags[-1]==0
                scores=pre[0].float().square().sum(-1).mean(0).cpu()
            for start in range(0,N,128):
                end=min(N,start+128);positions=torch.arange(start,end);av=positions[tags[start:end]!=0];count=int((tags[start:end]==1).sum())
                if not count:continue
                if self.policy=='hash':
                    candidates=av.tolist();random.Random(14261021+N*1000003+start).shuffle(candidates);chosen=torch.tensor(candidates[:count])
                else:
                    assert int(av.max())<len(scores);chosen=av[torch.argsort(scores[av],descending=True,stable=True)[:count]]
                mask[chosen]=True
            assert all(int((mask[s:s+128]&(~text[s:s+128])).sum())==int((tags[s:s+128]==1).sum()) for s in range(0,N,128))
        print(json.dumps({'stage_intervention_route':self.policy,'side':self.side,'request_index':self.request_index,'tokens':N,
            'mask_SHA256':hashlib.sha256(mask.numpy().tobytes()).hexdigest(),'retained_audio':int((mask&(tags==1)).sum()),
            'retained_video':int((mask&(tags==2)).sum()),'all_text_retained':bool(mask[text].all()),'no_gold_or_outputs_used':True}),flush=True)
        return mask.to(dtype=torch.uint8,device=pre.device)
    @torch.no_grad()
    def append(self,l,state,kpost,v,pre=None):
        if l==0 and self.keep is None:
            assert self.tags is not None and state is None;self.keep=self.make_mask(pre)
        state=super().append(l,state,kpost,v,pre);assert state['n']<=len(self.keep)
        state['tail_keep_mask']=self.keep[:state['n']]
        if l==0:state['policy_context_CPU']=self.tags
        return state
    @torch.no_grad()
    def masked_raw(self,index,book,keep):
        h,n,r=index.shape;out=torch.zeros(h,n,128,device=index.device)
        for stage in range(r):
            ix=index[...,stage].long();cb=book[:,stage].float();chosen=torch.gather(cb,1,ix[...,None].expand(-1,-1,128))
            if stage>=self.low:chosen=chosen*keep[None,:,None].float()
            out+=chosen
        return out
    @torch.no_grad()
    def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
        raw={}
        for kind in ['K','V']:
            ix=state[kind+'_index'];book=self.tables[l][kind]
            raw[kind]=self.masked_raw(ix,book,state['tail_keep_mask']) if kind==self.side else self.raw_decode(ix,book)
        pos=positions.long();assert pos.shape==(3,state['n']);phase=pos[:,:,None].float()*inv_freq[None,None,:]
        cos=(phase.cos()*attention_scaling).bfloat16();sin=(phase.sin()*attention_scaling).bfloat16();widths=[16,24,24]
        sc=torch.cat([cos[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1);ss=torch.cat([sin[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1)
        sc=torch.cat([sc,sc],-1)[None,None];ss=torch.cat([ss,ss],-1)[None,None]
        return apply_key_rope(raw['K'][None].bfloat16(),sc,ss),raw['V'][None].bfloat16()
