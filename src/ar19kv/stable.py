"""Preserve mapped arithmetic schedule; store no persistent compact-slot map."""
import bisect,torch
from .refinement import RateMatchedPacked
from ar18kv.hybrid import DenseFirst
from r14kv.common import apply_key_rope

class StablePacked(RateMatchedPacked):
    method="AR19-stable-map-free-temporal-RVQ"
    @torch.no_grad()
    def append(self,l,state,kpost,v,pre=None):
        assert pre is not None;start=state['n'];n=v.shape[-2];t=self.tables[0];raw={'K':pre[0].float(),'V':v[0].float()};heads=torch.arange(2,device='cuda')[:,None]
        a=self.plan['anchor_tokens'];nr=self.plan['innovation_slots'];aslots=torch.arange(a,device='cuda')
        if start==0:
            assert n==self._end
            anchor_ord=torch.tensor(self._anchor_ord,device='cuda');aseq=self._vid_gpu.long()[anchor_ord]
            for kind in ['K','V']:
                self._innovation(l,state,raw[kind][:,aseq]-t[kind+'_mean'][l].float()[:,None],aslots,kind)
            if self.policy.get('mode')!='compact_independent':
                p=self._h*self._w;gop=int(self.policy.get('gop',2));order=torch.arange(len(self._video_list),device='cuda');targets=order[(order//p)%gop!=0];seq=self._vid_gpu.long()[targets]
                # Dense scratch covers ONE layer, allocated/freed within append.
                decoded={}
                for kind in ['K','V']:
                    z=torch.zeros(2,n,128,device='cuda',dtype=torch.bfloat16);z[:,aseq]=self._read_slots(l,state,kind,aslots[None].expand(2,-1),True);decoded[kind]=z
                best=torch.full((2,len(targets)),float('inf'),device='cuda');codes=torch.full_like(best,4,dtype=torch.uint8)
                for c in range(9):
                    refs=self._reference_indices(targets,torch.full_like(codes,c));score=0
                    for kind in ['K','V']:
                        pred=decoded[kind].gather(1,refs[...,None].expand(-1,-1,128)).float();score=score+((raw[kind][:,seq]-pred)/t[kind+'_scale'][l].float()[:,None]).square().mean(-1)
                    take=score<best;codes=torch.where(take,torch.full_like(codes,c),codes);best=torch.minimum(best,score)
                refs=self._reference_indices(targets,codes);state['T_code'][:,seq]=codes
                if nr:
                    picked=torch.argsort(best,dim=1,descending=True,stable=True)[:,:nr];chosen=seq[picked];selectedrefs=refs.gather(1,picked);slots=torch.arange(a,a+nr,device='cuda')
                    state['T_code'][heads,chosen]=codes.gather(1,picked)|128
                    for kind in ['K','V']:
                        current=raw[kind].gather(1,chosen[...,None].expand(-1,-1,128));pred=decoded[kind].gather(1,selectedrefs[...,None].expand(-1,-1,128)).float()
                        self._innovation(l,state,current-pred,slots,kind)
                    # Preserve legacy floating-point quantization order, then
                    # reorder only stored integer indices / BF16 norms.
                    order=torch.argsort(picked,dim=1,stable=True)
                    for kind in ['K','V']:
                        ix=state[kind+'_index'][:,a:a+nr]
                        norm=state[kind+'_norm'][:,a:a+nr]
                        ix.copy_(ix.gather(1,order[...,None,None].expand_as(ix)))
                        norm.copy_(norm.gather(1,order[...,None].expand_as(norm)))
                state['temporal_targets']=len(targets);state['temporal_innovation_heads']=2*nr;state['temporal_copy_heads']=2*(len(targets)-nr);state['distortion_copy_before_innovation_sum']=float(best.sum())
            text=self._text_gpu.long();state['text_FP_K'].copy_(kpost[0][:,text]);state['text_FP_V'].copy_(v[0][:,text]);state['text_filled']=len(self._text_list)
        else:
            qs=start-self._end;slots=torch.arange(a+nr+qs,a+nr+qs+n,device='cuda');assert int(slots[-1])<self.plan['encoded_slots_per_head']
            for kind in ['K','V']:self._innovation(l,state,raw[kind]-t[kind+'_mean'][l].float()[:,None],slots,kind)
        state['n']+=n;return state
    def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
        n=state['n'];a=self.plan['anchor_tokens'];aslots=torch.arange(a,device='cuda')[None].expand(2,-1);anchor_ord=torch.tensor(self._anchor_ord,device='cuda');aseq=self._vid_gpu.long()[anchor_ord];out=[]
        order=torch.arange(len(self._video_list),device='cuda');p=self._h*self._w;gop=int(self.policy.get('gop',2));targets=order[(order//p)%gop!=0];seq=self._vid_gpu.long()[targets]
        for kind in ['K','V']:
            z=torch.zeros(2,n,128,device='cuda',dtype=torch.bfloat16);z[:,aseq]=self._read_slots(l,state,kind,aslots,True)
            if self.policy.get('mode')!='compact_independent' and len(targets):
                codes=state['T_code'][:,seq];refs=self._reference_indices(targets,codes&127);pred=z.gather(1,refs[...,None].expand(-1,-1,128)).float()
                active=(codes&128)!=0;rank=active.long().cumsum(1)-1;slot=torch.where(active,rank+a,0)
                residual=self._read_slots(l,state,kind,slot.clamp_min(0),False).float();z[:,seq]=(pred+torch.where(active[...,None],residual,0)).bfloat16()
            if n>self._end:
                slot=torch.arange(a+self.plan['innovation_slots'],a+self.plan['innovation_slots']+n-self._end,device='cuda')[None].expand(2,-1);z[:,self._end:n]=self._read_slots(l,state,kind,slot,True)
            out.append(z[None])
        phase=positions.float()[:,:,None]*inv_freq[None,None,:];trigs=[];widths=[16,24,24]
        for trig in [torch.cos,torch.sin]:
            aa=(trig(phase)*attention_scaling).bfloat16();bb=torch.cat([aa[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1);trigs.append(torch.cat([bb,bb],-1)[None,None])
        k=apply_key_rope(out[0],*trigs);v=out[1];m=bisect.bisect_left(self._text_list,n);assert m==state['text_filled']
        if m:
            ix=self._text_gpu[:m].long();k[0].index_copy_(1,ix,state['text_FP_K'][:,:m]);v[0].index_copy_(1,ix,state['text_FP_V'][:,:m])
        return k,v

class HybridStablePacked(DenseFirst,StablePacked):pass
