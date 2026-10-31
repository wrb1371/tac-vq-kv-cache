"""One-time same-cost V-block permutation; fixed A1 layer quotas and codec."""
import hashlib,json,torch
from .nova import NOVAAllocated,description
from r15kv.persistent_registry import Arena
class NOVAReordered(NOVAAllocated):
    method='NOVA-R16-same-cost-content-or-hash-block-routing'
    @torch.no_grad()
    def reorder(self,l,state,v):
        mode=self.policy['routing'];assert mode in ['content','hash']
        if self.plan.get('legacy_fallback'):return
        original=list(self.plan['modes'][l]);sink=self.plan['sink'];recent=self.plan['recent']
        available=max(0,v.shape[-2]-sink-recent)//128
        available=min(available,len(original));upgrades=sum(x==2 for x in original[:available])
        if not 0<upgrades<available:return
        scores=[];gain=float(self.policy['layer_gains'][l])
        for j in range(available):
            if mode=='hash':score=int(hashlib.sha256(f'R16-fixed-hash-20261022/{l}/{j}'.encode()).hexdigest(),16)
            else:
                vr=self.rotate_v(l,v[0,:,sink+128*j:sink+128*(j+1)])
                threshold=vr.abs().sort(-1).values[...,int(.92*128):int(.92*128)+1];z=vr.clamp(-threshold,threshold);lo=z.amin(-1,keepdim=True);hi=z.amax(-1,keepdim=True)
                scale=(hi-lo).clamp_min(1e-8).bfloat16();zero=(-lo/scale.float()).bfloat16();q=(z/scale.float()+zero.float()).round().clamp(0,1)
                reconstructed=(q-zero.float())*scale.float();error=(vr-reconstructed).square().sum();assert torch.isfinite(error);score=float(error.item())*gain
            scores.append(score)
        chosen=set(sorted(range(available),key=lambda j:(-scores[j],j))[:upgrades]);modes=[2 if j in chosen else 1 for j in range(available)]+original[available:]
        # Equal-length completed blocks only. Tail and unknown future positions
        # retain their original precision, including their non128 token fee.
        old=self.plan['descriptions'][l];new=description(self.meta.capacity,sink,recent,self.groups,self.index_bits,modes);assert old==new and Arena.bytes(old)==state['arena'].tensor.numel()
        self.plan['modes'][l]=modes;offsets=[];counts={1:0,2:0};low=max(0,self.meta.capacity-sink-recent)
        for j,m in enumerate(modes):offsets.append(counts[m]);counts[m]+=min(128,low-j*128)
        state['V_block_mode'].copy_(torch.tensor(modes,device='cuda',dtype=torch.uint8));state['V_block_offset'].copy_(torch.tensor(offsets,device='cuda',dtype=torch.int32))
        self.plan.setdefault('routing_trace',[]).append({'layer':l,'routing':mode,'eligible_full_blocks':available,'upgraded_eligible_blocks':upgrades,'changed_blocks':sum(a!=b for a,b in zip(modes,original)),'unchanged_per_layer_V2_token_quota':True,'unchanged_all_registered_arena_bytes':True,'selected_blocks':sorted(chosen),'selected_blocks_SHA256':hashlib.sha256(json.dumps(sorted(chosen)).encode()).hexdigest(),'content_score_min':min(scores) if mode=='content' else None,'content_score_max':max(scores) if mode=='content' else None,'score':'actualV1 reconstruction residual energy × fixedCAL layer gain' if mode=='content' else 'deterministic global layer/block hash, no input content/answers','scope':'one-time routing before initial history encoding; no model-query logits, no gradients or fitted test parameters'})
        self.plan['no_current_video_semantic_score']=mode!='content'
    def append(self,l,state,kpost,v,pre=None):
        if state['n']==0:self.reorder(l,state,v)
        return super().append(l,state,kpost,v,pre)
