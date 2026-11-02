"""Matched removals of local correspondence or decoded-anchor feedback.

The frozen planner/observer/reader are inherited without modification. Only the
first media encoding differs. Neither control changes persistent capacity.
"""
import os
import torch
from ar22kv.joint import JointTemporal
from ar22kv.matching import fixed_reference

DECODE_CHECKS=[]

class RepresentationControl(JointTemporal):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.intervention=self.policy['ar23_intervention']
        assert self.intervention in ['same_spatial_reference','original_anchor_feedback']
        self.metadata['ar23_intervention']=self.intervention
        self._decode_checked=set()

    @torch.no_grad()
    def append(self,layer,state,kpost,value,pre=None):
        if self._hybrid_dense or state['n']:
            return super().append(layer,state,kpost,value,pre)
        with self._layer(layer):
            assert pre is not None
            n=value.shape[-2];assert n==self._end
            table=self.tables[0];raw={'K':pre[0].float(),'V':value[0].float()}
            a,nr=self.plan['anchor_tokens'],self.plan['innovation_slots']
            aslots=torch.arange(a,device=value.device)
            aseq=self._vid_gpu.long()[torch.tensor(self._anchor_ord,device=value.device)]
            decoded={}
            for side in ['K','V']:
                self._innovation(layer,state,raw[side][:,aseq]-table[side+'_mean'][layer].float()[:,None],aslots,side)
                decoded[side]=torch.zeros((2,n,128),device=value.device,dtype=torch.bfloat16)
                decoded[side][:,aseq]=self._read_slots(layer,state,side,aslots[None].expand(2,-1),True)
            order=torch.arange(len(self._video_list),device=value.device)
            targets=order[(order//(self._h*self._w))%int(self.policy['gop'])!=0]
            seq=self._vid_gpu.long()[targets]
            if len(targets):
                if self.intervention=='same_spatial_reference':
                    codes=torch.full((2,len(targets)),4,device=value.device,dtype=torch.uint8)
                    refs=self._reference_indices(targets,codes)
                    scores=torch.zeros((2,len(targets)),device=value.device,dtype=torch.float32)
                    for side in ['K','V']:
                        pred=raw[side].gather(1,refs[...,None].expand(-1,-1,128))
                        scores+=((raw[side][:,seq]-pred)/table[side+'_scale'][layer].float()[:,None]).square().mean(-1)
                else:
                    importance=None
                    if self.policy.get('ranking','norm')=='sensitivity':
                        importance=[torch.tensor(self.conditional['ranking_importance'][s][layer],device=value.device,dtype=torch.float32) for s in ['K','V']]
                    codes,scores=fixed_reference(raw,self._vid_gpu,targets,[table['K_scale'][layer],table['V_scale'][layer]],self._h,self._w,int(self.policy['gop']),importance)
                    refs=self._reference_indices(targets,codes)
                state['T_code'][:,seq]=codes
                if nr:
                    picked=torch.argsort(scores,dim=1,descending=True,stable=True)[:,:nr]
                    picked=picked.sort(1).values;chosen,ref=seq[picked],refs.gather(1,picked)
                    heads=torch.arange(2,device=value.device)[:,None]
                    state['T_code'][heads,chosen]=codes.gather(1,picked)|128
                    slots=torch.arange(a,a+nr,device=value.device)
                    for side in ['K','V']:
                        current=raw[side].gather(1,chosen[...,None].expand(-1,-1,128))
                        source=raw[side] if self.intervention=='original_anchor_feedback' else decoded[side].float()
                        prediction=source.gather(1,ref[...,None].expand(-1,-1,128)).float()
                        self._innovation(layer,state,current-prediction,slots,side)
                state['temporal_targets']=len(targets);state['temporal_innovation_heads']=2*nr
                state['temporal_copy_heads']=2*(len(targets)-nr)
                state['distortion_raw_reference_score_sum']=float(scores.sum())
            text=self._text_gpu.long();state['text_FP_K'].copy_(kpost[0][:,text]);state['text_FP_V'].copy_(value[0][:,text])
            state['text_filled']=len(self._text_list);state['n']=n
            return state

    def decode(self,layer,state,positions,inv_freq,attention_scaling=1.):
        observed=super().decode(layer,state,positions,inv_freq,attention_scaling)
        if os.environ.get('AR23_INDEPENDENT_DECODE_GATE')=='1' and layer in [0,18,35] and layer not in self._decode_checked:
            from ar19kv.packed_sparse import HybridPackedSparse
            slow=object.__new__(HybridPackedSparse);slow.__dict__.update(self.__dict__)
            slow.plan={**self.plan,'innovation_slots':self.plan['layer_innovation_slots'][layer],'encoded_slots_per_head':self.plan['layer_encoded_slots_per_head'][layer]}
            expected=slow.decode(layer,state,positions,inv_freq,attention_scaling)
            diffs=[int((a!=b).sum()) for a,b in zip(expected,observed)];assert max(diffs)==0,(self.intervention,layer,diffs)
            DECODE_CHECKS.append({'intervention':self.intervention,'layer':layer,'BF16_differences':diffs});self._decode_checked.add(layer)
        return observed
