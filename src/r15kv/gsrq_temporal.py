"""G2-R1: V-only decoded-anchor prediction with a paid small residual dictionary."""
import json,pathlib,torch
from r14kv.common import apply_key_rope
from .codecs import GSRQBudget
from .budget import gs_split,estimate_resident_bytes
from .persistent_registry import Arena
from .temporal_mapping import video_mapping
class GSRQTemporalV(GSRQBudget):
    method='GSRQ-R15-G2-V-only-decoded-anchor-C128-R7'
    def __init__(self,package,budget_mib,**policy):
        super().__init__(package,budget_mib,**policy);self.parent=json.loads((pathlib.Path(policy['parent_package'])/'completion.json').read_text())
        assert self.tables[0]['V_residual'].shape==(2,7,128,128)
    def make_plan(self,meta):
        source=self.context['item_id'].rsplit('_',1)[0];mapping,identity=video_mapping(self.context['tags'],self.context['positions'],self.context['video_grid_thw'].cpu(),meta.capacity,source,bool(self.policy.get('hash_pairing',False)));self._mapping=mapping
        dtype=torch.int16 if max(meta.position_max)+max(0,meta.generation_cap-1)<=32767 else torch.int32;posbytes=3*meta.capacity*torch.empty((),dtype=dtype).element_size()+256
        total=min(self.parent['Rmax_K']+self.parent['Rmax_V'],(self.budget_bytes-self.parent['static_tensor_bytes']-posbytes)//(72*meta.capacity));rk,rv=gs_split(total,self.parent['Rmax_K'],self.parent['Rmax_V'],0)
        na=identity['anchor_tokens'];npred=identity['predicted_tokens'];nnon=identity['nonvideo_capacity'];desc=[('K_index',(2,meta.capacity,rk),torch.uint8,'payload'),('V_nonvideo_index',(2,nnon,rv),torch.uint8,'payload'),('V_anchor_index',(2,na,12),torch.uint8,'payload'),('V_residual_index',(2,npred,7),torch.uint8,'payload')]
        routes=[(k,tuple(v.shape),v.dtype,'metadata') for k,v in mapping.items()];self._route_description=routes;est=estimate_resident_bytes(self.static_bytes,posbytes+Arena.bytes(routes),desc)
        if est['resident_capacity_bytes']>self.budget_bytes:raise RuntimeError('NOT_FEASIBLE_BY_INTEGER_ORACLE G2_R1 '+str(est))
        return {'R_K':rk,'R_nonvideo_V':rv,'R_anchor_V':12,'R_residual_V':7,'residual_centroids':128,'residual_index_storage_bits':8,'small_dictionary_only_saves_static_not7bit_payload':True,'description':desc,'estimate':est,'position_dtype':str(dtype),'capacity':meta.capacity,'video_mapping':identity,'K_unchanged_from_parent':True,'audio_text_V_unchanged_from_parent':True,'decoded_anchor_persistent':False,'causal4096_supported':True,'routing_features':'known processor grid and earlier temporal anchor only; no questionQ or future KV values'}
    def prepare_request(self,meta,registry):
        plan=super().prepare_request(meta,registry);self.routing=Arena(self._route_description,'cuda',registry,'cache/G2_real_spatiotemporal_mapping');self.routes=self.routing.views
        for key,value in self._mapping.items():self.routes[key].copy_(value)
        self._mapping=None;return plan
    def allocate_state(self,l,registry):
        s=super().allocate_state(l,registry);s.update(nonvideo_seen=0,anchor_seen=0,pred_seen=0);return s
    @torch.no_grad()
    def append(self,l,state,kpost,v,pre=None):
        assert pre is not None;start=state['n'];n=v.shape[-2];self.encode_into(pre[0],self.tables[l]['K'],state['K_index'][:,start:start+n]);kind=self.routes['kind'][start:start+n];raw=v[0]
        non=(kind==0).nonzero(as_tuple=True)[0];anc=(kind==1).nonzero(as_tuple=True)[0];pred=(kind==2).nonzero(as_tuple=True)[0]
        if len(non):
            b=state['nonvideo_seen'];self.encode_into(raw.index_select(1,non),self.tables[l]['V'],state['V_nonvideo_index'][:,b:b+len(non)]);state['nonvideo_seen']+=len(non)
        if len(anc):
            b=state['anchor_seen'];self.encode_into(raw.index_select(1,anc),self.tables[l]['V'][:,:12],state['V_anchor_index'][:,b:b+len(anc)]);state['anchor_seen']+=len(anc)
        if len(pred):
            b=state['pred_seen'];refs=self.routes['anchor_ref'][b:b+len(pred)].long();assert refs.max()<state['anchor_seen'],'no future or unavailable anchor'
            anchor=self.raw_decode(state['V_anchor_index'][:,:state['anchor_seen']],self.tables[l]['V'][:,:12]);residual=raw.index_select(1,pred).float()-anchor.index_select(1,refs)
            self.encode_into(residual,self.tables[l]['V_residual'],state['V_residual_index'][:,b:b+len(pred)]);state['pred_seen']+=len(pred)
        state['n']+=n;assert state['n']==state['nonvideo_seen']+state['anchor_seen']+state['pred_seen'];return state
    @torch.no_grad()
    def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
        n=state['n'];kr=self.raw_decode(state['K_index'][:,:n],self.tables[l]['K']);vr=torch.empty(2,n,128,device='cuda');nn=state['nonvideo_seen'];na=state['anchor_seen'];npred=state['pred_seen']
        if nn:vr[:,self.routes['nonvideo_native'][:nn].long()]=self.raw_decode(state['V_nonvideo_index'][:,:nn],self.tables[l]['V'])
        anchor=self.raw_decode(state['V_anchor_index'][:,:na],self.tables[l]['V'][:,:12])
        if na:vr[:,self.routes['anchor_native'][:na].long()]=anchor
        if npred:vr[:,self.routes['pred_native'][:npred].long()]=anchor.index_select(1,self.routes['anchor_ref'][:npred].long())+self.raw_decode(state['V_residual_index'][:,:npred],self.tables[l]['V_residual'])
        phase=positions.float()[:,:,None]*inv_freq[None,None,:];cos=(phase.cos()*attention_scaling).bfloat16();sin=(phase.sin()*attention_scaling).bfloat16();widths=[16,24,24];sc=torch.cat([cos[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1);ss=torch.cat([sin[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1)
        return apply_key_rope(kr[None].bfloat16(),torch.cat([sc,sc],-1)[None,None],torch.cat([ss,ss],-1)[None,None]),vr[None].bfloat16()
    def extra_receipt(self,states):
        actual=self.registry.observe('G2_receipt');used=sum(2*(s['n']*self.plan['R_K']+s['nonvideo_seen']*self.plan['R_nonvideo_V']+s['anchor_seen']*12+s['pred_seen']*7) for s in states.values());allocated=sum(s['arena'].payload_bytes for s in states.values())
        return {'cache_payload_bytes':used,'cache_metadata_and_capacity_bytes':actual['resident_bytes']-actual['codec_static_bytes']-used,'allocated_index_capacity_bytes':allocated,'unused_index_capacity_bytes':allocated-used,'per_layer_V_nonvideo_tokens':{str(l):s['nonvideo_seen'] for l,s in states.items()},'per_layer_V_anchor_tokens':{str(l):s['anchor_seen'] for l,s in states.items()},'per_layer_V_predicted_tokens':{str(l):s['pred_seen'] for l,s in states.items()},'persistent_decoded_anchor_bytes':0,'residual_codebook_C':128,'actual_residual_index_dtype':'uint8,8bits','extra_model_passes':0}
    def finish_request(self):
        self.routing=None;self.routes=None;self._mapping=None;super().finish_request()
