"""N1: eagerly sealed local per-channel V in the frozen NOVA V basis."""
import collections,hashlib,math,torch
from .codecs import NOVABudget
from .budget import nova_description,estimate_resident_bytes,pack_indices,unpack_indices
from .persistent_registry import Arena
class NOVALocalV(NOVABudget):
    method='NOVA-R15-N1-local-channel-V'
    def window_groups(self,meta,sink,recent):
        n=max(0,meta.prompt_tokens-1-sink-recent);w=int(self.policy.get('window',256));mode=self.policy.get('grouping','continuous')
        tags=self.context['tags'][sink:sink+n].cpu();ix=torch.arange(n,dtype=torch.long)
        if mode=='continuous':groups=[ix]
        elif mode in ['modality','hash']:
            groups=[ix[tags==m] for m in [0,1,2,3] if (tags==m).any()]
            if mode=='hash':
                gen=torch.Generator().manual_seed(int(hashlib.sha256(('R15-N1:'+meta.input_sha256).encode()).hexdigest()[:15],16));perm=ix[torch.randperm(n,generator=gen)];offset=0;shuffled=[]
                for g in groups:shuffled.append(perm[offset:offset+len(g)].sort().values);offset+=len(g)
                groups=shuffled
        else:raise ValueError(mode)
        return [g[b:b+w] for g in groups for b in range(0,len(g),w)],n
    def make_plan(self,meta):
        for bits in [2,1]:
            for sink,recent in self.policy.get('bands',[[64,256],[32,128]]):
                windows,ninitial=self.window_groups(meta,sink,recent);cap=max(0,meta.capacity-sink-recent);nw=len(windows)+max(0,cap-ninitial)
                desc=[x for x in nova_description(meta.capacity,sink,recent,self.groups,self.index_bits,bits) if x[0] not in ['V_scale','V_zero']]
                desc += [('channel_scale',(2,nw,128),torch.bfloat16,'metadata'),('channel_offset',(2,nw,128),torch.bfloat16,'metadata')]
                routing=[('window_id',(cap,),torch.int16,'metadata'),('token_order',(cap,),torch.int32,'metadata'),('window_offset',(nw+1,),torch.int32,'metadata')]
                migration=2*2*min(recent,max(0,meta.capacity-sink))*128*2;est=estimate_resident_bytes(self.static_bytes,migration+Arena.bytes(routing),desc)
                if est['resident_capacity_bytes']<=self.budget_bytes:
                    self._windows=windows;self._ninitial=ninitial;self._routing_desc=routing
                    return {'v_mode':'local_channel_affine'+str(bits),'v_bits':bits,'v1_quantizer':self.policy.get('v1_quantizer','minmax'),'sink':sink,'recent':recent,'description':desc,'estimate':est,'capacity':meta.capacity,'window':int(self.policy.get('window',256)),'grouping':self.policy.get('grouping','continuous'),'initial_low_tokens':ninitial,'initial_windows':len(windows),'max_windows':nw,'append_rule':'each migrated token seals immediately as an independent short window; no free FP pending and no repeated re-quantization of old V','causal4096_supported':False,'migration_bytes':migration}
        raise RuntimeError('NOT_FEASIBLE_BY_INTEGER_ORACLE NOVA_N1')
    def prepare_request(self,meta,registry):
        plan=super().prepare_request(meta,registry);self.routing=Arena(self._routing_desc,'cuda',registry,'cache/N1_shared_routing');self.routes=self.routing.views
        order=[];wid=[];offset=[0]
        for i,g in enumerate(self._windows):order+=g.tolist();offset.append(len(order))
        for j in range(plan['initial_low_tokens'],self.routes['window_id'].numel()):order.append(j);offset.append(len(order))
        if order:
            self.routes['token_order'].copy_(torch.tensor(order,dtype=torch.int32));self.routes['window_offset'].copy_(torch.tensor(offset,dtype=torch.int32))
            for i in range(len(offset)-1):self.routes['window_id'][self.routes['token_order'][offset[i]:offset[i+1]].long()]=i
        else:self.routes['window_offset'].zero_()
        return plan
    def allocate_state(self,l,registry):
        state=super().allocate_state(l,registry);state['sealed_windows']=0;return state
    @torch.no_grad()
    def encode_window_batch(self,l,state,v,window_ids,start):
        ids=torch.tensor(window_ids,device='cuda',dtype=torch.long);offsets=self.routes['window_offset'];starts=offsets.index_select(0,ids).long();length=int(offsets[window_ids[0]+1]-offsets[window_ids[0]])
        order_pos=(starts[:,None]+torch.arange(length,device='cuda')[None,:]).reshape(-1);native=self.routes['token_order'].index_select(0,order_pos).long()
        source=native-start;assert source.min()>=0 and source.max()<v.shape[1]
        vr=self.rotate_v(l,v.index_select(1,source),start).reshape(2,len(ids),length,128)
        lo=vr.amin(2,keepdim=True);hi=vr.amax(2,keepdim=True);scale=((hi-lo).clamp_min(1e-8)/(2**self.plan['v_bits']-1)).bfloat16();offset=lo.bfloat16()
        if self.plan['v_bits']==1 and self.policy.get('v1_quantizer')=='lloyd2':
            # One explicit repair of observed minmax-V1 bias. Fit two scalar
            # levels locally, snapping actual BF16 offset/scale before each assignment.
            low=lo;high=hi
            for _ in range(4):
                scale=(high-low).clamp_min(1e-8).bfloat16();offset=low.bfloat16();q=(vr>offset.float()+.5*scale.float());count=q.sum(2,keepdim=True);other=length-count
                low=torch.where(other>0,(vr*(~q)).sum(2,keepdim=True)/other.clamp_min(1),offset.float());high=torch.where(count>0,(vr*q).sum(2,keepdim=True)/count.clamp_min(1),offset.float()+scale.float())
            scale=(high-low).clamp_min(1e-8).bfloat16();offset=low.bfloat16();q=(vr>offset.float()+.5*scale.float()).to(torch.uint8)
        else:q=((vr-offset.float())/scale.float()).round().clamp(0,2**self.plan['v_bits']-1).to(torch.uint8)
        state['channel_scale'][:,ids]=scale.squeeze(2);state['channel_offset'][:,ids]=offset.squeeze(2);state['V_index'][:,native]=pack_indices(q,self.plan['v_bits']).reshape(2,-1,128*self.plan['v_bits']//8)
    @torch.no_grad()
    def write_values(self,l,state,v,start):
        if start==0 and self._ninitial>0:
            assert v.shape[1]==self._ninitial,'N1 initial complete postfill prefix required; causal4096 uses a separate ordinary parent'
            buckets=collections.defaultdict(list)
            for i,g in enumerate(self._windows):buckets[len(g)].append(i)
            for group in buckets.values():
                for b in range(0,len(group),16):self.encode_window_batch(l,state,v,group[b:b+16],start)
            state['sealed_windows']=len(self._windows)
        else:
            assert start>=self._ninitial and v.shape[1]==1,'append eagerly seals each aged token'
            wid=len(self._windows)+start-self._ninitial;self.encode_window_batch(l,state,v,[wid],start);state['sealed_windows']+=1
    @torch.no_grad()
    def synthesize_low(self,l,state):
        n=state['nlow'];t=self.tables[l];ix=unpack_indices(state['K_index'][:,:n],self.index_bits,self.groups).long();rows=[]
        for h in range(2):rows.append(t['centroids'][h].float()[torch.arange(self.groups,device='cuda')[None,:],ix[h]])
        r=torch.stack(rows).reshape(2,n,128)*state['rho'][:,:n].float();k=torch.bmm(r,t['A'].float().transpose(1,2))+t['mean'].float()[:,None]
        wid=self.routes['window_id'][:n].long();scale=state['channel_scale'].index_select(1,wid).float();offset=state['channel_offset'].index_select(1,wid).float();q=unpack_indices(state['V_index'][:,:n],self.plan['v_bits'],128).float();vr=q*scale+offset
        return k.bfloat16(),self.inverse_v(l,vr).bfloat16()
    def finish_request(self):
        self.routing=None;self.routes=None;self._windows=None;super().finish_request()
    def extra_receipt(self,states):
        return {'V_quantization_axis':'channel within actually sealed local window','initial_window_token_counts':[len(w) for w in self._windows],'per_layer_sealed_windows':{str(l):s['sealed_windows'] for l,s in states.items()},'per_layer_low_tokens':{str(l):s['nlow'] for l,s in states.items()},'per_layer_unique_FP_band_tokens':{str(l):s['nf']+s['nr'] for l,s in states.items()},'unsealed_FP_pending_tokens':0,'V1_local_centroid_iterations':4 if self.policy.get('v1_quantizer')=='lloyd2' else 0,'metadata_dtype':'BF16 actual offset and scale used before assignments','append_short_tail_policy':'eager sealing, full short-window metadata charged'}
