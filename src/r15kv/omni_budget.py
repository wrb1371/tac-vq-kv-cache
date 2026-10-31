"""Finite Omni-derived budget adaptation; author local MXFP4 and modality V recipe."""
import collections,math,torch
from .codecs import PackageCodec
from .budget import pack_indices,unpack_indices,estimate_resident_bytes
from .persistent_registry import Arena
from model.codec.packed_cache import _encode_mxfp4,_decode_mxfp4
from model.codec.turboquant import hadamard_cpu,lloyd_max
class OmniBudget(PackageCodec):
    method='Omni-derived-R15-budget-adaptation';needs_positions=False
    def __init__(self,package,budget_mib,**policy):
        super().__init__(package,budget_mib,**policy);dtype=self.tables[0]['V_rotation'].dtype
        c2,b2=lloyd_max(128,2);c1=torch.tensor([-1.,1.])*math.sqrt(2/math.pi/128)
        self.aux={'H':hadamard_cpu(128).to('cuda',dtype=dtype),'centroids2':c2.cuda(),'boundaries2':b2.cuda(),'centroids1':c1.cuda(),'boundaries1':torch.zeros(1,device='cuda')}
        for bits,c in [(1,c1),(2,c2)]:
            z=(c-c[0])/(c[-1]-c[0]);self.aux['levels'+str(bits)]=z.cuda();self.aux['local_boundaries'+str(bits)]=((z[:-1]+z[1:])/2).cuda()
        r=self.load_registry.own('codec/shared_H_LM_tables',self.aux,'codec');self.static_bytes=r['resident_bytes'];self.metadata.update(codec_static_bytes=self.static_bytes,identity='Omni-derived, actual1/2bit packing and bounded eager-sealed local groups; not official original K2V2 configuration',key_rotation='one truly shared Hadamard storage across all layers and physical heads',position_storage='native-order payload, shared routing; no unnecessary per-layer repeated positions',local_metadata='real author MXFP4 min/span, rounded FP16 metadata in assignment; normFP16',tail_rule='postfill short groups sealed; each appended text token is a charged new short group; no FP pending retained')
    def set_context(self,**kwargs):super().set_context(**kwargs);self.source_tags=kwargs['tags'].cpu()
    def make_plan(self,meta):
        n=meta.prompt_tokens-1;cap=meta.capacity
        for kb,vb in self.policy.get('bit_pairs',[[2,2],[2,1],[1,2],[1,1]]):
            for g in self.policy.get('groups',[32,128,512,2048]):
                ix=torch.arange(n);windows=[ids[b:b+g] for m in [0,1,2,3] for ids in [ix[self.source_tags[:n]==m]] for b in range(0,len(ids),g)];nbase=len(windows);windows += [torch.tensor([j]) for j in range(n,cap)];nw=len(windows)
                desc=[('K_index',(2,cap,128*kb//8),torch.uint8,'payload'),('V_index',(2,cap,128*vb//8),torch.uint8,'payload'),('K_norm',(2,cap,1),torch.float16,'metadata'),('V_norm',(2,cap,1),torch.float16,'metadata')]
                for key,width in [('K_min',64),('K_span',64),('K_min_exp',4),('K_span_exp',4)]:desc.append((key,(2,nw,width),torch.uint8,'metadata'))
                routes=[('tags',(cap,),torch.uint8,'metadata'),('window_id',(cap,),torch.int16,'metadata'),('order',(cap,),torch.int32,'metadata'),('offset',(nw+1,),torch.int32,'metadata')]
                est=estimate_resident_bytes(self.static_bytes,Arena.bytes(routes),desc)
                if est['resident_capacity_bytes']<=self.budget_bytes:
                    self._windows=windows;self._route_desc=routes;self._nbase=nbase
                    return {'K_bits':kb,'V_bits':vb,'K_local_group':g,'initial_windows':nbase,'max_windows':nw,'initial_prefix_tokens':n,'description':desc,'estimate':est,'capacity':cap,'append_rule':'eager-sealed one-token text group, no retained pendingFP','official_original_configuration':False}
        raise RuntimeError('NOT_FEASIBLE_BY_INTEGER_ORACLE Omni-derived')
    def prepare_request(self,meta,registry):
        registry.own('codec/shared_H_LM_tables',self.aux,'codec');plan=super().prepare_request(meta,registry);self.routing=Arena(self._route_desc,'cuda',registry,'cache/Omni_shared_routing');self.routes=self.routing.views
        self.routes['tags'].zero_();self.routes['tags'][:len(self.source_tags)].copy_(self.source_tags);order=[];offsets=[0]
        for i,w in enumerate(self._windows):order+=w.tolist();offsets.append(len(order))
        self.routes['order'].copy_(torch.tensor(order,dtype=torch.int32));self.routes['offset'].copy_(torch.tensor(offsets,dtype=torch.int32))
        for i in range(len(offsets)-1):self.routes['window_id'][self.routes['order'][offsets[i]:offsets[i+1]].long()]=i
        self._offsets=offsets;return plan
    @torch.no_grad()
    def write_key_groups(self,l,state,k,start):
        n=k.shape[1];norm=k.float().norm(dim=-1,keepdim=True).clamp_min(1e-12);state['K_norm'][:,start:start+n].copy_(norm.half());direction=torch.matmul(k.float()/norm,self.aux['H'].float().T).half()
        if start==0:
            assert n==self.plan['initial_prefix_tokens'];groups=range(self._nbase)
        else:
            assert n==1;groups=[self._nbase+start-self.plan['initial_prefix_tokens']]
        by=collections.defaultdict(list)
        for j in groups:by[len(self._windows[j])].append(j)
        for length,js in by.items():
            for b in range(0,len(js),16):
                batch=js[b:b+16];ids=torch.tensor(batch,device='cuda');beg=self.routes['offset'].index_select(0,ids).long();op=(beg[:,None]+torch.arange(length,device='cuda')[None,:]).flatten();native=self.routes['order'].index_select(0,op).long();z=direction.index_select(1,native-start).reshape(2,len(batch),length,128).float();minimum=z.amin(2);span=(z.amax(2)-minimum).clamp_min(1e-12);mp,me,mr=_encode_mxfp4(minimum);sp,se,sr=_encode_mxfp4(span)
                state['K_min'][:,ids]=mp;state['K_min_exp'][:,ids]=me;state['K_span'][:,ids]=sp;state['K_span_exp'][:,ids]=se
                mr=mr.half().float();sr=sr.half().float();unit=((z-mr[:,:,None])/torch.where(sr==0,torch.ones_like(sr),sr)[:,:,None]).clamp(0,1);q=torch.bucketize(unit.contiguous(),self.aux['local_boundaries'+str(self.plan['K_bits'])],right=True);state['K_index'][:,native]=pack_indices(q,self.plan['K_bits']).reshape(2,-1,128*self.plan['K_bits']//8)
    @torch.no_grad()
    def append(self,l,state,kpost,v,pre=None):
        start=state['n'];k=kpost[0];v=v[0];n=k.shape[1];self.write_key_groups(l,state,k,start);bits=self.plan['V_bits']
        for b in range(0,n,1024):
            x=v[:,b:b+1024].float();norm=x.norm(dim=-1,keepdim=True).clamp_min(1e-12);state['V_norm'][:,start+b:start+b+x.shape[1]].copy_(norm.half());direction=x/norm;tags=self.routes['tags'][start+b:start+b+x.shape[1]];rot=torch.empty_like(direction)
            for mod in [0,1,2]:
                ids=(tags==mod).nonzero(as_tuple=True)[0]
                if ids.numel():rot[:,ids]=torch.bmm(direction.index_select(1,ids),self.tables[l]['V_rotation'][:,mod].float().transpose(1,2))
            q=torch.bucketize(rot.contiguous(),self.aux['boundaries'+str(bits)],right=True);state['V_index'][:,start+b:start+b+x.shape[1]].copy_(pack_indices(q,bits))
        state['n']+=n;return state
    @torch.no_grad()
    def decode(self,l,state,positions=None,inv_freq=None,attention_scaling=1.):
        n=state['n'];wid=self.routes['window_id'][:n].long();kb=self.plan['K_bits'];vb=self.plan['V_bits'];ix=unpack_indices(state['K_index'][:,:n],kb,128).long()
        m=_decode_mxfp4(state['K_min'],state['K_min_exp'],128).index_select(1,wid);s=_decode_mxfp4(state['K_span'],state['K_span_exp'],128).index_select(1,wid);kr=(m+s*self.aux['levels'+str(kb)][ix])*state['K_norm'][:,:n].float();k=torch.matmul(kr,self.aux['H'].float())
        vi=unpack_indices(state['V_index'][:,:n],vb,128).long();vr=self.aux['centroids'+str(vb)][vi];vr=vr/vr.norm(dim=-1,keepdim=True).clamp_min(1e-12);vr*=state['V_norm'][:,:n].float();v=torch.empty_like(vr);tags=self.routes['tags'][:n]
        for mod in [0,1,2]:
            ids=(tags==mod).nonzero(as_tuple=True)[0]
            if ids.numel():v[:,ids]=torch.bmm(vr.index_select(1,ids),self.tables[l]['V_rotation'][:,mod].float())
        return k.bfloat16()[None],v.bfloat16()[None]
    def finish_request(self):
        self.routing=None;self.routes=None;self.source_tags=None;self._windows=None;super().finish_request()
