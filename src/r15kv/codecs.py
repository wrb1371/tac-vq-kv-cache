"""Fixed-package ordinary budget parents, packed into bounded persistent arenas."""
import json,math,pathlib,time,torch
from r14kv.common import sha,apply_key_rope
from r14kv.fitting import nearest
from .budget import pack_indices,unpack_indices,nova_description,gsrq_description,gs_split,estimate_resident_bytes
from .persistent_registry import PersistentRegistry,Arena

class PackageCodec:
    read_backend='original-space-BF16-FlashAttention2-reference'
    needs_positions=False
    def __init__(self,package,budget_mib,**policy):
        self.directory=pathlib.Path(package);self.package=json.loads((self.directory/'completion.json').read_text());assert self.package['completed']
        self.budget_bytes=int(budget_mib)*2**20;self.policy=policy;self.tables=[];self.load_registry=PersistentRegistry(self.budget_bytes);self.load_registry.own('codec/tables',self.tables,'codec');t=time.perf_counter()
        for rec in self.package['records']:
            p=self.directory/rec['file'];assert sha(p)==rec['SHA256'];cpu=torch.load(p,map_location='cpu',weights_only=True)
            for key,x in cpu.items():assert x.untyped_storage().nbytes()==rec['tensors'][key]['storage_bytes']
            self.tables.append({key:x.to('cuda') for key,x in cpu.items()});self.load_registry.observe('direct_target_package_layer_'+str(rec['layer']));del cpu
        torch.cuda.synchronize();self.load_ms=(time.perf_counter()-t)*1000
        self.static_bytes=self.load_registry.end;assert self.static_bytes==self.package['static_tensor_bytes']
        self.package_sha256=sha(self.directory/'completion.json');self.metadata={'family':self.package['family'],'package_SHA256':self.package_sha256,'fixed_GPU_package':True,'CPU_export_before_GPU_load':True,'codec_static_bytes':self.static_bytes,'policy':policy,'read_backend':self.read_backend}
        self.context={}
    def set_context(self,**kwargs):self.context=kwargs
    def allocate_state(self,l,registry):
        arena=Arena(self.plan['description'],'cuda',registry,f'cache/layer{l:02}/arena')
        return {'arena':arena,'n':0,**arena.views}
    def prepare_request(self,meta,registry):
        self.meta=meta;self.registry=registry;registry.own('codec/tables',self.tables,'codec')
        self.plan=self.make_plan(meta);self.context={}
        assert self.plan['estimate']['resident_capacity_bytes']<=self.budget_bytes,self.plan
        return self.plan
    def finish_request(self):self.context={};self.registry=None;self.plan=None

class NOVABudget(PackageCodec):
    method='NOVA-R15-ordinary-budget-parent'
    def __init__(self,package,budget_mib,**policy):
        super().__init__(package,budget_mib,**policy)
        cb=self.tables[0]['centroids'];self.groups=cb.shape[1];self.gdim=cb.shape[-1];self.index_bits=int(math.log2(cb.shape[2]));assert cb.shape[2] in [16,256]
    def make_plan(self,meta):
        modes=self.policy.get('v_modes',['affine2','affine1','sign1']);bands=self.policy.get('bands',[[64,256],[32,128]])
        for mode in modes:
            for sink,recent in bands:
                bits=2 if mode=='affine2' else 1;description=nova_description(meta.capacity,sink,recent,self.groups,self.index_bits,bits,mode=='sign1')
                migration=2*2*min(recent,max(0,meta.capacity-sink))*128*2
                extra=self.extra_global_bytes(meta,description)
                est=estimate_resident_bytes(self.static_bytes,migration+extra,description)
                if est['resident_capacity_bytes']<=self.budget_bytes:return {'v_mode':mode,'v_bits':bits,'sink':sink,'recent':recent,'description':description,'estimate':est,'global_extra_bytes':extra,'migration_bytes':migration,'capacity':meta.capacity}
        raise RuntimeError('NOT_FEASIBLE_BY_INTEGER_ORACLE NOVA '+str({'T':meta.capacity,'budget':self.budget_bytes,'package':self.package_sha256}))
    def extra_global_bytes(self,meta,description):return 0
    def prepare_request(self,meta,registry):
        plan=super().prepare_request(meta,registry);rs=min(plan['recent'],max(0,meta.capacity-plan['sink']))
        self.migration=torch.empty(2,2,rs,128,device='cuda',dtype=torch.bfloat16);registry.own('cache/FP_band_migration_buffer',self.migration)
        return plan
    def allocate_state(self,l,registry):
        s=super().allocate_state(l,registry);s.update(nf=0,nr=0,nlow=0);return s
    @torch.no_grad()
    def write_keys(self,l,state,k,start):
        table=self.tables[l];n=k.shape[1]
        for b in range(0,n,1024):
            x=k[:,b:b+1024];r=torch.bmm((x-table['mean'][:,None]).bfloat16(),table['F']).float()
            rho=r.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8).bfloat16();z=r/rho.float();nn=z.shape[1]
            grouped=z.reshape(2,nn,self.groups,self.gdim).permute(0,2,1,3).reshape(2*self.groups,nn,self.gdim)
            cb=table['centroids'].float().reshape(2*self.groups,-1,self.gdim);ix=nearest(grouped,cb).reshape(2,self.groups,nn).transpose(1,2)
            state['K_index'][:,start+b:start+b+nn].copy_(pack_indices(ix,self.index_bits));state['rho'][:,start+b:start+b+nn].copy_(rho)
    @torch.no_grad()
    def rotate_v(self,l,v,start=0):return torch.bmm(v,self.tables[l]['V_rotation']).float()
    @torch.no_grad()
    def inverse_v(self,l,vr,start=0):return torch.bmm(vr,self.tables[l]['V_rotation'].float().transpose(1,2))
    @torch.no_grad()
    def write_values(self,l,state,v,start):
        mode=self.plan['v_mode'];bits=self.plan['v_bits'];levels=2**bits
        for b in range(0,v.shape[1],1024):
            vr=self.rotate_v(l,v[:,b:b+1024],start+b);n=vr.shape[1];sl=slice(start+b,start+b+n)
            if mode=='sign1':
                scale=vr.abs().mean(-1,keepdim=True).bfloat16();q=(vr>=0).to(torch.uint8)
            else:
                threshold=vr.abs().sort(-1).values[...,int(.92*128):int(.92*128)+1];clipped=vr.clamp(-threshold,threshold)
                lo=clipped.amin(-1,keepdim=True);hi=clipped.amax(-1,keepdim=True);scale=((hi-lo).clamp_min(1e-8)/(levels-1)).bfloat16();zero=(-lo/scale.float()).bfloat16()
                q=(clipped/scale.float()+zero.float()).round().clamp(0,levels-1).to(torch.uint8);state['V_zero'][:,sl].copy_(zero)
            state['V_scale'][:,sl].copy_(scale);state['V_index'][:,sl].copy_(pack_indices(q,bits))
    def write_low(self,l,state,k,v):
        n=k.shape[1];start=state['nlow'];assert start+n<=state['K_index'].shape[1]
        self.write_keys(l,state,k,start);self.write_values(l,state,v,start);state['nlow']+=n
    @torch.no_grad()
    def append(self,l,state,kpost,v,pre=None):
        kpost=kpost[0];v=v[0];nnew=kpost.shape[1];take=min(state['firstK'].shape[1]-state['nf'],nnew)
        if take:
            state['firstK'][:,state['nf']:state['nf']+take].copy_(kpost[:,:take]);state['firstV'][:,state['nf']:state['nf']+take].copy_(v[:,:take]);state['nf']+=take
        nk=kpost[:,take:];nv=v[:,take:];oldr=state['nr'];cap=state['recentK'].shape[1];age=max(0,oldr+nk.shape[1]-cap);aold=min(oldr,age);anew=age-aold
        if aold:self.write_low(l,state,state['recentK'][:,:aold],state['recentV'][:,:aold])
        if anew:self.write_low(l,state,nk[:,:anew],nv[:,:anew])
        oldkeep=oldr-aold
        if oldkeep:
            self.migration[0,:,:oldkeep].copy_(state['recentK'][:,aold:oldr]);self.migration[1,:,:oldkeep].copy_(state['recentV'][:,aold:oldr]);state['recentK'][:,:oldkeep].copy_(self.migration[0,:,:oldkeep]);state['recentV'][:,:oldkeep].copy_(self.migration[1,:,:oldkeep])
        newkeep=nk.shape[1]-anew
        if newkeep:state['recentK'][:,oldkeep:oldkeep+newkeep].copy_(nk[:,anew:]);state['recentV'][:,oldkeep:oldkeep+newkeep].copy_(nv[:,anew:])
        state['nr']=oldkeep+newkeep;state['n']+=nnew;assert state['n']==state['nf']+state['nlow']+state['nr'];return state
    @torch.no_grad()
    def synthesize_low(self,l,state):
        n=state['nlow'];t=self.tables[l];ix=unpack_indices(state['K_index'][:,:n],self.index_bits,self.groups).long();rows=[]
        for h in range(2):rows.append(t['centroids'][h].float()[torch.arange(self.groups,device='cuda')[None,:],ix[h]])
        r=torch.stack(rows).reshape(2,n,128)*state['rho'][:,:n].float();k=torch.bmm(r,t['A'].float().transpose(1,2))+t['mean'].float()[:,None]
        q=unpack_indices(state['V_index'][:,:n],self.plan['v_bits'],128).float()
        vr=(q*2-1)*state['V_scale'][:,:n].float() if self.plan['v_mode']=='sign1' else (q-state['V_zero'][:,:n].float())*state['V_scale'][:,:n].float()
        return k.bfloat16(),self.inverse_v(l,vr).bfloat16()
    @torch.no_grad()
    def decode(self,l,state,positions=None,inv_freq=None,attention_scaling=1.):
        ks=[state['firstK'][:,:state['nf']]];vs=[state['firstV'][:,:state['nf']]]
        if state['nlow']:
            k,v=self.synthesize_low(l,state);ks.append(k);vs.append(v)
        ks.append(state['recentK'][:,:state['nr']]);vs.append(state['recentV'][:,:state['nr']]);return torch.cat(ks,1)[None],torch.cat(vs,1)[None]
    def finish_request(self):
        self.migration=None;super().finish_request()

class GSRQBudget(PackageCodec):
    method='GSRQ-R15-ordinary-budget-parent';needs_positions=True
    def make_plan(self,meta):
        dtype=torch.int16 if max(meta.position_max)+max(0,meta.generation_cap-1)<=32767 else torch.int32;posbytes=3*meta.capacity*torch.empty((),dtype=dtype).element_size()+256
        km=self.package['Rmax_K'];vm=self.package['Rmax_V'];available=self.budget_bytes-self.static_bytes-posbytes
        total=min(km+vm,available//(72*meta.capacity));split=gs_split(total,km,vm,int(self.policy.get('K_minus_V',0)))
        if split is None:raise RuntimeError('NOT_FEASIBLE_BY_INTEGER_ORACLE GSRQ')
        rk,rv=split;description=gsrq_description(meta.capacity,rk,rv);est=estimate_resident_bytes(self.static_bytes,posbytes,description)
        return {'R_K':rk,'R_V':rv,'loaded_Rmax_K':km,'loaded_Rmax_V':vm,'description':description,'estimate':est,'position_dtype':str(dtype),'capacity':meta.capacity}
    @torch.no_grad()
    def encode_into(self,x,book,dest):
        # Exact deployed FP8 centroids for assignments and residuals. Chunked search scratch only.
        for b in range(0,x.shape[1],4096):
            residual=x[:,b:b+4096].float().clone()
            for stage in range(dest.shape[-1]):
                cb=book[:,stage].float();ix=nearest(residual,cb);dest[:,b:b+residual.shape[1],stage].copy_(ix.to(torch.uint8));residual-=torch.gather(cb,1,ix[...,None].expand(-1,-1,128))
    @torch.no_grad()
    def append(self,l,state,kpost,v,pre=None):
        assert pre is not None and pre.shape==kpost.shape;start=state['n'];n=pre.shape[-2];assert start+n<=self.meta.capacity
        self.encode_into(pre[0],self.tables[l]['K'],state['K_index'][:,start:start+n]);self.encode_into(v[0],self.tables[l]['V'],state['V_index'][:,start:start+n]);state['n']+=n;return state
    @torch.no_grad()
    def raw_decode(self,index,book):
        out=torch.zeros(2,index.shape[1],128,device=index.device)
        for stage in range(index.shape[-1]):
            cb=book[:,stage].float();ix=index[...,stage].long();out+=torch.gather(cb,1,ix[...,None].expand(-1,-1,128))
        return out
    @torch.no_grad()
    def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
        n=state['n'];k=self.raw_decode(state['K_index'][:,:n],self.tables[l]['K'])[None];v=self.raw_decode(state['V_index'][:,:n],self.tables[l]['V'])[None]
        assert positions.shape==(3,n);phase=positions.float()[:,:,None]*inv_freq[None,None,:];cos=(phase.cos()*attention_scaling).bfloat16();sin=(phase.sin()*attention_scaling).bfloat16();widths=[16,24,24]
        sc=torch.cat([cos[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1);ss=torch.cat([sin[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1)
        return apply_key_rope(k.bfloat16(),torch.cat([sc,sc],-1)[None,None],torch.cat([ss,ss],-1)[None,None]),v.bfloat16()
