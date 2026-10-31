"""Physically paid native7B NOVA/NSN states, original-space reference readers."""
import hashlib,json,math,pathlib,time,os
import torch
from ar26kv.runtime import ROOT
from ar26kv.codec import ReadCache
from r15kv.persistent_registry import PersistentRegistry,Arena
from r15kv.budget import pack_indices,unpack_indices
from r14kv.fitting import nearest
from r16kv.nsn64 import NSN64Reference
from ar30kv.planning import select,nbytes,L,H,D
DT={'u8':torch.uint8,'bf16':torch.bfloat16,'f16':torch.float16,'i16':torch.int16,'i32':torch.int32,'f32':torch.float32}
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

class NativeControl:
    def __init__(self,name,model,item,pos,vi,ti,end,teacher):
        self.name=name;self.model=model;self.item=item;self.end=end;self.q=int(item['query_reservation_tokens']);self.capacity=end+self.q
        self.T=len(ti);self.M=len(vi);self.budget=int(item['budget_mib']*2**20);started=time.time()
        info={'media_end':end,'capacity':self.capacity,'protected_text':self.T,'position_bytes':2 if int(pos.max())<=32767 else 4}
        assert (info['media_end'],info['capacity'],info['protected_text'])==(end,self.capacity,self.T)
        chosen,options=select(name,self.capacity,self.T,info['position_bytes'],self.budget)
        if chosen is None:raise RuntimeError('INFEASIBLE_BY_INTEGER_ORACLE '+name+' '+str(min(x['resident_bytes']for x in options))+' > '+str(self.budget))
        self.plan={k:v for k,v in chosen.items()if k!='description'};self.plan.update(method=name,query_slots=self.q,media_original=self.M,geometry=[L,28,H,D])
        self.registry=PersistentRegistry(self.budget);self.states={};self.table=None
        if name.startswith('NOVA'):
            g=chosen['group_size'];nc=chosen['centroids'];directory=pathlib.Path(os.environ.get('STIRVQ_BASELINE_ROOT',str(ROOT)))/f'artifacts/ar30/packages/NOVA_g{g}_C{nc}'
        else:directory=ROOT/'artifacts/round16/packages/nsn_global'
        receipt=json.loads((directory/'completion.json').read_text());assert receipt['completed'];self.tables=[]
        for row in receipt['records']:
            p=directory/row['file'];assert sha(p)==row['SHA256']
            self.tables.append({k:v.cuda()for k,v in torch.load(p,map_location='cpu',weights_only=True).items()})
        self.registry.own('codec/tables',self.tables,'codec');assert self.registry.end==chosen['codec_bytes']==receipt['static_tensor_bytes']
        self.plan.update(package=str(directory),package_SHA256=sha(directory/'completion.json'),static_bytes=self.registry.end)
        self.global_maps={'text':ti.to(torch.int32)}
        if name=='NSNQuant':
            dtype=torch.int16 if info['position_bytes']==2 else torch.int32
            coordinates=torch.zeros((3,self.capacity),device='cuda',dtype=dtype);coordinates[:,:end].copy_(pos[:,0,:end])
            assert int(pos[:,:,:end].max())<=32767 or dtype==torch.int32
            rotary=model.model.language_model.rotary_emb
            self.global_maps.update(positions=coordinates,freq=rotary.inv_freq.detach().float().clone())
            self.position_storage=coordinates;self.position_inv_freq=self.global_maps['freq'];self.position_attention_scaling=float(rotary.attention_scaling)
            assert model.config.text_config.rope_scaling['mrope_section']==[16,24,24]
        else:
            self.migration=torch.zeros((2,H,min(chosen['recent'],max(0,self.capacity-chosen['sink'])),D),device='cuda',dtype=torch.bfloat16)
            self.registry.own('cache/migration',self.migration)
        self.registry.own('cache/maps',self.global_maps)
        desc=[(k,tuple(shape),DT[dtype],kind)for k,shape,dtype,kind in chosen['description']]
        assert Arena.bytes(desc)==nbytes(chosen['description'])
        self.plan['planning_seconds']=time.time()-started;started=time.time();ti_cpu=ti.cpu()
        for l in range(L):
            arena=Arena(desc,'cuda',self.registry,f'cache/layer{l}');arena.tensor.zero_();st={'arena':arena,**arena.views};self.states[l]=st
            self.encode(l,teacher.states[l]);st['text_FP_K'].copy_(teacher.states[l]['K_post'][:,ti_cpu].cuda());st['text_FP_V'].copy_(teacher.states[l]['V'][:,ti_cpu].cuda())
        torch.cuda.synchronize();self.plan['encoding_seconds']=time.time()-started;self.receipt=self.registry.receipt()
        assert self.receipt['resident_bytes']==chosen['resident_bytes'];self.plan['estimated_resident_bytes']=chosen['resident_bytes']

    @torch.no_grad()
    def encode(self,l,native):
        if self.name=='NSNQuant':return self.encode_nsn(l,native)
        st=self.states[l];t=self.tables[l];fs=min(self.plan['sink'],self.end);rs=min(self.plan['recent'],self.end-fs);n=self.end-fs-rs
        st.update(nlow=n,nf=fs,nr=rs)
        k=native['K_post'].cuda();v=native['V'].cuda()
        for s,x in [('K',k),('V',v)]:
            st['first'+s][:,:fs].copy_(x[:,:fs]);st['recent'+s][:,:rs].copy_(x[:,self.end-rs:])
        g=self.plan['group_size'];groups=D//g;nc=self.plan['centroids'];ib=self.plan['K_index_bits'];bits=self.plan['v_bits'];mode=self.plan['v_mode']
        for b in range(0,n,1024):
            kk=k[:,fs+b:fs+b+min(1024,n-b)];vv=v[:,fs+b:fs+b+kk.shape[1]];nn=kk.shape[1];sl=slice(b,b+nn)
            r=torch.bmm((kk-t['mean'][:,None]).bfloat16(),t['F']).float();rho=r.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8).bfloat16()
            z=(r/rho.float()).reshape(H,nn,groups,g).permute(0,2,1,3).reshape(H*groups,nn,g)
            cb=t['centroids'].float().reshape(H*groups,nc,g);ix=nearest(z,cb).reshape(H,groups,nn).transpose(1,2)
            st['K_index'][:,sl].copy_(pack_indices(ix,ib));st['rho'][:,sl].copy_(rho)
            vr=torch.bmm(vv,t['V_rotation']).float()
            if mode=='sign1':scale=vr.abs().mean(-1,keepdim=True).bfloat16();q=(vr>=0).to(torch.uint8)
            else:
                threshold=vr.abs().sort(-1).values[...,int(.92*D):int(.92*D)+1];zz=vr.clamp(-threshold,threshold)
                lo=zz.amin(-1,keepdim=True);hi=zz.amax(-1,keepdim=True);scale=((hi-lo).clamp_min(1e-8)/(2**bits-1)).bfloat16();zero=(-lo/scale.float()).bfloat16()
                q=(zz/scale.float()+zero.float()).round().clamp(0,2**bits-1).to(torch.uint8);st['V_zero'][:,sl].copy_(zero)
            st['V_scale'][:,sl].copy_(scale);st['V_index'][:,sl].copy_(pack_indices(q,bits))

    rope=NSN64Reference.rope
    @staticmethod
    def pair_state(st,h):
        return {k:(v[h:h+2]if torch.is_tensor(v)else v)for k,v in st.items()if k!='arena'}
    @torch.no_grad()
    def encode_nsn(self,l,native):
        st=self.states[l];n=self.end//64*64;pending=self.end-n;raw={'K':native['K_pre'].cuda(),'V':native['V'].cuda()}
        st.update(quantized=n,pending=pending,n=self.end)
        for b in range(0,n,1024):
            length=min(1024,n-b)
            for h in [0,2]:
                sub=self.pair_state(st,h)
                for s in ['K','V']:NSN64Reference._encode(self,l,sub,s,raw[s][h:h+2,b:b+length],b)
        for s in ['K','V']:st[s+'_pending'][:,:pending].copy_(raw[s][:,n:])

    @torch.no_grad()
    def decode(self,l):
        st=self.states[l]
        if self.name=='NSNQuant':
            n=st['quantized'];pending=st['pending'];restored={}
            for s in ['K','V']:
                z=torch.cat([NSN64Reference._restore(self,self.pair_state(st,h),s)for h in [0,2]],0)
                tail=st[s+'_pending'][:,:pending]
                if pending:z=torch.cat([z,self.rope(tail,self.position_storage[:,n:n+pending])if s=='K'else tail.half()],1)
                restored[s]=z.bfloat16()
            k,v=restored['K'],restored['V']
        else:
            t=self.tables[l];n=st['nlow'];groups=D//self.plan['group_size'];ix=unpack_indices(st['K_index'][:,:n],self.plan['K_index_bits'],groups).long()
            r=torch.stack([t['centroids'][h].float()[torch.arange(groups,device='cuda')[None],ix[h]]for h in range(H)]).reshape(H,n,D)*st['rho'][:,:n].float()
            k=(torch.bmm(r,t['A'].float().transpose(1,2))+t['mean'][:,None].float()).bfloat16()
            q=unpack_indices(st['V_index'][:,:n],self.plan['v_bits'],D).float()
            vr=(q*2-1)*st['V_scale'][:,:n].float()if self.plan['v_mode']=='sign1'else(q-st['V_zero'][:,:n].float())*st['V_scale'][:,:n].float()
            v=torch.bmm(vr,t['V_rotation'].float().transpose(1,2)).bfloat16()
            k=torch.cat([st['firstK'][:,:st['nf']],k,st['recentK'][:,:st['nr']]],1);v=torch.cat([st['firstV'][:,:st['nf']],v,st['recentV'][:,:st['nr']]],1)
        ti=self.global_maps['text'].long();k[:,ti]=st['text_FP_K'];v[:,ti]=st['text_FP_V'];assert k.shape==v.shape==(H,self.end,D)
        return k,v
    def cache(self):return ReadCache(self)
