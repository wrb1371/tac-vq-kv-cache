"""All-position temporal memory with genuinely compact innovation storage."""
import bisect,math,torch
from .temporal import Temporal
from ar18kv.hybrid import DenseFirst
from r15kv.budget import estimate_resident_bytes
from r14kv.common import apply_key_rope

class SparseTemporal(Temporal):
    method='AR19-sparse-motion-innovation-PSRVQ'
    def make_plan(self,meta):
        m=len(self._video_list);p=self._h*self._w;gop=int(self.policy.get('gop',2));ind=self.policy.get('mode')=='compact_independent'
        self._end=self._video_list[-1]+1;self._anchor_ord=[i for i in range(m) if ind or (i//p)%gop==0]
        a=len(self._anchor_ord);r=math.ceil((m-a)*float(self.policy.get('keep_fraction',.5)));q=meta.capacity-self._end;e=a+r+q
        dtype=torch.int16 if max(meta.position_max)+max(0,meta.generation_cap-1)<=32767 else torch.int32
        global_bytes=3*meta.capacity*torch.empty((),dtype=dtype).element_size()+256+4*m+4*len(self._text_list);options=[];g=128//self.package['group_dim']
        for rk in range(1,self.package['Rmax_K']+1):
            for rv in range(1,self.package['Rmax_V']+1):
                desc=[('K_index',(2,e,rk,g),torch.uint8,'payload'),('V_index',(2,e,rv,g),torch.uint8,'payload')]
                desc += [(side+'_norm',(2,e,1),torch.bfloat16,'metadata') for side in ['K','V']]
                desc += [('T_code',(2,meta.capacity),torch.uint8,'metadata'),('T_slot',(2,meta.capacity),torch.int32,'metadata')]
                desc += [(side,(2,len(self._text_list),128),torch.bfloat16,'payload') for side in ['text_FP_K','text_FP_V']]
                est=estimate_resident_bytes(self.static_bytes,global_bytes,desc)
                if est['resident_capacity_bytes']<=self.budget_bytes:options.append((rk,rv,desc,est))
        if not options:raise RuntimeError('INFEASIBLE_BY_INTEGER_ORACLE_SPARSE_INNOVATION')
        rk,rv,desc,est=max(options,key=lambda z:(z[0]+z[1],-abs(z[0]-z[1]),z[0]))
        return {'description':desc,'capacity':meta.capacity,'R_K':rk,'R_V':rv,'position_dtype':str(dtype),'group_dim':self.package['group_dim'],'groups':g,'anchor_tokens':a,'innovation_slots':r,'query_slots':q,'encoded_slots_per_head':e,'media_original':m,'protected_prefix_text_tokens':len(self._text_list),'keep_fraction':self.policy.get('keep_fraction',.5),'gop':gop,'mode':self.policy.get('mode'),'merged_video_grid':[self._nt,self._h,self._w],'estimate':est,'representation':'all native positions; compact anchors and selected residuals; rest copy matched reconstructed anchor','allocator':'balanced maximum feasible exact paid capacity'}
    def allocate_state(self,l,registry):
        s=super().allocate_state(l,registry);s['T_slot'].fill_(-1);return s
    def _read_slots(self,l,s,kind,slots,independent):
        t=self.tables[0];d=self.package['group_dim'];g=128//d;heads=torch.arange(2,device='cuda')[:,None];idx=s[kind+'_index'][heads,slots]
        z=torch.zeros(2,slots.shape[1],g,d,device='cuda')
        for rr in range(idx.shape[2]):z+=t[kind+'_books'][rr].float()[idx[:,:,rr].long()]
        norm=s[kind+'_norm'][heads,slots].float();z=(z.reshape(2,-1,128)@t['H'].T)*norm*t[kind+'_scale'][l].float()[:,None]
        if independent:z+=t[kind+'_mean'][l].float()[:,None]
        return z.bfloat16()
    @torch.no_grad()
    def append(self,l,state,kpost,v,pre=None):
        assert pre is not None;start=state['n'];n=v.shape[-2];t=self.tables[0];raw={'K':pre[0].float(),'V':v[0].float()};heads=torch.arange(2,device='cuda')[:,None]
        a=self.plan['anchor_tokens'];nr=self.plan['innovation_slots'];aslots=torch.arange(a,device='cuda')
        if start==0:
            assert n==self._end
            anchor_ord=torch.tensor(self._anchor_ord,device='cuda');aseq=self._vid_gpu.long()[anchor_ord];state['T_slot'][:,aseq]=aslots[None].to(torch.int32)
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
                    state['T_slot'][heads,chosen]=slots[None].to(torch.int32)
                    for kind in ['K','V']:
                        current=raw[kind].gather(1,chosen[...,None].expand(-1,-1,128));pred=decoded[kind].gather(1,selectedrefs[...,None].expand(-1,-1,128)).float()
                        self._innovation(l,state,current-pred,slots,kind)
                state['temporal_targets']=len(targets);state['temporal_innovation_heads']=2*nr;state['temporal_copy_heads']=2*(len(targets)-nr);state['distortion_copy_before_innovation_sum']=float(best.sum())
            text=self._text_gpu.long();state['text_FP_K'].copy_(kpost[0][:,text]);state['text_FP_V'].copy_(v[0][:,text]);state['text_filled']=len(self._text_list)
        else:
            qs=start-self._end;slots=torch.arange(a+nr+qs,a+nr+qs+n,device='cuda');assert int(slots[-1])<self.plan['encoded_slots_per_head']
            state['T_slot'][:,start:start+n]=slots[None].to(torch.int32)
            for kind in ['K','V']:self._innovation(l,state,raw[kind]-t[kind+'_mean'][l].float()[:,None],slots,kind)
        state['n']+=n;return state
    def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
        n=state['n'];a=self.plan['anchor_tokens'];aslots=torch.arange(a,device='cuda')[None].expand(2,-1);anchor_ord=torch.tensor(self._anchor_ord,device='cuda');aseq=self._vid_gpu.long()[anchor_ord];out=[]
        order=torch.arange(len(self._video_list),device='cuda');p=self._h*self._w;gop=int(self.policy.get('gop',2));targets=order[(order//p)%gop!=0];seq=self._vid_gpu.long()[targets]
        for kind in ['K','V']:
            z=torch.zeros(2,n,128,device='cuda',dtype=torch.bfloat16);z[:,aseq]=self._read_slots(l,state,kind,aslots,True)
            if self.policy.get('mode')!='compact_independent' and len(targets):
                codes=state['T_code'][:,seq];refs=self._reference_indices(targets,codes);pred=z.gather(1,refs[...,None].expand(-1,-1,128)).float();slot=state['T_slot'][:,seq].long();active=slot>=a
                residual=self._read_slots(l,state,kind,slot.clamp_min(0),False).float();z[:,seq]=(pred+torch.where(active[...,None],residual,0)).bfloat16()
            if n>self._end:
                slot=state['T_slot'][:,self._end:n].long();z[:,self._end:n]=self._read_slots(l,state,kind,slot,True)
            out.append(z[None])
        phase=positions.float()[:,:,None]*inv_freq[None,None,:];trigs=[];widths=[16,24,24]
        for trig in [torch.cos,torch.sin]:
            aa=(trig(phase)*attention_scaling).bfloat16();bb=torch.cat([aa[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1);trigs.append(torch.cat([bb,bb],-1)[None,None])
        k=apply_key_rope(out[0],*trigs);v=out[1];m=bisect.bisect_left(self._text_list,n);assert m==state['text_filled']
        if m:
            ix=self._text_gpu[:m].long();k[0].index_copy_(1,ix,state['text_FP_K'][:,:m]);v[0].index_copy_(1,ix,state['text_FP_V'][:,:m])
        return k,v

class HybridSparse(DenseFirst,SparseTemporal):pass
