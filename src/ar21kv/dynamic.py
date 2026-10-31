"""Physical allocation with runtime layout RVQ/lookup/compose kernels."""
import torch
import triton
from .dynamic_kernels import read_slots,encode_rvq,match9,compose,_tail
from .allocated import AllocatedPacked

class DynamicRVQ:
    @torch.no_grad()
    def _innovation(self, l, state, raw, slots, kind):
        t = self.tables[0]
        for b in range(0, len(slots), 512):
            x = raw[:, b:b + 512].float()
            nn = x.shape[1]
            u = x / t[kind + '_scale'][l].float()[:, None]
            rho = u.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6).bfloat16()
            where = slots[b:b + nn]
            state[kind + '_norm'][:, where] = rho
            z = ((u / rho.float()) @ t['H']).reshape(1, 2 * nn * 8, 16)
            encode_rvq(z, t[kind + '_books'], state[kind + '_index'], where)


class DynamicMatchedAppend:
    @torch.no_grad()
    def append(self, l, state, kpost, v, pre=None):
        if self._hybrid_dense:
            return super().append(l, state, kpost, v, pre)
        assert pre is not None
        start = state['n']; n = v.shape[-2]; t = self.tables[0]
        raw = {'K': pre[0].float(), 'V': v[0].float()}
        a = self.plan['anchor_tokens']; nr = self.plan['innovation_slots']
        heads = torch.arange(2, device='cuda')[:, None]
        if start == 0:
            assert n == self._end
            aslots = torch.arange(a, device='cuda')
            aseq = self._vid_gpu.long()[torch.tensor(self._anchor_ord, device='cuda')]
            for kind in ['K', 'V']:
                self._innovation(l, state, raw[kind][:, aseq] - t[kind + '_mean'][l].float()[:, None], aslots, kind)
            order = torch.arange(len(self._video_list), device='cuda')
            targets = order[(order // (self._h * self._w)) % int(self.policy['gop']) != 0]
            seq = self._vid_gpu.long()[targets]
            if len(targets):
                decoded = {}
                for kind in ['K', 'V']:
                    z = torch.zeros((2, n, 128), device='cuda', dtype=torch.bfloat16)
                    z[:, aseq] = self._read_slots(l, state, kind, aslots[None].expand(2, -1), True)
                    decoded[kind] = z
                codes, best = match9(raw, decoded, self._vid_gpu, targets, [t['K_scale'][l], t['V_scale'][l]], self._h, self._w, int(self.policy['gop']))
                refs = self._reference_indices(targets, codes)
                state['T_code'][:, seq] = codes
                if nr:
                    picked = torch.argsort(best, dim=1, descending=True, stable=True)[:, :nr]
                    if not self._stable_order:
                        picked = picked.sort(1).values
                    chosen = seq[picked]; selected = refs.gather(1, picked)
                    slots = torch.arange(a, a + nr, device='cuda')
                    state['T_code'][heads, chosen] = codes.gather(1, picked) | 128
                    for kind in ['K', 'V']:
                        cur = raw[kind].gather(1, chosen[..., None].expand(-1, -1, 128))
                        pred = decoded[kind].gather(1, selected[..., None].expand(-1, -1, 128)).float()
                        self._innovation(l, state, cur - pred, slots, kind)
                    if self._stable_order:
                        integer_order = torch.argsort(picked, dim=1, stable=True)
                        for kind in ['K', 'V']:
                            ix = state[kind + '_index'][:, a:a+nr]; norm = state[kind + '_norm'][:, a:a+nr]
                            ix.copy_(ix.gather(1, integer_order[..., None, None].expand_as(ix)))
                            norm.copy_(norm.gather(1, integer_order[..., None].expand_as(norm)))
                state['temporal_targets'] = len(targets)
                state['temporal_innovation_heads'] = 2 * nr
                state['temporal_copy_heads'] = 2 * (len(targets) - nr)
                state['distortion_copy_before_innovation_sum'] = float(best.sum())
            text = self._text_gpu.long()
            state['text_FP_K'].copy_(kpost[0][:, text]); state['text_FP_V'].copy_(v[0][:, text]); state['text_filled'] = len(self._text_list)
        else:
            slots = torch.arange(a+nr+start-self._end, a+nr+start-self._end+n, device='cuda')
            assert int(slots[-1]) < self.plan['encoded_slots_per_head']
            for kind in ['K', 'V']:
                self._innovation(l, state, raw[kind] - t[kind + '_mean'][l].float()[:, None], slots, kind)
        state['n'] += n
        return state


class DynamicReadDecode:
    def _read_slots(self,l,state,kind,slots,independent):
        return read_slots(l,state,kind,slots,independent,self.tables[0])

    def decode(self, l, state, positions, inv_freq, attention_scaling=1.):
        if self._hybrid_dense:
            return super().decode(l, state, positions, inv_freq, attention_scaling)
        a = self.plan['anchor_tokens']; nr = self.plan['innovation_slots']; nq = state['n'] - self._end
        arrays = []
        for start, count, independent in [(0, a, True), (a, nr, False), (a+nr, nq, True)]:
            slots = torch.arange(start, start+count, device='cuda')[None].expand(2, -1)
            for kind in ['K', 'V']:
                arrays.append(self._read_slots(l, state, kind, slots, independent) if count else torch.empty((2, 0, 128), device='cuda', dtype=torch.bfloat16))
        return compose(self, state, positions, inv_freq, attention_scaling, arrays)


class DynamicExactGemm:
    @torch.no_grad()
    def _innovation(self,l,state,raw,slots,kind):
        t=self.tables[0];out=state[kind+'_index']
        for b in range(0,len(slots),512):
            x=raw[:,b:b+512].float();n=x.shape[1];u=x/t[kind+'_scale'][l].float()[:,None]
            rho=u.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6).bfloat16();where=slots[b:b+n];state[kind+'_norm'][:,where]=rho
            z=((u/rho.float())@t['H']).reshape(1,2*n*8,16)
            for stage in range(out.shape[2]):
                cb=t[kind+'_books'][stage][None].float()
                cn=cb.square().sum(-1);xx=z.square().sum(-1)
                dot=torch.bmm(z,cb.transpose(1,2))
                _tail[(2*n*8,)](dot,xx,cn,cb,z,out,where.contiguous(),2*n*8,n,out.stride(0),out.stride(1),stage,num_warps=4,enable_fp_fusion=False)


class DynamicGainSelection:
    @torch.no_grad()
    def append(self,l,state,kpost,v,pre=None):
        if self._hybrid_dense or state['n']!=0:return super().append(l,state,kpost,v,pre)
        assert pre is not None;n=v.shape[-2];assert n==self._end;t=self.tables[0];raw={'K':pre[0].float(),'V':v[0].float()};heads=torch.arange(2,device=v.device)[:,None]
        a=self.plan['anchor_tokens'];nr=self.plan['innovation_slots'];aslots=torch.arange(a,device=v.device);aseq=self._vid_gpu.long()[torch.tensor(self._anchor_ord,device=v.device)]
        for kind in ['K','V']:self._innovation(l,state,raw[kind][:,aseq]-t[kind+'_mean'][l].float()[:,None],aslots,kind)
        order=torch.arange(len(self._video_list),device=v.device);targets=order[(order//(self._h*self._w))%int(self.policy['gop'])!=0];seq=self._vid_gpu.long()[targets]
        if len(targets):
            decoded={}
            for kind in ['K','V']:
                z=torch.zeros(2,n,128,device=v.device,dtype=torch.bfloat16);z[:,aseq]=self._read_slots(l,state,kind,aslots[None].expand(2,-1),True);decoded[kind]=z
            codes,best=match9(raw,decoded,self._vid_gpu,targets,[t['K_scale'][l],t['V_scale'][l]],self._h,self._w,int(self.policy['gop']));refs=self._reference_indices(targets,codes);state['T_code'][:,seq]=codes
            if nr:
                count=min(len(targets),int(self.policy.get('candidate_factor',4))*nr);pool=torch.argsort(best,dim=1,descending=True,stable=True)[:,:count];chosen=seq[pool];reference=refs.gather(1,pool);local=torch.arange(count,device=v.device);temporary={};reconstructed={};before=torch.zeros(2,count,device=v.device);after=torch.zeros_like(before)
                for kind in ['K','V']:
                    stages=state[kind+'_index'].shape[2];temporary[kind+'_index']=torch.empty(2,count,stages,8,device=v.device,dtype=torch.uint8);temporary[kind+'_norm']=torch.empty(2,count,1,device=v.device,dtype=torch.bfloat16)
                    current=raw[kind].gather(1,chosen[...,None].expand(-1,-1,128));prediction=decoded[kind].gather(1,reference[...,None].expand(-1,-1,128)).float();innovation=current-prediction
                    self._innovation(l,temporary,innovation,local,kind);residual=self._read_slots(l,temporary,kind,local[None].expand(2,-1),False).float();recon=(prediction+residual).bfloat16().float();scale=t[kind+'_scale'][l].float()[:,None]
                    before+=(innovation/scale).square().mean(-1);after+=((current-recon)/scale).square().mean(-1)
                gains=before-after;pick_local=torch.argsort(gains,dim=1,descending=True,stable=True)[:,:nr];target=pool.gather(1,pick_local);sort=torch.argsort(target,dim=1,stable=True);pick_local=pick_local.gather(1,sort);target=target.gather(1,sort);actual=seq[target]
                state['T_code'][heads,actual]=codes.gather(1,target)|128
                for kind in ['K','V']:
                    ix=temporary[kind+'_index'];norm=temporary[kind+'_norm'];state[kind+'_index'][:,a:a+nr].copy_(ix.gather(1,pick_local[...,None,None].expand(-1,-1,ix.shape[2],8)));state[kind+'_norm'][:,a:a+nr].copy_(norm.gather(1,pick_local[...,None]))
                selected_gain=gains.gather(1,pick_local).sum();norm_gain=gains[:,:nr].sum();improvement=float(selected_gain-norm_gain);assert improvement>=-1e-4*max(1.,float(selected_gain.abs()))
                state['temporal_candidate_slots_per_head']=count;state['distortion_gain_vs_norm_same_pool_sum']=improvement;state['distortion_selected_actual_gain_sum']=float(selected_gain)
            state['temporal_targets']=len(targets);state['temporal_innovation_heads']=2*nr;state['temporal_copy_heads']=2*(len(targets)-nr);state['distortion_copy_before_innovation_sum']=float(best.sum())
        text=self._text_gpu.long();state['text_FP_K'].copy_(kpost[0][:,text]);state['text_FP_V'].copy_(v[0][:,text]);state['text_filled']=len(self._text_list);state['n']+=n;return state

class DynamicAllocated(DynamicMatchedAppend,DynamicReadDecode,DynamicRVQ,AllocatedPacked):_stable_order=False
class DynamicExactAllocated(DynamicMatchedAppend,DynamicReadDecode,DynamicExactGemm,AllocatedPacked):_stable_order=False
class DynamicGain(DynamicGainSelection,DynamicReadDecode,DynamicRVQ,AllocatedPacked):pass
class DynamicExactGain(DynamicGainSelection,DynamicReadDecode,DynamicExactGemm,AllocatedPacked):pass
