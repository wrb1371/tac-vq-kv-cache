"""Budgeted finite-prefix ports of published video KV selectors.

Official selector functions are retained; this is NOT an end-to-end streaming
reported-number reproduction. Native post-MRoPE K and V are physically packed.
"""
import importlib.util,types,pathlib,torch
from r15kv.codecs import PackageCodec
from r15kv.persistent_registry import Arena
from r14kv.common import ROOT

def reference_module(name):
    path=ROOT/'third_party/ar19_selector_ports'/name/'kvcache_utils.py'
    spec=importlib.util.spec_from_file_location('ar19_official_'+name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class VideoPrune(PackageCodec):
    method='AR19-video-KV-selector-port';needs_positions=False;needs_pre_rope_K=False
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.selector=self.policy['selector']
        self.official=None if self.selector=='uniform' else reference_module(self.selector)
        self.metadata.update(port_scope='one-shot finite prefix, all 36 layers, actual GPU budget; not full streaming reproduction',selection_K_space='native post-MRoPE',protected='all native text/special positions; question retained')
    def prepare_request(self,meta,registry):
        tags=self.context['tags'];self._video_list=(tags==2).nonzero().flatten().tolist();self._end=self._video_list[-1]+1
        self._text_list=(tags[:self._end]==0).nonzero().flatten().tolist();grid=self.context['video_grid_thw'];assert grid.shape==(1,3)
        self._nt,self._h,self._w=map(int,grid[0].tolist());self._h//=2;self._w//=2;assert len(self._video_list)==self._nt*self._h*self._w
        plan=super().prepare_request(meta,registry)
        self._video_gpu=torch.tensor(self._video_list,device='cuda',dtype=torch.int32);self._text_gpu=torch.tensor(self._text_list,device='cuda',dtype=torch.int32)
        registry.own('cache/video_sequence_positions',self._video_gpu);registry.own('cache/text_sequence_positions',self._text_gpu)
        return plan
    def make_plan(self,meta):
        extra=4*(len(self._video_list)+len(self._text_list));slots=min(meta.capacity,(self.budget_bytes-extra)//(36*1024));qreserve=meta.capacity-self._end
        keep=min(len(self._video_list),slots-len(self._text_list)-qreserve)
        if keep<1:raise RuntimeError('INFEASIBLE_BY_INTEGER_ORACLE_VIDEO_KV')
        desc=[(k,(2,slots,128),torch.bfloat16,'payload') for k in ['K_BF16','V_BF16']]
        total=extra+36*Arena.bytes(desc);assert total<=self.budget_bytes
        return {'description':desc,'capacity':meta.capacity,'physical_slots_per_layer':slots,'media_kept_per_head':keep,'media_original':len(self._video_list),'protected_prefix_text':len(self._text_list),'reserved_query_slots':qreserve,'merged_video_grid':[self._nt,self._h,self._w],'selector':self.selector,'all_layers_compressed_when_needed':True,'query_agnostic':True,'estimate':{'codec_static_bytes':0,'global_request_bytes':extra,'layer_arena_bytes':Arena.bytes(desc),'resident_capacity_bytes':total,'kind':'INTEGER_ESTIMATE_NOT_MEASUREMENT'}}
    def allocate_state(self,l,registry):
        s=super().allocate_state(l,registry);s['nk']=0;return s
    def _selected(self,l,k,v):
        keep=self.plan['media_kept_per_head'];p=self._h*self._w;n=k.shape[-2]
        if keep==n:return torch.arange(n,device=k.device)[None].expand(2,-1)
        if self.selector=='uniform':
            ix=(torch.arange(keep,device='cuda')*n//keep).long();return ix[None].expand(2,-1)
        # At very tight budget, the published fixed recent fraction can exceed
        # the entire admitted cache. Reduce it before selection, never overrun.
        recent=min(int(self._nt*.25),(keep-round(keep*.5))//p)
        qr=(recent+1e-8)/self._nt if recent else 0.0;tar=.5 if recent else 0.0
        dummy=types.SimpleNamespace(config=types.SimpleNamespace(height_width=self._h),model=types.SimpleNamespace(language_model=types.SimpleNamespace(layers=[None])))
        if self.selector=='infinipot':
            layer=types.SimpleNamespace(keys=k,values=v)
            class Cache:
                layers=[layer]
                def get_seq_length(self):return self.layers[0].keys.shape[-2]
            cache,_=self.official.process_kv_cache(Cache(),dummy,0,0,p,keep/p,method='infinipot-v',tar_ratio=tar,query_ratio=qr,adaptive_pooling=False)
            return cache.selection_indices
        assert self.selector=='cords'
        ix=self.official._cords_select_indices(k,v,2,p,self._nt,keep/p,tar,qr,-1,1,dummy,8,'cords',.5,1.0,'maxcos',.25,0.0,1.0,'none',False,1)
        assert ix.shape==(2,keep)
        return ix
    @torch.no_grad()
    def append(self,l,state,kpost,v,pre=None):
        start=state['n'];n=kpost.shape[-2]
        if start==0:
            assert n==self._end
            vi=self._video_gpu.long();ti=self._text_gpu.long();selected=self._selected(l,kpost[:,:,vi],v[:,:,vi])
            ordered=torch.cat([ti[None].expand(2,-1),vi[selected]],1).sort(1).values
            count=ordered.shape[1];gather=ordered[None,...,None].expand(1,2,count,128)
            k=kpost.gather(2,gather);vv=v.gather(2,gather)
            state['K_BF16'][:,:count].copy_(k[0]);state['V_BF16'][:,:count].copy_(vv[0]);state['nk']=count
        else:
            nk=state['nk'];assert nk+n<=state['K_BF16'].shape[1]
            state['K_BF16'][:,nk:nk+n].copy_(kpost[0]);state['V_BF16'][:,nk:nk+n].copy_(v[0]);state['nk']+=n
        state['n']+=n;return state
    def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
        return state['K_BF16'][:,:state['nk']][None],state['V_BF16'][:,:state['nk']][None]
    def finish_request(self):
        self._video_gpu=None;self._text_gpu=None;self._video_list=[];self._text_list=[];super().finish_request()
