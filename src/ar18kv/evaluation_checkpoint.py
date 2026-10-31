"""Quality-only evaluation accelerator: one CPU compressed pre-question checkpoint.

No deployed policy change; no original dense full-history checkpoint. Cold latency
must come from the unaccelerated evaluator, never from restored requests here.
"""
import hashlib,json,time,torch
class PrefixCheckpoint:
    def __init__(self):self.saved=None;self.hits=0;self.misses=0;self.peak_CPU_bytes=0
    def key(self,cache,inputs,positions,split,imeta):
        if not hasattr(cache,'codec') or getattr(cache.codec,'_hybrid_dense',False):return None
        p={k:v for k,v in cache.codec.plan.items() if k not in ['capacity','description','descriptions','estimate','base_V1_resident_bytes','V2_tokens_across_layers','low_tokens_across_layers','remaining_bytes','minimum_legal_upgrade_bytes']}
        payload={'prefix_tokens_SHA256':hashlib.sha256(inputs['input_ids'][:,:split].cpu().numpy().tobytes()).hexdigest(),'prefix_positions_SHA256':hashlib.sha256(positions[:,:split].cpu().numpy().tobytes()).hexdigest(),'features':imeta['native_media_feature_SHA256'],'raw_media':imeta['raw_media_SHA256'],'package':cache.codec.package_sha256,'policy':cache.codec.policy,'representation':p,'split':split}
        return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    def restore(self,key,cache,positions,split):
        if key is None or self.saved is None or self.saved['key']!=key:self.misses+=1;return False
        saved=self.saved
        assert len(saved['layers'])==36 and not cache.states and cache.get_seq_length()==0
        for l,record in enumerate(saved['layers']):
            state=cache.codec.allocate_state(l,cache.registry);views=state['arena'].views
            assert set(views)==set(record['views'])
            for name,dst in views.items():
                src=record['views'][name];assert dst.dtype==src.dtype and dst.ndim==src.ndim;sl=tuple(slice(0,min(a,b)) for a,b in zip(dst.shape,src.shape));dst[sl].copy_(src[sl])
            state.update(record['scalars']);assert state['n']==split;cache.states[l]=state
        cache.key_cache=[torch.empty(0,device='cuda',dtype=torch.bfloat16) for _ in range(36)];cache.value_cache=[torch.empty(0,device='cuda',dtype=torch.bfloat16) for _ in range(36)];cache._seen_tokens=split;cache.capture_positions(positions[:,:split][:,None]);cache.registry.observe('evaluation_compressed_prefix_checkpoint_restored');self.hits+=1;return True
    def capture(self,key,cache,split):
        self.saved=None
        if key is None:return
        layers=[];total=0
        for l in range(36):
            state=cache.states[l];assert state['n']==split and not cache.pre and not cache.incoming
            views={name:t.detach().to('cpu',copy=True) for name,t in state['arena'].views.items()};scalars={}
            for name,value in state.items():
                if name=='arena' or name in views:continue
                assert isinstance(value,(int,float,bool,str,type(None))),(name,type(value));scalars[name]=value
            total+=sum(t.untyped_storage().nbytes() for t in views.values());layers.append({'views':views,'scalars':scalars})
        assert total<=256*2**20;self.saved={'key':key,'layers':layers,'CPU_bytes':total};self.peak_CPU_bytes=max(self.peak_CPU_bytes,total)
