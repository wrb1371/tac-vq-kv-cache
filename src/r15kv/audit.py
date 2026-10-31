"""At compact boundaries, every reachable GPU tensor storage must be registered."""
import torch
from .persistent_registry import storages
def audit_compact_storage(cache):
    tensors={};seen=set()
    def walk(x,path):
        if torch.is_tensor(x):
            s=x.untyped_storage()
            if s.nbytes():tensors.setdefault((str(x.device),s.data_ptr(),s.nbytes()),{'path':path,'bytes':s.nbytes(),'device':str(x.device),'dtype':str(x.dtype),'shape':list(x.shape)})
            return
        if id(x) in seen:return
        seen.add(id(x))
        if isinstance(x,dict):
            for k,v in x.items():walk(v,path+'/'+str(k))
        elif isinstance(x,(list,tuple)):
            for k,v in enumerate(x):walk(v,path+'/'+str(k))
        elif hasattr(x,'__dict__') and not isinstance(x,type):
            for k,v in vars(x).items():walk(v,path+'/'+k)
    walk(cache,'cache_and_codec')
    registered={}
    def keys(x):
        if torch.is_tensor(x):
            s=x.untyped_storage()
            if s.nbytes():registered[(str(x.device),s.data_ptr(),s.nbytes())]=True
        elif isinstance(x,dict):
            for v in x.values():keys(v)
        elif isinstance(x,(list,tuple)):
            for v in x:keys(v)
    for x,kind in cache.registry.roots.values():keys(x)
    unknown=[v for k,v in tensors.items() if k[0].startswith('cuda') and k not in registered]
    assert not unknown,('UNREGISTERED_PERSISTENT_GPU_TENSOR',unknown)
    cpu=sum(v['bytes'] for k,v in tensors.items() if k[0]=='cpu');gpu=sum(v['bytes'] for k,v in tensors.items() if k[0].startswith('cuda'));assert gpu==cache.registry.end,(gpu,cache.registry.end)
    return {'all_reachable_persistent_GPU_storages_registered':True,'audited_unique_GPU_storages':sum(k[0].startswith('cuda') for k in tensors),'audited_GPU_storage_bytes':gpu,'cpu_codec_resident_bytes':cpu,'CPU_scope':'reachable tensors of cache+codec; excludes Python-object overhead, input preprocessing caches, model weights and module-global helper LRU tables'}
