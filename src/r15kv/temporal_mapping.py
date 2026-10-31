"""Actual merged spatial support, fixed four-temporal-group anchor dependencies."""
import hashlib,torch
def spatial_permutation(source,window,spatial):
    seed=int(hashlib.sha256(('R15-G2-VR1:'+source+':'+str(window)).encode()).hexdigest()[:15],16)
    return torch.randperm(spatial,generator=torch.Generator().manual_seed(seed))
def video_mapping(tags,positions,grid,capacity,source,hash_pairing=False):
    tags=tags.cpu();positions=positions.cpu();vid=(tags==2).nonzero(as_tuple=True)[0];t,h,w=map(int,grid[0]);s=(h//2)*(w//2)
    assert len(vid)==t*s and t>=1;native=vid.reshape(t,s)
    for j in range(1,t):assert torch.equal(positions[1:,native[0]],positions[1:,native[j]]),'merged spatial support changed within video'
    kind=torch.zeros(capacity,dtype=torch.uint8);anchor=native[::4].flatten();kind[vid]=2;kind[anchor]=1;pred=(kind==2).nonzero(as_tuple=True)[0];non=(kind==0).nonzero(as_tuple=True)[0];refs=[]
    perms={j:spatial_permutation(source,j,s) for j in range((t+3)//4)}
    for frame in range(t):
        if frame%4==0:continue
        group=frame//4;spatial=perms[group] if hash_pairing else torch.arange(s);refs.extend((group*s+spatial).tolist())
    refs=torch.tensor(refs,dtype=torch.int32);assert len(refs)==len(pred)
    if len(pred):assert torch.all(anchor[refs.long()]<pred),'predictor may only use earlier temporal anchors'
    return {'kind':kind,'anchor_native':anchor.to(torch.int32),'pred_native':pred.to(torch.int32),'nonvideo_native':non.to(torch.int32),'anchor_ref':refs}, {'temporal_groups':t,'spatial_support_tokens':s,'anchor_stride_temporal_groups':4,'anchor_tokens':len(anchor),'predicted_tokens':len(pred),'nonvideo_capacity':len(non),'hash_pairing':hash_pairing}
