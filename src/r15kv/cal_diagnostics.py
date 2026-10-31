"""Small CAL-only output-space probes; no T² tensor or task-result substitution."""
import collections,hashlib,math,torch
from r14kv.fitting import nearest
def hash_modalities(tags,input_sha,block=2048):
    """Match each input block's true group counts, not its semantic membership."""
    tags=tags.cpu();out=torch.empty_like(tags)
    for b in range(0,len(tags),block):
        z=tags[b:b+block];gen=torch.Generator().manual_seed(int(hashlib.sha256(('R15-N2-hash:'+input_sha+str(b)).encode()).hexdigest()[:15],16));ix=torch.randperm(len(z),generator=gen);offset=0
        for mod in [0,1,2,3]:
            count=int((z==mod).sum());out[b+ix[offset:offset+count]]=mod;offset+=count
    return out
def quant_token(x,bits,sign=False):
    if sign:return (torch.where(x>=0,1.,-1.)*x.abs().mean(-1,keepdim=True).bfloat16().float())
    threshold=x.abs().sort(-1).values[...,int(.92*128):int(.92*128)+1];clipped=x.clamp(-threshold,threshold);lo=clipped.amin(-1,keepdim=True);hi=clipped.amax(-1,keepdim=True)
    scale=((hi-lo).clamp_min(1e-8)/(2**bits-1)).bfloat16().float();zero=(-lo/scale).bfloat16().float();q=(clipped/scale+zero).round().clamp(0,2**bits-1);return (q-zero)*scale
def quant_channel(x,tags,bits,w,grouping='continuous',input_sha=''):
    n=x.shape[1];ix=torch.arange(n);groups=[ix] if grouping=='continuous' else [ix[tags.cpu()==m] for m in [0,1,2,3] if (tags.cpu()==m).any()]
    if grouping=='hash':
        gen=torch.Generator().manual_seed(int(hashlib.sha256(('R15-N1:'+input_sha).encode()).hexdigest()[:15],16));perm=ix[torch.randperm(n,generator=gen)];start=0;new=[]
        for g in groups:new.append(perm[start:start+len(g)].sort().values);start+=len(g)
        groups=new
    out=torch.empty_like(x);windows=0
    for g in groups:
        for b in range(0,len(g),w):
            ids=g[b:b+w].to(x.device);z=x.index_select(1,ids);lo=z.amin(1,keepdim=True).bfloat16().float();scale=((z.amax(1,keepdim=True)-z.amin(1,keepdim=True)).clamp_min(1e-8)/(2**bits-1)).bfloat16().float();q=((z-lo)/scale).round().clamp(0,2**bits-1);out[:,ids]=q*scale+lo;windows+=1
    return out,windows
def project_heads(x,weight):
    # [Hkv,8,Q,D] -> true concatenated heads -> W_O.
    x=x.reshape(16,x.shape[2],128).transpose(0,1).reshape(x.shape[2],2048);return x.float()@weight.float().T
def attention_probabilities(q,k,qindices):
    keypos=torch.arange(k.shape[1],device=k.device);qidx=torch.tensor(qindices,device=k.device);out=[]
    for h in range(2):
        logits=torch.matmul(q[h*8:(h+1)*8].float(),k[h].float().T)/math.sqrt(128);logits.masked_fill_(keypos[None,None,:]>qidx[None,:,None],-torch.inf);out.append(logits.softmax(-1))
    return torch.stack(out)
def value_probe(v,probs,basis,tags,weight,input_sha):
    lo=32;hi=max(lo,v.shape[1]-128);rows=[];reference=torch.stack([torch.matmul(probs[h],v[h].float()) for h in range(2)]);den=project_heads(reference,weight).square().sum().clamp_min(1e-30)
    for space in ['raw','rotated']:
        x=v[:,lo:hi].float() if space=='raw' else torch.bmm(v[:,lo:hi],basis).float()
        for name,bits,w,grouping in [('token_affine2',2,0,''),('token_affine1',1,0,''),('token_sign1',1,0,''),('channel_W128_V2',2,128,'continuous'),('channel_W256_V2',2,256,'continuous'),('channel_W256_V1',1,256,'continuous'),('modal_W256_V1',1,256,'modality'),('hash_W256_V1',1,256,'hash')]:
            if w:rec,nw=quant_channel(x,tags[lo:hi],bits,w,grouping,input_sha)
            else:rec=quant_token(x,bits,name=='token_sign1');nw=0
            restored=rec if space=='raw' else torch.bmm(rec,basis.float().transpose(1,2));dv=restored.bfloat16().float()-v[:,lo:hi].float();delta=torch.stack([torch.matmul(probs[h,:,:,lo:hi],dv[h]) for h in range(2)]);post=project_heads(delta,weight)
            wt=probs[:,:,:,lo:hi].square().sum((1,2));rows.append({'space':space,'quantizer':name,'bits':bits,'windows':nw,'raw_reconstruction_MSE':float(dv.square().mean()),'attention_weighted_reconstruction':float((dv.square().sum(-1)*wt).sum()/wt.sum().clamp_min(1e-30)),'attention_output_squared_error':float(delta.square().sum()),'relative_post_WO_squared_error':float(post.square().sum()/den),'quantizer_payload_and_metadata_bytes_analytical':2*(x.shape[1]*128*bits//8+(nw*128*4 if w else x.shape[1]*(2 if name=='token_sign1' else 4))),'scope':'CAL proxy; full deployment routing/band/codec/capacity separately measured in real candidates'})
    return rows
def rq_reconstruct(x,book,r=8):
    residual=x.float().clone();result=torch.zeros_like(residual)
    for stage in range(r):
        cb=book[:,stage].float();ix=nearest(residual,cb);chosen=torch.gather(cb,1,ix[...,None].expand(-1,-1,128));result+=chosen;residual-=chosen
    return result
def asymmetric_probe(pre,v,postk,probs,q,qidx,cos,sin,books,weight):
    from r14kv.common import apply_key_rope
    ids=torch.linspace(0,v.shape[1]-1,min(4096,v.shape[1]),device=v.device).round().long();rows=[];reference=torch.stack([torch.matmul(probs[h],v[h].float()) for h in range(2)]);den=project_heads(reference,weight).square().sum().clamp_min(1e-30);keypos=torch.arange(v.shape[1],device=v.device);qpositions=torch.tensor(qidx,device=v.device)
    for rk,rv in [(12,12),(13,11),(11,13)]:
        kr=rq_reconstruct(pre.index_select(1,ids),books['K'],rk);vr=rq_reconstruct(v.index_select(1,ids),books['V'],rv);kp=apply_key_rope(kr[None].bfloat16(),cos.index_select(2,ids),sin.index_select(2,ids))[0].float();dv=vr.bfloat16().float()-v.index_select(1,ids).float();err=[]
        for h in range(2):
            logits=torch.matmul(q[h*8:(h+1)*8].float(),postk[h].float().T)/math.sqrt(128);logits[:,:,ids]=torch.matmul(q[h*8:(h+1)*8].float(),kp[h].T)/math.sqrt(128);logits.masked_fill_(keypos[None,None,:]>qpositions[None,:,None],-torch.inf);pp=logits.softmax(-1);err.append(torch.matmul(pp-probs[h],v[h].float())+torch.matmul(pp.index_select(-1,ids),dv[h]))
        rows.append({'R_K':rk,'R_V':rv,'stage_sum':24,'matched_codec_MiB':54,'sampled_all_modality_tokens':len(ids),'relative_post_WO_squared_error':float(project_heads(torch.stack(err),weight).square().sum()/den),'scope':'CAL same-sample partial intervention proxy, not main task or full deployment measurement'})
    return rows
def temporal_probe(pre,v,postk,probs,q,qidx,positions,cos,sin,tags,grid,books,weight,input_sha):
    from r14kv.common import apply_key_rope
    vid=(tags==2).nonzero(as_tuple=True)[0];t,h,w=map(int,grid[0]);s=(h//2)*(w//2);assert len(vid)==t*s
    mapping=vid.reshape(t,s);assert t>=4
    for j in range(1,t):assert torch.equal(positions[1:,mapping[0]],positions[1:,mapping[j]]),'spatial support must match, not neighboring sequence positions'
    possible=list(range(0,t-3,4));starts=sorted(set([possible[0],possible[len(possible)//2],possible[-1]]));groups=torch.stack([mapping[b:b+4] for b in starts]);ids=groups.flatten().to(v.device);nblock=len(starts);nq=len(qidx);rows=[]
    reference=torch.stack([torch.matmul(probs[h],v[h].float()) for h in range(2)]);den=project_heads(reference,weight).square().sum().clamp_min(1e-30)
    gen=torch.Generator().manual_seed(int(hashlib.sha256(('R15-G2:'+input_sha).encode()).hexdigest()[:15],16));perm=torch.randperm(s,generator=gen).to(v.device)
    for kind,full in [('K',pre),('V',v)]:
        x=full.index_select(1,ids).float();shape=(2,nblock,4,s,128);xb=x.reshape(shape);anchor=rq_reconstruct(xb[:,:,0].reshape(2,-1,128),books[kind]).reshape(2,nblock,1,s,128)
        independent=rq_reconstruct(x,books[kind]);candidates={'independent':independent};ratios={'independent':None}
        for mode in ['true_time','hash_pairing']:
            a=anchor if mode=='true_time' else anchor.index_select(3,perm);expanded=a.expand(-1,-1,3,-1,-1);residual=xb[:,:,1:]-expanded;ratios[mode]=float(residual.square().sum()/xb[:,:,1:].square().sum().clamp_min(1e-30));tail=rq_reconstruct(residual.reshape(2,-1,128),books[kind]).reshape(2,nblock,3,s,128);rec=torch.cat([anchor,a.expand(-1,-1,3,-1,-1)+tail],2).reshape(2,-1,128);candidates[mode]=rec
        for mode,rec in candidates.items():
            delta_raw=rec-x;wt=probs.index_select(-1,ids).square().sum((1,2));err=torch.zeros(2,8,nq,128,device=v.device)
            if kind=='V':
                for h0 in range(2):err[h0]=torch.matmul(probs[h0].index_select(-1,ids),delta_raw[h0])
            else:
                kpost=apply_key_rope(rec[None].bfloat16(),cos.index_select(2,ids),sin.index_select(2,ids))[0].float();keypos=torch.arange(v.shape[1],device=v.device);qpositions=torch.tensor(qidx,device=v.device)
                for h0 in range(2):
                    logits=torch.matmul(q[h0*8:(h0+1)*8].float(),postk[h0].float().T)/math.sqrt(128);logits[:,:,ids]=torch.matmul(q[h0*8:(h0+1)*8].float(),kpost[h0].T)/math.sqrt(128);logits.masked_fill_(keypos[None,None,:]>qpositions[None,:,None],-torch.inf);pp=logits.softmax(-1);err[h0]=torch.matmul(pp-probs[h0],v[h0].float())
            rows.append({'kind':kind,'pairing':mode,'groups':nblock,'temporal_groups_per_anchor':4,'spatial_tokens':s,'probe_video_tokens':len(ids),'R_anchor':8,'R_residual':8,'same_raw_FP8_books_reused':True,'weighted_reconstruction':float((delta_raw.square().sum(-1)*wt).sum()/wt.sum().clamp_min(1e-30)),'relative_post_WO_squared_error':float(project_heads(err,weight).square().sum()/den),'raw_predictor_residual_norm_ratio':ratios[mode],'scope':'CAL small full-input intervention probe; decoded anchor; no fitted residual dictionary or task-quality claim'})
    return rows
