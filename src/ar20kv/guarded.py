"""Conservative distance-margin guard for FP32 nearest-centroid ambiguity.

Triton and cuBLAS can disagree near a boundary. Re-evaluate flagged groups with
the original PyTorch routine; also provide whole-chunk repair as an ablation.
The guard is not a proof for every possible hardware/input reduction order.
"""
import torch
import triton
import triton.language as tl
from r14kv.fitting import nearest
from .codec import FullStable, FullPacked


@triton.jit
def _guard_rvq(X,B,O,Slots,Bad,ROWS:tl.constexpr,NT:tl.constexpr,OS0:tl.constexpr,OS1:tl.constexpr,R:tl.constexpr,BM:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM);d=tl.arange(0,16);c=tl.arange(0,256)
    z=tl.load(X+row[:,None]*16+d[None,:],row[:,None]<ROWS,0)
    head=row//(NT*8);tok=row//8%NT;group=row%8
    slot=tl.load(Slots+tok,row<ROWS,0);ambiguous=tl.full((BM,),False,tl.int1)
    for r in tl.static_range(R):
        book=tl.load(B+r*4096+c[None,:]*16+d[:,None]).to(tl.float32)
        norm=tl.sum(book*book,0);xx=tl.sum(z*z,1);dot=tl.dot(z,book,input_precision='ieee')
        dist=(xx[:,None]+norm[None,:])-2.0*dot
        chosen=tl.argmin(dist,1,tie_break_left=True);best=tl.min(dist,1)
        second=tl.min(tl.where(c[None,:]==chosen[:,None],float('inf'),dist),1)
        normmax=tl.max(norm,0)
        bound=4.e-6*(xx+normmax+2.0*tl.sqrt(xx*normmax))
        ambiguous=ambiguous|((second-best)<=bound)
        tl.store(O+head*OS0+slot*OS1+r*8+group,chosen.to(tl.uint8),row<ROWS)
        picked=tl.load(B+r*4096+chosen[:,None]*16+d[None,:]).to(tl.float32)
        z=z-picked
    tl.store(Bad+row,ambiguous.to(tl.uint8),row<ROWS)


class GuardRVQ:
    _whole_chunk_guard=False
    @torch.no_grad()
    def _innovation(self,l,state,raw,slots,kind):
        t=self.tables[0];out=state[kind+'_index'];books=t[kind+'_books'];r=out.shape[2]
        for b in range(0,len(slots),512):
            x=raw[:,b:b+512].float();n=x.shape[1];u=x/t[kind+'_scale'][l].float()[:,None]
            rho=u.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6).bfloat16();where=slots[b:b+n];state[kind+'_norm'][:,where]=rho
            z=((u/rho.float())@t['H']).reshape(1,2*n*8,16)
            flags=torch.empty(2*n*8,device='cuda',dtype=torch.uint8)
            _guard_rvq[(triton.cdiv(2*n*8,32),)](z,books,out,where.contiguous(),flags,2*n*8,n,out.stride(0),out.stride(1),r,32,num_warps=4,enable_fp_fusion=False)
            ids=flags.nonzero().flatten()
            state['temporal_guard_chunks']=state.get('temporal_guard_chunks',0)+1
            state['temporal_guard_groups']=state.get('temporal_guard_groups',0)+len(ids)
            if not len(ids):continue
            state['temporal_guard_repair_chunks']=state.get('temporal_guard_repair_chunks',0)+1
            if self._whole_chunk_guard:
                for stage in range(r):
                    cb=books[stage][None].float();ix=nearest(z,cb)
                    out[:,where,stage]=ix.reshape(2,n,8).to(torch.uint8)
                    z-=torch.gather(cb,1,ix[...,None].expand(-1,-1,16))
            else:
                zz=z[:,ids].clone();heads=ids//(n*8);tokens=ids//8%n;groups=ids%8
                for stage in range(r):
                    cb=books[stage][None].float();ix=nearest(zz,cb)
                    out[heads,where[tokens],stage,groups]=ix[0].to(torch.uint8)
                    zz-=torch.gather(cb,1,ix[...,None].expand(-1,-1,16))


class GuardStable(GuardRVQ,FullStable): pass
class GuardPacked(GuardRVQ,FullPacked): pass
class StrictGuardStable(GuardRVQ,FullStable): _whole_chunk_guard=True
class StrictGuardPacked(GuardRVQ,FullPacked): _whole_chunk_guard=True
