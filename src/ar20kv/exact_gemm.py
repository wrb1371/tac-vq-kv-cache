"""Keep original cuBLAS GEMM shapes; fuse only its distance/search/update tail."""
import torch
import triton
import triton.language as tl
from .codec import FullDecode,FusedRead,MatchedAppend
from ar19kv.stable import HybridStablePacked
from ar19kv.packed_sparse import HybridPackedSparse


@triton.jit
def _tail(Dot,XX,CN,Book,Z,Out,Slots,ROWS:tl.constexpr,NT:tl.constexpr,OS0:tl.constexpr,OS1:tl.constexpr,STAGE:tl.constexpr):
    row=tl.program_id(0);c=tl.arange(0,256);d=tl.arange(0,16)
    dot=tl.load(Dot+row*256+c);xx=tl.load(XX+row);cn=tl.load(CN+c)
    distance=(xx+cn)-2.0*dot
    index=tl.argmin(distance,0,tie_break_left=True)
    head=row//(NT*8);token=row//8%NT;group=row%8;slot=tl.load(Slots+token)
    tl.store(Out+head*OS0+slot*OS1+STAGE*8+group,index.to(tl.uint8))
    z=tl.load(Z+row*16+d)
    reconstruction=tl.load(Book+index*16+d).to(tl.float32)
    tl.store(Z+row*16+d,z-reconstruction)


class ExactGemm:
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


class ExactStable(FullDecode,FusedRead,ExactGemm,HybridStablePacked): pass
class ExactPacked(FullDecode,FusedRead,ExactGemm,HybridPackedSparse): pass
class ExactMatchStable(FullDecode,MatchedAppend,FusedRead,ExactGemm,HybridStablePacked): _stable_order=True
class ExactMatchPacked(FullDecode,MatchedAppend,FusedRead,ExactGemm,HybridPackedSparse): _stable_order=False
