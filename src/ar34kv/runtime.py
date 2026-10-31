"""Native prefix teacher plus full pre-RoPE query sufficient statistics."""
import torch
from ar32kv.runtime import Teacher as OriginalTeacher,geometry

class Teacher(OriginalTeacher):
    def __init__(self,model,kind):
        self.model=model;self.kind=kind;self.language,self.geo=geometry(model,kind);self.active=False;self.handles=[]
        L,N,H,D=self.geo
        for l,block in enumerate(self.language.layers):
            def qhook(module,args,out,l=l):
                if not self.active:return
                q=out.detach().view(1,-1,N,D).transpose(1,2)
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
                recent=q[:,:,-64:]
                recent,_=apply_multimodal_rotary_pos_emb(recent,recent,self.cos[:,:,-64:],self.sin[:,:,-64:],self.sections)
                self.queries[l]=recent[0].contiguous().cpu()
                # All prefix rows after the four sink positions; no sampling.
                z=q[0,:,4:].float();mu=z.mean(1);centered=z-mu[:,None]
                cov=torch.matmul(centered.transpose(1,2),centered)/z.shape[1]
                R=self.future_R
                self.statistics[l]={'mean':(mu@R.T).cpu(),'cov':(R@cov@R.T).cpu()}
                if self.gate:
                    from .gates import statistics_gate
                    self.stat_checks.append(statistics_gate(z,mu,cov,R,self.statistics[l]))
            self.handles.append(block.self_attn.q_proj.register_forward_hook(qhook))
        self.gate=False
    @torch.inference_mode()
    def capture(self,x,pos,end):
        D=self.geo[-1];self.statistics={};self.stat_checks=[]
        pp=pos[:,:end][:,None]if self.kind=='Omni'else pos[:,:,:end]
        start=int(pp.max().item())+1
        future=torch.arange(start,start+512,device='cuda').view(1,1,-1).expand(3,1,-1)
        c,s=self.language.rotary_emb(torch.empty((1,1,D),device='cuda',dtype=torch.float32),future)
        # Future text positions advance all three native MRoPE axes together.
        sections=self.language.layers[0].self_attn.rope_scaling['mrope_section']*2
        c=torch.cat([a[i%3]for i,a in enumerate(c.split(sections,dim=-1))],dim=-1)[0].float()
        s=torch.cat([a[i%3]for i,a in enumerate(s.split(sections,dim=-1))],dim=-1)[0].float()
        P=torch.zeros((D,D),device='cuda');P[D//2:,:D//2]=torch.eye(D//2,device='cuda');P[:D//2,D//2:]=-torch.eye(D//2,device='cuda')
        self.future_R=(c[:,None]*torch.eye(D,device='cuda')+s[:,None]*P).mean(0)
        self.future_start=start
        super().capture(x,pos,end)
        # Original capture selects a union of rows; this teacher retains only64.
        self.rows=torch.arange(max(0,end-64),end,device='cuda')
