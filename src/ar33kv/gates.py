"""Paid-format, frozen-plan and independent residual-encoding checks."""
import torch
from ar26kv.gates import reader_gate
from ar26kv.matching import references
from r14kv.fitting import nearest

def nbytes(c):
    if c.plan['hybrid_dense']:return c.static+28*4*4*128*c.capacity
    positions=c.global_maps['positions'];total=c.static+3*c.capacity*positions.element_size()+256+4*(c.M+c.T)
    total+=28*(4*4*128*c.T+4*c.capacity)
    for z in c.plan['layer_choices']:total+=4*z['slots']*(8*(z['rK']+z['rV'])+4)
    return total

@torch.no_grad()
def gate(c,teacher,parent):
    assert c.receipt['resident_bytes']==nbytes(c)<=c.budget
    out=[{'profile':c.profile,'independent_physical_bytes_exact':True}]
    if c.plan['hybrid_dense']:return out
    for l in [0,13,27]:out.extend({'profile':c.profile,**r}for r in reader_gate(c,l))
    z=c.plan['layer_choices'];g=c.plan['gop'];a=c.plan['anchors'];p=c.grid[1]*c.grid[2]
    if c.profile in ['NoDecodedFeedback','NoLocalMatch']:
        assert c.plan['layer_choices']==parent['plan']['layer_choices']and g==parent['plan']['gop']
        for l in [0,13,27]:
            for s in ['K','V']:
                for k in [s+'_index',s+'_norm']:assert torch.equal(c.states[l][k][:,:a].cpu(),parent['states'][l][k][:,:a])
        out.append({'profile':c.profile,'parent_plans_and_anchor_codes_exact':True})
    if c.profile=='NoLocalMatch':
        vi=c.global_maps['video'].long();o=torch.arange(c.M,device='cuda');to=o[(o//p)%g!=0]
        assert all(bool(((st['T_code'][:,vi[to]]&127)==4).all())for st in c.states.values())
    if c.profile in ['NoInnovation','Independent']:assert all(v['innovations']==0 for v in z)
    if c.profile=='Independent':assert g==1 and a==c.M
    if c.profile in ['A25_S100','FixedGIStableProbe']:assert g==8
    if c.profile in ['FixedIStableProbe','FixedGIStableProbe']:assert all(v['innovations']==int(__import__('math').ceil((c.M-a)/8))for v in z)
    if c.profile in ['JointContent','UniformJoint']:assert c.plan['floors']==[(1,1)]*28
    if c.profile=='UniformJoint':assert len({(v['rK'],v['rV'],v['innovations'])for v in z})==1
    if c.profile=='AR24_NoProbe':assert all(v=={'K':1.,'V':1.}for v in c.plan['content_alpha'].values())
    if c.profile in ['NoDecodedFeedback','NoLocalMatch']:
        l=0;st=c.states[l];vi=c.global_maps['video'].long();o=torch.arange(c.M,device='cuda');to=o[(o//p)%g!=0];raw={'K':teacher.states[l]['K_pre'].cuda().float(),'V':teacher.states[l]['V'].cuda().float()};codes=st['T_code'][:,vi[to]];refs=references(to,codes&127,c.grid[1],c.grid[2],g);nr=z[l]['innovations']
        for h in range(4):
            ix=torch.nonzero((codes[h]&128)!=0).flatten()[:32]
            if not len(ix):continue
            for s in ['K','V']:
                slot=(refs[h,ix]//p//g)*p+refs[h,ix]%p
                pred=raw[s][h,vi[refs[h,ix]]]if c.profile=='NoDecodedFeedback'else c.read_product(l,s,st,a,True)[h,slot].float()
                residual=raw[s][h,vi[to[ix]]]-pred;t=c.table;u=residual/t[s+'_scale'][l,h].float();rho=u.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6).bfloat16();zz=((u/rho.float())@t['H']).reshape(1,-1,16)
                assert torch.equal(st[s+'_norm'][h,a:a+len(ix)],rho)
                for rr in range(st[s+'_index'].shape[2]):
                    book=t[s+'_books'][rr][None].float();ids=nearest(zz,book)
                    assert torch.equal(st[s+'_index'][h,a:a+len(ix),rr].long(),ids.reshape(len(ix),8))
                    zz-=torch.gather(book,1,ids[...,None].expand(-1,-1,16))
        out.append({'profile':c.profile,'residual_reference_and_stage_indices_independently_checked':True})
    return out
