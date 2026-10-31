"""Integer physical-layout admission, independent of scores and GPU fitting."""
import math
L,H,D=28,4,128
METHODS=['NOVA-KV-g8','NOVA-KV-g4','NSNQuant']
SIZE={'u8':1,'bf16':2,'f16':2,'i16':2,'i32':4,'f32':4}
NOVA_STATIC=L*H*(3*D*D*2+D*2+D*16)
NSN_STATIC=40960
def nova_static(centroids):return L*H*(3*D*D*2+D*2+D*centroids)
def nbytes(desc):
    total=0
    for _,shape,dtype,_ in desc:
        b=SIZE[dtype];total=(total+b-1)//b*b+math.prod(shape)*b
    return total
def text_desc(t):return [(s,(H,t,D),'bf16','payload')for s in ['text_FP_K','text_FP_V']]
def nova_desc(P,t,group,sink,recent,mode,centroids=16):
    a=min(sink,P);b=min(recent,P-a);n=P-a-b;vb=2 if mode=='affine2'else 1
    rows=[(s,(H,a,D),'bf16','payload')for s in ['firstK','firstV']]
    rows +=[(s,(H,b,D),'bf16','payload')for s in ['recentK','recentV']]
    rows +=[('K_index',(H,n,(D//group*int(math.log2(centroids))+7)//8),'u8','payload'),('rho',(H,n,1),'bf16','metadata'),
            ('V_index',(H,n,D*vb//8),'u8','payload'),('V_scale',(H,n,1),'bf16','metadata')]
    if mode!='sign1':rows.append(('V_zero',(H,n,1),'bf16','metadata'))
    return rows+text_desc(t)
def nsn_desc(P,t,bits):
    blocks=P//64;n=blocks*64;rows=[]
    for s in ['K','V']:
        rows +=[(s+'_index',(H,n,16),'u8','payload'),(s+'_sign',(H,n,16 if bits==2 else 0),'u8','payload'),
                 (s+'_norm1_index',(H,blocks,32),'u8','metadata'),(s+'_norm1_scale',(H,blocks,1),'f16','metadata'),
                 (s+'_norm1_offset',(H,blocks,1),'f16','metadata'),(s+'_mean_index',(H,blocks,64),'u8','metadata'),
                 (s+'_mean_scale',(H,blocks,4),'f16','metadata'),(s+'_mean_offset',(H,blocks,4),'f16','metadata'),
                 (s+'_norm2',(H,n,1),'f16','metadata'),(s+'_pending',(H,64,D),'bf16','payload')]
    return rows+text_desc(t)
def candidates(method,P,t,bpos):
    if method.startswith('NOVA'):
        g=8 if method.endswith('g8')else 4
        bands=[(64,256),(32,128),(0,0)]if g==8 else[(8,32)]
        for centroids in [256,16]:
            for mode in ['affine2','affine1','sign1']:
                for sink,recent in bands:
                    desc=nova_desc(P,t,g,sink,recent,mode,centroids);migration=4*H*D*min(recent,max(0,P-sink));static=nova_static(centroids)
                    yield {'group_size':g,'centroids':centroids,'K_index_bits':int(math.log2(centroids)),
                           'v_mode':mode,'v_bits':2 if mode=='affine2'else 1,'sink':sink,'recent':recent,
                           'description':desc,'codec_bytes':static,'migration_bytes':migration,'global_bytes':4*t+migration,
                           'resident_bytes':static+4*t+migration+L*nbytes(desc)}
    else:
        for bits in [2,1]:
            desc=nsn_desc(P,t,bits);global_bytes=3*P*bpos+256+4*t
            yield {'bits':bits,'window':64,'description':desc,'codec_bytes':NSN_STATIC,'global_bytes':global_bytes,
                   'migration_bytes':0,'resident_bytes':NSN_STATIC+global_bytes+L*nbytes(desc)}
def select(method,P,t,bpos,budget_bytes):
    rows=list(candidates(method,P,t,bpos));feasible=[r for r in rows if r['resident_bytes']<=budget_bytes]
    selected=dict(feasible[0])if feasible else None
    if selected:selected.update(capacity=P,budget_bytes=budget_bytes,position_bytes=bpos,hybrid_dense=False)
    return selected,rows
