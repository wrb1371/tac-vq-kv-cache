"""G1: real AV-time windows and paid, genuinely sparse residual-stage tails."""
import collections,hashlib,json,math,pathlib,torch
from .codecs import GSRQBudget
from .budget import gsrq_description,gs_split,estimate_resident_bytes
from .persistent_registry import Arena
ACTIVE=None
def install_query_capture():
    import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as hf
    if hasattr(hf,'_r15_original_rope'):return
    hf._r15_original_rope=hf.apply_multimodal_rotary_pos_emb
    def wrapped(q,k,cos,sin,sections,**kwargs):
        qp,kp=hf._r15_original_rope(q,k,cos,sin,sections,**kwargs)
        if ACTIVE is not None and qp.shape[-2]>1:ACTIVE.capture_question_q(qp)
        return qp,kp
    hf.apply_multimodal_rotary_pos_emb=wrapped
def temporal_windows(tags,positions,capacity,seconds=2,shuffled=False,input_sha=''):
    tags=tags.cpu();positions=positions.cpu();n=len(tags);media=(tags==1)|(tags==2);origin=int(positions[0,media].min()) if media.any() else 0
    groups=collections.OrderedDict()
    for i in range(n):
        mod=int(tags[i]);key=(mod,int((int(positions[0,i])-origin)//(25*seconds))) if mod in [1,2] else (0,i//128)
        groups.setdefault(key,[]).append(i)
    for i in range(n,capacity):groups.setdefault((0,i//128),[]).append(i)
    if shuffled:
        for mod in [1,2]:
            keys=[k for k in groups if k[0]==mod];flat=[i for k in keys for i in groups[k]];gen=torch.Generator().manual_seed(int(hashlib.sha256(('R15-G1-shuffle:'+input_sha+str(mod)).encode()).hexdigest()[:15],16));perm=torch.tensor(flat)[torch.randperm(len(flat),generator=gen)].tolist();offset=0
            for k in keys:c=len(groups[k]);groups[k]=perm[offset:offset+c];offset+=c
    return list(groups.values()),[int(k[0]) for k in groups],{'temporal_axis':'native MRoPE temporal coordinates,25 ticks/s; video grid supplied and validated separately','seconds':seconds,'media_origin':origin,'shuffled_codec_grouping_only':shuffled}
def question_token_indices(input_ids,item_id,limit=16):
    # The official question body, never gold or a distinguished correct option.
    from r14kv.common import ROOT
    from transformers import AutoTokenizer
    if not hasattr(question_token_indices,'annotations'):
        p=ROOT.parent/'exploration_round13_20261020/artifacts/round13/network/worldsense_qa.json';question_token_indices.annotations=json.loads(p.read_text());question_token_indices.tokenizer=AutoTokenizer.from_pretrained(ROOT/'assets/round13/models/Qwen2.5-Omni-3B',local_files_only=True)
    source,task=item_id.rsplit('_',1);question=question_token_indices.annotations[source][task]['question'];hay=input_ids[0].cpu().tolist();tail=hay[-2048:];tokenizer=question_token_indices.tokenizer
    # Boundary BPE can merge '?' with the following newline. Align the actual
    # token stream by character offsets rather than tokenizing the body alone.
    text=tokenizer.decode(tail,skip_special_tokens=False,clean_up_tokenization_spaces=False);encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    assert encoded['input_ids']==tail,'decoded-tail tokenizer roundtrip must be exact'
    begin=text.find(question);assert begin>=0 and text.find(question,begin+1)<0,'question body must occur exactly once in actual prompt tail';end=begin+len(question)
    selected=[len(hay)-len(tail)+i for i,(s,e) in enumerate(encoded['offset_mapping']) if e>begin and s<end and e>s]
    assert selected,'empty question-body alignment';return selected[-limit:]
class GSRQWindowStages(GSRQBudget):
    method='GSRQ-R15-G1-question-time-window-stage-allocation'
    def __init__(self,package,budget_mib,**policy):
        super().__init__(package,budget_mib,**policy);install_query_capture();self.current_q=None
        self.parent=json.loads(pathlib.Path(policy['parent_package']).joinpath('completion.json').read_text())
    def set_context(self,**kwargs):
        global ACTIVE
        super().set_context(**kwargs);self.question_indices=question_token_indices(kwargs['input_ids'],kwargs['item_id']);self.q_layer=0;self.embedding_window_scores=None;self._route_events=[];ACTIVE=self
    def install_model_hooks(self,model):
        def observe_input_features(module,args,kwargs):
            if ACTIVE is not self:return
            x=kwargs.get('hidden_states',args[0] if args else None)
            if x.shape[1]<=1:return
            def compute():
                z=x[0].float();unit=torch.nn.functional.normalize(z,dim=-1);pick=torch.linspace(0,len(self.question_indices)-1,min(4,len(self.question_indices))).round().long().tolist();index=torch.tensor([self.question_indices[i] for i in pick],device=z.device);qmean=unit.index_select(0,index).mean(0);scores=unit@qmean;return self.aggregate(scores),self.aggregate(scores,self._native_windows) if self._native_windows is not None else None
            cache=kwargs['past_key_value'];self.embedding_window_scores,self.embedding_native_window_scores=cache._time(compute,'route');self._route_events.append(cache.events[-1])
        self._feature_handle=model.thinker.model.layers[0].self_attn.register_forward_pre_hook(observe_input_features,with_kwargs=True)
    def capture_question_q(self,q):
        assert q.shape[-2]>=max(self.question_indices)+1,'G1 complete-postfill future questions are unavailable in causal4096'
        self.current_q=q[0].index_select(1,torch.tensor(self.question_indices,device=q.device));self.current_q_layer=self.q_layer;self.q_layer+=1
    def make_plan(self,meta):
        windows,mods,identity=temporal_windows(self.context['tags'],self.context['positions'],meta.capacity,int(self.policy.get('seconds',2)),self.policy.get('score')=='shuffled',meta.input_sha256)
        self._native_windows=temporal_windows(self.context['tags'],self.context['positions'],meta.capacity,int(self.policy.get('seconds',2)),False,meta.input_sha256)[0] if self.policy.get('score')=='shuffled' else None
        self._windows=windows;self._mods=mods;self._offsets=[0]
        for w in windows:self._offsets.append(self._offsets[-1]+len(w))
        self._order=[i for w in windows for i in w];nw=len(windows);pdtype=torch.int16 if max(meta.position_max)+max(0,meta.generation_cap-1)<=32767 else torch.int32;posbytes=3*meta.capacity*torch.empty((),dtype=pdtype).element_size()+256
        self._route_description=[('order',(meta.capacity,),torch.int32,'metadata'),('offset',(nw+1,),torch.int32,'metadata')]
        globalbytes=posbytes+Arena.bytes(self._route_description)
        parenttotal=min(self.parent['Rmax_K']+self.parent['Rmax_V'],(self.budget_bytes-self.parent['static_tensor_bytes']-posbytes)//(72*meta.capacity));parentrk,parentrv=gs_split(parenttotal,self.parent['Rmax_K'],self.parent['Rmax_V'],int(self.policy.get('parent_K_minus_V',0)))
        delta=int(self.policy.get('extra_stages',2));fixedmeta=36*(nw*5+4);available=self.budget_bytes-self.static_bytes-globalbytes-fixedmeta
        low=min(parentrk,self.package['Rmax_K']-delta,int(available//(72*meta.capacity))-parentrv-1)
        assert low>=1,'insufficient base capacity for G1'
        base=gsrq_description(meta.capacity,low,parentrv);base += [('window_stage_count',(nw,),torch.uint8,'metadata'),('window_tail_offset',(nw+1,),torch.int32,'metadata')]
        remaining=self.budget_bytes-self.static_bytes-globalbytes-36*Arena.bytes(base);tailcap=min(meta.capacity,remaining//(36*2*delta))
        desc=base+[('K_tail',(2,tailcap,delta),torch.uint8,'payload')];est=estimate_resident_bytes(self.static_bytes,globalbytes,desc)
        return {'R_K':low,'R_V':parentrv,'parent_R_K':parentrk,'parent_R_V':parentrv,'extra_stages':delta,'tail_token_capacity_per_layer':tailcap,'loaded_Rmax_K':self.package['Rmax_K'],'loaded_Rmax_V':self.package['Rmax_V'],'window_count':nw,'window_identity':identity,'score':self.policy.get('score','relevance'),'position_dtype':str(pdtype),'description':desc,'estimate':est,'capacity':meta.capacity,'causal4096_supported':False,'Q_coordinates':'post-MRoPE question Q against same-layer native post-MRoPE K; all8 query heads per physicalKVhead; causal mask for sampled question queries','generation_rule':'routing frozen after postfill; future generated TEXT tokens use base stages only'}
    def prepare_request(self,meta,registry):
        plan=super().prepare_request(meta,registry);self.routing=Arena(self._route_description,'cuda',registry,'cache/G1_shared_real_time_windows');self.routing.views['order'].copy_(torch.tensor(self._order,dtype=torch.int32));self.routing.views['offset'].copy_(torch.tensor(self._offsets,dtype=torch.int32));return plan
    def allocate_state(self,l,registry):
        s=super().allocate_state(l,registry);s['window_stage_count'].zero_();s['window_tail_offset'].fill_(-1);s.update(selected_windows=[],selected_tail_tokens=0);return s
    def indices(self,selected):
        views=[self.routing.views['order'][self._offsets[j]:self._offsets[j+1]] for j in selected]
        return torch.cat(views).long() if views else torch.empty(0,dtype=torch.long,device='cuda')
    @torch.no_grad()
    def relevance(self,kpost):
        n=kpost.shape[-2];q=self.current_q.float();keypos=torch.arange(n,device='cuda');qidx=torch.tensor(self.question_indices,device='cuda');score=torch.zeros(n,device='cuda')
        for h in range(2):
            qs=q[h*8:(h+1)*8];logits=torch.matmul(qs,kpost[0,h].float().T)/math.sqrt(128);logits.masked_fill_(keypos[None,None,:]>qidx[None,:,None],-torch.inf);score+=logits.softmax(-1).mean((0,1))
        return score
    def aggregate(self,scores,windows=None):
        return [float(scores[w].mean()) if self._mods[j] in [1,2] else -float('inf') for j,w in enumerate(self._windows if windows is None else windows)]
    def select(self,scores,capacity):
        remaining=capacity;selected=[]
        for j in sorted(range(len(scores)),key=lambda j:(-scores[j],j)):
            if self._mods[j] not in [1,2]:continue
            if len(self._windows[j])<=remaining:selected.append(j);remaining-=len(self._windows[j])
        return selected
    def matched_hash(self,selected):
        quotas=collections.Counter(len(self._windows[j]) for j in selected);by=collections.defaultdict(list)
        for j,w in enumerate(self._windows):
            if self._mods[j] in [1,2]:by[len(w)].append(j)
        result=[]
        for length,count in quotas.items():result += sorted(by[length],key=lambda j:hashlib.sha256(('R15-G1-control:'+self.meta.input_sha256+str(j)).encode()).hexdigest())[:count]
        assert sum(len(self._windows[j]) for j in result)==sum(len(self._windows[j]) for j in selected);return result
    @torch.no_grad()
    def append(self,l,state,kpost,v,pre=None):
        first=state['n']==0;super().append(l,state,kpost,v,pre)
        if not first:return state
        assert self.current_q is not None and self.current_q_layer==l
        residual=pre[0].float()-self.raw_decode(state['K_index'][:,:state['n']],self.tables[l]['K'])
        mode=self.policy.get('score','relevance')
        if mode=='energy':scores=self.aggregate(residual.square().mean((0,2)))
        elif mode=='embedding' or (mode in ['hash','shuffled'] and self.policy.get('control_route','embedding')=='embedding'):
            assert self.embedding_window_scores is not None,'must capture merged input features before layer0 packing';scores=self.embedding_window_scores
        else:
            native_scores=self.relevance(kpost);scores=self.aggregate(native_scores)
        selected=self.select(scores,self.plan['tail_token_capacity_per_layer'])
        if mode=='hash':selected=self.matched_hash(selected)
        if mode=='shuffled':
            reference_scores=self.embedding_native_window_scores if self.policy.get('control_route','embedding')=='embedding' else self.aggregate(native_scores,self._native_windows)
            reference=self.select(reference_scores,self.plan['tail_token_capacity_per_layer']);quotas=collections.Counter(len(self._windows[j]) for j in reference);selected=[]
            for length,count in quotas.items():
                choices=[j for j,w in enumerate(self._windows) if self._mods[j] in [1,2] and len(w)==length];selected+=sorted(choices,key=lambda j:(-scores[j],j))[:count]
            assert sum(len(self._windows[j]) for j in selected)==sum(len(self._windows[j]) for j in reference)
        selected=sorted(selected);index=self.indices(selected);nt=index.numel();delta=self.plan['extra_stages'];rk=self.plan['R_K']
        if nt:self.encode_into(residual.index_select(1,index),self.tables[l]['K'][:,rk:rk+delta],state['K_tail'][:,:nt])
        offset=0
        for j in selected:state['window_stage_count'][j]=delta;state['window_tail_offset'][j]=offset;offset+=len(self._windows[j])
        state.update(selected_windows=selected,selected_tail_tokens=nt,score_scope=mode);self.current_q=None;return state
    def extra_receipt(self,states):
        allocated=sum(s['arena'].payload_bytes for s in states.values());used=sum(2*s['n']*(self.plan['R_K']+self.plan['R_V'])+2*s['selected_tail_tokens']*self.plan['extra_stages'] for s in states.values());actual=self.registry.observe('G1_receipt_breakdown')
        return {'cache_payload_bytes':used,'cache_metadata_and_capacity_bytes':actual['resident_bytes']-actual['codec_static_bytes']-used,'allocated_index_capacity_bytes':allocated,'unused_index_capacity_bytes':allocated-used,'real_window_token_counts':[len(w) for w in self._windows],'real_window_modalities':self._mods,'per_layer_selected_windows':{str(l):s['selected_windows'] for l,s in states.items()},'per_layer_selected_tail_tokens':{str(l):s['selected_tail_tokens'] for l,s in states.items()},'per_layer_window_stage_count':{str(l):s['window_stage_count'].cpu().tolist() for l,s in states.items()},'per_layer_window_tail_offset':{str(l):s['window_tail_offset'].cpu().tolist() for l,s in states.items()},'selected_capacity_charged_in_full':True,'question_body_Q_indices':self.question_indices,'route_feature_ms':sum(s.elapsed_time(e) for kind,s,e in self._route_events),'extra_model_passes':0,'feature_scope':'embedding first; QK alternative is post-MRoPE matched native coordinates, postfill only'}
    @torch.no_grad()
    def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
        # Base decode handles the exact same MRoPE and original V as ordinary GSRQ.
        n=state['n'];kr=self.raw_decode(state['K_index'][:,:n],self.tables[l]['K']);nt=state['selected_tail_tokens'];rk=self.plan['R_K']
        if nt:
            ix=self.indices(state['selected_windows']);kr[:,ix]+=self.raw_decode(state['K_tail'][:,:nt],self.tables[l]['K'][:,rk:rk+self.plan['extra_stages']])
        v=self.raw_decode(state['V_index'][:,:n],self.tables[l]['V'])[None];phase=positions.float()[:,:,None]*inv_freq[None,None,:];cos=(phase.cos()*attention_scaling).bfloat16();sin=(phase.sin()*attention_scaling).bfloat16();widths=[16,24,24]
        sc=torch.cat([cos[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1);ss=torch.cat([sin[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1)
        from r14kv.common import apply_key_rope
        return apply_key_rope(kr[None].bfloat16(),torch.cat([sc,sc],-1)[None,None],torch.cat([ss,ss],-1)[None,None]),v.bfloat16()
    def finish_request(self):
        global ACTIVE
        ACTIVE=None;self.current_q=None;self.routing=None;self._windows=None;self._native_windows=None;self._order=None;super().finish_request()
