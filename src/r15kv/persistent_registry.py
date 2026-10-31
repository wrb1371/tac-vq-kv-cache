"""Actual-storage registry at every persistent allocation/update boundary."""
import torch
def storages(roots):
    found={}
    def walk(x,path,category):
        if torch.is_tensor(x):
            s=x.untyped_storage()
            if not s.nbytes():return
            key=(str(x.device),s.data_ptr(),s.nbytes())
            if key not in found:found[key]={'owner':path,'category':category,'device':str(x.device),'dtype':str(x.dtype),'shape':list(x.shape),'bytes':s.nbytes()}
        elif isinstance(x,dict):
            for k,v in x.items():walk(v,path+'/'+str(k),category)
        elif isinstance(x,(list,tuple)):
            for k,v in enumerate(x):walk(v,path+'/'+str(k),category)
    for name,(x,category) in roots.items():walk(x,name,category)
    return list(found.values())
class PersistentRegistry:
    def __init__(self,budget_bytes):
        self.roots={};self.budget=int(budget_bytes);self.peak=0;self.events=[];self.end=0
    def own(self,name,value,category='cache'):
        self.roots[name]=(value,category);return self.observe('register:'+name)
    def release(self,name):
        self.roots.pop(name,None);return self.observe('release:'+name)
    def observe(self,event):
        rows=storages(self.roots);gpu=[r for r in rows if r['device'].startswith('cuda')];total=sum(r['bytes'] for r in gpu);static=sum(r['bytes'] for r in gpu if r['category']=='codec')
        self.end=total
        if total>self.peak or not self.events:
            self.peak=max(self.peak,total);self.events.append({'event':event,'resident_bytes':total,'codec_bytes':static,'storage_count':len(gpu)})
        if total>self.budget:raise RuntimeError(f'PERSISTENT_BUDGET_OVERFLOW {event}: {total} > {self.budget}')
        return {'resident_bytes':total,'codec_static_bytes':static,'cache_bytes':total-static,'CPU_bytes':sum(r['bytes'] for r in rows if r['device']=='cpu')}
    def receipt(self):
        x=self.observe('request_end');return {**x,'resident_peak_bytes':self.peak,'resident_end_bytes':self.end,'budget_bytes':self.budget,'budget_pass':self.peak<=self.budget,'budget_overflow_bytes':max(0,self.peak-self.budget),'high_water_events':self.events}
class Arena:
    """One allocation; all payload/metadata views retain and charge its full capacity."""
    def __init__(self,description,device,registry,name):
        self.views={};offset=0;layout=[]
        for key,shape,dtype,kind in description:
            size=torch.empty((),dtype=dtype).element_size();offset=(offset+size-1)//size*size
            n=1
            for d in shape:n*=d
            layout.append((key,shape,dtype,kind,offset,n*size));offset+=n*size
        self.tensor=torch.empty(offset,dtype=torch.uint8,device=device);registry.own(name,self.tensor)
        for key,shape,dtype,kind,start,count in layout:self.views[key]=self.tensor[start:start+count].view(dtype).reshape(shape)
        self.payload_bytes=sum(row[-1] for row in layout if row[3]=='payload');self.metadata_capacity_bytes=offset-self.payload_bytes
        self.layout=[{'key':k,'shape':list(s),'dtype':str(d),'category':c,'offset':o,'bytes':n} for k,s,d,c,o,n in layout]
    @staticmethod
    def bytes(description):
        out=0
        for _,shape,dtype,_ in description:
            size=torch.empty((),dtype=dtype).element_size();out=(out+size-1)//size*size;n=1
            for d in shape:n*=d
            out+=n*size
        return out
