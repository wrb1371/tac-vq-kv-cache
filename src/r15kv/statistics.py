"""Paired micro accuracy with source-cluster duration-stratified bootstrap."""
import collections,numpy as np
def paired(rows_a,rows_b,replicates=10000,seed=15261021):
    a={r['id']:r for r in rows_a};b={r['id']:r for r in rows_b};assert a.keys()==b.keys()
    grouped=collections.defaultdict(list)
    for q in a:
        assert a[q]['input_sha256']==b[q]['input_sha256'];grouped[a[q]['source_id']].append(q)
    strata=collections.defaultdict(list);gains=[];losses=[]
    for source,qs in grouped.items():
        duration=a[qs[0]].get('duration_group','unknown');assert all(a[q].get('duration_group','unknown')==duration for q in qs)
        strata[duration].append((sum(int(bool(a[q].get('correct')))-int(bool(b[q].get('correct'))) for q in qs),len(qs)))
        gains += [q for q in qs if a[q].get('correct') and not b[q].get('correct')];losses += [q for q in qs if b[q].get('correct') and not a[q].get('correct')]
    rng=np.random.default_rng(seed);num=np.zeros(replicates);den=np.zeros(replicates)
    for key in sorted(strata):
        values=np.asarray(strata[key],dtype=np.int64);index=rng.integers(0,len(values),size=(replicates,len(values)));chosen=values[index];num+=chosen[:,:,0].sum(1);den+=chosen[:,:,1].sum(1)
    dif=num/den*100;return {'difference_pp':100*(len(gains)-len(losses))/len(a),'CI95_pp':np.percentile(dif,[2.5,97.5]).tolist(),'gains':len(gains),'losses':len(losses),'gain_question_ids':sorted(gains),'loss_question_ids':sorted(losses),'sources':len(grouped),'questions':len(a),'source_strata':{k:len(v) for k,v in strata.items()},'bootstrap_seed':seed,'replicates':replicates,'interpretation':'descriptive source-paired stratified interval; crossing0 is not statistical noninferiority; no uncorrected six-comparison significance claim'}
def distribution(values,scale=1.):
    x=np.asarray(values,dtype=float)/scale
    return {'mean':float(x.mean()),'p50':float(np.percentile(x,50)),'p95':float(np.percentile(x,95)),'max':float(x.max()),'min':float(x.min())} if len(x) else None
