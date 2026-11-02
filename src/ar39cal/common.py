import hashlib,json,os,pathlib,time
ROOT=pathlib.Path(__file__).resolve().parents[2]
ALLOWED=[0,2,3,4,6,7]
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def read(p):return json.loads(pathlib.Path(p).read_text())
def device(cap=26):
 import torch
 g=int(os.environ['CUDA_VISIBLE_DEVICES']);assert g in ALLOWED
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(cap*2**30/torch.cuda.get_device_properties(0).total_memory)
 return g
def base(rep):return ROOT/f'artifacts/ar39/replicas/r{rep}'
def pool(kind):return ROOT/f'artifacts/ar39/calibration/{kind}'
def geometry(kind):return (36,2,16,2048)if kind=='Omni'else(28,4,28,3584)
def capture_run(kind):return ROOT/'data/runs'/read(ROOT/'data/ar39/CAPTURE_JOBS.json')[kind]
def receipt(kind):
 p=capture_run(kind)/'completion.json';d=read(p);assert d['completed']and len(d['rows'])==64;return d
