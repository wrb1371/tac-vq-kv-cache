"""Route every replica, including replica zero, to freshly fitted AR39 files."""
import builtins,contextlib,functools,io,os,pathlib,sys
from ar39cal.common import ROOT,base
from ar38replica.paths import PREFIXES
@contextlib.contextmanager
def use(rep):
 mapping=[(ROOT/p,base(rep)/p)for p in PREFIXES]
 mapping.append((ROOT/'data/ar22/conditional_fixedI125.json',base(rep)/'data/ar22/conditional_fixedI125.json'))
 mapping.append((ROOT/'artifacts/ar29/packages/GSRQExport',base(rep)/'artifacts/ar31/packages/GSRQExport'))
 b=builtins.open;i=io.open;po=pathlib.Path.open
 def remap(f,mode):
  if isinstance(f,int)or any(x in mode for x in ['w','a','x','+']):return f
  try:p=pathlib.Path(os.fspath(f));p=p if p.is_absolute()else pathlib.Path.cwd()/p
  except TypeError:return f
  for old,new in mapping:
   try:q=p.relative_to(old)
   except ValueError:continue
   target=new/q
   # Schema-only overlays preserve immutable fitting receipts and learned values.
   relative=target.relative_to(base(rep))
   overlay=ROOT/f'artifacts/ar39/runtime_metadata/r{rep}'/relative
   return overlay if overlay.is_file()else target
  return f
 def clear():
  for mod in list(sys.modules.values()):
   if mod and str(getattr(mod,'__file__','')).startswith(str(ROOT/'src')):
    for v in list(vars(mod).values()):
     if isinstance(v,functools._lru_cache_wrapper):v.cache_clear()
 clear();builtins.open=lambda f,mode='r',*a,**k:b(remap(f,mode),mode,*a,**k);io.open=lambda f,mode='r',*a,**k:i(remap(f,mode),mode,*a,**k)
 # Python3.10's pathlib accessor captured io.open at module import. Rebind
 # Path.open itself so read_text/read_bytes reach the same actual file as
 # torch.load/builtins.open on every supported interpreter.
 pathlib.Path.open=lambda p,mode='r',*a,**k:po(pathlib.Path(remap(p,mode)),mode,*a,**k)
 try:yield
 finally:builtins.open=b;io.open=i;pathlib.Path.open=po;clear()
