"""Explicit read-only redirection of learned artifacts; historical files stay immutable."""
import builtins,contextlib,functools,io,os,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[2]
PREFIXES=['artifacts/ar17/packages/PSRVQ_d16','artifacts/ar22/private_tiers','artifacts/round15/packages/nova_g8c16','artifacts/ar26/packages/STI_RVQ','artifacts/ar26/packages/GSRQExport','artifacts/ar31/packages/GSRQExport','artifacts/ar30/packages/NOVA_g8_C16','artifacts/ar30/packages/NOVA_g8_C256','artifacts/ar37/packages','data/ar21/rate_curves.json','data/ar22/conditional_curves.json','data/ar26/VL_conditional_curves.json','data/ar26/VL_rate_curves.json']
def target(rep,relative):return ROOT/f'artifacts/ar38/replicas/r{rep}'/relative
@contextlib.contextmanager
def use(rep,prefixes=PREFIXES):
 if rep==0:yield;return
 mapping=[(ROOT/p,target(rep,p))for p in prefixes]
 if 'artifacts/ar31/packages/GSRQExport'in prefixes:mapping.append((ROOT/'artifacts/ar29/packages/GSRQExport',target(rep,'artifacts/ar31/packages/GSRQExport')))
 original_builtin=builtins.open;original_io=io.open
 def remap(file,mode='r'):
  if isinstance(file,int)or any(x in mode for x in ['w','a','x','+']):return file
  try:p=pathlib.Path(os.fspath(file));p=p if p.is_absolute()else pathlib.Path.cwd()/p
  except TypeError:return file
  for old,new in mapping:
   try:rel=p.relative_to(old)
   except ValueError:continue
   return new/rel
  return file
 def bopen(file,mode='r',*args,**kw):return original_builtin(remap(file,mode),mode,*args,**kw)
 def iopen(file,mode='r',*args,**kw):return original_io(remap(file,mode),mode,*args,**kw)
 def clear():
  for mod in list(sys.modules.values()):
   if mod and str(getattr(mod,'__file__','')).startswith(str(ROOT/'src')):
    for v in list(vars(mod).values()):
     if isinstance(v,functools._lru_cache_wrapper):v.cache_clear()
 clear();builtins.open=bopen;io.open=iopen
 try:yield
 finally:builtins.open=original_builtin;io.open=original_io;clear()
