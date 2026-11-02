"""Check the actual opened files, rather than trusting run configuration flags."""
import builtins,io,json,pathlib,sys
from .common import ROOT,base,read,sha
from .paths import use

def resolved(rep,relative):
    overlay=ROOT/f'artifacts/ar39/runtime_metadata/r{rep}'/relative
    return overlay if overlay.is_file()else base(rep)/relative

def audit(rep,check_torch=False):
    records=[]
    with use(rep):
        files=sorted(p for p in base(rep).rglob('*')if p.is_file()and p.suffix in ['.json','.pt'])
        for actual in files:
            relative=actual.relative_to(base(rep))
            logical=ROOT/relative
            expected=resolved(rep,relative)
            # Every API must open exactly the same file, not just equal contents.
            for op in [builtins.open,io.open,lambda p,mode:pathlib.Path(p).open(mode)]:
                with op(logical,'rb')as f:
                    assert pathlib.Path(f.name)==expected,(rep,logical,f.name,expected)
            assert logical.read_bytes()==expected.read_bytes(),(rep,relative)
            if actual.name=='completion.json':
                metadata=read(logical)
                for rec in metadata.get('records',[]):
                    assert sha(logical.parent/rec['file'])==rec['SHA256'],(rep,relative,rec['file'])
            records.append({'logical':str(relative),'actual':str(expected.relative_to(ROOT)),'SHA256':sha(expected)})
        for bank,conditional in [('artifacts/ar17/packages/PSRVQ_d16','data/ar22/conditional_curves.json'),
                                  ('artifacts/ar26/packages/STI_RVQ','data/ar26/VL_conditional_curves.json')]:
            meta=read(ROOT/bank/'completion.json');table=read(ROOT/conditional)
            assert meta['group_dim']==16 and meta['stages']==6
            assert table['calibration_dataset']=='WorldSense'and table['actual_media']==64
            assert table['bank_completion_SHA256']==sha(ROOT/bank/'completion.json')
            assert table['bank_SHA256']==sha(ROOT/bank/'global.pt')
            if check_torch:
                import torch
                loaded=torch.load(ROOT/bank/'global.pt',weights_only=True,map_location='cpu')
                direct=torch.load(base(rep)/bank/'global.pt',weights_only=True,map_location='cpu')
                assert loaded.keys()==direct.keys()and all(torch.equal(loaded[k],direct[k])for k in loaded)
        fixed=read(ROOT/'data/ar22/conditional_fixedI125.json')
        assert fixed['source_full_conditional_SHA256']==sha(ROOT/'data/ar22/conditional_curves.json')
        assert fixed['bank_completion_SHA256']==sha(ROOT/'artifacts/ar17/packages/PSRVQ_d16/completion.json')
        assert read(ROOT/'data/ar21/rate_curves.json')['fixed_bank_completion_SHA256']==sha(ROOT/'artifacts/ar17/packages/PSRVQ_d16/completion.json')
    return {'replicate':rep,'python':sys.version,'files':records,'actual_open_bindings_verified':True,
            'torch_direct_tensor_equality_checked':check_torch}

def codec_binding(codec,kind,rep):
    """Bind an instantiated codec/plan to the effective metadata and losses."""
    bank='artifacts/ar17/packages/PSRVQ_d16'if kind=='Omni'else'artifacts/ar26/packages/STI_RVQ'
    conditional='data/ar22/conditional_curves.json'if kind=='Omni'else'data/ar26/VL_conditional_curves.json'
    for attr in ['conditional','curves']:
        value=getattr(codec,attr,None)
        if isinstance(value,dict)and 'bank_SHA256'in value:
            assert value['calibration_dataset']=='WorldSense'and value['actual_media']==64
            assert value['bank_SHA256']==sha(ROOT/bank/'global.pt')
            assert value['bank_completion_SHA256']==sha(ROOT/bank/'completion.json')
    path=getattr(codec,'policy',{}).get('conditional_file',conditional)
    if getattr(codec,'conditional_sha',None)is not None:
        assert codec.conditional_sha==sha(ROOT/path)
    record={'replicate':rep,'shared_tensor_SHA256':sha(ROOT/bank/'global.pt'),
        'shared_metadata_SHA256':sha(ROOT/bank/'completion.json'),
        'conditional_SHA256':sha(ROOT/path),'calibration_dataset':'WorldSense64',
        'actual_paths_verified_at_worker_start':True}
    codec.plan['calibration_binding']=record
    return record
