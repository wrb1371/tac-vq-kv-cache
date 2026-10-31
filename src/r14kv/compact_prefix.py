"""Legal R12 /0.75-index-BPA GSRQ budget point, with real54MiB FP8 books."""
import torch
from .codecs import GSRQ

class GSRQCompactR12(GSRQ):
    def __init__(self,directory,device='cuda'):
        super().__init__(directory,stages=16,device=device);errors=[]
        for l,tables in enumerate(self.tables):
            for kind in ['K','V']:
                parent=tables[kind];prefix=parent[:,:12].contiguous();deployed=prefix.to(torch.float8_e4m3fn)
                assert torch.isfinite(deployed.float()).all() and deployed.untyped_storage().nbytes()==deployed.numel()
                errors.append({'layer':l,'kind':kind,'relative_RMSE':float(((deployed.float()-prefix.float()).square().sum()/prefix.float().square().sum().clamp_min(1e-30)).sqrt())})
                tables[kind]=deployed
        self.stages=12;self.method='GSRQ-Omni-BM-R12-FP8'
        self.metadata={**self.metadata,'engineering_control_only':True,'index_BPA':.75,'stages':12,'loaded_max_stages':12,'codebook_dtype':'float8_e4m3fn','persistent_FP16_or_R16_mirror':False,
            'R12_independent_deployed_storage':True,'R12_source':'greedyR32 first12, loaded through the independently exportedR16 prefix; training never depends on finalR',
            'nearest_and_decode':'actual deployed FP8 bytes; original-scale additive residual and true native MRoPE',
            'parent_FP16_to_FP8_errors':errors,'static_books_MiB_theory':54,'full_M0_parent_retained':True}
        self.metadata.pop('R16_independent_deployed_storage',None)
