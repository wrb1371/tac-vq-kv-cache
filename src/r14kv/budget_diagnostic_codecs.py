"""Same K/V intervention for the legal complete NOVA K1/V2 budget point."""
from .diagnostic_codecs import _SideOverride
from .engineering_codecs import NOVAK1V2
class NOVAK1Side(_SideOverride,NOVAK1V2):
    def __init__(self,directory,fp_side,device='cuda'):
        NOVAK1V2.__init__(self,directory,device);self.setup(fp_side)
