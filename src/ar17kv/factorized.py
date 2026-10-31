"""Factor components persist; effective full books are transient search scratch."""
import torch
from r16kv.gsrq import GSRQA0
from r15kv.codecs import GSRQBudget

class CompactPrivate(GSRQA0):
    method='GSRQ-AR-private-prefix'
    make_plan=GSRQBudget.make_plan

class Factorized(CompactPrivate):
    method='GSRQ-AR-factorized-tail'
    def book(self,l,kind,stage):
        private=int(self.package['private_stages']);t=self.tables[l]
        if stage<private:return t[kind+'_private'][:,stage].float()
        s=stage-private
        delta=torch.bmm(t[kind+'_coeff'][:,s].float(),t[kind+'_basis'][:,s].float())
        if self.package['recipe']=='tail_independent_r16':return delta
        assert self.package['recipe']=='tail_matched_difference_r4'
        mean=t[kind+'_mean'][:,s].float()
        return torch.cat((mean+delta,mean-delta),0)
