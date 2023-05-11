# Ed Callaghan
# A collection of everybody :)
# June 2019; June 2021

import NLOPTMinimizers
from SBPLXBasinhopper import SBPLXBasinhopper
from NELDERMEADBasinhopper import NELDERMEADBasinhopper

class NoMinimizer():
    def __init__(self):
        pass

    def minimize(self, f, seeds, bounds):
        return seeds, None

minimizers = {
    'NLOPT_L_SBPLX':        NLOPTMinimizers.L_SBPLX,
    'NLOPT_L_NELDERMEAD':   NLOPTMinimizers.L_NELDERMEAD,

    'SBPLX_BASINHOPPING':   SBPLXBasinhopper,
    'NELDERMEAD_BASINHOPPING':   NELDERMEADBasinhopper,

    'NONE': NoMinimizer,
}
