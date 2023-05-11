# Ed Callaghan
# A few gradient-less nlopt-based minimizers
# June 2019; June 2021

import nlopt
from Minimizer import Minimizer
from Objects import BuildObject

SECOND  = 1.0
MINUTE  = 60.0*SECOND
HOUR    = 60.0*MINUTE

def nlopt_f_wrap(f):
    tmp = f
    rv = lambda x, grad: tmp(x)
    return rv

defaults = {'ftol_rel': 1e-7,
            'xtol_rel': None,
            'max_eval': None,
            'max_time': 8.0*HOUR,
           'nlopt_obj': None,
           }
class NLOPTMinimizer(Minimizer):
    def __init__(self, **kwargs):
        BuildObject(self, defaults, kwargs)
        super(NLOPTMinimizer, self).__init__(**kwargs)

    def minimize(self, f, seeds, bounds):
        # check that seed satisfies bounds
        for s,b in zip(seeds, bounds):
            if not b[0] <= s <= b[1]:
                print('warning: seeds do not satisy bounds')
                return seeds, float('inf')

        # wrap function in nlopt-friendly call
        f = nlopt_f_wrap(f)

        # define minimizer
        dim = len(seeds)
        opt = nlopt.opt(self.nlopt_obj, dim)
        opt.set_min_objective(f)
        opt.set_lower_bounds([x[0] for x in bounds])
        opt.set_upper_bounds([x[1] for x in bounds])
        if not (self.ftol_rel == None):
            opt.set_ftol_rel(self.ftol_rel)
        if not (self.xtol_rel == None):
            opt.set_xtol_rel(self.xtol_rel)
        if not (self.max_eval == None):
            opt.set_maxeval(self.max_eval)
        if not (self.max_time == None):
            opt.set_maxtime(self.max_time)

        # perform minimization, and store minimum
        xopt = opt.optimize(seeds)
        fmin = f(xopt, None)

        return xopt, fmin

class L_SBPLX(NLOPTMinimizer):
    def __init__(self, **kwargs):
        super(L_SBPLX, self).__init__(**kwargs)
        self.nlopt_obj = nlopt.LN_SBPLX

class L_NELDERMEAD(NLOPTMinimizer):
    def __init__(self, **kwargs):
        super(L_NELDERMEAD, self).__init__(**kwargs)
        self.nlopt_obj = nlopt.LN_NELDERMEAD
