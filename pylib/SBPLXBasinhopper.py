# Ed Callaghan
# scipy.basinhopping with nlopt.L_SBPLX at each step... robust, and fast
# June, July 2019; June 2021

import scipy.optimize as so
import nlopt
import numpy as np
from Minimizer import Minimizer
from NLOPTMinimizers import nlopt_f_wrap, L_SBPLX
from Objects import BuildObject
from SCIPYCallable import SCIPYCallable

def f_bounds_wrap(f, bounds):
    inf = float('inf')
    def rv(xx):
        frv = f(xx)
#           if not b[0] < x < b[1]:
#               return inf
        for x,b in zip(xx, bounds):
            if x < b[0]:
                frv += pow(x - b[0], 2)
            elif b[1] < x:
                frv += pow(b[1] - x, 2)
        return frv
    return rv

defaults = {'iterations': 10000,
'iterations_to_terminate': 1000,
           'temperature': 1000.0,
        'local_ftol_rel': 1e-3,
        'local_xtol_rel': 1e-3,
        'local_max_eval': int(1e5),
           }
class SBPLXBasinhopper(Minimizer):
    def __init__(self, **kwargs):
        BuildObject(self, defaults, kwargs)
        super(SBPLXBasinhopper, self).__init__(**kwargs)

    def minimize(self, f, seeds, bounds):
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        diff = ub - lb
        def info_cb(args, val, accepted):
            print('Basinhopper found a minimum at', args)
            print('Minimum value is', val)
            if accepted:
                print('Accepting this minimum') 
            else:
                print('Rejecting this minimum')
        def take_step(x):
            delta = np.random.random(len(bounds))
            xnew = lb + diff*delta
            return xnew

        local_minimizer = L_SBPLX(ftol_rel=self.local_ftol_rel,
                                  xtol_rel=self.local_xtol_rel,
                                  max_eval=self.local_max_eval)
        local_minimizer = SCIPYCallable(local_minimizer)
#       f = f_bounds_wrap(f, bounds)
        res = so.basinhopping(f, seeds, \
                              niter=self.iterations, \
                              niter_success=self.iterations_to_terminate, \
                              T=self.temperature, \
                              callback=info_cb, \
                              take_step=take_step, \
                              minimizer_kwargs={'method': local_minimizer, \
                                                'options': {'param_bounds': \
                                                                    bounds, \
                                                           }
                                               },\
                              disp=True)
        xopt = res.x
        fmin = f(xopt)

        return xopt, fmin
