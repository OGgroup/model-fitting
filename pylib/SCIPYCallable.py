# Ed Callaghan
# A wrapper class to allow forwarding of basinhopper minimization requests
# July 2019

import scipy.optimize as so

class SCIPYCallable():
    def __init__(self, minimizer):
        self.minimizer = minimizer

    # from basinhopping, kwargs will contain both kwargs + extra options
    def __call__(self, f, x0, args, **kwargs):
        try:
            xopt, fmin = self.minimizer.minimize(f, x0, kwargs['param_bounds'])
            rv = so.OptimizeResult({'x': xopt, 'fun': fmin, 'success': True})
        except Exception as e:
            print('encountered exception: %s' % str(e))
            rv = so.OptimizeResult({'x': x0, 'fun': float('inf'), 'success': False})
        return rv
