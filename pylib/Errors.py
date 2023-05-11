# Ed Callaghan
# Computing errors via covariance matrices / profile likelihoods
# June 2019; June 2021

import iminuit
import numpy as _np
import numpy as np
import scipy.optimize as so

def Covariance(f, x):
    names = ['%d' % i for i in range(x.size)]
    dopt = {k: x[i] for i,k in enumerate(names)}
    m = iminuit.Minuit(f, pedantic=True, name=names, use_array_call=True, **dopt)
    m.hesse()
    mc = m.covariance
    rv = np.array([[mc[(m,n)] for n in names] for m in names])
    return rv

def ErrorsFromCovarianceMatrix(cov, params):
    # take the max projection on _any_ eigenvector...
    eigvals, eigvecs = np.linalg.eig(cov)
    for i in range(eigvecs.shape[0]):
        eigvalmax = eigvals[i]
        eigvecmax = eigvecs[:,i]

        rv = np.repeat(-1.0, len(params))
        # construct unit vectors
        for i,ip in enumerate(params):
            phat = np.zeros(cov.shape[0])
            phat[ip] = 1

            # project eigenvector onto unit vector
            rotation = np.dot(phat, eigvecmax)
            error = rotation*eigvalmax
            error = np.sqrt(np.abs(error))

            rv[i] = max(rv[i], error)
    return rv

def AsymptoticErrors(f, x, params=[0]):
    try:
        cov = Covariance(f, x)
        return ErrorsFromCovarianceMatrix(cov, params)

    except Exception as e:
        print('Exception: %s' % e)

    return np.repeat(-1.0, len(params))

def take_steps_to_bound(f, seed, threshold, takestep, **kwargs):
    keys = kwargs.keys()
    history = True # maybe this should be false?
    if 'history' in keys:
        history = kwargs['history']
    scale = 1.5
    if 'scale' in keys:
        scale = kwargs['scale']
    bisect_rtol = 1e-7
    if 'bisect_rtol' in keys:
        bisect_rtol = kwargs['bisect_rtol']

    dx = (1e-4)*seed
    if 'initial_step' in keys:
        dx = kwargs['initial_step']

    # TODO there must be a more transparent way to write this...
    crossed_limit = lambda x: False
    bisect = True
    if 'limit' in keys:
        limit = kwargs['limit']
        at_limit = False
        if limit < seed:
            crossed_limit = lambda x: x < limit
        else:
            crossed_limit = lambda x: limit < x

    old = None
    x = seed
    cache = []

    fval = -float('inf')
    while fval <= threshold:
        old = x
        x = takestep(x, dx)
        if crossed_limit(x):
            x = limit
            if not at_limit: # if we just hit the limit, compute fval < thresh?
                at_limit = True
            elif at_limit:   # if fval < thresh, break the loop and stop here
                rv = x
                bisect = False
                break
        fval = f(x)
        if history:
            cache.append((x, fval))
        dx *= scale

    # if the limit wasn't reached, or is above threshold
    if bisect:
        g = lambda y: f(y) - threshold
        rv = so.bisect(g, old, x, rtol=bisect_rtol)

    if history:
        return rv, cache
    return rv

# assumes that f is increasing on [seed, infinity)
# threshold is the value to ``capture''
def locate_upper_bound(f, seed, threshold, **kwargs):
    step = lambda x, dx: x + dx
    if 'limit' not in kwargs.keys():
        kwargs['limit'] = +float('inf')
    return take_steps_to_bound(f, seed, threshold, step, **kwargs)

# assumes that f is decreasing on (-infinity, seed]
# threshold is the value to ``capture''
def locate_lower_bound(f, seed, threshold, **kwargs):
    step = lambda x, dx: x - dx
    if 'limit' not in kwargs.keys():
        kwargs['limit'] = -float('inf')
    return take_steps_to_bound(f, seed, threshold, step, **kwargs)

def get_bounds_by_delta(f, xmin, delta, **kwargs):
    fmin = f(xmin)
    ftarget = fmin + delta
    try:
        limits = kwargs['limits']
    except:
        limits = (-float('inf'), +float('inf'))
    return locate_lower_bound(f, xmin, ftarget, limit=limits[0], **kwargs), \
           locate_upper_bound(f, xmin, ftarget, limit=limits[1], **kwargs)

def ProfiledErrors(f, x, params, **kwargs):
    rv = []
    minimizer = kwargs['minimizer']
    full_bounds = np.array(kwargs['bounds'])
    for i in params:
        nominal = x[i]
        seeds = np.delete(x, i)
        bounds = np.delete(full_bounds, i, axis=0)
        def profiled_likelihood(fixed):
            def minimizable(rest):
                full = np.insert(rest, i, fixed)
                rv = f(full)
                return rv
            _, rv = minimizer.minimize(minimizable, seeds, bounds)
            return rv
        step = 1e-2 * x[i]
        limits = full_bounds[i]
        bracket = get_bounds_by_delta(profiled_likelihood, x[i], 1.0, history=False, initial_step=step, limits=limits, **kwargs)
        errors = [b - nominal for b in bracket]
        rv.append(errors)
    return rv
