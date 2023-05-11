# Ed Callaghan
# numpy wrappers around safe functions

#import jax
#import jax.numpy as np
import numpy as np
import numpy as _np
import SafeMathC as smc

xlogx = np.vectorize(smc.xlogx, otypes=[float])

'''
xlogx_c  = smc.xlogx
xlogx_p = jax.core.Primitive("xlogx")
def xlogx_impl(x):
    rv = xlogx_c(x)
    return rv
xlogx_p.def_impl(xlogx_impl)
def xlogx(x):
    rv = xlogx_p.bind(x)
    return rv
def xlogx_batch(vector_args, batch_axes): # TODO this is SLOW
    assert batch_axes == (0,)
    xx = vector_args[0]
    rv = np.array([xlogx(x) for x in xx])
    return rv, batch_axes[0]
jax.interpreters.batching.primitive_batchers[xlogx_p] = xlogx_batch
xlogx = np.vectorize(xlogx)

xlogx_d0_c = smc.xlogx_d0
xlogx_d0_p = jax.core.Primitive("xlogx_d0")
def xlogx_d0_impl(x):
    rv = xlogx_d0_c(x)
    return rv
xlogx_d0_p.def_impl(xlogx_d0_impl)
def xlogx_d0(x):
    rv = xlogx_d0_p.bind(x)
    return rv
# batch rule / vectorization would go here
def xlogx_val_jvp(x_args, t_args):
    x = x_args[0]
    t = t_args[0]
    rv_x = xlogx(x)
    rv_t = xlogx_d0(x)*t
    rv = (rv_x, rv_t)
    return rv
jax.interpreters.ad.primitive_jvps[xlogx_p] = xlogx_val_jvp

################################################################################

expo_c  = smc.expo
expo_p = jax.core.Primitive("expo")
def expo_impl(x):
    rv = expo_c(x)
    return rv
expo_p.def_impl(expo_impl)
def expo(x):
    rv = expo_p.bind(x)
    return rv
def expo_batch(vector_args, batch_axes): # TODO this is SLOW
    assert batch_axes == (0,)
    xx = vector_args[0]
    rv = np.array([expo(x) for x in xx])
    return rv, batch_axes[0]
jax.interpreters.batching.primitive_batchers[expo_p] = expo_batch
expo = np.vectorize(expo)

def expo_val_jvp(x_args, t_args):
    x = x_args[0]
    t = t_args[0]
    rv_x = expo(x)
    rv_t = expo(x)*t
    rv = (rv_x, rv_t)
    return rv
jax.interpreters.ad.primitive_jvps[expo_p] = expo_val_jvp

def expo_abstract_eval(x):
    rv = jax.abstract_arrays.ShapedArray(x.shape, x.dtype)
    return rv
expo_p.def_abstract_eval(expo_abstract_eval)

def expo_xla_translation(ctx, x):
    rv = [jax.lib.xla_client.ops.Exp(x)]
    return rv
jax.interpreters.xla.backend_specific_translations['cpu'][expo_p] = expo_xla_translation

################################################################################
'''

'''
experfc_c = smc.experfc
experfc_p = jax.core.Primitive("experfc")
def experfc_impl(x, a, b, c):
    rv = experfc_c(x, a, b, c)
    return rv
experfc_p.def_impl(experfc_impl)
def experfc(x, a, b, c):
    rv = experfc_p.bind(x, a, b, c)
    return rv
def experfc_batch(vector_args, batch_axes): # TODO this is SLOW
    assert batch_axes == (0,None,None,None)
    xx = vector_args[0]
    a, b, c = vector_args[1:]
    rv = np.array([experfc(x, a, b, c) for x in xx])
    return rv, batch_axes[0]
jax.interpreters.batching.primitive_batchers[experfc_p] = experfc_batch
experfc = np.vectorize(experfc)

experfc_d0_c = np.vectorize(smc.experfc_d0)
experfc_d0_p = jax.core.Primitive("experfc_d0")
def experfc_d0_impl(*args):
    rv = experfc_d0_c(*args)
    return rv
experfc_d0_p.def_impl(experfc_d0_impl)
def experfc_d0(*args):
    rv = experfc_d0_p.bind(*args)
    return rv
def experfc_val_jvp(x_args, t_args):
    x, a, b, c = x_args
    dx, da, db, dc = t_args
    rv_x = experfc(*x_args)
    rv_t = experfc_d0(*x_args)*dx # + ...
    rv = (rv_x, rv_t)
    return rv
jax.interpreters.ad.primitive_jvps[experfc_p] = experfc_val_jvp
'''
