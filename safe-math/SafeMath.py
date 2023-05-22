# Ed Callaghan
# numpy wrappers around safe functions

import numpy as np
import SafeMathC as smc

xlogx = np.vectorize(smc.xlogx, otypes=[float])
