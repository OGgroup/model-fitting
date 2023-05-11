# Ed Callaghan
# Set of minimizable objectives
# June 2021

#from Math import np
import numpy as np
from Objects import BuildObject
import SafeMath as sm

class Objective():
    def __init__(self, **kwargs):
        BuildObject(self, {}, kwargs)
    def Evaluate(self, *args):
        rv = self.MinusTwoLogL(*args)
        if np.isnan(rv):
            rv = float('infinity')
        return rv

class UnbinnedLikelihood(Objective):
    def __init__(self, **kwargs):
        super(UnbinnedLikelihood, self).__init__(**kwargs)

    # TODO there should be some threshold (effective 0) here
    def MinusTwoLogL(self, *args):
        rv = self.model.likelihood(self.samples, *args)
        rv = np.log(rv)
        rv = -2.0*np.sum(rv)
        return rv

class BinnedObjective(Objective):
    def __init__(self, **kwargs):
        super(BinnedObjective, self).__init__(**kwargs)
        self.normalization = 1.0
        self.normalization *= np.sum(self.histogram.counts)
        self.normalization *= np.diff(self.histogram.edges)

    def evaluate_model(self, *args):
        rv = self.model.likelihood(self.histogram.centers, *args)
        rv *= self.normalization
        return rv

class BinnedLikelihood(BinnedObjective):
    def __init__(self, **kwargs):
        super(BinnedLikelihood, self).__init__(**kwargs)

    def MinusTwoLogL(self, *args):
        eps = np.finfo(float).eps
        data = self.histogram.counts
        pred = self.evaluate_model(*args)
        pred[pred < eps] = eps
        rv = 0.0
        rv += pred - data
#       extra = data*np.log(data/pred)
#       extra[data == 0.0] = 0.0
#       rv += extra
        rv += sm.xlogx(data) - data*np.log(pred)
        rv = 2.0*np.sum(rv)
        return rv

class ChiSquared(BinnedObjective):
    def __init__(self, **kwargs):
        super(ChiSquared, self).__init__(**kwargs)

    def MinusTwoLogL(self, *args):
        data = self.histogram.counts
        pred = self.evaluate_model(*args)
        rv = pow(pred - data, 2)/pred
        rv[pred == 0.0] = 0.0
        rv[pred < 0] = 0.0
        rv = np.sum(rv)
        return rv

class DataChiSquared(BinnedObjective):
    def __init__(self, **kwargs):
        super(DataChiSquared, self).__init__(**kwargs)

    def MinusTwoLogL(self, *args):
        data = self.histogram.counts
        pred = self.evaluate_model(*args)
        rv = pow(pred - data, 2)/data
        rv[data == 0.0] = 0.0
        rv[data < 0] = 0.0
        rv = np.sum(rv)
        return rv
