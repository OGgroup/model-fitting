# Ed Callaghan
# A wrapper for a complete histogram
# January 2021

import numpy as np

class Histogram():
    def __init__(self, counts, edges):
        self.counts = counts
        self.edges = edges
        self.centers = edges[:-1] + 0.5*np.diff(edges)
        self.nbins = len(self.counts)
        self.area = np.sum(counts * np.diff(edges))
    def Get(self):
        rvc = self.counts
        rvs = np.sqrt(self.counts)
        return rvc, rvs
    def GetNormalized(self):
        norm = np.sum(self.counts)
        rvc, rvs = self.Get()
        rvc = rvc/norm
        rvs = rvs/norm
        return rvc, rvs
    def __add__(self, rhs):
        rv = Histogram(self.counts + rhs.counts, self.edges)
        return rv
    def BaselineSubtract(self, window):
        wlo = window[0]
        whi = window[1]
        mask = (wlo < self.centers) & (self.centers < whi)
        avg = np.mean(self.counts[mask])
        self.counts = self.counts - avg
