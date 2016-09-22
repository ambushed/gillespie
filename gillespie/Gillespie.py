__author__ = 'Vlad'

import os
import numpy.random as nprandom
import autograd.numpy as np
from functools import partial
from autograd import jacobian
from autograd import primitive
from pathos.multiprocessing import ProcessingPool as Pool

class Gillespie(object):

    def __init__(self,a,b,propensities,increments):
        self.a = a
        self.b = b
        self.propensities = propensities
        self.increments = increments

    def __init__(self,*args,**kwargs):
        # args -- tuple of anonymous arguments
        # kwargs -- dictionary of named arguments

        self.a = kwargs.get('a', 0)
        self.b = kwargs.get('b', 0)
        self.propensities = kwargs.get('propensities', None)
        self.increments = kwargs.get('increments', None)
        self.nPaths = kwargs.get('nPaths', 10)
        self.T = kwargs.get('T', 10)
        self.gen1 = kwargs.get('gen1', None)
        self.gen2 = kwargs.get('gen2', None)
        self.useSmoothing = kwargs.get('useSmoothing',False)
        self.numSteps = kwargs.get('numSteps',51)
        self.numProc = kwargs.get('numProc',1)
        self.seed = kwargs.get('seed',5)

        if self.numProc>1:
            self.pool = Pool(self.numProc)
            self.mapFunc = self.pool.map
        else:
            self.mapFunc = map


    def get_tau(self, x, y, r_1):
        a = self.alpha0(x,y)
        if a==0:
            return 10**50
        return r_1/a

    def evolve_species(self, r, m, o):
        a = self.alpha0(m, o)
        left = 0

        originalM = m
        originalO = o
        for i, propensity in enumerate(self.propensities):
            param = self.parameters[i]
            right = left + propensity(param, m, o)
            if left / a <= r < right / a:
                m = m + self.increments[0][i]
                o = o + self.increments[1][i]
                break
            left = right

        meanM = 0
        meanO = 0

        for i, propensity in enumerate(self.propensities):
            param = self.parameters[i]
            meanM += self.increments[0][i] * propensity(param, m, o) / a
            meanO += self.increments[1][i] * propensity(param, m, o) / a

        return max(m, 0), max(o, 0), originalM + meanM, originalO + meanO

    def generateR1(self,state):
        r1 = state.random_sample()
        logR1 = np.log(r1) * -1
        return logR1

    def generateR2(self,state):
        r2 = state.random_sample()
        return r2

    def run_path(self, *parameters):

        self.timeGrid = np.linspace(0,self.T,self.numSteps)
        self.parameters = parameters[:-1]
        self.alpha0 = lambda x,y: sum([pair[1](pair[0],x,y) for pair in zip(parameters,self.propensities)])

        tauSamples = []
        tauSamples.append(0.0)
        aSamples = []
        bSamples = []
        aSamples.append(0)
        bSamples.append(0)
        nA = self.a
        nB = self.b

        seed1 = parameters[-1]+self.seed
        seed2 = parameters[-1]+self.seed+100000

        state1 = nprandom.RandomState(seed1)
        state2 = nprandom.RandomState(seed2)

        while tauSamples[-1]<self.T:

            r1 = self.generateR1(state1)
            tau = self.get_tau(nA, nB, r1)
            if (tau+tauSamples[-1]>self.T):
                tauSamples.append(self.T)
                aSamples.append(aSamples[-1])
                bSamples.append(bSamples[-1])
                break
            tauSamples.append(tau+tauSamples[-1])

            r2 = self.generateR2(state2)
            nA,nB,expA,expB = self.evolve_species(r2, nA, nB)

            if self.useSmoothing:
                aSamples.append(expA)
                bSamples.append(expB)
            else:
                aSamples.append(nA)
                bSamples.append(nB)

        indices = np.searchsorted(tauSamples, self.timeGrid, side = 'right') - 1 # Get point before
        aPath = [aSamples[k] for k in indices]
        bPath = [bSamples[k] for k in indices]

        return np.array(aPath[1:]+bPath[1:])

    def run_simulation(self,*parameters):

        runner = partial(self.run_path, *parameters)
        paths = self.mapFunc(runner,range(0,self.nPaths))

        zipped_paths = zip(*paths)
        aResult = [sum(step) / len(step) for step in zipped_paths[:self.numSteps-1]]
        bResult = [sum(step) / len(step) for step in zipped_paths[self.numSteps-1:]]

        return aResult+bResult

    def take_gradients_path(self, *parameters):

        gr = jacobian(self.run_path, 0)
        gradient = gr(*parameters)
        return gradient

    def take_gradients(self, *parameters):

        runner = partial(self.take_gradients_path, *parameters)
        paths = self.mapFunc(runner,range(0,self.nPaths))

        zipped_paths = zip(*paths)
        aResult = [sum(step) / len(step) for step in zipped_paths[:self.numSteps-1]]
        bResult = [sum(step) / len(step) for step in zipped_paths[self.numSteps-1:]]

        return aResult+bResult

