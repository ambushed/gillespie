__author__ = 'Vlad'

import os
import numpy.random as nprandom
import autograd.numpy as np
from functools import partial
from autograd import jacobian
from autograd import primitive
from pathos.multiprocessing import ProcessingPool as Pool

class Gillespie(object):

    def __init__(self,species,propensities,increments):
        self.species = species
        self.propensities = propensities
        self.increments = increments

    def __init__(self,*args,**kwargs):
        # args -- tuple of anonymous arguments
        # kwargs -- dictionary of named arguments
        self.species = kwargs.get('species', None)
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


    def get_tau(self, nSpecies, r_1):
        a = self.alpha0(nSpecies)
        if a==0:
            return 10**50
        return r_1/a

    def evolve_species(self, r, nSpecies):
        a = self.alpha0(nSpecies)
        left = 0

        originalCount = list(nSpecies)
        for i, propensity in enumerate(self.propensities):
            param = self.parameters[i]
            right = left + propensity(param, nSpecies)
            if left / a <= r < right / a:
                for j in range(len(nSpecies)):
                    nSpecies[j]=nSpecies[j]+self.increments[j][i]
                break
            left = right

        means = [0]*len(nSpecies)

        for i, propensity in enumerate(self.propensities):
            param = self.parameters[i]
            for j in range(len(nSpecies)):
                means[j]+=self.increments[j][i] * propensity(param, nSpecies) / a

        for j in range(len(nSpecies)):
            nSpecies[j] = max(nSpecies[j], 0)

        return nSpecies, np.array(originalCount) + np.array(means)

    def generateR1(self,state):
        r1 = state.random_sample()
        logR1 = np.log(r1) * -1
        return logR1

    def generateR2(self,state):
        r2 = state.random_sample()
        return r2

    def run_path(self, parameters, seed_offset):


        self.timeGrid = np.linspace(0,self.T,self.numSteps)
        self.parameters = np.log(parameters)
        self.alpha0 = lambda x: sum(pair[1](pair[0],x) for pair in zip(self.parameters,self.propensities))

        tauSamples = []
        tauSamples.append(0.0)
        samples = []
        for i in range(len(self.species)):
            samples.append([0])
        nSpecies = list(self.species)

        seed1 = seed_offset+self.seed
        seed2 = seed_offset+self.seed+100000

        state1 = nprandom.RandomState(seed1)
        state2 = nprandom.RandomState(seed2)

        while tauSamples[-1]<self.T:

            r1 = self.generateR1(state1)
            tau = self.get_tau(nSpecies, r1)
            if (tau+tauSamples[-1]>self.T):
                tauSamples.append(self.T)
                for i in range(len(self.species)):
                    samples[i].append(samples[i][-1])
                break
            tauSamples.append(tau+tauSamples[-1])

            r2 = self.generateR2(state2)
            nSpecies,expSpecies = self.evolve_species(r2,nSpecies)

            for i in range(len(self.species)):
                samples[i].append(expSpecies[i] if self.useSmoothing else nSpecies[i])

        indices = np.searchsorted(tauSamples, self.timeGrid, side = 'right') - 1 # Get point before
        result = []
        for i in range(len(self.species)):
            res = [samples[i][k] for k in indices]
            result.extend(res[1:])

        return np.array(result)

    def run_simulation(self,parameters):

        runner = partial(self.run_path, parameters)
        paths = self.mapFunc(runner,range(0,self.nPaths))

        zipped_paths = zip(*paths)
        aResult = [sum(step) / len(step) for step in zipped_paths[:self.numSteps-1]]
        bResult = [sum(step) / len(step) for step in zipped_paths[self.numSteps-1:]]

        return aResult+bResult

    def take_gradients_path(self, parameters, seed_offset):

        gr = jacobian(self.run_path, 0)
        gradient = gr(parameters,seed_offset)
        return gradient

    def take_gradients(self, parameters):

        runner = partial(self.take_gradients_path, parameters)
        paths = self.mapFunc(runner,range(0,self.nPaths))

        zipped_paths = zip(*paths)
        aResult = [sum(step) / len(step) for step in zipped_paths[:self.numSteps-1]]
        bResult = [sum(step) / len(step) for step in zipped_paths[self.numSteps-1:]]

        return aResult+bResult

