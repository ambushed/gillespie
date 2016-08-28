import unittest
from gillespie import Gillespie
from gillespie import Setup
import autograd.numpy as np
from autograd import jacobian
import numpy.random as nprandom
from itertools import tee
import pylab as plt

#1.  Run simulation to get "observed data" given the seeked parameters
#2.  Bump one (then more) parameter value to take it away from true value
#3.  Calculate the dG/dw and value of the Gillespie sim at the new w value
#4.  Plug into the Loss function
#5.  Do Gradient descent using the derivative of a loss function

class GillespieTestSuite(unittest.TestCase):

    def testLotkaVolterraBumpsVsAutograd(self):

        def generateR1(seed):
            state = nprandom.RandomState(seed)
            while True:
                r1 = state.random_sample(1000)
                logR1 = np.log(r1) * -1
                for r in logR1:
                    yield r

        def generateR2(seed):
            state = nprandom.RandomState(seed)
            while True:
                r2 = state.random_sample(1000)
                for r in r2:
                    yield r


        setup = Setup(yaml_file_name="../models/lotka_volterra.yaml")
        propensities = setup.get_propensity_list()
        parameters = setup.get_parameter_list()
        species = setup.get_species()
        incr = setup.get_increments()
        nPaths = 100 #setup.get_number_of_paths()
        T = setup.get_time_horizon()
        idx = 0
        numSteps = 101

#observed data generation:

        g1= generateR1(5)
        g2= generateR2(10)


        timeGrid = np.linspace(0, T, numSteps)[1:]
        a_diff = []
        b_diff = []
        my_gillespie = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr,nPaths = nPaths,T=T,gen1=g1,gen2=g2,useSmoothing=False,numSteps = numSteps)
        discrete_sim_data = my_gillespie.run_simulation(*parameters)
        base_a = discrete_sim_data[:numSteps-1]
        base_b = discrete_sim_data[numSteps-1:]
        for i in range(25):
            my_gillespie = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr,nPaths = nPaths,T=T,gen1=g1,gen2=g2,useSmoothing=False,numSteps = numSteps)
            discrete_sim_data = my_gillespie.run_simulation(*parameters)
            discrete_a = discrete_sim_data[:numSteps-1]
            discrete_b = discrete_sim_data[numSteps-1:]
            a_diff.append([a - b for a, b in zip(discrete_a, base_a)])
            b_diff.append([a - b for a, b in zip(discrete_b, base_b)])
            if i % 5 == 0:
                print("!")

        a_matrix = np.array(a_diff)
        b_matrix = np.array(b_diff)

        a_mean = a_matrix.mean(axis=0)
        a_std = a_matrix.std(axis=0)*2
        b_mean = b_matrix.mean(axis=0)
        b_std = b_matrix.std(axis=0)*2

        fig,(ax1,ax2) = plt.subplots(nrows=2,sharex=True)
        ax1.errorbar(timeGrid, a_mean, yerr=a_std)
        ax1.set_title("A species random numbers 2*std ")
        ax2.errorbar(timeGrid, b_mean, yerr=b_std)
        ax2.set_title("B species random numbers 2*std")

        plt.show()

