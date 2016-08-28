import unittest
from gillespie import Gillespie
from gillespie import Setup
import autograd.numpy as np
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
        nPaths = 2#setup.get_number_of_paths()
        T = 2.0#setup.get_time_horizon()

        tau = np.linspace(0,T,51)
        idx = 0

#observed data generation:

        g1,g1grad,g1sim = tee(generateR1(5),3)
        g2,g2grad,g2sim = tee(generateR1(10),3)
        my_gillespie = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr,nPaths = nPaths,T=T,gen1=g1,gen2=g2,useSmoothing=False)
        discrete_sim_data = my_gillespie.run_simulation(*parameters)
        discrete_a = discrete_sim_data[:50]
        discrete_b = discrete_sim_data[50:]

        a_data = []
        b_data = []
        offset = -0.5
        step = 0.05
        for i in range(20):
            starting_parameters = [x for x in parameters]
            starting_parameters[idx] = parameters[idx]+step*i+offset
            my_gillespie = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr,nPaths = nPaths,T=T,gen1=g1,gen2=g2,useSmoothing=True)
            res = my_gillespie.run_simulation(*starting_parameters)
            a_data.append(res[:50])
            b_data.append(res[50:])
            print(starting_parameters[idx])

        timeGrid = np.linspace(0,T,51)[1:]

        for i in range(len(a_data)):
            plt.plot(timeGrid, a_data[i], 'green', label = "smooth {}".format(i))

        plt.plot(timeGrid, discrete_a, 'red', label = "discrete", marker='o')

        plt.show()
