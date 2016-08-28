import unittest
from gillespie import Gillespie
from gillespie import Setup
import autograd.numpy as np
from autograd import jacobian
import numpy.random as nprandom
from itertools import tee
import seaborn as sns
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
        my_gillespie = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr,nPaths = nPaths,T=T,gen1=g1,gen2=g2,useSmoothing=True)
        observed_data = my_gillespie.run_simulation(*parameters)

        starting_parameters = [x for x in parameters]
        starting_parameters[idx] = parameters[idx]+0.2
        dw = 0.00001
        prev_loss_function_grad = 100001.0
        loss_function_grad = 100000.0
        print observed_data[:10],"\n"
        cnt=0

        while cnt<20:
            print "prev grad {} current grad {} starting_parameters[0] {}".format(prev_loss_function_grad,loss_function_grad,starting_parameters[0])
            prev_loss_function_grad = loss_function_grad

            gillespieGrad = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr, nPaths = nPaths,T=T,gen1=g1grad,gen2=g2grad,useSmoothing=True)

            gr = jacobian(gillespieGrad.run_simulation,idx)
            gradient = gr(*starting_parameters)

            gillespieSim = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr, nPaths = nPaths,T=T,gen1=g1sim,gen2=g2sim,useSmoothing=True)
            simulated_data = gillespieSim.run_simulation(*starting_parameters)
            print "\n Simulated Data: \n {} \n Gradients: \n {} ".format(simulated_data[:10],gradient[:10])
            element_wise_grad = [(observed_data[i]-simulated_data[i])*gradient[i] for i in range(len(gradient))]

            loss_function_grad = 2*sum(element_wise_grad)
            starting_parameters[idx] = starting_parameters[idx]-loss_function_grad*dw


            g1grad,g1sim = tee(generateR1(5),2)
            g2grad,g2sim = tee(generateR2(10),2)
            cnt+=1

if __name__ == "__main__":



    suite = unittest.TestSuite()
    suite.addTest(GillespieTestSuite())
    unittest.TextTestRunner().run(suite)
