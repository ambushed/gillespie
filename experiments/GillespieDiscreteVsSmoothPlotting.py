from gillespie import Gillespie
from gillespie import Setup
import autograd.numpy as np
import numpy.random as nprandom
from itertools import tee
import pylab as plt

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

g1,g1S = tee(generateR1(200),2)
g2,g2S = tee(generateR2(10),2)

setup = Setup(yaml_file_name="../models/lotka_volterra.yaml")
propensities = setup.get_propensity_list()
parameters = setup.get_parameter_list()
species = setup.get_species()
incr = setup.get_increments()
nPaths = 2
T = 2
my_gillespieUp = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr, nPaths = nPaths,T=T,gen1=g1,gen2=g2,useSmoothing = False)
aMean = my_gillespieUp.run_simulation(*parameters)
my_gillespieSmooth = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr, nPaths = nPaths,T=T,gen1=g1S,gen2=g2S,useSmoothing = True)
aMeanS = my_gillespieSmooth.run_simulation(*parameters)

tau = np.linspace(0,T,51)

plt.plot(tau[1:],aMeanS,label="Smooth A")
plt.step(tau[1:],aMean,label='Step A')
legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

