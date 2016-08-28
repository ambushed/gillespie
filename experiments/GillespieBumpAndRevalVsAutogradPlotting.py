from gillespie import Gillespie
from gillespie import Setup
import autograd.numpy as np
from autograd import jacobian
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

g1,g1Up,g1Down = tee(generateR1(5),3)
g2,g2Up,g2Down = tee(generateR2(10),3)

setup = Setup(yaml_file_name="../models/lotka_volterra.yaml")
propensities = setup.get_propensity_list()
parameters = setup.get_parameter_list()
species = setup.get_species()
incr = setup.get_increments()
nPaths = 1
T = 2.0

tau = np.linspace(0,T,51)

delta = 0.000001
idx = 0

my_gillespie = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr, nPaths = nPaths,T=T,gen1=g1,gen2=g2,useSmoothing=True)
gradients = my_gillespie.take_gradients(*parameters)
aGradient = gradients[:50]
bGradient = gradients[50:]

parameters1 = [x for x in parameters]
parameters1[idx] = parameters[idx]+delta
my_gillespieUp = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr, nPaths = nPaths,T=T,gen1=g1Up,gen2=g2Up,useSmoothing=True)
bumpUp = my_gillespieUp.run_simulation(*parameters1)
parameters1[idx] = parameters[idx]-delta
my_gillespieDown = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr, nPaths = nPaths,T=T,gen1=g1Down,gen2=g2Down,useSmoothing=True)
bumpDown = my_gillespieDown.run_simulation(*parameters1)
aBumpUp = bumpUp[:50]
aBumpDown = bumpDown[:50]
bBumpUp = bumpUp[50:]
bBumpDown = bumpDown[50:]
diffA = [(up-down)/(2*delta) for up,down in zip(aBumpUp,aBumpDown)]
diffB = [(up-down)/(2*delta) for up,down in zip(bBumpUp,bBumpDown)]

fig,(ax0, ax1) = plt.subplots(nrows=2,sharex=True)
ax0.plot(tau[1:],diffA,label="BumpAndReval ",linewidth=10)
ax0.plot(tau[1:],aGradient,label="Autograd ",linewidth=5)
ax0.set_title("Species A. AutoGrad vs BumpAndReval T: {} nPaths: {}".format(T,nPaths))

ax1.plot(tau[1:],diffB,label="BumpAndReval",linewidth=10)
ax1.plot(tau[1:],bGradient,label="Autograd ",linewidth=5)
ax1.set_title("Species B. AutoGrad vs BumpAndReval T: {} nPaths: {}".format(T,nPaths))
plt.show()
