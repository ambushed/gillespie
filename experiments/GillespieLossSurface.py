from gillespie import Gillespie
from gillespie import Setup
import autograd.numpy as np
from autograd import value_and_grad
from gillespie.GillespieAdam import adam
import matplotlib
import math
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from pyevtk.hl import gridToVTK

import vtk

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
np.set_printoptions(precision=4)
import decimal

def drange(x,y,jump):
    while x<y:
        yield x
        x+=jump

def gillespieGradientWalk():

    np.set_printoptions(precision=4)
    setup = Setup(yaml_file_name="lotka_volterra.yaml")
    propensities = setup.get_propensity_list()
    parameters = np.array(setup.get_parameter_list())
    species = setup.get_species()
    incr = setup.get_increments()
    nPaths = 1 #setup.get_number_of_paths()
    T = 1.0 #setup.get_time_horizon()
    seed = 100
    numProc = 1
    idx = 0

    my_gillespie = Gillespie(species=species,propensities=propensities,
                             increments=incr,nPaths = nPaths,T=T,useSmoothing=True, seed = seed, numProc = numProc)

    observed_data = my_gillespie.run_simulation(parameters)

    param_0_range = [x_arr for x_arr in drange(0.001,3.0,1.0)]
    param_1_range = [x_arr for x_arr in drange(0.001,3.0,1.0)]
    param_2_range = [x_arr for x_arr in drange(0.001,3.0,1.0)]

    def lossFunction(parameters,dummy=None):

        gillespieGrad = Gillespie(species=species,propensities=propensities,increments=incr,
                                  nPaths = nPaths,T=T,useSmoothing=True, seed = seed, numProc = numProc )

        simulated_data = gillespieGrad.run_simulation(parameters)

        #return sum(0.5*(np.array(simulated_data)-np.array(observed_data))**2)
        return np.sum(np.square( (np.array(simulated_data)-np.array(observed_data))))

    from math import log

    n = len(param_0_range)
    res = np.empty((n,n))
    for x in param_0_range:
        print "\n\n"
        for y in param_1_range:
            print "\n"
            #for z in param_2_range:
            #    print "."
            res[x,y]= log(lossFunction([x,y,2.0]))
                #results.append("{} {} {} {}".format(log(x),log(y),log(x),log(lossFunction([x,y,z]))))

    return param_0_range,param_1_range,param_2_range,res

def testLinePlot():
    "Test if line plots can be built with python"

    # Set up a 2D scene, add an XY chart to it
    view = vtk.vtkContextView()
    view.GetRenderer().SetBackground(1.0, 1.0, 1.0)
    view.GetRenderWindow().SetSize(400, 300)
    chart = vtk.vtkChartXYZ()
    plot = vtk.vtkPlotSurface()
    view.GetScene().AddItem(chart)

    # Create a table with some points in it
    table = vtk.vtkTable()

    arrX = vtk.vtkFloatArray()
    arrX.SetName("X Axis")

    arrC = vtk.vtkFloatArray()
    arrC.SetName("Cosine")

    arrS = vtk.vtkFloatArray()
    arrS.SetName("Sine")

    arrS2 = vtk.vtkFloatArray()
    arrS2.SetName("Sine2")

    numPoints = 69
    inc = 7.5 / (numPoints - 1)

    for i in range(0, numPoints):
        arrX.InsertNextValue(i * inc)
        arrC.InsertNextValue(math.cos(i * inc) + 0.0)
        arrS.InsertNextValue(math.sin(i * inc) + 0.0)
        arrS2.InsertNextValue(math.sin(i * inc) + 0.5)

    table.AddColumn(arrX)
    table.AddColumn(arrC)
    table.AddColumn(arrS)
    table.AddColumn(arrS2)

    plot.SetInputData(table)
    # Now add the line plots with appropriate colors
    line = chart.AddPlot(plot)
    #line.SetInputData(table, 0, 1)
    #line.SetColor(0, 255, 0, 255)
    #line.SetWidth(1.0)

    #line = chart.AddPlot(0)
    #line.SetInputData(table, 0, 2)
    #line.SetColor(255, 0, 0, 255)
    #line.SetWidth(5.0)

    #line = chart.AddPlot(0)
    #line.SetInputData(table, 0, 3)
    #line.SetColor(0, 0, 255, 255)
    #line.SetWidth(4.0)

    view.GetRenderWindow().SetMultiSamples(0)
    view.GetRenderWindow().Render()
    view.GetInteractor().Start()


if __name__ == "__main__":

    X,Y,Z,res = gillespieGradientWalk()

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #X = np.arange(-5, 5, 0.25)
    #Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    #R = np.sqrt(X ** 2 + Y ** 2)
    #Z = np.sin(R)
    surf = ax.plot_surface(X, Y, res, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

    #testLinePlot()

    chart = vtk.vtkChartXYZ()
    plot = vtk.vtkPlotSurface()
    view = vtk.vtkContextView()
    view.GetRenderWindow().SetSize(400, 300)
    view.GetScene().AddItem(chart)

    # create a surface vtkTable
    table = vtk.vtkTable()
    numPoints = 70
    inc = 9.424778 / (numPoints - 1)
    for x in range(numPoints):
        arr = vtk.vtkDoubleArray()
        for y in range(numPoints):
            z = math.sin(math.sqrt(x * x + y * y))
            arr.InsertNextValue(z)
        table.AddColumn(arr)

    plot.SetYRange(0, 9.424778)
    plot.SetXRange(0, 9.424778)
    plot.SetInputData(table)
    chart.AddPlot(plot)

    view.GetRenderWindow().SetMultiSamples(0)
    view.GetInteractor().Initialize()
    view.GetRenderWindow().Render()

    view.GetInteractor().Start()