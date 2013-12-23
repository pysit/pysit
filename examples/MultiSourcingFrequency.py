# Std import block
import time

import numpy as np
import matplotlib.pyplot as plt

from pysit import *
from pysit.gallery import horizontal_reflector

if __name__ == '__main__':
    #    Define Domain
    pmlx = PML(0.1, 100)
    pmlz = PML(0.1, 100)

    x_config = (0.1, 1.0, pmlx, pmlx)
    z_config = (0.1, 0.8, pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)

    m = CartesianMesh(d, 90, 70)

    #    Generate true wave speed
    C, C0, m, d = horizontal_reflector(m)

    # Set up shots
    Nshots = 3
    shots = []

    xmin = d.x.lbound
    xmax = d.x.rbound
    nx   = m.x.n
    zmin = d.z.lbound
    zmax = d.z.rbound

    source_list = []
    intensitylist = [1.0,2.0,1.5]

    source_list.append(PointSource(m, (xmax*1.0/3.0, 0.1), RickerWavelet(10.0), intensity = intensitylist[0]))
    source_list.append(PointSource(m, (xmax*2.0/3.0, 0.1), RickerWavelet(10.0), intensity = intensitylist[1]))
    source_list.append(PointSource(m, (xmax*0.5, 0.7), RickerWavelet(10.0), intensity = intensitylist[2])) #intensity of sources is different

    #2 PointSource objects are defined above. Group them together in a single SourceSet
    source_set = SourceSet(m,source_list)

    # Define set of receivers
    zpos = zmin + (1./9.)*zmax
    xpos = np.linspace(xmin, xmax, nx)
    receivers = ReceiverSet(m, [PointReceiver(m, (x, zpos)) for x in xpos])

    # Create and store the shot
    shot = Shot(source_set, receivers)




    # Define and configure the wave solver
    trange = (0.0,3.0)

    solver = ConstantDensityHelmholtz(m, model_parameters={'C': C}, spatial_shifted_differences=True, spatial_accuracy_order=4)

    modeling_tools = FrequencyModeling(solver)
    frequencies = [1.0, 3.0, 5.0,10.0]
    ret = modeling_tools.forward_model(shot, solver.model_parameters.without_padding(), frequencies=frequencies, return_parameters=['wavefield'])

    arr1 = np.reshape(ret['wavefield'][1.0], (90,70))
    arr3 = np.reshape(ret['wavefield'][3.0], (90,70))
    arr5 = np.reshape(ret['wavefield'][5.0], (90,70))
    arr10 = np.reshape(ret['wavefield'][10.0], (90,70))

    #arrreallim = arr.real.min(),arr.real.max()
    #arrimaglim = arr.imag.min(),arr.imag.max()

    ax1 = plt.subplot2grid((4,3), (0,0))
    plt.ylabel("1Hz", size=45)
    plt.title("real part", size=45)
    vis.plot(arr1.real, m)
    plt.colorbar()
    ax2 = plt.subplot2grid((4,3), (0,1))
    plt.title("imag part", size=45)
    vis.plot(arr1.imag, m)
    plt.colorbar()
    ax3 = plt.subplot2grid((4,3), (0,2))
    plt.title("absolute value", size=45)
    vis.plot(np.abs(arr1), m)
    plt.colorbar()

    ax4 = plt.subplot2grid((4,3), (1,0))
    plt.ylabel("3Hz", size=45)
    vis.plot(arr3.real, m)
    plt.colorbar()
    ax5 = plt.subplot2grid((4,3), (1,1))
    vis.plot(arr3.imag, m)
    plt.colorbar()
    ax6 = plt.subplot2grid((4,3), (1,2))
    vis.plot(np.abs(arr3), m)
    plt.colorbar()

    ax7 = plt.subplot2grid((4,3), (2,0))
    plt.ylabel("5Hz", size=45)
    vis.plot(arr5.real, m)
    plt.colorbar()
    ax8 = plt.subplot2grid((4,3), (2,1))
    vis.plot(arr5.imag, m)
    plt.colorbar()
    ax9 = plt.subplot2grid((4,3), (2,2))
    vis.plot(np.abs(arr5), m)
    plt.colorbar()

    ax10 = plt.subplot2grid((4,3), (3,0))
    plt.ylabel("10Hz", size=45)
    vis.plot(arr10.real, m)
    plt.colorbar()
    ax11 = plt.subplot2grid((4,3), (3,1))
    vis.plot(arr10.imag, m)
    plt.colorbar()
    ax12 = plt.subplot2grid((4,3), (3,2))
    vis.plot(np.abs(arr10), m)
    plt.colorbar()

