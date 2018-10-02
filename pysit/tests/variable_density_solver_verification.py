#1: I will show that for second order accuracy on a homogeneous model the CDA and VDA solver are exactly the same on a constant density medium

#2: In this example I will show that the Laplacian of the variable density acoustics (VDA) solver is self-adjoint 
#With the exception of the boundary nodes, which is also the case in the CDA solver.

#3: For higher spatial accuracy orders the variable density solver on a constant density medium is more expensive.
#It has almost twice as many bands in the laplacian as the CDA solver because the heterogeneous Laplacian
#is implemented as div*((1/rho)grad) = -D^T((1/rho) *D). This process will create extra bands at higher orders of spatial accuracy.
#This is the price we pay for a self-adjoint Laplacian in VDA

#4: Simple demonstration. Uniform velocity, but density gradient. We observe reflection

#5: Laplacian does not deviate from symmetry at the pixels centered around the density jump (i.e. pixels 50, 50+91, 50+182, 50+273, ...)


from pysit import *
from pysit_extensions.convenient_plot_functions.plot_functions import *
import copy
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import scipy.io as spio



#   Define Domain
dx = 12.5 
dz = 12.5

nx = 51 
nz = 51

x_min = 0.0
x_max = (nx-1)*dx

z_min = 0.0
z_max = (nz-1)*dz

pmlx = PML(20*dx, 50)
pmlz = PML(20*dz, 50)

x_config = (x_min, x_max, pmlx, pmlx)
z_config = (z_min, z_max, pmlz, pmlz)

d = RectangularDomain(x_config, z_config)

m = CartesianMesh(d, nx, nz)

#Define velocity and density
background_vel = 1500.0
C_2d = background_vel*np.ones((nz,nx)) #UNIFORM
rho_2d = 1.0*np.ones((nz,nx))          #UNIFORM. Using 1.0 so results VDA solver should be the same as CDA solver

kappa_2d = rho_2d*C_2d**2 # vp**2 = kappa/rho for acoustic medium. Kappa = lamba + 2\3 * mu = lambda when mu = 0 in acoustic medium  

#create input arrays
kappa = np.reshape(kappa_2d, (nx*nz,1),'F')
rho   = np.reshape(rho_2d, (nx*nz,1),'F')
C     = np.reshape(C_2d, (nx*nz,1),'F')

#define model parameters
model_param_cda = {'C': C}
model_param_vda = {'kappa': kappa, 'rho': rho}

# Set up shot
peakfreq = 6.0
depth_source_receiver = 12.5

x_pos_source = 100.0 
z_pos_source = depth_source_receiver

x_pos_receivers = np.linspace(x_min, x_max, nx)
z_pos_receivers = depth_source_receiver #I guess it will always record 0.0 because that is what this Dirichlet boundary is 

shots_time_cda = []
source_approx = 'delta'
receiver_approx = 'delta'
# Define source location and type
source = PointSource(m, (x_pos_source, z_pos_source), RickerWavelet(peakfreq), approximation = source_approx)
receivers = ReceiverSet(m, [PointReceiver(m, (x, z_pos_receivers), approximation = receiver_approx) for x in x_pos_receivers])
shot = Shot(source, receivers)

#Both start without any recorded data
shots_time_cda.append(shot)
shots_time_vda = copy.deepcopy(shots_time_cda)

# Define and configure the wave solver
trange = (0.0,2.0)

accuracy_order = 2

solver_time_cda = ConstantDensityAcousticWave(m,
                                              spatial_accuracy_order=accuracy_order,
                                              trange=trange,
                                              kernel_implementation = 'numpy',
                                              )

solver_time_vda = VariableDensityAcousticWave(m,
                                              spatial_accuracy_order=accuracy_order,
                                              trange=trange,
                                              kernel_implementation = 'numpy',
                                              )

# Generate synthetic Seismic data
base_model_cda = solver_time_cda.ModelParameters(m,model_param_cda)
base_model_vda = solver_time_vda.ModelParameters(m,model_param_vda)

sys.stdout.write('Generating CDA solver data uniform model... \n')
generate_seismic_data(shots_time_cda, solver_time_cda, base_model_cda)
sys.stdout.write('Generating VDA solver data uniform model... \n')
generate_seismic_data(shots_time_vda, solver_time_vda, base_model_vda)

shotgather_time_cda = shots_time_cda[0].receivers.data
shotgather_time_vda = shots_time_vda[0].receivers.data

#divide by max: scalar difference due to density
#shotgather_time_vda = shotgather_time_vda * (shotgather_time_cda.max()/shotgather_time_vda.max())
rel_dif_time = np.linalg.norm(shotgather_time_vda - shotgather_time_cda)/np.linalg.norm(shotgather_time_vda)
print("1: Relative difference in CDA and VDA solver on uniform grid and 2nd order in space: " + str(rel_dif_time)) 
print("Exactly the same for second order accuracy.")

L_cda = solver_time_cda.operator_components.L
L_vda = solver_time_vda.operator_components.L

#Laplacians are not exactly self adjoint because of boundary conditions. 
CDA_deviation_self_adjoint = L_cda-L_cda.T
VDA_deviation_self_adjoint = L_vda-L_vda.T 

#remove some tiny deviations due to rounding errors
eps = 1e-14

#Remove elements between -eps and +eps
elements_within_plus_min_eps = np.logical_and(CDA_deviation_self_adjoint.data<=eps, CDA_deviation_self_adjoint.data>=-eps)
elements_not_within_plus_min_eps = np.logical_not(elements_within_plus_min_eps)
CDA_deviation_self_adjoint.data *= elements_not_within_plus_min_eps 
CDA_deviation_self_adjoint.eliminate_zeros()

#Remove elements between -eps and +eps
elements_within_plus_min_eps = np.logical_and(VDA_deviation_self_adjoint.data<=eps, VDA_deviation_self_adjoint.data>=-eps)
elements_not_within_plus_min_eps = np.logical_not(elements_within_plus_min_eps)
VDA_deviation_self_adjoint.data *= elements_not_within_plus_min_eps 
VDA_deviation_self_adjoint.eliminate_zeros()


print("2: Displaying entries with deviation from self adjoint larger than %e \n"%eps)
plt.figure(1); plt.spy(CDA_deviation_self_adjoint, markersize=3); plt.title('CDA entries deviating from self-adjoint')
plt.figure(2); plt.spy(VDA_deviation_self_adjoint, markersize=3); plt.title('VDA entries deviating from self-adjoint')

###############################################################
############### NOW REPEAT FOR 8-TH ORDER ACCURACY ############
###############################################################

accuracy_order = 8

solver_time_cda = ConstantDensityAcousticWave(m,
                                              spatial_accuracy_order=accuracy_order,
                                              trange=trange,
                                              kernel_implementation = 'numpy',
                                              )

solver_time_vda = VariableDensityAcousticWave(m,
                                              spatial_accuracy_order=accuracy_order,
                                              trange=trange,
                                              kernel_implementation = 'numpy',
                                              )

# Generate synthetic Seismic data
base_model_cda = solver_time_cda.ModelParameters(m,model_param_cda)
base_model_vda = solver_time_vda.ModelParameters(m,model_param_vda)

sys.stdout.write('Generating CDA solver data uniform model... \n')
generate_seismic_data(shots_time_cda, solver_time_cda, base_model_cda)
sys.stdout.write('Generating VDA solver data uniform model... \n')
generate_seismic_data(shots_time_vda, solver_time_vda, base_model_vda)

shotgather_time_cda = shots_time_cda[0].receivers.data
shotgather_time_vda = shots_time_vda[0].receivers.data

rel_dif_time = np.linalg.norm(shotgather_time_vda - shotgather_time_cda)/np.linalg.norm(shotgather_time_vda)
print("3: Relative difference in CDA and VDA solver on uniform grid and 8th order in space: " + str(rel_dif_time))
print("Slightly different because VDA solver has more bands. But difference is tiny \n")

#######################################################################
############### NOW LOOK AT A CASE WITH A DENSITY CONTRAST ############
############### still a PML which is too thin so some      ############ 
############### artificial reflections.                    ############
#######################################################################

#Define velocity and density
background_vel = 1500.0
C_2d = background_vel*np.ones((nz,nx)) #UNIFORM
rho_2d = 1000.0*np.ones((nz,nx))       
rho_2d[30:,:] = 5000.0                 #INTRODUCE DENSITY JUMP AT DEPTH PIXEL 30 (50 IN PML PADDED MODEL, 20 PML NODES)

kappa_2d = rho_2d*C_2d**2 # vp**2 = kappa/rho for acoustic medium. Kappa = lamba + 2\3 * mu = lambda when mu = 0 in acoustic medium  

kappa = np.reshape(kappa_2d, (nx*nz,1),'F')
rho   = np.reshape(rho_2d, (nx*nz,1),'F')
C     = np.reshape(C_2d, (nx*nz,1),'F')

model_param_cda = {'C': C}
model_param_vda = {'kappa': kappa, 'rho': rho}

# Set up shot
shots_time_vda_bump = []
source_approx = 'delta'
receiver_approx = 'delta'
# Define source location and type
source = PointSource(m, (x_pos_source, z_pos_source), RickerWavelet(peakfreq), approximation = source_approx)
receivers = ReceiverSet(m, [PointReceiver(m, (x, z_pos_receivers), approximation = receiver_approx) for x in x_pos_receivers])
shot = Shot(source, receivers)
shots_time_vda_bump.append(shot)

# Define and configure the wave solver
trange = (0.0,2.0)
solver_time_vda_bump = VariableDensityAcousticWave(m,
                                              spatial_accuracy_order=accuracy_order,
                                              trange=trange,
                                              kernel_implementation = 'numpy',
                                              )

# Generate synthetic Seismic data
wavefields = []
base_model_vda_bump = solver_time_vda_bump.ModelParameters(m,model_param_vda)
generate_seismic_data(shots_time_vda_bump, solver_time_vda_bump, base_model_vda_bump, wavefields=wavefields)

print("4: Plotting trace, constant velocity but density bump introduces reflection as expected")

shotgather_time_vda_bump = shots_time_vda_bump[0].receivers.data
plt.figure(4)
plt.plot(shotgather_time_vda_bump[:,10])


#Now that there is a density gradient, differences from self-adjoint could be introduced at the pixels surrounding the jump
#Here in this Laplacian corresponding to the padded 91*91 model  

L_vda_dense_bump = solver_time_vda_bump.operator_components.L
VDA_bump_deviation_self_adjoint = L_vda_dense_bump - L_vda_dense_bump.T

#Remove elements between -eps and +eps
elements_within_plus_min_eps = np.logical_and(VDA_bump_deviation_self_adjoint.data<=eps, VDA_bump_deviation_self_adjoint.data>=-eps)
elements_not_within_plus_min_eps = np.logical_not(elements_within_plus_min_eps)
VDA_bump_deviation_self_adjoint.data *= elements_not_within_plus_min_eps 
VDA_bump_deviation_self_adjoint.eliminate_zeros()

print("5: Displaying entries with deviation from self adjoint larger than for density jump model %e \n"%eps)
plt.figure(5); plt.spy(VDA_bump_deviation_self_adjoint, markersize=3); plt.title('VDA entries deviating from self-adjoint dens bump model')
plt.show()

print("6: Plotting wavefield, constant velocity but density bump introduces reflection as expected")  
vis.animate(wavefields, m, display_rate = 3)
