# test for generate seismic data
# Std import block
import time

import numpy as np
import matplotlib.pyplot as plt

from pysit import *
from pysit.gallery import horizontal_reflector
import os
import shutil

#   Define Domain
pmlx = PML(0.1, 100)
pmlz = PML(0.1, 100)
os.environ["OMP_NUM_THREADS"] = "12"
x_config = (0.1, 1.0, pmlx, pmlx)
z_config = (0.1, 0.8, pmlz, pmlz)

d = RectangularDomain(x_config, z_config)

m = CartesianMesh(d, 91, 71)

#   Generate true wave speed
C, C0, m, d = horizontal_reflector(m)

# Set up shots
zmin = d.z.lbound
zmax = d.z.rbound
zpos = zmin + (1./9.)*zmax

shots = equispaced_acquisition(m,
                               RickerWavelet(10.0),
                               sources=5,
                               source_depth=zpos,
                               source_kwargs={},
                               receivers='max',
                               receiver_depth=zpos,
                               receiver_kwargs={},
                               )

shots_pickle = equispaced_acquisition(m,
                               RickerWavelet(10.0),
                               sources=5,
                               source_depth=zpos,
                               source_kwargs={},
                               receivers='max',
                               receiver_depth=zpos,                               
                               receiver_kwargs={},
                               )

shots_savemat = equispaced_acquisition(m,
                               RickerWavelet(10.0),
                               sources=5,
                               source_depth=zpos,
                               source_kwargs={},
                               receivers='max',
                               receiver_depth=zpos,                               
                               receiver_kwargs={},
                               )

shots_h5py = equispaced_acquisition(m,
                               RickerWavelet(10.0),
                               sources=5,
                               source_depth=zpos,
                               source_kwargs={},
                               receivers='max',
                               receiver_depth=zpos,                               
                               receiver_kwargs={},
                               )


# Define and configure the wave solver
trange = (0.0,3.0)

solver_time = ConstantDensityAcousticWave(m,
                                          spatial_accuracy_order=6,
                                          kernel_implementation='cpp',
                                          trange=trange)
# Generate Seismic data/ Loading it

print("shot TIME generation...")
base_model = solver_time.ModelParameters(m,{'C':C})
generate_seismic_data(shots, solver_time, base_model)
print("DONE")

print("pickle TIME generation...")
generate_seismic_data(shots, solver_time, base_model,save_method='pickle')
generate_seismic_data_from_file(shots_pickle, solver_time, save_method='pickle')
print("DONE")

print("savemat TIME generation...")
generate_seismic_data(shots, solver_time, base_model,save_method='savemat')
generate_seismic_data_from_file(shots_savemat, solver_time, save_method='savemat')
print("DONE")
print("h5py TIME generation...")
generate_seismic_data(shots, solver_time, base_model,save_method='h5py')
generate_seismic_data_from_file(shots_h5py, solver_time, save_method='h5py')
print("DONE")

# Now compare the result to be sure of your code catches all the exceptions
for i in range(1,len(shots)+1):
    if (shots[i-1].receivers.data == shots_pickle[i-1].receivers.data).all():
        print("Test for receivers %d data of pickle : OK" % i) 
    else:
        print(("Test for receivers %d data of pickle : fail") % i) 
    if (shots[i-1].receivers.data == shots_savemat[i-1].receivers.data).all():
        print("Test for receivers %d data of savemat : OK" % i) 
    else:
        print(("Test for receivers %d data of savemat : fail") % i) 
    if (shots[i-1].receivers.data == shots_h5py[i-1].receivers.data).all():
        print("Test for receivers %d data of hdf5 : OK" % i) 
    else:
        print(("Test for receivers %d data of hdf5 : fail") % i) 
    if (shots[i-1].receivers.ts == shots_pickle[i-1].receivers.ts).all():
        print("Test for receivers %d ts of pickle : OK" % i) 
    else:
        print(("Test for receivers %d ts of pickle : fail") % i)
    if (shots[i-1].receivers.ts == shots_savemat[i-1].receivers.ts).all():
        print("Test for receivers %d ts of savemat : OK" % i) 
    else:
        print(("Test for receivers %d ts of savemat : fail") % i) 
    if (shots[i-1].receivers.ts == shots_h5py[i-1].receivers.ts).all():
        print("Test for receivers %d ts of hdf5 : OK" % i) 
    else:
        print(("Test for receivers %d ts of hdf5 : fail") % i) 

#########################################################
# raise some error to verify that there are well caught #
#########################################################
# os.remove("./shots/shot_2.hdf5")
# generate_seismic_data_from_file(shots_h5py, save_method='h5py')
# generate_seismic_data_from_file(shots_h5py, save_method='pickle')
# generate_seismic_data_from_file(shots_h5py, save_method='petsc')
# generate_seismic_data_from_file(shots_h5py)


##############################################################
#    Now Frequencies we save direct fourier transform data   #
##############################################################
solver = ConstantDensityHelmholtz(m)
frequencies = [2.0, 3.5, 5.0, 6.5, 8.0, 9.5]

# Generate synthetic Seismic data
base_model = solver.ModelParameters(m,{'C': C})
tt = time.time()

print("shot FREQUENCY generation...")
generate_seismic_data(shots, solver, base_model, frequencies=frequencies)
print("DONE")

print("pickle FREQUENCY generation...")
generate_seismic_data(shots, solver, base_model,save_method='pickle', frequencies=frequencies)
generate_seismic_data_from_file(shots_pickle, solver, save_method='pickle')
print("DONE")

print("savemat FREQUENCY generation...")
generate_seismic_data(shots, solver, base_model,save_method='savemat', frequencies=frequencies)
generate_seismic_data_from_file(shots_savemat, solver, save_method='savemat', frequencies=frequencies)
print("DONE")

def compare_dict(a,b):
  if list(a.keys()) == list(b.keys()):
    same = True
    for key in a:
      same = (same and ((a[key]-b[key]) < np.finfo(float).eps).all())
      if same == False : break
    return same
  else:
    return False


# Now compare the result to be sure of your code catches all the exceptions
for i in range(1,len(shots)+1):
    if compare_dict(shots[i-1].receivers.data_dft, shots_pickle[i-1].receivers.data_dft):
      print("Test for receivers %d data_dft of pickle : OK" % i) 
    else:
      print(("Test for receivers %d data_dft of pickle : fail") % i) 
    if compare_dict(shots[i-1].receivers.data_dft, shots_savemat[i-1].receivers.data_dft):
      print("Test for receivers %d data_dft of savemat : OK" % i) 
    else:
      print(("Test for receivers %d data_dft of savemat : fail") % i) 
    