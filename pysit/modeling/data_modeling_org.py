

import time

import numpy as np

import shutil #folder deletion

from scipy.interpolate import interp1d

import os

__all__ = ['generate_seismic_data', 'generate_seismic_data_from_file', 'generate_shot_data_time', 'generate_shot_data_frequency']
__docformat__ = "restructuredtext en"


def generate_seismic_data(shots, solver, model, verbose=False, frequencies=None, save_method=None, **kwargs):
    """Given a list of shots and a solver, generates seismic data.

    Parameters
    ----------
    shots : list of pysit.Shot
        Collection of shots to be processed
    solver : pysit.WaveSolver
        Instance of wave solver to be used.
    **kwargs : dict, optional
        Optional arguments.

    Notes
    -----
    `kwargs` may be used to specify `C0` and `wavefields` to    `generate_shot_data`.

    """
    
    if verbose:
        print('Generating data...')
        tt = time.time()

    if solver.supports['equation_dynamics'] == "time":
        for shot in shots:
            generate_shot_data_time(shot, solver, model, verbose=verbose, **kwargs)
    elif solver.supports['equation_dynamics'] == "frequency":
        if frequencies is None:
            raise TypeError('A frequency solver is passed, but no frequencies are given')
        elif 'petsc' in kwargs and kwargs['petsc'] is not None:
            # solve the Helmholtz operator for several rhs                
            generate_shot_data_frequency_list(shots, solver, model, frequencies, verbose=verbose, **kwargs)
        else:
            for shot in shots:            
                generate_shot_data_frequency(shot, solver, model, frequencies, verbose=verbose, **kwargs)
    else:
        raise TypeError("A time or frequency solver must be specified.")

    if verbose:
        data_tt = time.time() - tt
        print('Data generation: {0}s'.format(data_tt))
        print('Data generation: {0}s/shot'.format(data_tt/len(shots)))

    if verbose:
        print('Saving data...')
        tt = time.time()
    if solver.supports['equation_dynamics'] == "time":
        # store shots in memory for a next simulation
        if save_method is not None:
            if save_method=='pickle':
                try:
                    import pickle
                except ImportError:
                    raise ImportError('cPickle is not installed please install it and try again')

                newpath = r'./shots_time'
                if os.path.exists(newpath):
                    shutil.rmtree(newpath)

                if not os.path.exists(newpath):
                    os.makedirs(newpath)

                for i in range(len(shots)):
                    fshot =newpath+'/shot_'+str(i+1)+'.save'
                    f = file(fshot, 'wb')
                    pickle.dump(shots[i].receivers.data,f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.close()
                    fts =newpath+'/ts_'+str(i+1)+'.save'
                    f = file(fts, 'wb')
                    pickle.dump(shots[i].receivers.ts,f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.close()
            elif save_method=='savemat':
                try:
                    import scipy.io as io
                except ImportError:
                    raise ImportError('scipy.io is not installed please install it and try again')

                newpath = r'./shots_time'
                if os.path.exists(newpath):
                    shutil.rmtree(newpath)

                if not os.path.exists(newpath):
                    os.makedirs(newpath)

                for i in range(len(shots)):
                    fshot =newpath+'/shot_'+str(i+1)+'.mat'
                    io.savemat(fshot,mdict={fshot:shots[i].receivers.data})
                    fts =newpath+'/ts_'+str(i+1)+'.mat'
                    io.savemat(fts,mdict={fts:shots[i].receivers.ts})
            elif save_method=='h5py':
                try:
                    import h5py
                except ImportError:
                    raise ImportError('h5py is not installed please install it and try again')

                newpath = r'./shots_time'

                if os.path.exists(newpath):
                    shutil.rmtree(newpath)
                
                if not os.path.exists(newpath):
                    os.makedirs(newpath)

                for i in range(len(shots)):
                    fshot =newpath+'/shot_'+str(i+1)+'.hdf5'
                    f = h5py.File(fshot,"w")
                    dts = f.create_dataset(fshot,shots[i].receivers.data.shape,shots[i].receivers.data.dtype)
                    dts[...] = shots[i].receivers.data
                    f.close()
                    fts =newpath+'/ts_'+str(i+1)+'.hdf5'
                    f = h5py.File(fts,'w')
                    dts = f.create_dataset(fts,shots[i].receivers.ts.shape,shots[i].receivers.ts.dtype)
                    dts[...] = shots[i].receivers.ts
                    f.close()
            else:
                raise TypeError('Unknown save_method')
    if solver.supports['equation_dynamics'] == "frequency":
        # store shots in memory for a next simulation
        if save_method is not None:
            if save_method=='pickle':
                try:
                    import pickle
                except ImportError:
                    raise ImportError('cPickle is not installed please install it and try again')

                newpath = r'./shots_frequency'
                if os.path.exists(newpath):
                    shutil.rmtree(newpath)

                if not os.path.exists(newpath):
                    os.makedirs(newpath)

                for i in range(len(shots)):
                    fshot =newpath+'/shot_'+str(i+1)+'.save'
                    f = file(fshot, 'wb')
                    pickle.dump(shots[i].receivers.data_dft,f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.close()
            elif save_method=='savemat':
                try:
                    import scipy.io as io
                except ImportError:
                    raise ImportError('scipy.io is not installed please install it and try again')

                newpath = r'./shots_frequency'
                if os.path.exists(newpath):
                    shutil.rmtree(newpath)

                if not os.path.exists(newpath):
                    os.makedirs(newpath)

                for i in range(len(shots)):
                    data_frequency = shots[i].receivers.data_dft
                    for nu in data_frequency:
                        fshot =newpath+'/shot_'+str(i+1)+'_nu_'+str(nu)+'.mat'
                        io.savemat(fshot,mdict={fshot:shots[i].receivers.data_dft[nu]})
            elif save_method=='h5py':
                raise NotImplementedError('h5py is not efficient with frequency data please use another data container')
            else:
                raise TypeError('Unknown save_method')
    if verbose:
        save_tt = time.time() - tt
        print('Data saving: {0}s'.format(save_tt))
        print('Data saving: {0}s/shot'.format(save_tt/len(shots)))

def generate_seismic_data_from_file(shots, solver, verbose=False, save_method=None, **kwargs):
    if verbose:
        print('Loading data...')
        tt = time.time()

    if save_method is not None:
        if solver.supports['equation_dynamics'] == "time":
            if save_method=='pickle':
                try:
                    import pickle
                except ImportError:
                    raise ImportError('cPickle is not installed please install it and try again')

                path = r'./shots_time'
                if os.path.exists(path):
                    try:
                        for i in range(len(shots)):                
                            filename =path+'/ts_'+str(i+1)+'.save'
                            f = file(filename, 'rb')
                            ts = pickle.load(f)
                            f.close()
                            shots[i].receivers.clear_data(len(ts))
                            shots[i].receivers.ts = ts
                            filename =path+'/shot_'+str(i+1)+'.save'
                            f = file(filename, 'rb')
                            shots[i].receivers.data = pickle.load(f)
                            f.close()
                            shots[i].receivers.interpolator = interp1d(ts, np.zeros_like(shots[i].receivers.data),
                                                                       axis=0, kind='linear', copy=False, bounds_error=False,
                                                                       fill_value=0.0)
                    except IOError:
                        raise IOError('Files are corrupted or generated with different save_method argument')
                else:
                    raise ImportError('There is no shot directory, please relaunch generate_seismic_data with save_method argument')
            
            elif save_method=='savemat':
                try:
                    import scipy.io as io
                except ImportError:
                    raise ImportError('scipy.io is not installed please install it and try again')

                path = r'./shots_time'
                if os.path.exists(path):
                    try:
                        for i in range(len(shots)):                
                            filename = path+'/ts_'+str(i+1)+'.mat'
                            ts = io.loadmat(filename)
                            ts = ts[filename]
                            ts = ts.flatten()
                            shots[i].receivers.clear_data(len(ts))
                            shots[i].receivers.ts = ts
                            filename = path+'/shot_'+str(i+1)+'.mat'
                            data = io.loadmat(filename)
                            data = data[filename]
                            shots[i].receivers.data = data
                            shots[i].receivers.interpolator = interp1d(ts, np.zeros_like(shots[i].receivers.data),
                                                                       axis=0, kind='linear', copy=False, bounds_error=False,
                                                                       fill_value=0.0)              
                    except IOError:
                        raise IOError('Files are corrupted or generated with different save_method argument')
                else:
                    raise ImportError('There is no shot directory, please relaunch generate_seismic_data with save_method argument')

            elif save_method=='h5py':
                try:
                    import h5py
                except ImportError:
                    raise ImportError('h5py is not installed please install it and try again')

                path = r'./shots_time'
                if os.path.exists(path):
                    try:
                        for i in range(len(shots)):
                            filename =path+'/ts_'+str(i+1)+'.hdf5'
                            f = h5py.File(filename, 'r')
                            dts = f[filename]
                            ts = dts[...]              
                            f.close()
                            shots[i].receivers.clear_data(len(ts))
                            shots[i].receivers.ts = ts
                            filename = path+'/shot_'+str(i+1)+'.hdf5'
                            f = h5py.File(filename, 'r')
                            dts = f[filename]
                            shots[i].receivers.data = dts[...]
                            f.close()
                            shots[i].receivers.interpolator = interp1d(ts, np.zeros_like(shots[i].receivers.data),
                                                                       axis=0, kind='linear', copy=False, bounds_error=False,
                                                                       fill_value=0.0)              
                    except IOError:
                        raise IOError('Files are corrupted or generated with different save_method argument')
                else:
                    raise ImportError('There is no shot directory, please relaunch generate_seismic_data with save_method argument')        
            else:
                raise TypeError('Unknown save_method')
        elif solver.supports['equation_dynamics'] == "frequency":
            if save_method=='pickle':
                try:
                    import pickle
                except ImportError:
                    raise ImportError('cPickle is not installed please install it and try again')

                path = r'./shots_frequency'
                if os.path.exists(path):
                    try:
                        for i in range(len(shots)):                
                            filename =path+'/shot_'+str(i+1)+'.save'
                            f = file(filename, 'rb')
                            shots[i].receivers.data_dft = pickle.load(f)
                            f.close()

                    except IOError:
                        raise IOError('Files are corrupted or generated with different save_method argument')
                else:
                    raise ImportError('There is no shot directory, please relaunch generate_seismic_data with save_method argument')
            
            elif save_method=='savemat':
                try:
                    import scipy.io as io
                except ImportError:
                    raise ImportError('scipy.io is not installed please install it and try again')

                path = r'./shots_frequency'
                if os.path.exists(path):
                    try:
                        if 'frequencies' in kwargs :
                            data_frequency = kwargs['frequencies']
                            for i in range(len(shots)):
                                shots[i].receivers.data_dft = dict()
                                for nu in data_frequency:
                                    fshot =path+'/shot_'+str(i+1)+'_nu_'+str(nu)+'.mat'
                                    data = io.loadmat(fshot)
                                    data = data[fshot]
                                    shots[i].receivers.data_dft[nu] = data
                        else :
                            raise AttributeError("""No frequencies specified in generate_seismic_data_from_file arguments""")
                    except IOError:
                        raise IOError('Files are corrupted or generated with different save_method argument')
                else:
                    raise ImportError('There is no shot directory, please relaunch generate_seismic_data with save_method argument')

            elif save_method=='h5py':
                raise NotImplementedError('h5py is not efficient with frequency data please use another data container')
        else:
            raise AttributeError('No solver specified cannot load data')
    else:
        raise AttributeError('No save_method specified cannot load data')

    if verbose:
        load_tt = time.time() - tt
        print('Data Loading: {0}s'.format(load_tt))
        print('Data Loading: {0}s/shot'.format(load_tt/len(shots)))




def generate_shot_data_time(shot, solver, model, wavefields=None, wavefields_padded=None, verbose=False, **kwargs):
    """Given a shots and a solver, generates seismic data at the specified
    receivers.

    Parameters
    ----------
    shots : list of pysit.Shot
        Collection of shots to be processed
    solver : pysit.WaveSolver
        Instance of wave solver to be used.
    model_parameters : solver.ModelParameters, optional
        Wave equation parameters used for generating data.
    wavefields : list, optional
        List of wave states.
    verbose : boolean
        Verbosity flag.

    Notes
    -----
    An empty list passed as `wavefields` will be populated with the state of the wave
    field at each time index.

    """

    solver.model_parameters = model

    # Ensure that the receiver data is empty.  And that interpolator is setup.
    # shot.clear_data(solver.nsteps)
    ts = solver.ts()
    shot.reset_time_series(ts)

    # Populate some stuff that will probably not exist soon anyway.
    shot.dt = solver.dt
    shot.trange = solver.trange

    if solver.supports['equation_dynamics'] != "time":
        raise TypeError('Solver must be a time solver to generate data.')

    if(wavefields is not None):
        wavefields[:] = []
    if(wavefields_padded is not None):
        wavefields_padded[:] = []

    #Frequently used local references are faster than remote references
    mesh = solver.mesh
    dt = solver.dt
    source = shot.sources

    # Step k = 0
    # p_0 is a zero array because if we assume the input signal is causal and we
    # assume that the initial system (i.e., p_(-2) and p_(-1)) is uniformly
    # zero, then the leapfrog scheme would compute that p_0 = 0 as well.
    # This is stored in this solver specific data structure.
    solver_data = solver.SolverData()

    rhs_k   = np.zeros(mesh.shape(include_bc=True))
    rhs_kp1 = np.zeros(mesh.shape(include_bc=True))

    # k is the t index.  t = k*dt.
    for k in range(solver.nsteps):
#       print "  Computing step {0}...".format(k)
        uk = solver_data.k.primary_wavefield

        # Extract the primary, non ghost or boundary nodes from the wavefield
        # uk_bulk is a view so no new storage
        # also, this is a bad way to do things.  Somehow the BC padding needs to be better hidden.  Maybe it needs to always be a part of the mesh?
        uk_bulk = mesh.unpad_array(uk)

        # Record the data at t_k
        shot.receivers.sample_data_from_array(uk_bulk, k, **kwargs)

        if(wavefields is not None):
            wavefields.append(uk_bulk.copy())
        if(wavefields_padded is not None):
            wavefields_padded.append(uk.copy())

        # When k is the nth step, the next time step is not needed, so save
        # computation and break out early.
        if(k == (solver.nsteps-1)): break

        if k == 0:
            rhs_k = mesh.pad_array(source.f(k*dt), out_array=rhs_k)
            rhs_kp1 = mesh.pad_array(source.f((k+1)*dt), out_array=rhs_kp1)
        else:
            # shift time forward
            rhs_k, rhs_kp1 = rhs_kp1, rhs_k
            rhs_kp1 = mesh.pad_array(source.f((k+1)*dt), out_array=rhs_kp1)

        # Given the state at k and k-1, compute the state at k+1
        solver.time_step(solver_data, rhs_k, rhs_kp1)

        # Don't know what data is needed for the solver, so the solver data
        # handles advancing everything forward by one time step.
        # k-1 <-- k, k <-- k+1, etc
        solver_data.advance()

def generate_shot_data_frequency(shot, solver, model, frequencies, verbose=False, **kwargs):
        """most of this is copied from frequency_modeling.forward_model

        Parameters
        ----------
        shot : pysit.Shot
            Gives the source signal approximation for the right hand side.
        frequencies : list of 2-tuples
            2-tuple, first element is the frequency to use, second element the weight.
        return_parameters : list of {'wavefield', 'simdata', 'simdata_time', 'dWaveOp'}

        Returns
        -------
        retval : dict
            Dictionary whose keys are return_parameters that contains the specified data.

        Notes
        -----
        * u is used as the target field universally.  It could be velocity potential, it could be displacement, it could be pressure.
        * uhat is used to generically refer to the DFT of u that is needed to compute the imaging condition.

        """

        # Local references
        solver.model_parameters = model

        mesh = solver.mesh

        source = shot.sources

        # Sanitize the input
        if not np.iterable(frequencies):
            frequencies = [frequencies]

        solver_data = solver.SolverData()
        rhs = solver.WavefieldVector(mesh,dtype=solver.dtype)
        for nu in frequencies:
            rhs = solver.build_rhs(mesh.pad_array(source.f(nu=nu)), rhs_wavefieldvector=rhs)
            solver.solve(solver_data, rhs, nu)
            uhat = solver_data.k.primary_wavefield

            # Record the data at frequency nu
            shot.receivers.sample_data_from_array(mesh.unpad_array(uhat), nu=nu)

def generate_shot_data_frequency_list(shots, solver, model, frequencies, verbose=False, **kwargs):
        """most of this is copied from frequency_modeling.forward_model

        Parameters
        ----------
        shots : list of pysit.Shot
            Gives the source signal approximation for the right hand side.
        frequencies : list of 2-tuples
            2-tuple, first element is the frequency to use, second element the weight.
        return_parameters : list of {'wavefield', 'simdata', 'simdata_time', 'dWaveOp'}

        Returns
        -------
        retval : dict
            Dictionary whose keys are return_parameters that contains the specified data.

        Notes
        -----
        * u is used as the target field universally.  It could be velocity potential, it could be displacement, it could be pressure.
        * uhat is used to generically refer to the DFT of u that is needed to compute the imaging condition.

        """
        #number of shot varaible
        nshot = len(shots)
        # Local references
        solver.model_parameters = model

        mesh = solver.mesh

        # Sanitize the input
        if not np.iterable(frequencies):
            frequencies = [frequencies]

        rhs = solver.WavefieldVector(mesh,dtype=solver.dtype)
        # decalare a list for the Right Hand Side
        RHS = list()

        for nu in frequencies:
            del RHS[:]
            for k in range(nshot):
                shot = shots[k]
                source = shot.sources
                rhs = solver.build_rhs(mesh.pad_array(source.f(nu=nu)), rhs_wavefieldvector=rhs)
                RHS.append(rhs.data.copy())
                
            Uhat = solver.solve_petsc_uhat(solver, RHS, nu, **kwargs)
            for k in range(nshot):
                    shot = shots[k]
                    uhat = Uhat[:,k]
                    # Giving the good dimension to the array
                    uhat = np.expand_dims(uhat, axis=1)
                    # Record the data at frequency nu
                    shot.receivers.sample_data_from_array(mesh.unpad_array(uhat), nu=nu)
        