import numpy as np

import obspy.segy.core as segy

__all__ = ['read_model']

def read_model(fname):
	""" Reads a model in segy format and returns it as an array."""	
	
	data = segy.readSEGY(fname)
	
	return np.array([tr.data for tr in data.traces])