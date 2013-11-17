# Derived from sunpy/util/config/config.py which is licensed under a BSD license
import os
import os.path

import ConfigParser

import pysit

__all__ = ['load_configuration', 'get_user_pysit_path', 'get_gallery_data_path']

def load_configuration():
	
	cfg_files  = list()
	
	# The default configuration
	default_config = os.path.join(os.path.dirname(pysit.__file__), 'pysit.cfg')
	cfg_files.append(default_config)
	
	# Check for an override in ~/.pysit/pysit.cfg
	user_pysit_path = get_user_pysit_path()
	user_config = os.path.join(user_pysit_path, 'pysit.cfg')
	if os.path.isfile(user_config):
		cfg_files.append(user_config)
		
	# Load the configuration
	config = ConfigParser.ConfigParser()
	config.read(cfg_files)
	
	# Ensure that there is a gallery section with a data dir
	if not config.has_option('pysit.gallery', 'gallery_data_path'):
		print config.get('pysit.gallery', 'gallery_data_path')
		raise Exception('A proper gallery data path should have been specified in the default pysit.cfg.')
		
	return config

def get_user_pysit_path():
	""" Returns the full path to the users .pysit directory and creates it if
	    it does not exist."""
	path = os.path.join(os.path.expanduser('~'), '.pysit')
	if not os.path.isdir(path):
		os.mkdir(path)
		
	return path


def get_gallery_data_path():
	path = os.path.expanduser(pysit.config.get('pysit.gallery', 'gallery_data_path'))
	if not os.path.exists(path):
		os.mkdir(path)
	return path
