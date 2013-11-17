version = '0.4'
release = False

if not release:
	import os
	version+= 'dev'
#	hg_version_file = os.path.join(os.path.dirname(__file__), '__hg_version__.py')
#
#	if os.path.isfile(hg_version_file):
#		import imp
#		hg = imp.load_module('pysit.__hg_version__', open(hg_version_file), hg_version_file, ('.py','U',1))
#		version += hg.version

