import urllib2
import shutil
import os
import os.path

__all__ = ['download_file']

def download_file(url, destination):
		
	request = urllib2.urlopen(url)
	
	with open(destination, 'wb') as outfile:
		shutil.copyfileobj(request, outfile)
