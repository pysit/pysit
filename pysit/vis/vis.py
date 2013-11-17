import itertools
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation

__all__ = ['animate', 'plot']

def animate(data, mesh, display_rate=30,pause=1, scale=None, show=True, **kwargs):
				
	if mesh.dim == 1:
		
		fig = plt.figure()
		plt.clf()
		
		line = plot(data[0], mesh, **kwargs)
		title = plt.title('{0:4}/{1}'.format(0,len(data)-1))
		
		if scale == 'fixed':
			pltscale = 1.1*min([d.min() for d in data]), 1.1*max([d.max() for d in data])
		else:
			pltscale = scale
		
		def _animate(i, line, data, mesh, title, pltscale):
				
			line = plot(data[i], mesh, update=line, **kwargs)
			
			if pltscale is None:
				plt.gca().set_ylim(data[i].min(),data[i].max())
			else:
				plt.gca().set_ylim(*pltscale)
			
			title.set_text('{0:4}/{1}'.format(i,len(data)-1))
			plt.draw()   
				
#			return line, 
		
		_animate_args = (line, data, mesh, title, pltscale)
	
	if mesh.dim == 2:

		fig = plt.figure()
		plt.clf()
		
		im = plot(data[0], mesh, **kwargs)
		cbar = plt.colorbar()
		title = plt.title('{0:4}/{1}'.format(0,len(data)-1))
	
		if scale == 'fixed':
			pltclim = 1.1*min([d.min() for d in data]), 1.1*max([d.max() for d in data])
		else:
			pltclim = None
			
		def _animate(i, im, data, mesh, title, cbar, pltclim):
			
			im = plot(data[i], mesh, update=im, **kwargs)
				
			if pltclim is None:
				clim = (data[i].min(),data[i].max())
				im.set_clim(clim)
				cbar.set_clim(clim)
			else:
				im.set_clim(pltclim)
				cbar.set_clim(pltclim)
			
			title.set_text('{0:4}/{1}'.format(i,len(data)-1))
			plt.draw()   
				
#			return im, 
		
		_animate_args = (im, data, mesh, title, cbar, pltclim)
		
	if mesh.dim == 3:

		fig = plt.figure()
		plt.clf()
		
		ims = plot(data[0], mesh, **kwargs)
		cax=plt.subplot(2,2,4)
		cbar = plt.colorbar(cax=cax)
		title = plt.title('{0:4}/{1}'.format(0,len(data)-1))
		
		if scale == 'fixed':
			pltclim = 1.1*min([d.min() for d in data]), 1.1*max([d.max() for d in data])
		else:
			pltclim = None
			
		def _animate(i, ims, data, mesh, title, cbar, pltclim):
			
			ims = plot(data[i], mesh, update=ims, **kwargs)
				
			if pltclim is None:
				clim = (data[i].min(),data[i].max())
				for im in ims:
					im.set_clim(clim)
				cbar.set_clim(clim)
			else:
				for im in ims:
					im.set_clim(pltclim)
				cbar.set_clim(pltclim)
			
#			plt.sca(cax)
			title.set_text('{0:4}/{1}'.format(i,len(data)-1))
			plt.draw()   
				
#			return ims, 
		
		_animate_args = (ims, data, mesh, title, cbar, pltclim)
					

	anim = animation.FuncAnimation(fig, _animate, fargs=_animate_args, #init_func=init,
			interval=pause, frames=xrange(0,len(data),display_rate), blit=False, repeat=False)
	
	if show:
		plt.show()
	
	return anim
	
		

def plot(data, mesh, shade_pml=False, axis=None, ticks=True, update=None, slice3d=(0,0,0), **kwargs):
	
	""" Assumes that data has no ghost padding."""
	
	data.shape = -1,1
	
	sh_bc = mesh.shape(include_bc=True)
	sh_primary = mesh.shape()
	
	if data.shape == sh_bc:
		has_bc = True
		plot_shape = mesh.shape(include_bc=True, as_grid=True)
	elif data.shape == sh_primary:
		has_bc = False
		plot_shape = mesh.shape(as_grid=True)
	else:
		raise ValueError('Shape mismatch between domain and data.')
	
	if axis is None:
		ax = plt.gca()
	else:
		ax = axis
		
	data = data.reshape(plot_shape)
	
	if mesh.dim == 1:			
		if update is None:
			zs, = mesh.mesh_coords()
			ret, = ax.plot(zs,data)
			
			if has_bc:
				ax.axvline(mesh.domain.parameters['z']['lbound'], color='r')
				ax.axvline(mesh.domain.parameters['z']['rbound'], color='r')
		else:
			update.set_ydata(data)
			ret = update
			
		
	if mesh.dim == 2:
		data = data.T
		
		if update is None:
			im = ax.imshow(data, interpolation='nearest', aspect='auto', **kwargs)
	
			if ticks:
				mesh_tickers(mesh)
			else:
				ax.xaxis.set_ticks([])
				ax.yaxis.set_ticks([])
			
			if has_bc:
				draw_pml(mesh, 'x', 'LR', shade_pml=shade_pml)
				draw_pml(mesh, 'z', 'TB', shade_pml=shade_pml)
		
		else:
			update.set_data(data)
			im = update
	
		# Update current image for the colorbar
		plt.sci(im)
	
		ret = im
	
	if mesh.dim == 3:
		
		if update is None:
			
			# X-Y plot
			ax = plt.subplot(2,2,1)
			imslice = int(slice3d[2]) # z slice
			pltdata = data[:,:,imslice:(imslice+1)].squeeze()
			imxy = ax.imshow(pltdata, interpolation='nearest', aspect='equal', **kwargs)
			
			if ticks:
				mesh_tickers(mesh, ('y', 'x'))
			else:
				ax.xaxis.set_ticks([])
				ax.yaxis.set_ticks([])
			if has_bc:
				draw_pml(mesh, 'y', 'LR', shade_pml=shade_pml)
				draw_pml(mesh, 'x', 'TB', shade_pml=shade_pml)
			
			# X-Z plot
			ax = plt.subplot(2,2,2)
			imslice = int(slice3d[1]) # y slice
			pltdata = data[:,imslice:(imslice+1),:].squeeze().T
			imxz = ax.imshow(pltdata, interpolation='nearest', aspect='equal', **kwargs)
			
			if ticks:
				mesh_tickers(mesh, ('x', 'z'))
			else:
				ax.xaxis.set_ticks([])
				ax.yaxis.set_ticks([])
			if has_bc:
				draw_pml(mesh, 'x', 'LR', shade_pml=shade_pml)
				draw_pml(mesh, 'z', 'TB', shade_pml=shade_pml)
			
			# Y-Z plot
			ax = plt.subplot(2,2,3)
			imslice = int(slice3d[0]) # x slice
			pltdata = data[imslice:(imslice+1),:,:].squeeze().T
			imyz = ax.imshow(pltdata, interpolation='nearest', aspect='equal', **kwargs)
			
			if ticks:
				mesh_tickers(mesh, ('y', 'z'))
			else:
				ax.xaxis.set_ticks([])
				ax.yaxis.set_ticks([])
			if has_bc:
				draw_pml(mesh, 'y', 'LR', shade_pml=shade_pml)
				draw_pml(mesh, 'z', 'TB', shade_pml=shade_pml)
				
			update = [imxy, imxz, imyz]
			plt.sci(imyz)
		else:
			imslice = int(slice3d[2]) # z slice
			pltdata = data[:,:,imslice:(imslice+1)].squeeze()
			update[0].set_data(pltdata)
			
			imslice = int(slice3d[1]) # y slice
			pltdata = data[:,imslice:(imslice+1),:].squeeze().T
			update[1].set_data(pltdata)
			
			imslice = int(slice3d[0]) # x slice
			pltdata = data[imslice:(imslice+1),:,:].squeeze().T
			update[2].set_data(pltdata)
			
		
		ret = update
	
	return ret

def mesh_tickers(mesh, dims=('x','z')):
	ax = plt.gca()
	
	xformatter = mpl.ticker.FuncFormatter(MeshFormatterHelper(mesh, dims[0]))
	ax.xaxis.set_major_formatter(xformatter)
	
	zformatter = mpl.ticker.FuncFormatter(MeshFormatterHelper(mesh, dims[1]))
	ax.yaxis.set_major_formatter(zformatter)
	
def draw_pml(mesh, axis, orientation='TB', shade_pml=False):
	
	ax = plt.gca()
	
	dim = mesh.parameters[axis]
	
	S = 0 #start left
	L = 0 #end left
	if dim.lbc.type is 'pml':
		L = dim.lbc.n #end left
	
	R = dim.n #start right
	N = dim.n #end right
	if dim.rbc.type is 'pml':
		N += dim.rbc.n #end right
			
	if orientation == 'LR':
		if shade_pml:
			ax.axvspan(S,L, hatch='\\', fill=False, color='r')
			ax.axvspan(R,N, hatch='\\', fill=False, color='r')
			
		else:
			ax.axvline(L, color='r')
			ax.axvline(R, color='r')
		
		plt.xlim(0,dim['n']-1)
		
	if orientation == 'TB':
		if shade_pml:
			ax.axhspan(S,L, hatch='/', fill=False, color='r')
			ax.axhspan(R,N, hatch='/', fill=False, color='r')
		else:
			ax.axhline(L, color='r')
			ax.axhline(R, color='r')
			
		plt.ylim(dim['n']-1,0)
	
class MeshFormatterHelper(object):
	def __init__(self, mesh, axis):
		self.lbound = mesh.domain.parameters[axis]['lbound']
		self.delta  = mesh.parameters[axis]['delta']
	def __call__(self, grid_point, pos):
		return '{0:.3}'.format(self.lbound + self.delta*grid_point)

if __name__ == '__main__':

	from pysit import Domain, PML

	pmlx = PML(0.1, 100)
	pmlz = PML(0.1, 100)
	
	x_config = (0.1, 1.0, 90, pmlx, pmlx)
	z_config = (0.1, 0.8, 70, pmlz, pmlz)

	d = Domain( (x_config, z_config) )

	#	Generate true wave speed
	#   (M = C^-2 - C0^-2)
	M, C0, C = horizontal_reflector(d)
				
	XX, ZZ = d.generate_grid()
	
	display_on_grid(XX, domain=d)
	display_on_grid(XX, domain=d, shade_pml=True)

