try:
	from mpi4py import MPI
	hasmpi = True
except:
	hasmpi = False

__all__ = ['hasmpi', 'ParallelWrapShotNull', 'ParallelWrapShot']
	
class ParallelWrapShotBase(object):
		
	def __init__(self, *args, **kwargs):
		raise NotImplementedError('ParallelWrapShotBase.__init__ should never be called.')
	
class ParallelWrapShotNull(ParallelWrapShotBase):
		
	def __init__(self, *args, **kwargs):
		self.comm = None
		self.use_parallel = False
		
		self.size = 1
		self.rank = 0

class ParallelWrapShot(ParallelWrapShotBase):
	
	def __new__(cls, *args, **kwargs):
		
		if not hasmpi:
			return ParallelWrapShotNull(*args, **kwargs)
		
		if MPI.COMM_WORLD.Get_size() <= 1:
			return ParallelWrapShotNull(*args, **kwargs)
		
		return super(ParallelWrapShot, cls).__new__(cls, *args, **kwargs)
		
	def __init__(self, comm=None, *args, **kwargs):
		if comm is None:
			self.comm = MPI.COMM_WORLD
		else:
			self.comm = comm
			
		self.use_parallel = True
		self.size = self.comm.Get_size()
		self.rank = self.comm.Get_rank()
			
	
	
if __name__ == '__main__':
	
#	x = ParallelWrapShotBase()
	y = ParallelWrapShotNull()
	z = ParallelWrapShot('foo')
