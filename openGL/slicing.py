import numpy as np




class MatrixBlockSlicer :
	''' a class to create blocks in the matrix and slice it 
		arguments: the matrix to slice
	'''
	def __init__(self, N, initial_block=2, rnd=0.4):
		self.N = N
		self.h_sections = []
		self.v_sections = []
		self.min_width = 2
		self.initial = self.N/initial_block
		self.rnd = rnd

	def create_h_sections(self):
		self.h_sections.append(0)
		prev_pointer = np.random.randint(self.min_width, self.initial) #initialization
		self.h_sections.append(prev_pointer)
		while prev_pointer < self.N and not prev_pointer+2 >= self.N :
			width = np.random.randint(min(prev_pointer+2,self.N+1), high=self.N+1)
			prev_pointer = width
			self.h_sections.append(prev_pointer)

	def create_v_sections(self) :
		'''create vertical sections by dividing the matrix in blocks of length equal to the number of horizontal blocks
			then move the section borders, using a method that depends on whether the number of blocks divides N exactly or not
		'''
		h_blocks = len(self.h_sections)
		v_blocks,remainder = divmod(self.N, h_blocks)
		self.v_sections = range(0,self.N,v_blocks)
		if remainder == 0 :
			for i, item in enumerate(self.v_sections[1:]) :
				self.v_sections[i+1] = item + 1 if i % 2 == 0 else item - 1
		else :
			for i in range(1, remainder):
				#randomly choose a block and add one column to the right
				rnd = np.random.randint(1,len(self.v_sections)-1)
				self.v_sections[rnd] = min(self.v_sections[rnd] + 1, self.N)

	def slice_matrix(self, matrix) :
		self.create_h_sections()
		self.create_v_sections()
		for i,width in enumerate(self.h_sections[1:-1]) :
			if np.random.rand() < self.rnd :
				matrix[width:self.h_sections[i+2], self.v_sections[i+1]:self.v_sections[i+2]] = 1

