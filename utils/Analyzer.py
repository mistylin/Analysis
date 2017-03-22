

class Analyzer(object):

	def __init__(self, subID, filename, verbosity = 0, **kwargs):

		#self.default_parameters = {}

		self.data_file = filename
		self.verbosity = verbosity
		self.subID = subID

		for k,v in kwargs.items():
			if self.verbosity > 0:
				print '[%s] Setting new value for parameter %s' % (self.__class__.__name__, k)
			setattr(self, k, v)

		# Set default values for any parameter that was not passed in
		self.set_defaults()			

		self.set_output_filename()

	def set_defaults(self):
		"""
		Simple mechanism to set default values 
		for required parameters (only works for numeric right now)
		"""			

		for k,v in self.default_parameters.items():

			if not hasattr(self, k):
				setattr(self, k, v)
				if self.verbosity > 0:
					print '[%s] Setting default value for parameter %s' % (self.__class__.__name__, k)
				

	def set_output_filename(self):
		self.output_filename = self.subID + '_databin.p'

	def load_data(self):
		"""
		placeholder for load_data
		needs to be implemented by subclass
		"""		
		pass

	def plot_result(self):
		"""
		placeholder for plot_result
		needs to be implemented by subclass
		"""				
		pass

	def store(self):
		pass