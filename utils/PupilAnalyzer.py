from __future__ import division

import os,glob,datetime

import numpy as np
import scipy as sp
import seaborn as sn
import matplotlib.pylab as pl
import cPickle as pickle
import pandas as pd
import tables

from scipy.signal import fftconvolve, resample

from math import *

from hedfpy.EDFOperator import EDFOperator
from hedfpy.HDFEyeOperator import HDFEyeOperator
from hedfpy.EyeSignalOperator import EyeSignalOperator

from fir import FIRDeconvolution

from IPython import embed

from Analyzer import Analyzer

class PupilAnalyzer(Analyzer):

	def __init__(self, subID, filename, edf_folder, sort_by_date = False, reference_phase = 7,verbosity = 0, **kwargs):

		# Setup default parameter values
		self.default_parameters = {'low_pass_pupil_f': 6.0,
								   'high_pass_pupil_f': 0.01,
								   'signal_sample_frequency': 1000,
								   'deconv_sample_frequency': 10}

		super(PupilAnalyzer, self).__init__(subID, filename, verbosity=verbosity, **kwargs)

		self.edf_folder = edf_folder
		self.data_folder = edf_folder
		self.sort_by_date = sort_by_date
		self.reference_phase = reference_phase
		self.fir_signal = {}

		# Initialize variables
		self.combined_h5_filename = os.path.join(self.edf_folder,self.subID + '-combined.h5')
		self.combined_data  = None
		self.h5_operator 	= None
		self.FIR_object 	= None
		self.pupil_signal 	= None
		self.events 		= {}
		self.pupil_data     = None

	def load_data(self):

		if self.h5_operator is None:
			self.get_h5_operator()

		if not hasattr(self.h5_operator, 'h5f'):
			self.h5_operator.open_hdf_file()

	def unload_data(self):

		if not (self.h5_operator is None):
			self.h5_operator.close_hdf_file()		

		if not (self.combined_data is None):
			self.combined_data.close()

	def load_combined_data(self, force_rebuild = False):

		# if self.combined_data is None:

		if (not os.path.isfile(self.combined_h5_filename)) or force_rebuild: 
			self.recombine_signal_blocks(force_rebuild = force_rebuild)

			#self.combined_data = tables.open_file(self.combined_h5_filename, mode = 'r')


	def read_pupil_data(self, input_file, signal_type = 'full_signal'):

		signal = []
		with tables.open_file(input_file, mode = 'r') as pfile:
			signal = eval('pfile.root.pupil.%s.read()'%signal_type)
			pfile.close()

		return signal

	def read_trial_data(self, input_file):

		with pd.get_store(input_file) as tfile:
			params = tfile['trials/full/table']

		return params

	def get_h5_operator(self):		

		self.h5_operator = HDFEyeOperator(self.data_file)

		if not os.path.isfile(self.data_file):

			edf_files = glob.glob(self.edf_folder + '/*.edf')
			if self.sort_by_date:
				edf_files.sort(key=lambda x: os.path.getmtime(x))
			else:
				edf_files.sort()

			for ii,efile in enumerate(edf_files):

				alias = '%s%d' % (self.subID, ii)

				# insert the edf file contents only when the h5 is not present.
				self.h5_operator.add_edf_file(efile)
				self.h5_operator.edf_message_data_to_hdf(alias = alias)
				self.h5_operator.edf_gaze_data_to_hdf(alias = alias, pupil_hp = self.high_pass_pupil_f, pupil_lp = self.low_pass_pupil_f)

	def get_aliases(self):

		edf_files = glob.glob(self.edf_folder + '/*.edf')
		edf_files.sort(key=lambda x: os.path.getmtime(x))

		self.aliases = ['%s%d' % (self.subID, i) for i in range(0, len(edf_files))]			


	def recode_trial_code(self, params, last_node = False):

		#
		# mapping:
		# 0 = expected, color
		# 1 = expected, orientation
		# 
		#     ATT, 		UNATT
		# 10 = pred, 	unpred (color)
		# 30 = unpred, 	pred (color)
		# 50 = unpred, 	unpred (color)
		#
		# 20 = pred, 	unpred (orientation)
		# 40 = unpred, 	pred (orientation)
		# 60 = unpred, 	unpred (orientation)
		#

		# Return array if multiple trials are provided
		if (len(params) > 1) & (~last_node):
			return np.array([self.recode_trial_code(p[1], last_node = True) for p in params.iterrows()], dtype=float)

		new_format = 'trial_cue' in params.keys()

		if not new_format:
			if np.array(params['trial_type'] == 1): # base trial (/expected)
			 	# if np.array(params['task'] == 1):
			 	# 	return 0
			 	# else:
			 	# 	return 1
			 	return np.array(params['task']==2, dtype=int)

			else: # non-base trial (/unexpected)
			 	# if np.array(params['task'] == 1):
			 	# 	return 1
			 	# else:
			 	# 	return 3
				
				if np.array(params['task'] == 1): # color task
					if np.array(params['stimulus_type'] == 0): # red45
						
						if np.array(params['base_color_a'] > 0):
							return 10

						else:
							if np.array(params['base_ori'] == 45):
								return 30
							else:
								return 50
					if np.array(params['stimulus_type'] == 1): # red135
						if np.array(params['base_color_a'] > 0):
							return 10

						else:
							if np.array(params['base_ori'] == 135):
								return 30
							else:
								return 50
					if np.array(params['stimulus_type'] == 2): # green45
						if np.array(params['base_color_a'] < 0):
							return 10

						else:
							if np.array(params['base_ori'] == 45):
								return 30
							else:
								return 50	
					if np.array(params['stimulus_type'] == 3): # green135
						if np.array(params['base_color_a'] < 0):
							return 10

						else:
							if np.array(params['base_ori'] == 135):
								return 30
							else:
								return 50

				else: # orientation task
					if np.array(params['stimulus_type'] == 0): # red45
						if np.array(params['base_ori'] == 45):
							return 20

						else:
							if np.array(params['base_color_a'] > 0):
								return 40
							else:
								return 60
					if np.array(params['stimulus_type'] == 1): # red135
						if np.array(params['base_ori'] == 135):
							return 20

						else:
							if np.array(params['base_color_a'] > 0):
								return 40
							else:
								return 60
					if np.array(params['stimulus_type'] == 2): # green45
						if np.array(params['base_ori'] == 45):
							return 20

						else:
							if np.array(params['base_color_a'] < 0):
								return 40
							else:
								return 60
					if np.array(params['stimulus_type'] == 3): # green135
						if np.array(params['base_ori'] == 135):
							return 20

						else:
							if np.array(params['base_color_a'] < 0):
								return 40
							else:
								return 60				
		else:

			stimulus_types = {'red45': 0,
							  'red135': 1,
							  'green45': 2,
							  'green135': 3}

			if params['trial_cue']==params['trial_stimulus_label']:
				return np.array(params['task']==2, dtype=int)

			else:
				if np.array(params['task'] == 1): # color task
					if np.array(stimulus_types[params['trial_cue']] == 0): # red45
						
						if np.array(params['base_color_a'] > 0):
							return 10

						else:
							if np.array(params['base_ori'] == 45):
								return 30
							else:
								return 50
					if np.array(stimulus_types[params['trial_cue']] == 1): # red135
						if np.array(params['base_color_a'] > 0):
							return 10

						else:
							if np.array(params['base_ori'] == 135):
								return 30
							else:
								return 50
					if np.array(stimulus_types[params['trial_cue']] == 2): # green45
						if np.array(params['base_color_a'] < 0):
							return 10

						else:
							if np.array(params['base_ori'] == 45):
								return 30
							else:
								return 50	
					if np.array(stimulus_types[params['trial_cue']] == 3): # green135
						if np.array(params['base_color_a'] < 0):
							return 10

						else:
							if np.array(params['base_ori'] == 135):
								return 30
							else:
								return 50

				else: # orientation task
					if np.array(stimulus_types[params['trial_cue']] == 0): # red45
						if np.array(params['base_ori'] == 45):
							return 20

						else:
							if np.array(params['base_color_a'] > 0):
								return 40
							else:
								return 60
					if np.array(stimulus_types[params['trial_cue']] == 1): # red135
						if np.array(params['base_ori'] == 135):
							return 20

						else:
							if np.array(params['base_color_a'] > 0):
								return 40
							else:
								return 60
					if np.array(stimulus_types[params['trial_cue']] == 2): # green45
						if np.array(params['base_ori'] == 45):
							return 20

						else:
							if np.array(params['base_color_a'] < 0):
								return 40
							else:
								return 60
					if np.array(stimulus_types[params['trial_cue']] == 3): # green135
						if np.array(params['base_ori'] == 135):
							return 20

						else:
							if np.array(params['base_color_a'] < 0):
								return 40
							else:
								return 60


	def recombine_signal_blocks(self, force_rebuild = False):

		if not hasattr(self, 'signal_downsample_factor'):
			self.signal_downsample_factor = 10

		self.get_aliases()

		self.load_data()

		if self.verbosity:
			print '[%s] Recombining signal and parameters from blocks/runs' % (self.__class__.__name__)

		full_signal = np.array([])

		run_signals = []

		trials = pd.DataFrame()

		run_trials = []

		trial_parameters = {}

		for ii,alias in enumerate(self.aliases):

			this_trial_parameters = self.h5_operator.read_session_data(alias, 'parameters')

			this_trial_phase_times = self.h5_operator.read_session_data(alias, 'trial_phases')
			this_trial_times = self.h5_operator.read_session_data(alias, 'trials')
			
			blocks = self.h5_operator.read_session_data(alias, 'blocks')
			block_start_times = blocks['block_start_timestamp']

			# this_trial_phase_times[this_trial_phase_times['trial_phase_index']==reference_phase]

			block_event_times = pd.DataFrame()

			this_run_signal = np.array([])

			# Kick out incomplete trials
			if len(this_trial_phase_times) < len(this_trial_times):
				this_trial_times = this_trial_times[0:len(this_trial_phase_times)]
				
			this_phase_times = np.array(this_trial_phase_times['trial_phase_EL_timestamp'].values, dtype=float)
			
			prev_signal_size = np.float(full_signal.shape[0])

			for bs,be in zip(blocks['block_start_timestamp'], blocks['block_end_timestamp']):

				this_block_signal = np.squeeze(self.h5_operator.signal_during_period(time_period = (bs, be), alias = alias, signal = 'pupil_bp_clean_zscore', requested_eye = self.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0]))

				full_signal = np.append(full_signal, this_block_signal)
				this_run_signal = np.append(this_run_signal, this_block_signal)
				
				this_phase_times[(this_phase_times >= bs) & (this_phase_times < be)] -= bs

			#this_trial_parameters['trial_codes'] = pd.Series(self.recode_trial_code(this_trial_parameters))
			for phase_index in np.unique(this_trial_phase_times['trial_phase_index']):
				this_trial_parameters['trial_phase_%i_within_run'%phase_index] = pd.Series(this_phase_times[np.array(this_phase_times['trial_phase_index']==phase_index, dtype=bool)])
				this_trial_parameters['trial_phase_%i_full_signal'%phase_index] = pd.Series(this_phase_times[np.array(this_phase_times['trial_phase_index']==phase_index, dtype=bool)] + prev_signal_size)	
			
			if ii > 0:

				# Kick out duplicate trials (do we need to do this actually??)
				if this_trial_parameters.iloc[0]['trial_nr'] == trials.iloc[-1]['trial_nr']:
					this_trial_parameters = this_trial_parameters[1:]

				trials = pd.concat([trials, this_trial_parameters], axis=0, ignore_index = True)
			else:
				trials = this_trial_parameters

			run_trials.append(this_trial_parameters)
			run_signals.append(this_run_signal)
			


		# Store in hdf5 format
		file_mode = 'a'

		if force_rebuild and os.path.isfile(self.combined_h5_filename):
			os.remove(self.combined_h5_filename)

		output_file = tables.open_file(self.combined_h5_filename, mode = file_mode, title = self.subID)

		pgroup = output_file.create_group("/","pupil","pupil")
		tgroup = output_file.create_group("/","trials","trials")

		output_file.create_array(pgroup, "long_signal", full_signal, "long_signal")

		for rii in range(len(run_signals)):
			output_file.create_array(pgroup, "r%i_signal"%rii, run_signals[rii], "r%i_signal"%rii)
		
		output_file.close()

		trials.to_hdf(self.combined_h5_filename, key = '/trials/full', mode = 'a', format = 't', data_columns = True)

		for rii in range(len(run_signals)):
			run_trials[rii].to_hdf(self.combined_h5_filename, key = '/trials/run%i'%rii)




	def signal_per_trial(self, reference_phase = 6, with_rt = False, only_correct = True):

		trial_start_offset = 0

		self.load_combined_data()

		recorded_pupil_signal = self.read_pupil_data(self.combined_h5_filename, signal_type = 'long_signal')
		trial_parameters = self.read_trial_data(self.combined_h5_filename)

		self.trial_signals = []

		if only_correct:
			selected_trials = np.array(trial_parameters['correct_answer']==1, dtype=bool)
		else:
			selected_trials = np.array(np.ones((trial_parameters.shape[0],1)), dtype=bool)			

		if with_rt:
			signal_intervals = trial_parameters['trial_phase_%i_full_signal'%reference_phase][selected_trials].values + (trial_parameters['reaction_time'][selected_trials].values*self.signal_sample_frequency) + ((self.deconvolution_interval-trial_start_offset)*self.signal_sample_frequency)			
		else:
			signal_intervals = trial_parameters['trial_phase_%i_full_signal'%reference_phase][selected_trials].values + ((self.deconvolution_interval-trial_start_offset)*self.signal_sample_frequency)

		for tii,(ts,te) in enumerate(signal_intervals):
			if (ts > 0) & (te < recorded_pupil_signal.size):
				self.trial_signals.append(sp.signal.decimate(recorded_pupil_signal[int(ts):int(te)], self.signal_downsample_factor, 1))


	def get_IRF(self, deconv_interval = None):

		self.load_combined_data()

		recorded_pupil_signal = self.read_pupil_data(self.combined_h5_filename, signal_type = 'long_signal')
		trial_parameters = self.read_trial_data(self.combined_h5_filename)

		events = np.array([trial_parameters['trial_response_phase_full_signal'] + 500,  			    # stimulus onset
				  		   trial_parameters['trial_response_phase_full_signal'] + (500+150+30+150) + self.signal_sample_frequency*trial_parameters['reaction_time']]) # button press

		if deconv_interval is None:
			deconv_interval = self.deconvolution_interval


		print('[%s] Starting FIR deconvolution' % (self.__class__.__name__))

		self.FIRo = FIRDeconvolution(
						signal = recorded_pupil_signal,
						events = events / self.signal_sample_frequency,
						event_names = ['stimulus','button_press'],
						#durations = {'response': self.events['durations']['response']},
						sample_frequency = self.signal_sample_frequency,
			            deconvolution_frequency = self.deconv_sample_frequency,
			        	deconvolution_interval = deconv_interval,
			        	#covariates = self.events['covariates']
					)

		self.FIRo.create_design_matrix()

		print '[%s] Fitting IRF' % (self.__class__.__name__)
		self.FIRo.regress(method = 'lstsq')
		self.FIRo.betas_for_events()
		self.FIRo.calculate_rsq()	

	def build_design_matrix(self, sub_IRF = None):
		
		print('[%s] Creating design matrix for GLM' % (self.__class__.__name__))

		if sub_IRF is None:
			sub_IRF = {'stimulus': [], 'button_press': []}

			self.get_IRF()

			for name,dec in zip(self.FIRo.covariates.keys(), self.FIRo.betas_per_event_type.squeeze()):
				sub_IRF[name] = resample(dec,int(len(dec)*(self.signal_sample_frequency/self.deconv_sample_frequency)))[:,np.newaxis]
				sub_IRF[name] /= max(abs(sub_IRF[name]))


		self.load_combined_data()

		recorded_pupil_signal = self.read_pupil_data(self.combined_h5_filename, signal_type = 'long_signal')
		trial_parameters = self.read_trial_data(self.combined_h5_filename)

		stim_times = trial_parameters['trial_response_phase_full_signal'] + (500+150+30+150)
		resp_times = trial_parameters['trial_response_phase_full_signal'] + self.signal_sample_frequency*trial_parameters['reaction_time']

		# embed()

		self.design_matrix = np.ones((recorded_pupil_signal.size,1))	
		tempX = np.zeros((recorded_pupil_signal.size,1))

		for tcode in np.unique(trial_parameters['trial_codes']):
			for trial in np.where(trial_parameters['trial_codes']==tcode):
			# tempX[stim_times[] = 1


				self.design_matrix = np.hstack([self.design_matrix, fftconvolve(tempX, sub_IRF['stimulus'])[:recorded_pupil_signal.size], fftconvolve(tempX, sub_IRF['stimulus'])[:recorded_pupil_signal.size]])


	def run_GLM(self):

		print('[%s] Running GLM analysis' % (self.__class__.__name__))

		if not hasattr(self, 'design_matrix'):
			self.build_design_matrix()

		self.load_combined_data()

		recorded_pupil_signal = self.read_pupil_data(self.combined_h5_filename, signal_type = 'long_signal')
		trial_parameters = self.read_trial_data(self.combined_h5_filename)

		trial_betas = np.linalg.pinv(self.design_matrix).dot(recorded_pupil_signal)

		embed()



	def store_pupil(self):
		# Simply store the relevant variables to save speed
		
		fieldnames = ['task_data','events','trial_signals','fir_signal','pupil_data']

		print '[%s] Storing pupil data' % (self.__class__.__name__)

		output = []

		for fname in fieldnames:
			if hasattr(self, fname):
				print fname
				eval('output.append(self.'+fname+')')

		pickle.dump(output,open(os.path.join(self.data_folder, 'pupil_' + self.output_filename),'wb'))





	########################################
	###  								 ###					
	###    ARCHIVE OF UNWANTED FUNCIONS  ###
	###									 ###
	########################################



	# def extract_signal_blocks(self):

	# 	self.get_aliases()

	# 	self.load_data()

	# 	pupil_signal = []

	# 	down_fs = 100

	# 	winlength = 4500
	# 	minwinlength = 4000

	# 	events = []

	# 	response_events = []
	# 	cue_events = []
	# 	task_events = []
	# 	stim1_events = []
	# 	stim2_events = []

	# 	trial_signals = []

	# 	trial_tasks = []
	# 	trial_color = []
	# 	trial_orientation = []
	# 	trial_correct = []
	# 	trial_codes = []
	# 	reaction_time = []


	# 	print '[%s] Extracting events and signals from data' % (self.__class__.__name__)
	# 	# embed()

	# 	for alias in self.aliases:

	# 		trial_parameters = self.h5_operator.read_session_data(alias, 'parameters')

	# 		trial_phase_times = self.h5_operator.read_session_data(alias, 'trial_phases')
	# 		trial_times = self.h5_operator.read_session_data(alias, 'trials')
	# 		#trial_types = self.h5_operator.read_session_data(alias, 'parameters')['trial_type']

	# 		blocks = self.h5_operator.read_session_data(alias, 'blocks')
	# 		block_start_times = blocks['block_start_timestamp']
	# 		trial_phase_times = trial_phase_times[trial_phase_times['trial_phase_index']==7]



	# 		# Kick out incomplete trials
	# 		if len(trial_phase_times) < len(trial_times):
	# 			trial_times = trial_times[0:len(trial_phase_times)]
	# 			#trial_types = trial_types[0:len(trial_phase_times)]		

	# 		for bs,be in zip(blocks['block_start_timestamp'], blocks['block_end_timestamp']):

	# 			# block_events = (trial_phase_times['trial_phase_EL_timestamp'][np.array((trial_times['trial_start_EL_timestamp'] >= bs) * (trial_times['trial_end_EL_timestamp'] < be))] - bs + len(pupil_signal))/self.signal_sample_frequency

	# 			#block_types = trial_types[np.array((trial_times['trial_start_EL_timestamp'] >= bs) & (trial_times['trial_end_EL_timestamp'] < be))]


	# 			#stim2_events.extend(block_events - (.15))

	# 			for trial_time,tii in zip(trial_phase_times['trial_phase_EL_timestamp'][np.array((trial_times['trial_start_EL_timestamp'] >= bs) * (trial_times['trial_end_EL_timestamp'] < be))],trial_phase_times['trial_phase_trial'][np.array((trial_times['trial_start_EL_timestamp'] >= bs) * (trial_times['trial_end_EL_timestamp'] < be))]):

	# 				block_event_time = (trial_time - bs + len(pupil_signal))/self.signal_sample_frequency 

	# 				psignal = self.h5_operator.signal_during_period(time_period = [trial_time - (self.signal_sample_frequency*(.15+.03+.15+0.5)), trial_time - (self.signal_sample_frequency*(.15+.03+.15+0.5)) + winlength], alias = alias, signal = 'pupil_lp_clean_zscore', requested_eye = self.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0]).values
	# 				bsignal = self.h5_operator.signal_during_period(time_period = [trial_time - (self.signal_sample_frequency*(.15+.03+.15+0.5+1.25+1.25+0.5)), trial_time - (self.signal_sample_frequency*(.15+.03+.15+0.5+1.25+1.25))], alias = alias, signal = 'pupil_lp_clean_zscore', requested_eye = self.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0]).values

	# 				if len(psignal) >= minwinlength:

	# 					response_events.extend([block_event_time])	

	# 					cue_events.append(block_event_time - (1.25+1.25+0.5+.15+.03+.15))					
	# 					task_events.append(block_event_time - (1.25+0.5+.15+.03+.15))	
	# 					stim1_events.append(block_event_time - (.15+.03+.15))	

	# 					trial_signals.append(sp.signal.decimate(psignal[:minwinlength] - np.nanmean(bsignal), down_fs, 1))

	# 					trial_tasks.extend([trial_parameters['task'][np.where(trial_parameters['trial_nr']==tii)[0][0]]])
	# 					trial_color.extend([trial_parameters['trial_color'][np.where(trial_parameters['trial_nr']==tii)[0][0]]])
	# 					trial_orientation.extend([trial_parameters['trial_orientation'][np.where(trial_parameters['trial_nr']==tii)[0][0]]])
	# 					trial_correct.extend([trial_parameters['correct_answer'][np.where(trial_parameters['trial_nr']==tii)[0][0]]])
	# 					reaction_time.extend([trial_parameters['reaction_time'][np.where(trial_parameters['trial_nr']==tii)[0][0]]])

	# 					trial_codes.extend([self.recode_trial_code(trial_parameters.iloc[np.array(trial_parameters['trial_nr']==tii),:])])

	# 			pupil_signal.extend(np.squeeze(self.h5_operator.signal_during_period(time_period = [bs,be], alias = alias, signal = 'pupil_lp_clean_zscore', requested_eye = self.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0])))			

	# 	self.pupil_signal = np.array(pupil_signal)

	# 	# embed()
	# 	timestamps = {}

	# 	tcode_pairs = [[0,1],[10,20,30,40,50,60]]
	# 	tcode_names = ['Pred','Unpred']#'P-UP','UP-P','UP-UP']
	# 	names = []
	# 	durations = {}
	# 	covariates = {}

	# 	task_difficulty = np.array([1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
	# 	uniq_ints = np.vstack([np.unique(np.abs(trial_color)), np.unique(np.abs(trial_orientation))])

	# 	timestamps['cue'] = np.array(cue_events)
	# 	names.append('cue')	
	# 	durations['cue'] = (np.array(response_events) + np.array(reaction_time)) - np.array(cue_events)	

	# 	timestamps['task'] = np.array(task_events)
	# 	names.append('task')
	# 	durations['task'] = (np.array(response_events) + np.array(reaction_time)) - np.array(task_events)	

	# 	timestamps['stim'] = np.array(stim1_events)
	# 	names.append('stim')
	# 	durations['stim'] = np.ones((timestamps['stim'].size)) * 0.15
	# 	# covariates['stim.gain'] = np.ones((timestamps['stim'].size))
	# 	# covariates['stim.corr'] = np.array([task_difficulty[taskval[int(taskval[0])]==uniq_ints[int(taskval[0])-1]] for taskval in zip(trial_tasks,np.abs(trial_color),np.abs(trial_orientation))])		

	# 	# for ii,tcode in enumerate(tcode_pairs):		
	# 	# 	timestamps['stim_' + tcode_names[ii]] = np.extract((np.array(trial_codes)==tcode[0]) | (np.array(trial_codes)==tcode[1]), stim1_events)
	# 	# 	names.append('stim_' + tcode_names[ii])
	# 	# 	durations['stim_' + tcode_names[ii]] = np.ones((timestamps[-1].size)) * 0.15
	# 	# 	covariates['stim_' + tcode_names[ii] + '.gain'] = np.ones((timestamps[-1].size))
	# 	# 	covariates['stim_' + tcode_names[ii] + '.corr'] = np.extract((np.array(trial_codes)==tcode[0]) | (np.array(trial_codes)==tcode[1]), trial_ints)

	# 	# timestamps.append(np.array(response_events))
	# 	# names.append('response')

	# 	timestamps['response'] = np.array(response_events) + np.array(reaction_time)
	# 	names.append('response')
	# 	durations['response'] = np.ones((timestamps['response'].size))
	# 	covariates['response.gain'] = np.ones((timestamps['response'].size))
	# 	covariates['response.corr'] = np.array([task_difficulty[taskval[int(taskval[0])]==uniq_ints[int(taskval[0])-1]] for taskval in zip(trial_tasks,np.abs(trial_color),np.abs(trial_orientation))])

	# 	self.events.update({'timestamps': timestamps,
	# 						'codes': np.unique(trial_codes),
	# 						'names': names,
	# 						'durations': durations,
	# 						'covariates': covariates
	# 						})

	# 	# stimulus types:
	# 	# 
	# 	# green, 45, expected
	# 	# green, 45, unexpected
	# 	# green, 135, expected
	# 	# green, 135, unexpected
	# 	# red, 45, expected
	# 	# red, 45, unexpected
	# 	# red, 135, expected
	# 	# red, 135, unexpected

	# 	self.pupil_data = np.array(trial_signals)

 

	# 	self.task_data.update({'coded_trials': np.array(trial_codes),
	# 						   'trial_tasks': trial_tasks,
	# 						   'trial_color': trial_color,
	# 						   'trial_orientation': trial_orientation,
	# 						   'trial_correct': trial_correct,
	# 						   'reaction_time': reaction_time})	
	# 	# self.unload_data()

	# def event_related_average(self):

	# 	down_fs = 100
	# 	winlength = 6500#4500
	# 	minwinlength = 6000#4000

	# 	baselength = 500

	# 	pretimewindow_stim = 1000 * (.15 + .03 + .15)# + 0.5)
	# 	pretimewindow_baseline = 1000 * (.15 + .03 + .15 + 1.25 + 1.25 + 0.5 + 0.5)
	# 	pretimewindow_trial = 1000 * (.15 + .03 + .15 + 1.25 + 1.25 + 0.5 + 0.5)

	# 	# if len(self.events)==0:
	# 	# 	self.extract_signal_blocks()

	# 	self.trial_signals = {}

	# 	pupil_signal = []
	# 	response_events = []
	# 	cue_events = []
	# 	task_events = []
	# 	stim1_events = []
	# 	stim2_events = []

	# 	trial_signals = []

	# 	trial_tasks = []
	# 	trial_color = []
	# 	trial_orientation = []
	# 	trial_correct = []
	# 	trial_codes = []
	# 	reaction_time = []


	# 	# embed()

	# 	codes =  [0,1,10,20,30,40,50,60]
	# 	for e in codes:
	# 		self.trial_signals[e] = {'cue':[],'stim':[],'trial':[]}		

	# 	self.get_aliases()

	# 	self.load_combined_data()

	# 	print '[%s] Extracting event-related signals per condition' % (self.__class__.__name__)

	# 	for ii,alias in enumerate(self.aliases):

	# 		trial_parameters = self.h5_operator.read_session_data(alias, 'parameters')

	# 		trial_phase_times = self.h5_operator.read_session_data(alias, 'trial_phases')
	# 		trial_times = self.h5_operator.read_session_data(alias, 'trials')
	# 		#trial_types = self.h5_operator.read_session_data(alias, 'parameters')['trial_type']

	# 		blocks = self.h5_operator.read_session_data(alias, 'blocks')
	# 		block_start_times = blocks['block_start_timestamp']
	# 		trial_phase_times = trial_phase_times[trial_phase_times['trial_phase_index']==7]



	# 		# Kick out incomplete trials
	# 		if len(trial_phase_times) < len(trial_times):
	# 			trial_times = trial_times[0:len(trial_phase_times)]
	# 			#trial_types = trial_types[0:len(trial_phase_times)]		

	# 		for bs,be in zip(blocks['block_start_timestamp'], blocks['block_end_timestamp']):

	# 			block_events = trial_phase_times['trial_phase_EL_timestamp'][np.array((trial_times['trial_start_EL_timestamp'] >= bs) * (trial_times['trial_end_EL_timestamp'] < be))]

	# 			block_trials = trial_phase_times['trial_phase_trial'][np.array((trial_times['trial_start_EL_timestamp'] >= bs) * (trial_times['trial_end_EL_timestamp'] < be))]

	# 			for ble,blt in zip(block_events,block_trials):
	# 				# pupil_signal = np.squeeze(self.h5_operator.signal_during_period(time_period = (ble-1000*(0.5+1.25+1.25+0.5+.15+.03+.15), ble - 1000*(0.5+1.25+1.25+0.5+.15+.03+.15) + winlength), alias = alias, signal = 'pupil_lp_psc', requested_eye = 'L'))
	# 				# pupil_signal_baseline = np.squeeze(self.h5_operator.signal_during_period(time_period = (ble-pretimewindow_baseline, ble - pretimewindow_baseline + baselength), alias = alias, signal = 'pupil_lp_clean_zscore', requested_eye = self.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0]))
	# 				# pupil_signal_stim = np.squeeze(self.h5_operator.signal_during_period(time_period = (ble-pretimewindow_stim, ble - pretimewindow_stim + winlength), alias = alias, signal = 'pupil_lp_clean_zscore', requested_eye = self.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0]))
					
	# 				pupil_signal_trial = np.squeeze(self.h5_operator.signal_during_period(time_period = (ble-pretimewindow_trial, ble - pretimewindow_trial + winlength), alias = alias, signal = 'pupil_lp_clean_zscore', requested_eye = self.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0]))
	# 				block_event_time = (ble - bs + len(pupil_signal))/self.signal_sample_frequency 

	# 				if len(pupil_signal_trial) >= minwinlength:

	# 					#pupil_signal_stim = pupil_signal_stim[0:minwinlength]
	# 					pupil_signal_trial = pupil_signal_trial[0:minwinlength]

	# 					#pupil_signal_stim -= np.nanmean(pupil_signal_baseline)
	# 					pupil_signal_trial -= np.nanmean(pupil_signal_trial[:baselength])

	# 					response_events.extend([block_event_time])	

	# 					cue_events.append(block_event_time - (1.25+1.25+0.5+.15+.03+.15))					
	# 					task_events.append(block_event_time - (1.25+0.5+.15+.03+.15))	
	# 					stim1_events.append(block_event_time - (.15+.03+.15))	


	# 					trial_code = self.recode_trial_code(trial_parameters.iloc[np.array(trial_parameters['trial_nr']==blt),:])

	# 					trial_tasks.extend([trial_parameters['task'][np.where(trial_parameters['trial_nr']==blt)[0][0]]])
	# 					trial_color.extend([trial_parameters['trial_color'][np.where(trial_parameters['trial_nr']==blt)[0][0]]])
	# 					trial_orientation.extend([trial_parameters['trial_orientation'][np.where(trial_parameters['trial_nr']==blt)[0][0]]])
	# 					trial_correct.extend([trial_parameters['correct_answer'][np.where(trial_parameters['trial_nr']==blt)[0][0]]])
	# 					reaction_time.extend([trial_parameters['reaction_time'][np.where(trial_parameters['trial_nr']==blt)[0][0]]])

	# 					trial_codes.extend([self.recode_trial_code(trial_parameters.iloc[np.array(trial_parameters['trial_nr']==blt),:])])

	# 					#self.trial_signals[trial_codes[-1]]['stim'].append(sp.signal.decimate(pupil_signal_stim, down_fs, 1))
	# 					self.trial_signals[trial_code]['trial'].append(sp.signal.decimate(pupil_signal_trial, down_fs, 1))					
	# 			pupil_signal.extend(np.squeeze(self.h5_operator.signal_during_period(time_period = [bs,be], alias = alias, signal = 'pupil_lp_clean_zscore', requested_eye = self.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0])))			

	# 	self.pupil_signal = np.array(pupil_signal)
						

	# 	# embed()
	# 	timestamps = {}

	# 	tcode_pairs = [[0,1],[10,20,30,40,50,60]]
	# 	tcode_names = ['P-P','P-UP','UP-P','UP-UP']
	# 	names = []
	# 	durations = {}
	# 	covariates = {}

	# 	task_difficulty = 1.0 / np.array([0.25,0.35,0.5, 0.7, 1.0, 1.41, 2.0, 2.83, 4])
	# 	uniq_ints = np.vstack([np.unique(np.abs(trial_color)), np.unique(np.abs(trial_orientation))])

	# 	timestamps['cue'] = np.array(cue_events)
	# 	names.append('cue')	
	# 	durations['cue'] = (np.array(response_events) + np.array(reaction_time)) - np.array(cue_events)	

	# 	timestamps['task'] = np.array(task_events)
	# 	names.append('task')
	# 	durations['task'] = (np.array(response_events) + np.array(reaction_time)) - np.array(task_events)	

	# 	timestamps['stim'] = np.array(stim1_events)
	# 	names.append('stim')
	# 	durations['stim'] = np.ones((timestamps['stim'].size)) * 0.15
	# 	# covariates['stim.gain'] = np.ones((timestamps['stim'].size))
	# 	# covariates['stim.corr'] = np.array([task_difficulty[taskval[int(taskval[0])]==uniq_ints[int(taskval[0])-1]] for taskval in zip(trial_tasks,np.abs(trial_color),np.abs(trial_orientation))])		

	# 	# for ii,tcode in enumerate(tcode_pairs):		
	# 	# 	timestamps['stim_' + tcode_names[ii]] = np.extract((np.array(trial_codes)==tcode[0]) | (np.array(trial_codes)==tcode[1]), stim1_events)
	# 	# 	names.append('stim_' + tcode_names[ii])
	# 	# 	durations['stim_' + tcode_names[ii]] = np.ones((timestamps[-1].size)) * 0.15
	# 	# 	covariates['stim_' + tcode_names[ii] + '.gain'] = np.ones((timestamps[-1].size))
	# 	# 	covariates['stim_' + tcode_names[ii] + '.corr'] = np.extract((np.array(trial_codes)==tcode[0]) | (np.array(trial_codes)==tcode[1]), trial_ints)

	# 	# timestamps.append(np.array(response_events))
	# 	# names.append('response')

	# 	timestamps['dec_interval'] = np.array(response_events)
	# 	names.append('dec_interval')
	# 	durations['dec_interval'] = np.array(reaction_time)
	# 	covariates['dec_interval.gain'] = np.ones((timestamps['dec_interval'].size))
	# 	covariates['dec_interval.corr'] = np.array([task_difficulty[taskval[int(taskval[0])]==uniq_ints[int(taskval[0])-1]] for taskval in zip(trial_tasks,np.abs(trial_color),np.abs(trial_orientation))])


	# 	timestamps['response'] = np.array(response_events) + np.array(reaction_time)
	# 	names.append('response')
	# 	durations['response'] = np.ones((timestamps['response'].size))
	# 	covariates['response.gain'] = np.ones((timestamps['response'].size))
	# 	covariates['response.corr'] = np.array([task_difficulty[taskval[int(taskval[0])]==uniq_ints[int(taskval[0])-1]] for taskval in zip(trial_tasks,np.abs(trial_color),np.abs(trial_orientation))])

	# 	self.events.update({'timestamps': timestamps,
	# 						'codes': np.unique(trial_codes),
	# 						'names': names,
	# 						'durations': durations,
	# 						'covariates': covariates
	# 						})

	# 	# stimulus types:
	# 	# 
	# 	# green, 45, expected
	# 	# green, 45, unexpected
	# 	# green, 135, expected
	# 	# green, 135, unexpected
	# 	# red, 45, expected
	# 	# red, 45, unexpected
	# 	# red, 135, expected
	# 	# red, 135, unexpected

	# 	# self.pupil_data = np.array(trial_signals)

 

	# 	self.task_data.update({'coded_trials': np.array(trial_codes),
	# 						   'trial_tasks': trial_tasks,
	# 						   'trial_color': trial_color,
	# 						   'trial_orientation': trial_orientation,
	# 						   'trial_correct': trial_correct,
	# 						   'reaction_time': reaction_time})				

	# def event_related_average(self):

	# 	down_fs = 100
	# 	winlength = 6500
	# 	minwinlength = 6000

	# 	self.extract_signal_blocks()

		

	# 	self.trial_signals = {}

	# 	for e in self.events['codes']:
	# 		self.trial_signals[self.events['names'][e]] = []

	# 		times = self.events['timestamps'][np.array(self.events['coded_trials'] == e)]

	# 		for t in times:



	# 			psignal = self.pupil_signal[]

	# 		self.trial_signals[self.events['names'][trial_code]].append(sp.signal.decimate(pupil_signal, down_fs, 1))




		
	# 	for ii,alias in enumerate(self.aliases):

	# 		trial_phase_times = self.h5_operator.read_session_data(alias, 'trial_phases')
	# 		# trial_times = ho.read_session_data(alias, 'trials')

	# 		period_start = trial_phase_times['trial_phase_EL_timestamp'][np.array(trial_phase_times['trial_phase_index']==7)] - 1000*(0.5+.15+.03+.15)
	# 		period_end = trial_phase_times['trial_phase_EL_timestamp'][np.array(trial_phase_times['trial_phase_index']==7)] - 1000*(0.5+.15+.03+.15) + winlength
	# 		#trial_types = ho.read_session_data(alias, 'parameters')['trial_type'][np.array(trial_phase_times['trial_phase_index']==7)]
	# 		# tasks = ho.read_session_data(alias, 'parameters')['task']
	# 		# buttons = ho.read_session_data(alias, 'parameters')['button']
	# 		# tasks = ho.read_session_data(alias, 'parameters')['task']

	# 		trial_parameters = self.h5_operator.read_session_data(alias, 'parameters')

	# 		trial_types = trial_parameters['trial_type'][np.array(trial_phase_times['trial_phase_index']==7)]


	# 		for ii,ps in enumerate(zip(period_start,period_end)):

	# 			# if ii not in [100, 183, 200]:

	# 			try:
	# 				pupil_signal = np.squeeze(self.h5_operator.signal_during_period(time_period = ps, alias = alias, signal = 'pupil_lp_clean_zscore', requested_eye = 'L'))
	# 			except:
	# 				embed()


	# 			# print str(len(pupil_signal))

	# 			if (len(np.array(pupil_signal)) >= minwinlength):
					
	# 				pupil_signal = pupil_signal[0:minwinlength]


	# 			trial_code = self.recode_trial_code(trial_parameters.loc[np.array(trial_parameters['trial_nr']==trial_phase_times['trial_phase_trial'][ii]),:])

	# 			self.trial_signals[self.events['names'][trial_code]].append(sp.signal.decimate(pupil_signal, down_fs, 1))	