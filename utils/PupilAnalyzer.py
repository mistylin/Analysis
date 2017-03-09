import os,glob,datetime

import numpy as np
import scipy as sp
import seaborn as sn
import matplotlib.pylab as pl
import cPickle as pickle
import pandas as pd

from math import *

from hedfpy.EDFOperator import EDFOperator
from hedfpy.HDFEyeOperator import HDFEyeOperator
from hedfpy.EyeSignalOperator import EyeSignalOperator

from fir import FIRDeconvolution

from IPython import embed

from Analyzer import Analyzer

class PupilAnalyzer(Analyzer):

	def __init__(self, subID, filename, edf_folder, **kwargs):

		# Setup default parameter values
		self.default_parameters = {'low_pass_pupil_f': 6.0,
								   'high_pass_pupil_f': 0.01}

		super(PupilAnalyzer, self).__init__(subID, filename, **kwargs)

		self.edf_folder = edf_folder
		self.data_folder = edf_folder
		
		self.fir_signal = {}

		# Initialize variables
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

	def get_h5_operator(self):		

		self.h5_operator = HDFEyeOperator(self.data_file)

		if not os.path.isfile(self.data_file):

			edf_files = glob.glob(self.edf_folder + '/*.edf')
			#edf_files.sort(key=lambda x: os.path.getmtime(x))
			edf_files.sort()

			# embed()

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

	def extract_signal_blocks(self):

		self.get_aliases()

		self.load_data()

		pupil_signal = []

		down_fs = 100

		winlength = 4500
		minwinlength = 4000

		events = []

		response_events = []
		cue_events = []
		task_events = []
		stim1_events = []
		stim2_events = []

		trial_signals = []

		trial_tasks = []
		trial_color = []
		trial_orientation = []
		trial_correct = []
		trial_codes = []
		reaction_time = []


		print '[%s] Extracting events and signals from data' % (self.__class__.__name__)
		# embed()

		for alias in self.aliases:

			trial_parameters = self.h5_operator.read_session_data(alias, 'parameters')

			trial_phase_times = self.h5_operator.read_session_data(alias, 'trial_phases')
			trial_times = self.h5_operator.read_session_data(alias, 'trials')
			#trial_types = self.h5_operator.read_session_data(alias, 'parameters')['trial_type']

			blocks = self.h5_operator.read_session_data(alias, 'blocks')
			block_start_times = blocks['block_start_timestamp']
			trial_phase_times = trial_phase_times[trial_phase_times['trial_phase_index']==7]



			# Kick out incomplete trials
			if len(trial_phase_times) < len(trial_times):
				trial_times = trial_times[0:len(trial_phase_times)]
				#trial_types = trial_types[0:len(trial_phase_times)]		

			for bs,be in zip(blocks['block_start_timestamp'], blocks['block_end_timestamp']):

				# block_events = (trial_phase_times['trial_phase_EL_timestamp'][np.array((trial_times['trial_start_EL_timestamp'] >= bs) * (trial_times['trial_end_EL_timestamp'] < be))] - bs + len(pupil_signal))/self.signal_sample_frequency

				#block_types = trial_types[np.array((trial_times['trial_start_EL_timestamp'] >= bs) & (trial_times['trial_end_EL_timestamp'] < be))]


				#stim2_events.extend(block_events - (.15))

				for trial_time,tii in zip(trial_phase_times['trial_phase_EL_timestamp'][np.array((trial_times['trial_start_EL_timestamp'] >= bs) * (trial_times['trial_end_EL_timestamp'] < be))],trial_phase_times['trial_phase_trial'][np.array((trial_times['trial_start_EL_timestamp'] >= bs) * (trial_times['trial_end_EL_timestamp'] < be))]):

					block_event_time = (trial_time - bs + len(pupil_signal))/self.signal_sample_frequency 

					psignal = self.h5_operator.signal_during_period(time_period = [trial_time - (self.signal_sample_frequency*(.15+.03+.15+0.5)), trial_time - (self.signal_sample_frequency*(.15+.03+.15+0.5)) + winlength], alias = alias, signal = 'pupil_lp_clean_zscore', requested_eye = self.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0]).values
					bsignal = self.h5_operator.signal_during_period(time_period = [trial_time - (self.signal_sample_frequency*(.15+.03+.15+0.5+1.25+1.25+0.5)), trial_time - (self.signal_sample_frequency*(.15+.03+.15+0.5+1.25+1.25))], alias = alias, signal = 'pupil_lp_clean_zscore', requested_eye = self.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0]).values

					if len(psignal) >= minwinlength:

						response_events.extend([block_event_time])	

						cue_events.append(block_event_time - (1.25+1.25+0.5+.15+.03+.15))					
						task_events.append(block_event_time - (1.25+0.5+.15+.03+.15))	
						stim1_events.append(block_event_time - (.15+.03+.15))	

						trial_signals.append(sp.signal.decimate(psignal[:minwinlength] - np.nanmean(bsignal), down_fs, 1))

						trial_tasks.extend([trial_parameters['task'][np.where(trial_parameters['trial_nr']==tii)[0][0]]])
						trial_color.extend([trial_parameters['trial_color'][np.where(trial_parameters['trial_nr']==tii)[0][0]]])
						trial_orientation.extend([trial_parameters['trial_orientation'][np.where(trial_parameters['trial_nr']==tii)[0][0]]])
						trial_correct.extend([trial_parameters['correct_answer'][np.where(trial_parameters['trial_nr']==tii)[0][0]]])
						reaction_time.extend([trial_parameters['reaction_time'][np.where(trial_parameters['trial_nr']==tii)[0][0]]])

						trial_codes.extend([self.recode_trial_code(trial_parameters.iloc[np.array(trial_parameters['trial_nr']==tii),:])])

				pupil_signal.extend(np.squeeze(self.h5_operator.signal_during_period(time_period = [bs,be], alias = alias, signal = 'pupil_lp_clean_zscore', requested_eye = self.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0])))			

		self.pupil_signal = np.array(pupil_signal)

		# embed()
		timestamps = {}

		tcode_pairs = [[0,1],[10,20,30,40,50,60]]
		tcode_names = ['Pred','Unpred']#'P-UP','UP-P','UP-UP']
		names = []
		durations = {}
		covariates = {}

		task_difficulty = np.array([1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
		uniq_ints = np.vstack([np.unique(np.abs(trial_color)), np.unique(np.abs(trial_orientation))])

		timestamps['cue'] = np.array(cue_events)
		names.append('cue')	
		durations['cue'] = (np.array(response_events) + np.array(reaction_time)) - np.array(cue_events)	

		timestamps['task'] = np.array(task_events)
		names.append('task')
		durations['task'] = (np.array(response_events) + np.array(reaction_time)) - np.array(task_events)	

		timestamps['stim'] = np.array(stim1_events)
		names.append('stim')
		durations['stim'] = np.ones((timestamps['stim'].size)) * 0.15
		# covariates['stim.gain'] = np.ones((timestamps['stim'].size))
		# covariates['stim.corr'] = np.array([task_difficulty[taskval[int(taskval[0])]==uniq_ints[int(taskval[0])-1]] for taskval in zip(trial_tasks,np.abs(trial_color),np.abs(trial_orientation))])		

		# for ii,tcode in enumerate(tcode_pairs):		
		# 	timestamps['stim_' + tcode_names[ii]] = np.extract((np.array(trial_codes)==tcode[0]) | (np.array(trial_codes)==tcode[1]), stim1_events)
		# 	names.append('stim_' + tcode_names[ii])
		# 	durations['stim_' + tcode_names[ii]] = np.ones((timestamps[-1].size)) * 0.15
		# 	covariates['stim_' + tcode_names[ii] + '.gain'] = np.ones((timestamps[-1].size))
		# 	covariates['stim_' + tcode_names[ii] + '.corr'] = np.extract((np.array(trial_codes)==tcode[0]) | (np.array(trial_codes)==tcode[1]), trial_ints)

		# timestamps.append(np.array(response_events))
		# names.append('response')

		timestamps['response'] = np.array(response_events) + np.array(reaction_time)
		names.append('response')
		durations['response'] = np.ones((timestamps['response'].size))
		covariates['response.gain'] = np.ones((timestamps['response'].size))
		covariates['response.corr'] = np.array([task_difficulty[taskval[int(taskval[0])]==uniq_ints[int(taskval[0])-1]] for taskval in zip(trial_tasks,np.abs(trial_color),np.abs(trial_orientation))])

		self.events.update({'timestamps': timestamps,
							'codes': np.unique(trial_codes),
							'names': names,
							'durations': durations,
							'covariates': covariates
							})

		# stimulus types:
		# 
		# green, 45, expected
		# green, 45, unexpected
		# green, 135, expected
		# green, 135, unexpected
		# red, 45, expected
		# red, 45, unexpected
		# red, 135, expected
		# red, 135, unexpected

		self.pupil_data = np.array(trial_signals)

 

		self.task_data.update({'coded_trials': np.array(trial_codes),
							   'trial_tasks': trial_tasks,
							   'trial_color': trial_color,
							   'trial_orientation': trial_orientation,
							   'trial_correct': trial_correct,
							   'reaction_time': reaction_time})	
		# self.unload_data()

	def recode_trial_code(self, params):

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

		if np.array(params['trial_type'] == 1): # base trial (/expected)
		 	if np.array(params['task'] == 1):
		 		return 0
		 	else:
		 		return 1			

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



	def event_related_average(self):

		down_fs = 100
		winlength = 6500#4500
		minwinlength = 6000#4000

		baselength = 500

		pretimewindow_stim = 1000 * (.15 + .03 + .15)# + 0.5)
		pretimewindow_baseline = 1000 * (.15 + .03 + .15 + 1.25 + 1.25 + 0.5 + 0.5)
		pretimewindow_trial = 1000 * (.15 + .03 + .15 + 1.25 + 1.25 + 0.5 + 0.5)

		# if len(self.events)==0:
		# 	self.extract_signal_blocks()

		self.trial_signals = {}

		pupil_signal = []
		response_events = []
		cue_events = []
		task_events = []
		stim1_events = []
		stim2_events = []

		trial_signals = []

		trial_tasks = []
		trial_color = []
		trial_orientation = []
		trial_correct = []
		trial_codes = []
		reaction_time = []


		# embed()

		codes =  [0,1,10,20,30,40,50,60]
		for e in codes:
			self.trial_signals[e] = {'cue':[],'stim':[],'trial':[]}		

		self.get_aliases()

		self.load_data()

		print '[%s] Extracting event-related signals per condition' % (self.__class__.__name__)

		for ii,alias in enumerate(self.aliases):

			trial_parameters = self.h5_operator.read_session_data(alias, 'parameters')

			trial_phase_times = self.h5_operator.read_session_data(alias, 'trial_phases')
			trial_times = self.h5_operator.read_session_data(alias, 'trials')
			#trial_types = self.h5_operator.read_session_data(alias, 'parameters')['trial_type']

			blocks = self.h5_operator.read_session_data(alias, 'blocks')
			block_start_times = blocks['block_start_timestamp']
			trial_phase_times = trial_phase_times[trial_phase_times['trial_phase_index']==7]



			# Kick out incomplete trials
			if len(trial_phase_times) < len(trial_times):
				trial_times = trial_times[0:len(trial_phase_times)]
				#trial_types = trial_types[0:len(trial_phase_times)]		

			for bs,be in zip(blocks['block_start_timestamp'], blocks['block_end_timestamp']):

				block_events = trial_phase_times['trial_phase_EL_timestamp'][np.array((trial_times['trial_start_EL_timestamp'] >= bs) * (trial_times['trial_end_EL_timestamp'] < be))]

				block_trials = trial_phase_times['trial_phase_trial'][np.array((trial_times['trial_start_EL_timestamp'] >= bs) * (trial_times['trial_end_EL_timestamp'] < be))]

				for ble,blt in zip(block_events,block_trials):
					# pupil_signal = np.squeeze(self.h5_operator.signal_during_period(time_period = (ble-1000*(0.5+1.25+1.25+0.5+.15+.03+.15), ble - 1000*(0.5+1.25+1.25+0.5+.15+.03+.15) + winlength), alias = alias, signal = 'pupil_lp_psc', requested_eye = 'L'))
					# pupil_signal_baseline = np.squeeze(self.h5_operator.signal_during_period(time_period = (ble-pretimewindow_baseline, ble - pretimewindow_baseline + baselength), alias = alias, signal = 'pupil_lp_clean_zscore', requested_eye = self.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0]))
					# pupil_signal_stim = np.squeeze(self.h5_operator.signal_during_period(time_period = (ble-pretimewindow_stim, ble - pretimewindow_stim + winlength), alias = alias, signal = 'pupil_lp_clean_zscore', requested_eye = self.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0]))
					
					pupil_signal_trial = np.squeeze(self.h5_operator.signal_during_period(time_period = (ble-pretimewindow_trial, ble - pretimewindow_trial + winlength), alias = alias, signal = 'pupil_lp_clean_zscore', requested_eye = self.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0]))
					block_event_time = (ble - bs + len(pupil_signal))/self.signal_sample_frequency 

					if len(pupil_signal_trial) >= minwinlength:

						#pupil_signal_stim = pupil_signal_stim[0:minwinlength]
						pupil_signal_trial = pupil_signal_trial[0:minwinlength]

						#pupil_signal_stim -= np.nanmean(pupil_signal_baseline)
						pupil_signal_trial -= np.nanmean(pupil_signal_trial[:baselength])

						response_events.extend([block_event_time])	

						cue_events.append(block_event_time - (1.25+1.25+0.5+.15+.03+.15))					
						task_events.append(block_event_time - (1.25+0.5+.15+.03+.15))	
						stim1_events.append(block_event_time - (.15+.03+.15))	


						trial_code = self.recode_trial_code(trial_parameters.iloc[np.array(trial_parameters['trial_nr']==blt),:])

						trial_tasks.extend([trial_parameters['task'][np.where(trial_parameters['trial_nr']==blt)[0][0]]])
						trial_color.extend([trial_parameters['trial_color'][np.where(trial_parameters['trial_nr']==blt)[0][0]]])
						trial_orientation.extend([trial_parameters['trial_orientation'][np.where(trial_parameters['trial_nr']==blt)[0][0]]])
						trial_correct.extend([trial_parameters['correct_answer'][np.where(trial_parameters['trial_nr']==blt)[0][0]]])
						reaction_time.extend([trial_parameters['reaction_time'][np.where(trial_parameters['trial_nr']==blt)[0][0]]])

						trial_codes.extend([self.recode_trial_code(trial_parameters.iloc[np.array(trial_parameters['trial_nr']==blt),:])])

						#self.trial_signals[trial_codes[-1]]['stim'].append(sp.signal.decimate(pupil_signal_stim, down_fs, 1))
						self.trial_signals[trial_code]['trial'].append(sp.signal.decimate(pupil_signal_trial, down_fs, 1))					
				pupil_signal.extend(np.squeeze(self.h5_operator.signal_during_period(time_period = [bs,be], alias = alias, signal = 'pupil_lp_clean_zscore', requested_eye = self.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0])))			

		self.pupil_signal = np.array(pupil_signal)
						

		# embed()
		timestamps = {}

		tcode_pairs = [[0,1],[10,20,30,40,50,60]]
		tcode_names = ['P-P','P-UP','UP-P','UP-UP']
		names = []
		durations = {}
		covariates = {}

		task_difficulty = 1.0 / np.array([0.25,0.35,0.5, 0.7, 1.0, 1.41, 2.0, 2.83, 4])
		uniq_ints = np.vstack([np.unique(np.abs(trial_color)), np.unique(np.abs(trial_orientation))])

		timestamps['cue'] = np.array(cue_events)
		names.append('cue')	
		durations['cue'] = (np.array(response_events) + np.array(reaction_time)) - np.array(cue_events)	

		timestamps['task'] = np.array(task_events)
		names.append('task')
		durations['task'] = (np.array(response_events) + np.array(reaction_time)) - np.array(task_events)	

		timestamps['stim'] = np.array(stim1_events)
		names.append('stim')
		durations['stim'] = np.ones((timestamps['stim'].size)) * 0.15
		# covariates['stim.gain'] = np.ones((timestamps['stim'].size))
		# covariates['stim.corr'] = np.array([task_difficulty[taskval[int(taskval[0])]==uniq_ints[int(taskval[0])-1]] for taskval in zip(trial_tasks,np.abs(trial_color),np.abs(trial_orientation))])		

		# for ii,tcode in enumerate(tcode_pairs):		
		# 	timestamps['stim_' + tcode_names[ii]] = np.extract((np.array(trial_codes)==tcode[0]) | (np.array(trial_codes)==tcode[1]), stim1_events)
		# 	names.append('stim_' + tcode_names[ii])
		# 	durations['stim_' + tcode_names[ii]] = np.ones((timestamps[-1].size)) * 0.15
		# 	covariates['stim_' + tcode_names[ii] + '.gain'] = np.ones((timestamps[-1].size))
		# 	covariates['stim_' + tcode_names[ii] + '.corr'] = np.extract((np.array(trial_codes)==tcode[0]) | (np.array(trial_codes)==tcode[1]), trial_ints)

		# timestamps.append(np.array(response_events))
		# names.append('response')

		timestamps['dec_interval'] = np.array(response_events)
		names.append('dec_interval')
		durations['dec_interval'] = np.array(reaction_time)
		covariates['dec_interval.gain'] = np.ones((timestamps['dec_interval'].size))
		covariates['dec_interval.corr'] = np.array([task_difficulty[taskval[int(taskval[0])]==uniq_ints[int(taskval[0])-1]] for taskval in zip(trial_tasks,np.abs(trial_color),np.abs(trial_orientation))])


		timestamps['response'] = np.array(response_events) + np.array(reaction_time)
		names.append('response')
		durations['response'] = np.ones((timestamps['response'].size))
		covariates['response.gain'] = np.ones((timestamps['response'].size))
		covariates['response.corr'] = np.array([task_difficulty[taskval[int(taskval[0])]==uniq_ints[int(taskval[0])-1]] for taskval in zip(trial_tasks,np.abs(trial_color),np.abs(trial_orientation))])

		self.events.update({'timestamps': timestamps,
							'codes': np.unique(trial_codes),
							'names': names,
							'durations': durations,
							'covariates': covariates
							})

		# stimulus types:
		# 
		# green, 45, expected
		# green, 45, unexpected
		# green, 135, expected
		# green, 135, unexpected
		# red, 45, expected
		# red, 45, unexpected
		# red, 135, expected
		# red, 135, unexpected

		# self.pupil_data = np.array(trial_signals)

 

		self.task_data.update({'coded_trials': np.array(trial_codes),
							   'trial_tasks': trial_tasks,
							   'trial_color': trial_color,
							   'trial_orientation': trial_orientation,
							   'trial_correct': trial_correct,
							   'reaction_time': reaction_time})							



	def run_FIR(self, deconv_interval = None):

		down_fs = 100

		if deconv_interval is None:
			deconv_interval = self.deconvolution_interval


		if 'timestamps' not in self.events.keys():
			self.extract_signal_blocks()
		embed()

		print '[%s] Starting FIR deconvolution' % (self.__class__.__name__)

		self.FIRo = FIRDeconvolution(
						signal = self.pupil_signal,
						events = [self.events['timestamps']['response']],
						event_names = ['response'],
						durations = {'response': self.events['durations']['response']},
						sample_frequency = self.signal_sample_frequency,
			            deconvolution_frequency = self.deconv_sample_frequency,
			        	deconvolution_interval = deconv_interval,
			        	covariates = self.events['covariates']
					)

		self.FIRo.create_design_matrix()

		print '[%s] Fitting IRF' % (self.__class__.__name__)
		self.FIRo.regress(method = 'lstsq')
		self.FIRo.betas_for_events()
		self.FIRo.calculate_rsq()	

		# embed()

		plot_time = len(self.pupil_signal)/self.signal_sample_frequency
		ordered_covariates = ['response.gain','response.corr']

		f = pl.figure(figsize = (10,8))
		s = f.add_subplot(311)
		s.set_title('FIR responses, R squared %1.3f'%self.FIRo.rsq)
		for dec in ordered_covariates:
		    pl.plot(self.FIRo.deconvolution_interval_timepoints, self.FIRo.betas_for_cov(dec))
		pl.legend(ordered_covariates)
		sn.despine(offset=10)

		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		s = f.add_subplot(312)
		s.set_title('design matrix')
		pl.imshow(self.FIRo.design_matrix[:,:plot_time], aspect = 0.075 * plot_time/self.FIRo.deconvolution_interval_size, 
		          cmap = 'RdBu', interpolation = 'nearest', rasterized = True)
		sn.despine(offset=10)

		s = f.add_subplot(313)
		s.set_title('data and predictions')
		pl.plot(np.linspace(0,plot_time, int(plot_time * self.FIRo.deconvolution_frequency/self.FIRo.sample_frequency)), 
		        self.FIRo.resampled_signal[:,:int(plot_time * self.FIRo.deconvolution_frequency/self.FIRo.sample_frequency)].T, 'r')
		pl.plot(np.linspace(0,plot_time, int(plot_time * self.FIRo.deconvolution_frequency/self.FIRo.sample_frequency)), 
		        self.FIRo.predict_from_design_matrix(self.FIRo.design_matrix[:,:int(plot_time * self.FIRo.deconvolution_frequency/self.FIRo.sample_frequency)]).T, 'k')
		pl.legend(['signal','explained'])
		sn.despine(offset=10)
		pl.tight_layout()		

		pl.show()

		# self.fir_signal = {}

		# pl.figure(figsize=(10,10))

		# for name,dec in zip(self.FIRo.covariates.keys(), self.FIRo.betas_per_event_type.squeeze()):
		# 	#self.fir_signal.update({name: [self.FIRo.deconvolution_interval_timepoints, dec]})
		# 	pl.plot(self.FIRo.deconvolution_interval_timepoints, dec, label = name)

		# pl.legend()
		# pl.show()

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
