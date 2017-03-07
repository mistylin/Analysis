from __future__ import division
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

from PupilAnalyzer import PupilAnalyzer

class BehaviorAnalyzer(PupilAnalyzer):

	def __init__(self, subID, csv_filename, h5_filename, raw_folder, **kwargs):

		self.default_parameters = {}

		super(BehaviorAnalyzer, self).__init__(subID, h5_filename, raw_folder, **kwargs)

		self.csv_file = csv_filename

		self.h5_operator = None
		self.task_data = {}

		self.task_performance = {}

	def load_data(self):

		super(BehaviorAnalyzer, self).load_data()

		self.csv_data = pd.read_csv(self.csv_file)

	def recode_trial_types(self):
		"""
		Provide a simple coding scheme to extract trial according to 
		the type of stimulus presented:
			0:	base stimulus
			1:	change in attended feature
			2:	change in unattended feature
			3: 	change in both features
		"""

		self.load_data()

		self.get_aliases()

		new_trial_types = []

		trial_tasks = []
		trial_color = []
		trial_orientation = []
		trial_correct = []
		reaction_time = []

		for alias in self.aliases:

			trial_parameters = self.h5_operator.read_session_data(alias, 'parameters')

			trial_phase_times = self.h5_operator.read_session_data(alias, 'trial_phases')
			trial_times = self.h5_operator.read_session_data(alias, 'trials')
			#trial_types = self.h5_operator.read_session_data(alias, 'parameters')['trial_type']

			trial_phase_times = trial_phase_times[trial_phase_times['trial_phase_index']==7]

			# Kick out incomplete trials
			#if len(trial_phase_times) < len(trial_times):
			#	trial_parameters = trial_parameters[0:len(trial_phase_times)]

			for tii in range(len(trial_phase_times)):

				if trial_parameters['trial_type'][tii] == 1: # base trial (/expected)
					new_trial_types.extend([0])

					trial_tasks.extend([trial_parameters['task'][tii]])
					trial_color.extend([trial_parameters['trial_color'][tii]])
					trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
					trial_correct.extend([trial_parameters['correct_answer'][tii]])
					reaction_time.extend([trial_parameters['reaction_time'][tii]])					

				else: # non-base trial (/unexpected)
					
					if trial_parameters['task'][tii] == 1: # color task
						if trial_parameters['stimulus_type'][tii] == 0: # green45
							if trial_parameters['base_color_a'][tii] < 0:
								new_trial_types.extend([2])

								trial_tasks.extend([trial_parameters['task'][tii]])
								trial_color.extend([trial_parameters['trial_color'][tii]])
								trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
								trial_correct.extend([trial_parameters['correct_answer'][tii]])
								reaction_time.extend([trial_parameters['reaction_time'][tii]])
							else:
								if trial_parameters['base_ori'][tii] == 45:
									new_trial_types.extend([1])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
								else:
									new_trial_types.extend([3])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
						if trial_parameters['stimulus_type'][tii] == 1: # green135
							if trial_parameters['base_color_a'][tii] < 0:
								new_trial_types.extend([2])

								trial_tasks.extend([trial_parameters['task'][tii]])
								trial_color.extend([trial_parameters['trial_color'][tii]])
								trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
								trial_correct.extend([trial_parameters['correct_answer'][tii]])
								reaction_time.extend([trial_parameters['reaction_time'][tii]])
							else:
								if trial_parameters['base_ori'][tii] == 135:
									new_trial_types.extend([1])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
								else:
									new_trial_types.extend([3])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
						if trial_parameters['stimulus_type'][tii] == 2: # red45
							if trial_parameters['base_color_a'][tii] > 0:
								new_trial_types.extend([2])

								trial_tasks.extend([trial_parameters['task'][tii]])
								trial_color.extend([trial_parameters['trial_color'][tii]])
								trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
								trial_correct.extend([trial_parameters['correct_answer'][tii]])
								reaction_time.extend([trial_parameters['reaction_time'][tii]])
							else:
								if trial_parameters['base_ori'][tii] == 45:
									new_trial_types.extend([1])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
								else:
									new_trial_types.extend([3])	

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
						if trial_parameters['stimulus_type'][tii] == 3: # red135
							if trial_parameters['base_color_a'][tii] > 0:
								new_trial_types.extend([2])

								trial_tasks.extend([trial_parameters['task'][tii]])
								trial_color.extend([trial_parameters['trial_color'][tii]])
								trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
								trial_correct.extend([trial_parameters['correct_answer'][tii]])
								reaction_time.extend([trial_parameters['reaction_time'][tii]])
							else:
								if trial_parameters['base_ori'][tii] == 135:
									new_trial_types.extend([1])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
								else:
									new_trial_types.extend([3])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])

					else: # orientation task
						if trial_parameters['stimulus_type'][tii] == 0: # green45
							if trial_parameters['base_ori'][tii] == 45:
								new_trial_types.extend([2])

								trial_tasks.extend([trial_parameters['task'][tii]])
								trial_color.extend([trial_parameters['trial_color'][tii]])
								trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
								trial_correct.extend([trial_parameters['correct_answer'][tii]])
								reaction_time.extend([trial_parameters['reaction_time'][tii]])
							else:
								if trial_parameters['base_color_a'][tii] < 0:
									new_trial_types.extend([1])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
								else:
									new_trial_types.extend([3])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
						if trial_parameters['stimulus_type'][tii] == 1: # green135
							if trial_parameters['base_ori'][tii] == 135:
								new_trial_types.extend([2])

								trial_tasks.extend([trial_parameters['task'][tii]])
								trial_color.extend([trial_parameters['trial_color'][tii]])
								trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
								trial_correct.extend([trial_parameters['correct_answer'][tii]])
								reaction_time.extend([trial_parameters['reaction_time'][tii]])
							else:
								if trial_parameters['base_color_a'][tii] < 0:
									new_trial_types.extend([1])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
								else:
									new_trial_types.extend([3])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
						if trial_parameters['stimulus_type'][tii] == 2: # red45
							if trial_parameters['base_ori'][tii] == 45:
								new_trial_types.extend([2])

								trial_tasks.extend([trial_parameters['task'][tii]])
								trial_color.extend([trial_parameters['trial_color'][tii]])
								trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
								trial_correct.extend([trial_parameters['correct_answer'][tii]])
								reaction_time.extend([trial_parameters['reaction_time'][tii]])
							else:
								if trial_parameters['base_color_a'][tii] > 0:
									new_trial_types.extend([1])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
								else:
									new_trial_types.extend([3])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
						if trial_parameters['stimulus_type'][tii] == 3: # red135
							if trial_parameters['base_ori'][tii] == 135:
								new_trial_types.extend([2])

								trial_tasks.extend([trial_parameters['task'][tii]])
								trial_color.extend([trial_parameters['trial_color'][tii]])
								trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
								trial_correct.extend([trial_parameters['correct_answer'][tii]])
								reaction_time.extend([trial_parameters['reaction_time'][tii]])
							else:
								if trial_parameters['base_color_a'][tii] > 0:
									new_trial_types.extend([1])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
								else:
									new_trial_types.extend([3])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])

		self.events.update({'codes': [0,1,2,3],
							'names': ['base','attended','unattended','both'],
							'coded_trials': np.array(new_trial_types)})

		self.task_data.update({'trial_tasks': trial_tasks,
							   'trial_color': trial_color,
							   'trial_orientation': trial_orientation,
							   'trial_correct': trial_correct,
							   'reaction_time': reaction_time})		

	def sort_events_by_type(self):

		if len(self.events) == 0:
			#self.recode_trial_types()
			self.extract_signal_blocks()

		sorted_times = []

		for etype in np.unique(self.events['coded_trials']): #self.events['codes']:

			# sorted_events.append(self.events['timestamps'][np.array(self.events['coded_trials'])==int(etype)])
			try:
				sorted_times.append(np.extract(np.array(self.events['coded_trials']) == etype, np.array(self.events['timestamps'])))
			except:
				embed()
		
		self.events.update({'sorted_timestamps': sorted_times})

	def run_FIR(self, deconv_interval = None):
		"""
		Estimate Finite Impulse Response function for pupil signal
		"""	

		if deconv_interval is None:
			deconv_interval = self.deconvolution_interval

		# self.sort_events_by_type()

		super(BehaviorAnalyzer, self).run_FIR(deconv_interval)

	# def collect_task_data(self):

	# 	self.load_data()

	# 	self.get_aliases()

	# 	trial_tasks = []
	# 	trial_color = []
	# 	trial_orientation = []
	# 	trial_correct = []

	# 	for alias in self.aliases:

	# 		trial_parameters = self.h5_operator.read_session_data(alias, 'parameters')

	# 		trial_tasks.extend(trial_parameters['task'])
	# 		trial_color.extend(trial_parameters['trial_color'])
	# 		trial_orientation.extend(trial_parameters['trial_orientation'])
	# 		trial_correct.extend(trial_parameters['correct_answer'])

	# 	self.task_data.update({'trial_tasks': trial_tasks,
	# 						   'trial_color': trial_color,
	# 						   'trial_orientation': trial_orientation,
	# 						   'trial_correct': trial_correct})


	def compute_performance(self):
		
		if len(self.events) == 0:
			#self.recode_trial_types()
			self.extract_signal_blocks()

		self.load_data()

		performance = []
		reaction_time = []

		# self.collect_task_data()

		trial_codes = []
		for ii in range(len(self.csv_data)):
			trial_codes.extend([self.recode_trial_code(self.csv_data.iloc[ii])])

		trial_codes = np.array(trial_codes)

		trial_tasks = np.array(self.csv_data['task'])#np.array(self.task_data['trial_tasks'])
		trial_color = abs(np.array(self.csv_data['trial_color']))#abs(np.array(self.task_data['trial_color']))
		trial_orientation = abs(np.array(self.csv_data['trial_orientation']))#abs(np.array(self.task_data['trial_orientation']))
		trial_correct = np.array(self.csv_data['correct_answer'])#self.compute_correct_responses()#np.array(self.task_data['trial_correct'])
		trial_rts = np.array(self.csv_data['reaction_time'])#np.array(self.task_data['reaction_time'])

		# embed()
		# trial_/correct[np.where((trial_tasks == 1) & (trial_color < 0))] = 1-trial_correct[np.where((trial_tasks == 1) & (trial_color < 0))]
		# trial_correct[np.where((trial_tasks == 2) & (trial_orientation < 0))] = 1-trial_correct[np.where((trial_tasks == 2) & (trial_orientation < 0))]

		int_steps = np.array([0.25,0.35,0.5, 0.7, 1.0, 1.41, 2.0, 2.83, 4])

		uniq_ints = np.vstack([np.unique(trial_color), np.unique(trial_orientation)])

		# trial_ints = np.zeros((trial_tasks.size,1))
		# for ii,taskval in enumerate(zip(trial_tasks,trial_color,trial_orientation)):
		# 	trial_ints[ii] = int_steps[taskval[taskval[0]]==uniq_ints[taskval[0]-1]]

		trial_ints = np.array([int_steps[taskval[int(taskval[0])]==uniq_ints[int(taskval[0])-1]] for taskval in zip(trial_tasks,trial_color,trial_orientation)])

		# COLOR

		self.task_performance.update({'pred-v-unpred': [[],[]],
									  'pred-v-unpred-rt': [[],[]]})

		unpred_trials = trial_codes>10
		pred_trials = trial_codes<10

		ints = trial_ints[pred_trials]
		correct = np.extract(pred_trials, trial_correct)
		rts = np.extract(pred_trials, trial_rts)
		self.task_performance['pred-v-unpred'][0] = np.array([[i, np.nanmean(np.extract(ints==i, correct))] for i in np.unique(ints)])	
		self.task_performance['pred-v-unpred-rt'][0] = [[i, np.extract(ints==i, rts)] for i in np.unique(ints)]

		ints = trial_ints[unpred_trials]
		correct = np.extract(unpred_trials, trial_correct)
		rts = np.extract(unpred_trials, trial_rts)
		self.task_performance['pred-v-unpred'][1] = np.array([[i, np.nanmean(np.extract(ints==i, correct))] for i in np.unique(ints)])	
		self.task_performance['pred-v-unpred-rt'][1] = [[i, np.extract(ints==i, rts)] for i in np.unique(ints)]

		for ii in range(2):
			self.task_performance['pred-v-unpred'][ii][self.task_performance['pred-v-unpred'][ii][:,1]<.5,1] = 1-self.task_performance['pred-v-unpred'][ii][self.task_performance['pred-v-unpred'][ii][:,1]<.5,1]



	def compute_performance_pred_unpred(self):

		if len(self.events) == 0:
			#self.recode_trial_types()
			self.extract_signal_blocks()

		self.load_data()

		performance = []
		reaction_time = []

		# self.collect_task_data()

		trial_codes = []
		for ii in range(len(self.csv_data)):
			trial_codes.extend([self.recode_trial_code(self.csv_data.iloc[ii])])

		trial_codes = np.array(trial_codes)

		trial_tasks = np.array(self.csv_data['task'])#np.array(self.task_data['trial_tasks'])
		trial_color = abs(np.array(self.csv_data['trial_color']))#abs(np.array(self.task_data['trial_color']))
		trial_orientation = abs(np.array(self.csv_data['trial_orientation']))#abs(np.array(self.task_data['trial_orientation']))
		trial_correct = np.array(self.csv_data['correct_answer'])#self.compute_correct_responses()#np.array(self.task_data['trial_correct'])
		trial_rts = np.array(self.csv_data['reaction_time'])#np.array(self.task_data['reaction_time'])

		# embed()
		# trial_/correct[np.where((trial_tasks == 1) & (trial_color < 0))] = 1-trial_correct[np.where((trial_tasks == 1) & (trial_color < 0))]
		# trial_correct[np.where((trial_tasks == 2) & (trial_orientation < 0))] = 1-trial_correct[np.where((trial_tasks == 2) & (trial_orientation < 0))]

		# COLOR

		self.task_performance.update({'col_pred-v-unpred': [[],[]]})

		unpred_trials = ((trial_codes/10)%2>0) & (trial_codes>10)
		pred_trials = trial_codes==0

		#ints = np.zeros((len(trial_codes)))
		#ints[np.where(trial_codes[pred_trials]==0)] = np.extract(trial_codes[pred_trials]==0, trial_color) #/ max(trial_color)
		# ints[np.where(trial_codes[pred_trials]==1)] = np.extract(trial_codes[pred_trials]==1, trial_orientation) #/ max(trial_orientation)
		ints = trial_color[pred_trials]
		correct = np.extract(pred_trials, trial_correct)
		# rts = np.extract(pred_trials, trial_rts)
		# embed()
		self.task_performance['col_pred-v-unpred'][0] = np.array([[i, np.nanmean(np.extract(ints==i, correct))] for i in np.unique(ints)])	
		# self.task_performance['RT_pred-v-unpred'][0] = np.array([[i, np.nanmean(np.extract(ints==i, trial_rts))] for i in np.unique(ints)])	

		ints = trial_color[unpred_trials]
		# ints[np.where(((trial_codes[unpred_trials]/10)%2)==0)] = np.extract(((trial_codes[unpred_trials]/10)%2)==0, trial_color) #/ max(trial_color)
		# ints[np.where(((trial_codes[unpred_trials]/10)%2)>0)] = np.extract(((trial_codes[unpred_trials]/10)%2)>0, trial_orientation) #/ max(trial_orientation)
		correct = np.extract(unpred_trials, trial_correct)
		# rts = np.extract(unpred_trials, trial_rts)
		self.task_performance['col_pred-v-unpred'][1] = np.array([[i, np.nanmean(np.extract(ints==i, correct))] for i in np.unique(ints)])	
		# self.task_performance['RT_pred-v-unpred'][1] = np.array([[i, np.nanmean(np.extract(ints==i, trial_rts))] for i in np.unique(ints)])	


		# ORIENTATION

		self.task_performance.update({'ori_pred-v-unpred': [[],[]]})

		unpred_trials = ((trial_codes/10)%2==0) & (trial_codes>10)
		pred_trials = trial_codes==1

		#ints = np.zeros((len(trial_codes)))
		#ints[np.where(trial_codes[pred_trials]==0)] = np.extract(trial_codes[pred_trials]==0, trial_color) #/ max(trial_color)
		# ints[np.where(trial_codes[pred_trials]==1)] = np.extract(trial_codes[pred_trials]==1, trial_orientation) #/ max(trial_orientation)
		ints = trial_orientation[pred_trials]
		correct = np.extract(pred_trials, trial_correct)
		# rts = np.extract(pred_trials, trial_rts)
		self.task_performance['ori_pred-v-unpred'][0] = np.array([[i, np.nanmean(np.extract(ints==i, correct))] for i in np.unique(ints)])	
		# self.task_performance['RT_pred-v-unpred'][0] = np.array([[i, np.nanmean(np.extract(ints==i, trial_rts))] for i in np.unique(ints)])	

		ints = trial_orientation[unpred_trials]
		# ints[np.where(((trial_codes[unpred_trials]/10)%2)==0)] = np.extract(((trial_codes[unpred_trials]/10)%2)==0, trial_color) #/ max(trial_color)
		# ints[np.where(((trial_codes[unpred_trials]/10)%2)>0)] = np.extract(((trial_codes[unpred_trials]/10)%2)>0, trial_orientation) #/ max(trial_orientation)
		correct = np.extract(unpred_trials, trial_correct)
		# rts = np.extract(unpred_trials, trial_rts)
		self.task_performance['ori_pred-v-unpred'][1] = np.array([[i, np.nanmean(np.extract(ints==i, correct))] for i in np.unique(ints)])	
		# self.task_performance['RT_pred-v-unpred'][1] = np.array([[i, np.nanmean(np.extract(ints==i, trial_rts))] for i in np.unique(ints)])	

		for ii in range(2):
			self.task_performance['col_pred-v-unpred'][ii][self.task_performance['col_pred-v-unpred'][ii][:,1]<.5,1] = 1-self.task_performance['col_pred-v-unpred'][ii][self.task_performance['col_pred-v-unpred'][ii][:,1]<.5,1]
			self.task_performance['ori_pred-v-unpred'][ii][self.task_performance['ori_pred-v-unpred'][ii][:,1]<.5,1] = 1-self.task_performance['ori_pred-v-unpred'][ii][self.task_performance['ori_pred-v-unpred'][ii][:,1]<.5,1]
			# self.task_performance['task_pred-v-unpred'][ii][:,0] = np.log10(self.task_performance['task_pred-v-unpred'][ii][:,0]) #/ np.log(2)
			# self.task_performance['task_pred-v-unpred'][ii] = self.task_performance['task_pred-v-unpred'][ii][1:,]

	def compute_performance_attended_unattended(self):


 		self.load_data()

		performance = []
		reaction_time = []

		# self.collect_task_data()
		# embed()
		trial_codes = []
		for ii in range(len(self.csv_data)):
			trial_codes.extend([self.recode_trial_code(self.csv_data.iloc[ii])])

		trial_codes = np.array(trial_codes)

		trial_tasks = np.array(self.csv_data['task'])#np.array(self.task_data['trial_tasks'])
		trial_color = abs(np.array(self.csv_data['trial_color']))#abs(np.array(self.task_data['trial_color']))
		trial_orientation = abs(np.array(self.csv_data['trial_orientation']))#abs(np.array(self.task_data['trial_orientation']))
		trial_correct = np.array(self.csv_data['correct_answer'])#self.compute_correct_responses()#np.array(self.task_data['trial_correct'])
		trial_rts = np.array(self.csv_data['reaction_time'])#np.array(self.task_data['reaction_time'])

		# COLOR

		self.task_performance.update({'col_att-v-unatt': [[],[],[]]})

		pred_unpred_trials = trial_codes == 10
		unpred_pred_trials = trial_codes == 30
		unpred_unpred_trials = trial_codes == 50

		ints = trial_color[pred_unpred_trials]
		correct = np.extract(pred_unpred_trials, trial_correct)
		self.task_performance['col_att-v-unatt'][0] = np.array([[i, np.nanmean(np.extract(ints==i, correct))] for i in np.unique(ints)])	

		ints = trial_color[unpred_pred_trials]
		correct = np.extract(unpred_pred_trials, trial_correct)
		self.task_performance['col_att-v-unatt'][1] = np.array([[i, np.nanmean(np.extract(ints==i, correct))] for i in np.unique(ints)])

		ints = trial_color[unpred_unpred_trials]
		correct = np.extract(unpred_unpred_trials, trial_correct)
		self.task_performance['col_att-v-unatt'][2] = np.array([[i, np.nanmean(np.extract(ints==i, correct))] for i in np.unique(ints)])

		# ORIENTATION

		self.task_performance.update({'ori_att-v-unatt': [[],[],[]]})

		pred_unpred_trials = trial_codes == 20
		unpred_pred_trials = trial_codes == 40
		unpred_unpred_trials = trial_codes == 60

		ints = trial_orientation[pred_unpred_trials]
		correct = np.extract(pred_unpred_trials, trial_correct)
		self.task_performance['ori_att-v-unatt'][0] = np.array([[i, np.nanmean(np.extract(ints==i, correct))] for i in np.unique(ints)])	

		ints = trial_orientation[unpred_pred_trials]
		correct = np.extract(unpred_pred_trials, trial_correct)
		self.task_performance['ori_att-v-unatt'][1] = np.array([[i, np.nanmean(np.extract(ints==i, correct))] for i in np.unique(ints)])

		ints = trial_orientation[unpred_unpred_trials]
		correct = np.extract(unpred_unpred_trials, trial_correct)
		self.task_performance['ori_att-v-unatt'][2] = np.array([[i, np.nanmean(np.extract(ints==i, correct))] for i in np.unique(ints)])
		# embed()
		for ii in range(3):
			self.task_performance['col_att-v-unatt'][ii][self.task_performance['col_att-v-unatt'][ii][:,1]<.5,1] = 1-np.extract(self.task_performance['col_att-v-unatt'][ii][:,1]<.5, self.task_performance['col_att-v-unatt'][ii][:,1])
			self.task_performance['ori_att-v-unatt'][ii][self.task_performance['ori_att-v-unatt'][ii][:,1]<.5,1] = 1-np.extract(self.task_performance['ori_att-v-unatt'][ii][:,1]<.5, self.task_performance['ori_att-v-unatt'][ii][:,1])
			# self.task_performance['task_pred-v-unpred'][ii][:,0] = np.log10(self.task_performance['task_pred-v-unpred'][ii][:,0]) #/ np.log(2)
			# self.task_performance['task_pred-v-unpred'][ii] = self.task_performance['task_pred-v-unpred'][ii][1:,]

	# def compute_performance_attended_unattended(self):

	# 	if len(self.events) == 0:
	# 		#self.recode_trial_types()
	# 		self.extract_signal_blocks()

	# 	self.load_data()

	# 	performance = []
	# 	reaction_time = []

	# 	# self.collect_task_data()

	# 	trial_codes = self.task_data['coded_trials']

	# 	# trial_codes[trial_codes==10 or trial_codes==11) = 10
	# 	# trial_codes(trial_codes==20 or trial_codes==21) = 11
	# 	# trial_codes(trial_codes==30 or trial_codes==31) = 12

	# 	trial_tasks = np.array(self.task_data['trial_tasks'])
	# 	trial_color = abs(np.array(self.task_data['trial_color']))
	# 	trial_orientation = abs(np.array(self.task_data['trial_orientation']))
	# 	trial_correct = np.array(self.task_data['trial_correct'])
	# 	trial_rts = np.array(self.task_data['reaction_time'])


	# 	# color_intensities = abs(np.unique(trial_color))
	# 	# orientation_intensities = abs(np.unique(trial_orientation))

	# 	self.task_performance.update({'task_att-v-unatt': [[],[],[]],
	# 							 	  'RT_att-v-unatt': [[],[],[]]})


	# 	pred_unpred_trials = (trial_codes >= 10) & (trial_codes <= 20)
	# 	unpred_pred_trials = (trial_codes >= 30) & (trial_codes <= 40)
	# 	unpred_unpred_trials = (trial_codes >= 50) & (trial_codes <= 60)

	# 	ints = np.zeros((len(trial_codes)))
	# 	ints[np.where(trial_codes[pred_unpred_trials]==0)] = np.extract(trial_codes[pred_unpred_trials]==0, trial_color)# / max(trial_color)
	# 	ints[np.where(trial_codes[pred_unpred_trials]==1)] = np.extract(trial_codes[pred_unpred_trials]==1, trial_orientation)# #/ max(trial_orientation)
	# 	correct = np.extract(pred_unpred_trials, trial_correct)
	# 	rts = np.extract(pred_unpred_trials, trial_rts)
	# 	self.task_performance['task_att-v-unatt'][0] = np.array([[i, np.mean(np.extract(ints==i, trial_correct))] for i in np.unique(ints)])	
	# 	self.task_performance['RT_att-v-unatt'][0] = np.array([[i, np.mean(np.extract(ints==i, trial_rts))] for i in np.unique(ints)])	

	# 	ints = np.zeros((len(trial_codes)))
	# 	ints[np.where(((trial_codes[unpred_pred_trials]/10)%2)==0)] = np.extract(((trial_codes[unpred_pred_trials]/10)%2)==0, trial_color) #/ max(trial_color)
	# 	ints[np.where(((trial_codes[unpred_pred_trials]/10)%2)>0)] = np.extract(((trial_codes[unpred_pred_trials]/10)%2)>0, trial_orientation)# / max(trial_orientation)
	# 	correct = np.extract(unpred_pred_trials, trial_correct)
	# 	rts = np.extract(unpred_pred_trials, trial_rts)
	# 	self.task_performance['task_att-v-unatt'][1] = np.array([[i, np.mean(np.extract(ints==i, trial_correct))] for i in np.unique(ints)])	
	# 	self.task_performance['RT_att-v-unatt'][1] = np.array([[i, np.mean(np.extract(ints==i, trial_rts))] for i in np.unique(ints)])		

	# 	ints = np.zeros((len(trial_codes)))
	# 	ints[np.where(((trial_codes[unpred_unpred_trials]/10)%2)==0)] = np.extract(((trial_codes[unpred_unpred_trials]/10)%2)==0, trial_color) #/ max(trial_color)
	# 	ints[np.where(((trial_codes[unpred_unpred_trials]/10)%2)>0)] = np.extract(((trial_codes[unpred_unpred_trials]/10)%2)>0, trial_orientation)# / max(trial_orientation)
	# 	correct = np.extract(unpred_unpred_trials, trial_correct)
	# 	rts = np.extract(unpred_unpred_trials, trial_rts)
	# 	self.task_performance['task_att-v-unatt'][2] = np.array([[i, np.mean(np.extract(ints==i, trial_correct))] for i in np.unique(ints)])	
	# 	self.task_performance['RT_att-v-unatt'][2] = np.array([[i, np.mean(np.extract(ints==i, trial_rts))] for i in np.unique(ints)])			

	# 	for ii in range(3):
	# 		self.task_performance['task_att-v-unatt'][ii][self.task_performance['task_att-v-unatt'][ii][:,1]<.5,1] = 1-self.task_performance['task_att-v-unatt'][ii][self.task_performance['task_att-v-unatt'][ii][:,1]<.5,1]
	# 		self.task_performance['task_att-v-unatt'][ii][:,0] = np.log10(self.task_performance['task_att-v-unatt'][ii][:,0])# / np.log(2)
	# 		self.task_performance['task_att-v-unatt'][ii] = self.task_performance['task_att-v-unatt'][ii][1:,]

	def compute_correct_responses(self):
		
		# Color task
		color_iis = self.task_data['trial_tasks'] == 1






	def fit_cum_vonmises(self):
		"""
		Fit Von Mises distribution to data
		"""
		params = []

		# embed()

		for ii in range(len(self.task_performance['percent_correct'])):
			try:
				params.append(sp.stats.vonmises.fit(self.task_performance['percent_correct'][ii]))
			except:
				embed()

		self.task_performance.update({'von_mises_params': params})


	def fit_cum_gauss(self, dataset = ''):

		if 'psych_curve_params' not in self.task_performance.keys():
			self.task_performance.update({'psych_curve_params': {}})

		self.task_performance['psych_curve_params'].update({dataset: []})

		if dataset[0:3] == 'col':
			refset = 'col_pred-v-unpred'
		elif dataset[0:3] == 'ori':
			refset = 'ori_pred-v-unpred'
		else:
			refset = 'pred-v-unpred'

		# embed()

		for trial_type in range(len(self.task_performance[dataset])):
			try:
				if trial_type == 0:
					self.task_performance['psych_curve_params'][dataset].append(sp.optimize.curve_fit(sp.stats.norm.cdf, self.task_performance[dataset][trial_type][:,0], self.task_performance[dataset][trial_type][:,1], p0=[0.5,1], bounds=(0,np.Inf))[0])
				else:
					self.task_performance['psych_curve_params'][dataset].append(sp.optimize.curve_fit(sp.stats.norm.cdf, self.task_performance[dataset][trial_type][:,0], self.task_performance[dataset][trial_type][:,1], p0=self.task_performance['psych_curve_params'][refset][0], bounds=(0,np.Inf))[0])
				# self.task_performance['cum_gauss_params']['orientation_task'].append(sp.optimize.curve_fit(sp.stats.norm.cdf, self.task_performance['orientation_task'][trial_type][:,0], self.task_performance['orientation_task'][trial_type][:,1], p0=[0,0.5])[0])
			except:
				embed()


	def fit_sig_fun(self, dataset = ''):

		if 'psych_curve_params' not in self.task_performance.keys():
			self.task_performance.update({'psych_curve_params': {}})

		self.task_performance['psych_curve_params'].update({dataset: []})


		if dataset[0:3] == 'col':
			refset = 'col_pred-v-unpred'
		elif dataset[0:3] == 'ori':
			refset = 'ori_pred-v-unpred'
		else:
			refset = 'pred-v-unpred'

		# embed()

		for trial_type in range(len(self.task_performance[dataset])):
			try:
				if trial_type == 0:
					self.task_performance['psych_curve_params'][dataset].append(sp.optimize.curve_fit(self.sigmoid, self.task_performance[dataset][trial_type][:,0], self.task_performance[dataset][trial_type][:,1], p0=[0,1,0.01], bounds=(0,[np.Inf, np.Inf, np.Inf]))[0])
				else:
					self.task_performance['psych_curve_params'][dataset].append(sp.optimize.curve_fit(self.sigmoid, self.task_performance[dataset][trial_type][:,0], self.task_performance[dataset][trial_type][:,1], p0=self.task_performance['psych_curve_params'][refset][0], bounds=(0,[np.Inf, np.Inf, np.Inf]))[0])
				# self.task_performance['cum_gauss_params']['orientation_task'].append(sp.optimize.curve_fit(sp.stats.norm.cdf, self.task_performance['orientation_task'][trial_type][:,0], self.task_performance['orientation_task'][trial_type][:,1], p0=[0,0.5])[0])
			except:
				embed()

	def sigmoid(self, x, a, b, l):
		g = 0.5
		# l = 0.1
		# l = 0
		return g + (1-g-l)/(1+np.exp(-(x-a)/b))

	def fit_psycho(self):

		params = []

		embed()

		base_color_ints = np.extract((trial_codes == 0) & (trial_tasks == 1.0), trial_color)
		base_color_correct = np.extract((trial_codes == 0) & (trial_tasks == 1.0), trial_correct)

		base_color_data = [[i, np.mean(np.extract(base_color_ints==i, base_color_correct)), np.sum(base_color_ints==i)] for i in np.unique(base_color_ints)]

		#base_color_data =  map(lambda i: [i, np.mean(np.extract(base_color_ints==i, base_color_correct)), np.sum(base_color_ints==i)], np.unique(base_color_ints))

		base_ori_ints = np.extract((trial_codes == 0) & (trial_tasks == 2.0), trial_orientation)
		base_ori_correct = np.extract((trial_codes == 0) & (trial_tasks == 2.0), trial_correct)

		base_ori_data = [[i, np.mean(np.extract(base_ori_ints==i, base_ori_correct)), np.sum(base_ori_ints==i)] for i in np.unique(base_ori_ints)]

		nafc = 2
		constraints = ( 'unconstrained', 'unconstrained', 'Beta(2,20)' )

		B_single_sessions = psi.BootstrapInference ( data_single_sessions, priors=constraints, nafc=nafc )

	def store_behavior(self):
		# Simply store the relevant variables to save speed
		print '[%s] Storing behavioural data' % (self.__class__.__name__)
		#pickle.dump([self.task_data,self.events,self.task_performance,self.trial_signals],open(os.path.join(self.data_folder, self.output_filename),'wb'))
		pickle.dump([self.task_data,self.events,self.task_performance],open(os.path.join(self.data_folder, 'behavior_' + self.output_filename),'wb'))