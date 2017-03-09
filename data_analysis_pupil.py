# Pupil analysis

import os,sys

from IPython import embed as shell

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

sys.path.append( 'utils' )

# Import hedfpy functions
# from hedfpy import EDFOperator
# from hedfpy import EyeSignalOperator
# from hedfpy import HDFEyeOperator

# Martijns analysis tools
# import Analyzer
# from BehaviorAnalyzer import BehaviorAnalyzer
from PupilAnalyzer import PupilAnalyzer # from the file import the method, the file is in the folder named 'utils'

# sublist = ['MS', 'SL']
# def load_eye_data():

sublist = ['MS', 'SL'] #, SL'
subID = 'MS'

# def load_eye_data (subID)

# 	data_folder = '/home/xiaomeng/Data/Pre_scan_data/' + str(subID) + '/'

# 	figure_dir = '/home/xiaomeng/Data/Pre_scan_data/pupil_fig/' 

# 	h5filename = os.path.join(data_folder, subID + '.h5')

# 	pa = PupilAnalyzer(subID, h5filename, data_folder) # create an object to store output of PupilAnalyzer Class

# 	pa.load_data() # object.method() >> create the h5 file

# 	pa.get_aliases()

# 	return pa

def get_minlength_sub(pa):
	# get minlength_run over runs, for one participant
	minlength_sub = []
	for aliasii, alias in enumerate(pa.aliases):
		#alias = pa.aliases[3] # get the first run

		trial_time_info = pa.h5_operator.read_session_data(alias, 'trials')

		trial_start_time = trial_time_info['trial_start_EL_timestamp']
		trial_end_time = trial_time_info['trial_end_EL_timestamp']

		trial_time = zip(trial_start_time, trial_end_time)

		minlength_run = np.int(np.min(trial_end_time - trial_start_time)-1)# minimum trial length +1 or not (3590,) (3589,)
		minlength_sub.append(minlength_run)

	minlength_sub = np.min(minlength_sub)

	return minlength_sub


def plot_trial_signal_mean_each_run(pa):
	for ts, te in trial_time:

		# np.squeeze gives you an array, so don't need np.array(np.squeeze(xxx))
		pupil_signal = np.squeeze(pa.h5_operator.signal_during_period(time_period = [ts, te], alias = alias, signal = 'pupil_bp_clean_psc', requested_eye = pa.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0]))
		
		if len(pupil_signal) >= minlength_sub:
			pupil_signal = pupil_signal[0: minlength_sub]

		trial_signal_run.append(pupil_signal)
	return trial_signal_run



#''' try to use functions'''
for subii,subID in enumerate(sublist):
	
	print '[main] Running analysis for %s' % (subID)

	data_folder = '/home/xiaomeng/Data/Pre_scan_data/' + str(subID) + '/'

	figure_dir = '/home/xiaomeng/Data/Pre_scan_data/pupil_fig/' 

	h5filename = os.path.join(data_folder, subID + '.h5')

	pa = PupilAnalyzer(subID, h5filename, data_folder) # create an object

	pa.load_data() # object.method() >> create the h5 file

	pa.get_aliases()

	minlength_sub = get_minlength_sub(pa)

	trial_signal_sub = []
	for aliasii, alias in enumerate(pa.aliases):
		#alias = pa.aliases[3] # get the first run

		trial_time_info = pa.h5_operator.read_session_data(alias, 'trials')

		trial_start_time = trial_time_info['trial_start_EL_timestamp']
		trial_end_time = trial_time_info['trial_end_EL_timestamp']

		trial_time = zip(trial_start_time, trial_end_time)

		trial_signal_run = []

		#trial_signal_run = plot_trial_signal_mean_each_run()
		for ts, te in trial_time:
			# np.squeeze gives you an array, so don't need np.array(np.squeeze(xxx))
			pupil_signal = np.squeeze(pa.h5_operator.signal_during_period(time_period = [ts, te], alias = alias, signal = 'pupil_bp_clean_psc', requested_eye = pa.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0]))
			
			if len(pupil_signal) >= minlength_sub:
				pupil_signal = pupil_signal[0: minlength_sub]

			trial_signal_run.append(pupil_signal)

		trial_signal_run_mean = np.mean (trial_signal_run, axis = 0) # mean signal for one run

		f = plt.figure(figsize = (8,6))
		plt.plot(trial_signal_run_mean)
		plt.savefig(figure_dir + '%s%d_pupil_size_plot.jpg'%(subID,aliasii))
		print 'finish run %d'%(aliasii)
		#plt.show()

		#shell()
		trial_signal_sub.extend(np.array(trial_signal_run)) # create a list of basic item: trial signal, don't generate the run dimension>> use extend >> get elements

	print 'yeah! all length are the same'
	trial_signal_sub_mean = np.mean(trial_signal_sub, axis =0) #shapes (3588,) (3589,) 

	#shell()
	f = plt.figure(figsize = (8,6))
	plt.plot(trial_signal_sub_mean)
	plt.savefig(figure_dir + '%s_mean_pupil_size_plot.jpg'%(subID))

	print 'finish one participant !!!'
	#plt.show()





# #''' loop over participants, to get the mean trial signal for each participant'''
# for subii,subID in enumerate(sublist):
	
# 	print '[main] Running analysis for %s' % (subID)

# 	data_folder = '/home/xiaomeng/Data/Pre_scan_data/' + str(subID) + '/'

# 	figure_dir = '/home/xiaomeng/Data/Pre_scan_data/pupil_fig/' 

# 	h5filename = os.path.join(data_folder, subID + '.h5')

# 	pa = PupilAnalyzer(subID, h5filename, data_folder) # create an object

# 	pa.load_data() # object.method() >> create the h5 file

# 	pa.get_aliases()

# 	# get minlength_run over runs, for one participant
# 	minlength_sub = []
# 	for aliasii, alias in enumerate(pa.aliases):
# 		#alias = pa.aliases[3] # get the first run

# 		trial_time_info = pa.h5_operator.read_session_data(alias, 'trials')

# 		trial_start_time = trial_time_info['trial_start_EL_timestamp']
# 		trial_end_time = trial_time_info['trial_end_EL_timestamp']

# 		trial_time = zip(trial_start_time, trial_end_time)

# 		minlength_run = np.int(np.min(trial_end_time - trial_start_time)-1)# minimum trial length +1 or not (3590,) (3589,)
# 		minlength_sub.append(minlength_run)

# 	minlength_sub = np.min(minlength_sub)


# 	trial_signal_sub = []
# 	for aliasii, alias in enumerate(pa.aliases):
# 		#alias = pa.aliases[3] # get the first run

# 		trial_time_info = pa.h5_operator.read_session_data(alias, 'trials')

# 		trial_start_time = trial_time_info['trial_start_EL_timestamp']
# 		trial_end_time = trial_time_info['trial_end_EL_timestamp']

# 		trial_time = zip(trial_start_time, trial_end_time)

# 		#minlength_run = np.int(np.min(trial_end_time - trial_start_time)-1)# minimum trial length +1 or not (3590,) (3589,)

# 		trial_signal_run = []

# 		for ts, te in trial_time:

# 			# np.squeeze gives you an array, so don't need np.array(np.squeeze(xxx))
# 			pupil_signal = np.squeeze(pa.h5_operator.signal_during_period(time_period = [ts, te], alias = alias, signal = 'pupil_bp_clean_psc', requested_eye = pa.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0]))
			
# 			if len(pupil_signal) >= minlength_sub:
# 				pupil_signal = pupil_signal[0: minlength_sub]

# 			trial_signal_run.append(pupil_signal)

# 		trial_signal_run_mean = np.mean (trial_signal_run, axis = 0) # mean signal for one run

# 		f = plt.figure(figsize = (8,6))
# 		plt.plot(trial_signal_run_mean)
# 		plt.savefig(figure_dir + '%s%d_pupil_size_plot.jpg'%(subID,aliasii))
# 		print 'finish run %d'%(aliasii)
# 		#plt.show()

# 		#shell()
# 		trial_signal_sub.extend(np.array(trial_signal_run)) # create a list of basic item: trial signal, don't generate the run dimension>> use extend >> get elements


# 	for ii, item in enumerate(trial_signal_sub):
# 		if item.shape != (3588,):
# 			print ii, item.shape



# 	print 'yeah! all length are the same'
# 	trial_signal_sub_mean = np.mean(trial_signal_sub, axis =0) #shapes (3588,) (3589,) 

# 	#shell()
# 	f = plt.figure(figsize = (8,6))
# 	plt.plot(trial_signal_sub_mean)
# 	plt.savefig(figure_dir + '%s_mean_pupil_size_plot.jpg'%(subID))

# 	print 'finish one participant !!!'
# 	#plt.show()


# ''' do it for one separate participant'''

# print '[main] Running analysis for %s' % (subID)

# data_folder = '/home/xiaomeng/Data/Pre_scan_data/' + str(subID) + '/'

# figure_dir = '/home/xiaomeng/Data/Pre_scan_data/pupil_fig/' 

# h5filename = os.path.join(data_folder, subID + '.h5')

# pa = PupilAnalyzer(subID, h5filename, data_folder) # create an object

# pa.load_data() # object.method() >> create the h5 file

# pa.get_aliases()


# # get minlength_run over runs, for one participant
# minlength_sub = []
# for aliasii, alias in enumerate(pa.aliases):
# 	#alias = pa.aliases[3] # get the first run

# 	trial_time_info = pa.h5_operator.read_session_data(alias, 'trials')

# 	trial_start_time = trial_time_info['trial_start_EL_timestamp']
# 	trial_end_time = trial_time_info['trial_end_EL_timestamp']

# 	trial_time = zip(trial_start_time, trial_end_time)

# 	minlength_run = np.int(np.min(trial_end_time - trial_start_time)-1)# minimum trial length +1 or not (3590,) (3589,)
# 	minlength_sub.append(minlength_run)

# minlength_sub = np.min(minlength_sub)


# trial_signal_sub = []
# for aliasii, alias in enumerate(pa.aliases):
# 	#alias = pa.aliases[3] # get the first run

# 	trial_time_info = pa.h5_operator.read_session_data(alias, 'trials')

# 	trial_start_time = trial_time_info['trial_start_EL_timestamp']
# 	trial_end_time = trial_time_info['trial_end_EL_timestamp']

# 	trial_time = zip(trial_start_time, trial_end_time)

# 	#minlength_run = np.int(np.min(trial_end_time - trial_start_time)-1)# minimum trial length +1 or not (3590,) (3589,)

# 	trial_signal_run = []

# 	for ts, te in trial_time:

# 		# np.squeeze gives you an array, so don't need np.array(np.squeeze(xxx))
# 		pupil_signal = np.squeeze(pa.h5_operator.signal_during_period(time_period = [ts, te], alias = alias, signal = 'pupil_bp_clean_psc', requested_eye = pa.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0]))
		
# 		if len(pupil_signal) >= minlength_sub:
# 			pupil_signal = pupil_signal[0: minlength_sub]

# 		trial_signal_run.append(pupil_signal)

# 	trial_signal_run_mean = np.mean (trial_signal_run, axis = 0) # mean signal for one run

# 	f = plt.figure(figsize = (8,6))
# 	plt.plot(trial_signal_run_mean)
# 	plt.savefig(figure_dir + '%s%d_pupil_size_plot.jpg'%(subID,aliasii))
# 	print 'finish run %d'%(aliasii)
# 	#plt.show()

# 	#shell()
# 	trial_signal_sub.extend(np.array(trial_signal_run)) # create a list of basic item: trial signal, don't generate the run dimension>> use extend >> get elements


# for ii, item in enumerate(trial_signal_sub):
# 	if item.shape != (3588,):
# 		print ii, item.shape



# print 'yeah! all length are the same'
# trial_signal_sub_mean = np.mean(trial_signal_sub, axis =0) #shapes (3588,) (3589,) 

# #shell()
# f = plt.figure(figsize = (8,6))
# plt.plot(trial_signal_sub_mean)
# plt.savefig(figure_dir + '%s_mean_pupil_size_plot.jpg'%(subID))
# #plt.show()

