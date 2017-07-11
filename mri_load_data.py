from __future__ import division
import os,sys,glob
import numpy as np
import scipy.stats as stats
from scipy.stats.stats import pearsonr
from scipy.signal import fftconvolve
from scipy import ndimage
import scipy

import nibabel as nib
import pickle
from IPython import embed as shell
import matplotlib.pyplot as plt
import pandas as pd
import copy


def voxel_filter(target_files_fmri, lh, rh):
	# #check nan values	
	nan_voxels = []
	same_voxels = []
	for filename_fmri in target_files_fmri:
		unmasked_fmri_data = nib.load(filename_fmri).get_data()
		fmri_data = np.vstack([unmasked_fmri_data[lh,:], unmasked_fmri_data[rh,:]])

		# if filename_fmri == target_files_fmri[0]:
		# 	fmri_data_first_run = copy.copy(fmri_data)

		# 	fmri_data_first_run_lh = unmasked_fmri_data[lh,:]
		#  	fmri_data_first_run_rh = unmasked_fmri_data[rh,:]

		n_all_voxels = fmri_data.shape[0] # 5728, 5728, 5728, 5728
		print 'n_all_voxels:', n_all_voxels 
		
		nan_voxels.extend(np.argwhere(fmri_data.sum(axis=1)==0).flatten())

	print 'found %i nan voxels'%len(nan_voxels)

	voxel_list_without_nan =[]
	for i in range(n_all_voxels):
		if i not in nan_voxels:
			voxel_list_without_nan.append(i)
	print 'finish checking nan values'



	# for filename_fmri in target_files_fmri:
	# 	unmasked_fmri_data = nib.load(filename_fmri).get_data()
	# 	fmri_data = np.vstack([unmasked_fmri_data[lh,:], unmasked_fmri_data[rh,:]])[voxel_list_without_nan,:]

		# if filename_fmri == target_files_fmri[0]:
	# for i in voxel_list_without_nan:
	# 	for j in voxel_list_without_nan:
	# 		if i != j:
	# 			b = pearsonr(fmri_data[i], fmri_data[j])[0]
	# 			if b ==1:
	# 				print 'find you! (%i, %i)' % (i, j)
	# 				same_voxels.append((i, j))
	# 			else:
	# 				print i, j, b


	for i in voxel_list_without_nan :
		for j in voxel_list_without_nan :
			if i < j:
				a = pearsonr(fmri_data[i], fmri_data[j])[0]
				if a ==1:
					print 'find you! (%i, %i)' % (i, j)
					same_voxels.append((i,j))
				else:
					print i,j,a
	print 'found %i doubled voxels'% len(same_voxels)

	duplicated_voxels = np.array(same_voxels)[:, 1] #.flatten()
	voxel_list = []
	for i in voxel_list_without_nan:
		if i not in duplicated_voxels:
			voxel_list.append(i)
	print 'finish checking freesurfer masks'


	# voxel_list = []
	# for i in voxel_list_without_nan:
	# 	if i not in same_voxels:
	# 		voxel_list.append(i)
	# print 'finish checking freesurfer masks'

	return voxel_list




def load_fmri(filename_fmri, voxel_list, lh, rh):
	unmasked_fmri_data = nib.load(filename_fmri).get_data()
	fmri_data = np.vstack([unmasked_fmri_data[lh,:], unmasked_fmri_data[rh,:]])[voxel_list,:]
	# another way to flatten ---- e.g. moco_params.reshape(-1, moco_params.shape[-1])

	return fmri_data

def load_event_64channels (filename_beh, fmri_data, empty_start = 15, empty_end = 15, number_of_stimuli = 64):
	empty_start = 15
	empty_end = 15
	number_of_stimuli = 64

	trial_order_run = pickle.load(open(filename_beh, 'rb'))[1]
	#create events with 1
	tmp_trial_order_run  = np.zeros((fmri_data.shape[1],1))
	#15 + 256( 2* 128) +15 =286
	tmp_trial_order_run[empty_start:-empty_end:2] = trial_order_run[:]+1 # [:,np.newaxis]+1
	events = np.hstack([np.array(tmp_trial_order_run == stim, dtype=int) for stim in np.arange(1,number_of_stimuli+1)])

	return events

def load_event_16channels (filename_beh, fmri_data, empty_start = 15, empty_end = 15, number_of_stimuli = 8):
	trial_order_run = pickle.load(open(filename_beh, 'rb'))[1]
	#  the first 8 values represent all orientations but only 1color
	empty_start = 15
	empty_end = 15
	number_of_stimuli = 8
	## for trial_order_col------------------------------------
	trial_order_col = np.zeros((len(trial_order_run),))

	for ii, stim_nr in enumerate(trial_order_run) :
		if (stim_nr >= 0 ) * (stim_nr < (8*1))  :
			trial_order_col[ii] = 1			
		elif (stim_nr >= (8*1)) and (stim_nr < (8*2)) :
			trial_order_col[ii] = 2
		elif (stim_nr >= (8*2)) and (stim_nr < (8*3)) :
			trial_order_col[ii] = 3
		elif (stim_nr >= (8*3)) and (stim_nr < (8*4)) :
			trial_order_col[ii] = 4
		elif (stim_nr >= (8*4)) and (stim_nr < (8*5)) :
			trial_order_col[ii] = 5
		elif (stim_nr >= (8*5)) and (stim_nr < (8*6)) :
			trial_order_col[ii] = 6							
		elif (stim_nr >= (8*6)) and (stim_nr < (8*7)) :
			trial_order_col[ii] = 7
		elif (stim_nr >= (8*7)) and (stim_nr < (8*8)) :
			trial_order_col[ii] = 8		

	trial_order_col = trial_order_col[:, np.newaxis]

	#create events with 1
	# empty_start = 15
	# empty_end = 15
	# number_of_stimuli = 8  # 64

	tmp_trial_order_col  = np.zeros((fmri_data.shape[1],1))
	#15 + 256( 2* 128) +15 =286, (286,1)
	tmp_trial_order_col[empty_start:-empty_end:2] = trial_order_col[:]   
	events_col = np.hstack([np.array(tmp_trial_order_col == stim, dtype=int) for stim in np.arange(1,number_of_stimuli+1)])


	## for trial_order_col------------------------------------
	trial_order_ori = np.zeros((len(trial_order_run),))

	for ii, stim_nr in enumerate(trial_order_run):

		if stim_nr < 64:
			trial_order_ori[ii] = stim_nr % 8 + 1
		else:
			trial_order_ori[ii] = 0

	trial_order_ori = trial_order_ori[:, np.newaxis]

	#create events with 1
	tmp_trial_order_ori  = np.zeros((fmri_data.shape[1],1))
	#15 + 256( 2* 128) +15 =286, (286,1)  new: 15+ 512(2*256) +15 =542
	tmp_trial_order_ori[empty_start:-empty_end:2] = trial_order_ori[:]   
	events_ori = np.hstack([np.array(tmp_trial_order_ori == stim, dtype=int) for stim in np.arange(1,number_of_stimuli+1)])


	events = np.hstack([events_ori, events_col])  # orientation + col (542,16)

	return events_ori, events_col



def load_key_press_regressor (filename_fixation, fmri_data): 
	fixation_order_run = pickle.load(open(filename_fixation, 'rb'))
	eventArray = fixation_order_run['eventArray']  # a list of lists
	key_press = np.zeros((fmri_data.shape[1],1))
	## old data set
	# for n_event, event in enumerate (eventArray):
	# 	for txt in event:
	# 		if 'key: y' in txt:
	# 			key_press[n_event] = 1

	# 		elif 'key: b' in txt:
	# 			key_press[n_event] = 1

	## new standard dataset
	for event in eventArray:
		# for txt in event:
			if 'key: y' in event:
				n_event = int(event.split( )[1])
				key_press[n_event] = 1

			elif 'key: b' in event:
				n_event = int(event.split( )[1])
				key_press[n_event] = 1
	return key_press

def load_stimuli_regressor (filename_beh, fmri_data, empty_start = 15, empty_end = 15): 
	trial_order_run = pickle.load(open(filename_beh, 'rb'))[1]
	trial_order_di = np.squeeze(trial_order_run[:]+1)
	trial_order_di[trial_order_di<=64] = 1
	trial_order_di[trial_order_di>64] = 0

	stim_regressor  = np.zeros((fmri_data.shape[1],1))
	#15 + 256( 2* 128) +15 =286
	stim_regressor[empty_start:-empty_end:2] = trial_order_di[:, np.newaxis] # [:,np.newaxis]+1

	return stim_regressor


