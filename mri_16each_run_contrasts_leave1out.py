from __future__ import division
import os,sys,glob

import numpy as np
import scipy.stats as stats
from scipy.signal import fftconvolve
import nibabel as nib
import pickle

from Staircase import ThreeUpOneDownStaircase
from tools import two_gamma as hrf

from IPython import embed as shell
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib.gridspec import GridSpec
from scipy import ndimage

from scipy.stats.stats import pearsonr
import pandas as pd

from sklearn.linear_model import RidgeCV
from fractions import Fraction
import scipy
import re
import ColorTools as ct


def roate_90_clockwise ( myarray ):

	x = np.arange(0, len(myarray) )
	y = myarray

	x_new = y
	y_new = len(myarray)-1 -x 

	# ax.set_xticklabels([7,6,5,4,3,2,1,0])

	plt.plot(x_new, y_new)

##-------------------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------------------
 # without def 


# # Load V1 mask
# data_dir_masks = '/home/shared/2017/visual/OriColorMapper/preproc/sub-002/masks/dc/'
# lh = np.array(nib.load(os.path.join(data_dir_masks, 'lh.V1.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)
# rh = np.array(nib.load(os.path.join(data_dir_masks, 'rh.V1.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)


sublist = [ ('sub-004', True, False), ('sub-001', False, True), ('sub-003', False, True), ]#, ('sub-001', False, True), ('sub-003', False, True)]#[('sub-001', False, True) ] [('sub-002', True, False)], [('sub-004', True, False)] [('sub-003', False, True) ]
#sublist = ['sub-001','sub-002']


data_dir_fmri = '/home/barendregt/disks/Aeneas_Shared/2017/visual/OriColorMapper/preproc/'#'/home/shared/2017/visual/OriColorMapper/preproc/'
#	data_dir_fmri = '/home/shared/2017/visual/OriColorMapper/preproc/sub-002/psc/'
data_dir_beh = '/home/barendregt/disks/Aeneas_Shared/2017/visual/OriColorMapper/bids_converted/'#/home/shared/2017/visual/OriColorMapper/bids_converted/'
#	data_dir_beh = '/home/shared/2017/visual/OriColorMapper/bids_converted/sub-002/func/'
#/Users/xiaomeng/subjects/XY_01052017/mri/brainmask.mgz  #or T1.mgz
data_dir_fixation = '/home/barendregt/disks/Aeneas_Shared/2017/visual/OriColorMapper/raw/'#/home/shared/2017/visual/OriColorMapper/raw/'

# get fullfield files
# sub-002_task-location_run-1_bold_brain_B0_volreg_sg.nii.gz
# sub-002_task-fullfield_run-2.pickle
# sub-002_task-fullfield_output_run-2.pickle ---output files

data_type = 'psc'#'tf'#'tf' #'psc'
each_run = True #False #True #False
ROI = 'V1' # 'V4'
regression = 'RidgeCV' #'GLM' #'RidgeCV'
# type_contrasts = 'full' # 'ori', 'color', 'full'
position_cen = 2 #'nan' #2 4  #'nan'



for subii, sub in enumerate(sublist):

	subname = sub[0]
	retinotopic = sub[1]
	exvivo = sub[2]
	
	print '[main] Running analysis for %s' % (str(subname))
	print ROI, regression, position_cen
	# '%s_%s_%s_%s_tValues_overVoxels.png'%(subname, ROI, data_type, regression)
	subject_dir_fmri= os.path.join(data_dir_fmri,subname)

	fmri_files = glob.glob(subject_dir_fmri  + '/' + data_type + '/*.nii.gz')
	fmri_files.sort()

	moco_files = glob.glob(subject_dir_fmri + '/mcf/parameter_info' + '/*.1D')
	moco_files.sort()

	#respiration&heart rate
		
	if retinotopic == True:
		if ROI == 'V1':
			lh = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/ret/lh.V1_vol_dil.nii.gz')).get_data(), dtype=bool)
			rh = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/ret/rh.V1_vol_dil.nii.gz')).get_data(), dtype=bool)
		elif ROI =='V4':
			lh = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/ret/lh.V4_vol_dil.nii.gz')).get_data(), dtype=bool)
			rh = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/ret/rh.V4_vol_dil.nii.gz')).get_data(), dtype=bool)
			# sub002:  V1: 10507 voxels  V4: 3863 voxels

	else: 
		if ROI == 'V1':
			if exvivo == True:
				lh = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/dc/lh.V1_exvivo.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)
				rh = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/dc/rh.V1_exvivo.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)
			
			else:
				lh = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/dc/lh.V1.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)
				rh = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/dc/rh.V1.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)
		else:
			print 'ROIs not found. Redefine ROIs!'

	subject_dir_beh = os.path.join(data_dir_beh,subname)
	beh_files = glob.glob(subject_dir_beh +'/func'+ '/*.pickle')
	beh_files.sort()


	subject_dir_fixation = os.path.join(data_dir_fixation, subname)
	fixation_files = glob.glob(subject_dir_fixation +'/output' +'/*.pickle')
	fixation_files.sort()

	target_files_fmri = []
	target_files_beh = []
	target_files_moco = []
	target_files_fixation = []

	target_condition = 'task-fullfield'
	for fmri_file in fmri_files:
		if os.path.split(fmri_file)[1].split('_')[1]== target_condition:
			target_files_fmri.append(fmri_file)

	for beh_file in beh_files:
		if os.path.split(beh_file)[1].split('_')[1]== target_condition:
			target_files_beh.append(beh_file)

	for moco_file in moco_files:
		if os.path.split(moco_file)[1].split('_')[1]== target_condition:
			target_files_moco.append(moco_file)

	for fixation_file in fixation_files:
		if os.path.split(fixation_file)[1].split('_')[1]== target_condition:
			target_files_fixation.append(fixation_file)

# #----------------------------------------------------------------------------------------------------------		
# #----------------------------------------------------------------------------------------------------------
###
###
###     load data for each run !!!
###
###
# #----------------------------------------------------------------------------------------------------------		
# #----------------------------------------------------------------------------------------------------------
	#if each_run == True: 

	# #check nan values	
	nan_voxels = []
	for filename_fmri in target_files_fmri:
		unmasked_fmri_data = nib.load(filename_fmri).get_data()
		fmri_data = np.vstack([unmasked_fmri_data[lh,:], unmasked_fmri_data[rh,:]])
		  
		n_all_voxels = fmri_data.shape[0] # 5728, 5728, 5728, 5728
		print 'n_all_voxels:', n_all_voxels 

		# if data_type == 'tf':
		# 	## name it with fmri_data, in fact it's for each run, namely(fmri_data_run)
		# 	fmri_data = (fmri_data - np.nanmean(fmri_data, axis = 1)[:, np.newaxis]) / np.nanstd(fmri_data, axis = 1)[:, np.newaxis]
		# 	#fmri_data shape: (5728, 286)
		# 	'finish normalization fmri data!'
		# elif data_type == 'psc':
		# 	print 'psc data type, normalization not needed!'
		
		# nan_voxels_run = 
		nan_voxels.extend(np.argwhere(fmri_data.sum(axis=1)==0).flatten())

	voxel_list =[]

	print 'found %i nan voxels'%len(nan_voxels)

	for i in range(n_all_voxels):
		if i not in nan_voxels:
			voxel_list.append(i)
	print 'finish checking nan values'



	## Load all types of data
	file_pairs_all = np.array(zip (target_files_fmri, target_files_beh, target_files_moco, target_files_fixation))
	
	# t_runs = [] 
	r_squareds_runs = []
	r_squareds_selection_runs = []
	betas_z_ori_runs = []
	betas_z_col_runs = []
	# for fileii, file_pair in enumerate(file_pairs):
		
	for fileii, (filename_fmri, filename_beh, filename_moco, filename_fixation) in enumerate(file_pairs_all):
		# if fileii == run_nr_leftOut			

		# filename_fmri = file_pair[0]
		# filename_beh = file_pair[1]
		# filename_moco = file_pair[2]
		# filename_fixation = file_pair[3]
	
	## Load fmri data--run
		unmasked_fmri_data = nib.load(filename_fmri).get_data()
		fmri_data = np.vstack([unmasked_fmri_data[lh,:], unmasked_fmri_data[rh,:]])[voxel_list,:]
		# another way to flatten ---- e.g. moco_params.reshape(-1, moco_params.shape[-1])

		# Z scored fmri_data, but with the same name
		# if data_type == 'tf':
		# 	#fmri_data = (fmri_data -fmri_data.mean()) / fmri_data.std()
		# 	## name it with fmri_data, in fact it's for each run, namely(fmri_data_run)
		# 	fmri_data = (fmri_data - np.nanmean(fmri_data, axis = 1)[:, np.newaxis]) / np.nanstd(fmri_data, axis = 1)[:, np.newaxis]
			#fmri_data shape: (5728, 286)


		# fmri_data = fmri_data[voxel_list,:]
		# fmri_data = fmri_data[np.isnan(fmri_data).sum(axis=1)==0,:]



	## Load stimuli order (events)-run
		trial_order_run = pickle.load(open(filename_beh, 'rb'))[1]
		#  the first 8 values represent all orientations but only 1color

		## for trial_order_col------------------------------------
		trial_order_col = np.zeros((len(trial_order_run),))

		for ii, stim_nr in enumerate(trial_order_run) :
			# if (stim_nr >= (8*ii) ) * (stim_nr < (8*(ii+1)))  :
			# 	trial_order_col[ii] = ii+1
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
		empty_start = 15
		empty_end = 15
		number_of_stimuli = 8  # 64

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

			# if stim_nr in np.arange(0, 64, 8):
			# 	trial_order_ori[ii] = 1

			# elif stim_nr in np.arange(1, 64, 8):
			# 	trial_order_ori[ii] = 2

			# elif stim_nr in np.arange(2, 64, 8):
			# 	trial_order_ori[ii] = 3

			# elif stim_nr in np.arange(3, 64, 8):
			# 	trial_order_ori[ii] = 4
				
			# elif stim_nr in np.arange(4, 64, 8):
			# 	trial_order_ori[ii] = 5

			# elif stim_nr in np.arange(5, 64, 8):
			# 	trial_order_ori[ii] = 6

			# elif stim_nr in np.arange(6, 64, 8):
			# 	trial_order_ori[ii] = 7

			# elif stim_nr in np.arange(7, 64, 8):
			# 	trial_order_ori[ii] = 8

		trial_order_ori = trial_order_ori[:, np.newaxis]

		#create events with 1
		# empty_start = 15
		# empty_end = 15
		# number_of_stimuli = 8  # 64

		tmp_trial_order_ori  = np.zeros((fmri_data.shape[1],1))
		#15 + 256( 2* 128) +15 =286, (286,1)
		tmp_trial_order_ori[empty_start:-empty_end:2] = trial_order_ori[:]   
		events_ori = np.hstack([np.array(tmp_trial_order_ori == stim, dtype=int) for stim in np.arange(1,number_of_stimuli+1)])


		events = np.hstack([events_col, events_ori])

	## Load motion correction parameters

		moco_params = pd.read_csv(filename_moco, delim_whitespace=True, header = None)
		# nib.load(filename_moco).get_data()
		# shape (286,1)///(286,6)

	# ## Load fixation task parameters

		fixation_order_run = pickle.load(open(filename_fixation, 'rb'))
		eventArray = fixation_order_run['eventArray']  # a list of lists

		key_press = np.zeros((fmri_data.shape[1],1))

		for n_event, event in enumerate (eventArray):
			for txt in event:
				if 'key: y' in txt:
					key_press[n_event] = 1

				elif 'key: b' in txt:
					key_press[n_event] = 1

	### Load stimulus regressor

		trial_order_run = pickle.load(open(filename_beh, 'rb'))[1]
		trial_order_di = np.squeeze(trial_order_run[:]+1)
		trial_order_di[trial_order_di<=64] = 1
		trial_order_di[trial_order_di>64] = 0
		# for ii, stim in enumerate(trial_order_di):
		# 	if stim >64:
		# 		trial_order_di[ii] = 0
		# 	else:
		# 		trial_order_di[ii] = 1

		#trial_order_di = trial_order_di[:, np.newaxis]
		stim_regressor  = np.zeros((fmri_data.shape[1],1))
		#15 + 256( 2* 128) +15 =286
		stim_regressor[empty_start:-empty_end:2] = trial_order_di[:, np.newaxis] # [:,np.newaxis]+1


	# convolve events with hrf, to get model_BOLD_timecourse
		TR = 0.945 #ms
		model_BOLD_timecourse = fftconvolve(events, hrf(np.arange(0,30)* TR)[:,np.newaxis],'full')[:fmri_data.shape[1],:]

		design_matrix = np.hstack([model_BOLD_timecourse, moco_params, key_press]) #np.ones((fmri_data.shape[1],1)), 
		# shape: (286,71--1+64+6)

		# for r_squareds selection
		model_BOLD_timecourse_selection = fftconvolve(stim_regressor, hrf(np.arange(0,30)* TR)[:,np.newaxis],'full')[:fmri_data.shape[1],:]
		design_matrix_selection = np.hstack([model_BOLD_timecourse_selection, moco_params, key_press]) # np.ones((fmri_data.shape[1],1)), 


		n_voxels = fmri_data.shape[0]
		n_TRs = fmri_data.shape[1]
		n_regressors = design_matrix.shape[1]  # 1+16+6+1 = 24
		df = (n_TRs-n_regressors)

		print 'n_voxels without nans', n_voxels

	# GLM to get betas
		if regression == 'GLM': #'RidgeCV'
			print 'start GLM fitting'
			betas, _sse, _r, _svs = np.linalg.lstsq(design_matrix, fmri_data.T )
			# betas shape (65, 9728--number of voxels)
			# _sse shape (10508)

			r_squareds = 1.0 - ((design_matrix.dot(betas).T -fmri_data)**2).sum(axis=1) / (fmri_data**2).sum(axis=1)

			betas = betas.T #(10508,72)
			print 'finish GLM'

		elif regression == 'RidgeCV':
			# ridge_fit = RidgeCV(alphas = np.linspace(1,50,50) , fit_intercept = False, normalize = True )
			alpha_range = [0.001,0.01,1,10,100,1000]#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
			# alpha_range = [0.5]
			ridge_fit = RidgeCV(alphas = alpha_range , fit_intercept = True, normalize = True)
			
			ridge_fit_selection = RidgeCV(alphas = alpha_range , fit_intercept = True, normalize = True)

			results = np.zeros((n_voxels,3))
			r_squareds =  np.zeros((n_voxels, ))
			alphas =  np.zeros((n_voxels, 1))
			intercept =  np.zeros((n_voxels, 1))
			betas = np.zeros((n_voxels, n_regressors ))
			betas_z_col = np.zeros((n_voxels, number_of_stimuli ))
			betas_z_ori = np.zeros((n_voxels, number_of_stimuli ))
			_sse = np.zeros((n_voxels, ))

			r_squareds_selection =  np.zeros((n_voxels, ))

			print 'start RidgeCV fitting'

			for x in range(n_voxels):
				
				ridge_fit.fit(design_matrix, fmri_data[x, :])
				# print x, ridge_fit.score(design_matrix, fmri_data[x,:]), ridge_fit.alpha_ #, ridge_fit.coef_.T
				# results[x] = [ridge_fit.score(design_matrix, fmri_data[x,:]), ridge_fit.alpha_, ridge_fit.coef_.T]	#ridge_fit.fit(design_matrix, fmri_data.T)
				r_squareds[x] = ridge_fit.score(design_matrix, fmri_data[x,:])
				alphas[x] = ridge_fit.alpha_
				betas[x, :] = ridge_fit.coef_.T
				intercept[x,:] = ridge_fit.intercept_
				_sse[x] = np.sqrt(np.sum((design_matrix.dot(betas[x]) - fmri_data[x,:])**2)/df)
				betas_z_ori [x, :] = (betas[x, 1:9] - np.mean(betas[x, 1:9])) / np.std(betas[x, 1:9])
				betas_z_col [x, :] = (betas[x, 9:17] - np.mean(betas[x, 9:17])) / np.std(betas[x, 9:17])

				# for selection
				ridge_fit_selection.fit(design_matrix_selection, fmri_data[x,:])
				r_squareds_selection[x] = ridge_fit_selection.score(design_matrix_selection, fmri_data[x,:])

# Do I need to z-core betas per voxel twice? Namely across eight orientation regressors and eight color regressors separately, without betas of other nuisance regressors and intercept?
	# betas shape: (97, 10655); fmri_data.shape: (10655,858); fmri_data_run.shape (10655,286)
	# average across channels? or voxels? voxels. because want to compare between voxels. fmri_data: average across time points, keep voxels intact. becuase we want to compare different runs--times.

	#fmri_data_run = (fmri_data_run - np.nanmean(fmri_data_run, axis = 1)[:, np.newaxis]) / np.nanstd(fmri_data_run, axis = 1)[:, np.newaxis]
	# betas_z = (betas - np.nanmean(betas, axis = 1)[:, np.newaxis]) / np.nanstd(betas, axis = 1)[:, np.newaxis]

			print 'finish RidgeCV'

		r_squareds_selection_runs.append(r_squareds_selection) 
		r_squareds_runs.append(r_squareds)
		betas_z_ori_runs.append(betas_z_ori)
		betas_z_col_runs.append(betas_z_col)


		# compute contrasts
		# if type_contrasts == 'full':
### compute contrasts. 
		# n_contrasts = 64
		# t = np.zeros((n_voxels, n_contrasts))
		# p = np.zeros((n_voxels, n_contrasts))

		# for i in range(n_contrasts):

		# 	c_moco = np.zeros(moco_params.shape[1])
		# 	c_key_press = np.zeros(key_press.shape[1])
		# 	a = np.ones(n_contrasts) *  -1/float(n_contrasts-1)  #-1/ 63.0
		# 	# a = np.zeros(64)
		# 	a[i] = 1

		# 	c = np.r_[0, a, c_moco, c_key_press] #.reshape(8,8) moco_params, key_press
		# 	design_var = c.dot(np.linalg.pinv(design_matrix.T.dot(design_matrix))).dot(c.T)
		# 	SE_c = np.sqrt(_sse * design_var)

		# 	t[:,i] = betas.dot(c) / SE_c  # SE_c (10508,)
		# 	p[:,i] = scipy.stats.t.sf(np.abs(t[:,i]), df)*2		

		# t_runs.append(t)

# start of leave-one-run-out-procedure.
	print 'prepare preference, & a set of tunings for leftout runs. '
# # prepare preference, & a set of tunings for leftout runs. 
	betas_z_ori_runs = np.array(betas_z_ori_runs)
	betas_z_col_runs = np.array(betas_z_col_runs)
	r_squareds_selection_runs = np.array(r_squareds_selection_runs)
	# shell()
	run_nr_all = np.arange(file_pairs_all.shape[0])

# #-----------------------------------------------------------------
#6.21 figure as Brouwer paper:
	betas_z_ori_across_voxels = np.mean (betas_z_ori_runs[r_squareds ~= 0], axis = 1)
	betas_z_col_across_voxels = np.mean (betas_z_col_runs, axis = 1)

	betas_z_ori_across_runs = np.mean( betas_z_ori_across_voxels , axis = 0)
	betas_z_col_across_runs = np.mean( betas_z_col_across_voxels, axis = 0)
	
	sd = np.array([np.std(betas_z_ori_across_voxels, axis = 0), np.std(betas_z_col_across_voxels, axis = 0)])
	n = len(run_nr_all)
	yerr = (sd/np.sqrt(n))

	f1 = plt.figure(figsize = (8,6))
	s1 = f1.add_subplot(211)
	x = range(len(betas_z_ori_across_runs))
	plt.bar(x, betas_z_ori_across_runs, color = 'blue', align = 'center', alpha= 0.4) #yerr = yerr[0]
	plt.errorbar(range(0,8), betas_z_ori_across_runs, yerr= yerr[0])
	# s1.set_title('orientation', fontsize = 10)
	s1.set_xticklabels(['placeholder', 'horizontal 0', 22.5, 45, 67.5, 'vertical 90', 112.5, 135, 157.5, 'horizontal 0'])
	s1.set_xlabel('orientation ')


	s2 = f1.add_subplot(212)

	color_theta = (np.pi*2)/8
	color_angle = color_theta * np.arange(0, 8,dtype=float)
	color_radius = 75

	color_a = color_radius * np.cos(color_angle)
	color_b = color_radius * np.sin(color_angle)

	colors2 = np.array([ct.lab2rgb((55, a, b)) for a,b in zip(color_a, color_b)])	
	colors2 = np.hstack((colors2/255, np.ones((8,1))))

	x = range(len(betas_z_col_across_runs))
	plt.bar(x, betas_z_col_across_runs, color = [colors2[t] for t in range(0,8)], align = 'center' )#yerr = yerr[1], alpha= 0.4)
	plt.errorbar(range(0,8), betas_z_col_across_runs, yerr= yerr[1])
	# s2.set_title('color', fontsize = 10)
	# s2.set_xticklabels(['placeholder', -2, -1, 0, 1, 2, 3, 4, -3, -2])
	s2.set_xlabel('color ')


	f1.savefig( '%s_%s_%s_%s_response_amplitudes.png'%(subname, ROI, data_type, regression))


# #--------------------------------------------------------------------

# when position_cen = 2
	# t_ori_mean_iterations = np.zeros((len(run_nr_all), 9)) 
	# t_col_mean_iterations = np.zeros((len(run_nr_all), 9)) 
	beta_ori_mean_iterations = np.zeros((len(run_nr_all), 9)) 
	beta_col_mean_iterations = np.zeros((len(run_nr_all), 9))

# when position _cen = nan
	# t_ori_cate_mean_iterations = np.zeros((len(run_nr_all), 8,9)) 
	# t_col_cate_mean_iterations = np.zeros((len(run_nr_all), 8,9)) 
	beta_ori_cate_mean_iterations = np.zeros((len(run_nr_all), 8,9)) 
	beta_col_cate_mean_iterations = np.zeros((len(run_nr_all), 8,9)) 

	for filepairii in run_nr_all :
	
		run_nr_leftOut = filepairii
		run_nr_rest = run_nr_all[~(run_nr_all == run_nr_leftOut)]
		# file_pairs = file_pairs_all[~(np.arange(file_pairs_all.shape[0]) == run_nr_leftOut)]

		# t_mean = np.mean(t_runs[run_nr_rest], axis = 0) # average across rest runs 		# t_mean shape: (5638,64)		
		betas_z_ori_mean = np.mean(betas_z_ori_runs[run_nr_rest], axis = 0)
		betas_z_col_mean = np.mean(betas_z_col_runs[run_nr_rest], axis = 0)
		r_squareds_mean = np.mean(r_squareds_selection_runs[run_nr_rest], axis = 0) 

		order = np.argsort(r_squareds_mean)
		voxels_all = sorted( zip(order, r_squareds_mean) , key = lambda tup: tup [1] )
		n_best = 100 # n_voxels #100
		voxels = voxels_all[-n_best:]

		voxel_indices_bestVox = np.array(voxels)[:,0]
		# t_pre_indices_bestVox = np.zeros((n_best, 2 ))
		beta_pre_ori_indices_bestVox = np.zeros((n_best, 1))
		beta_pre_col_indices_bestVox = np.zeros((n_best, 1)) 


	### prepare t_pre_index
	# position_cen =2 and 'nan' are the same!
		# if position_cen == 2: 
			
		# 	for voxelii, voxelIndex in enumerate(voxel_indices_bestVox):					

		# 		voxelIndex = int(voxelIndex)

		# 		t_matrix = t_mean [voxelIndex ].reshape(8,8)

		# 		t_pre_index = np.squeeze(np.where(t_matrix== t_matrix.max()))
		# 		# if get two max values, make the first one
		# 		if t_pre_index.size == 2:
		# 			pass
		# 		else:
		# 			t_pre_index = np.array([t_pre_index[0][0], t_pre_index[1][0]])

		# 		t_pre_indices_bestVox[voxelii, :] = t_pre_index

		# elif position_cen == 'nan': 
		for voxelii, voxelIndex in enumerate(voxel_indices_bestVox):

			voxelIndex = int(voxelIndex)
			# r_squared_best = voxel[1]

			# t_matrix = t_mean [voxelIndex ].reshape(8,8)
			beta_pre_ori = betas_z_ori_mean [voxelIndex]
			beta_pre_col = betas_z_col_mean [voxelIndex]

			#  the centers are green / vertical already.--4 (0-4) position
			# beta_ori_cen = np.roll(beta_pre_ori, 1)
			# beta_col_cen = np.roll(beta_pre_col, )  

			beta_pre_ori_index = [i for i, j in enumerate(beta_pre_ori) if j == beta_pre_ori.max() ]
			beta_pre_col_index =  [i for i, j in enumerate(beta_pre_col) if j == beta_pre_col.max() ]

			if len(beta_pre_ori_index) == 1:
				print 'only one preferred orientation'
			else:
				beta_pre_ori_index = beta_pre_ori_index[0]
				print 'more than one preferred ori'

			if len(beta_pre_col_index) == 1:
				print 'only one preferred color'
			else:
				beta_pre_col_index = beta_pre_col_index[0]
				print 'more than one preferred color'			
			
			beta_pre_ori_indices_bestVox[voxelii ] = beta_pre_ori_index
			beta_pre_col_indices_bestVox[voxelii ] = beta_pre_col_index

			# # t_pre_index = np.squeeze(np.where(t_matrix_cen== t_matrix_cen.max()))

			# # plt.imshow( t_matrix , cmap= plt.cm.ocean, interpolation = "None")

			# # # center --- exact values: make horizontal as the center of the x-axis, so the labels are still the extact positions.
			# # # so exact positions for x-axis
			# # # make the centers are green / horizontal. 
			# # t_matrix_cenRow = np.roll(t_matrix, 5, axis = 0) # roll downwards by 5 steps, to make the green one as the 4th(0,1,2,3,4), then in the next step, the green one will be at the center.
			# # t_matrix_cen= np.roll(t_matrix_cenRow, 1, axis = 1) # roll to right side by 1 step, to make horizontal be the 4th (be at center through next step)

			# # note that t_matrix_cen instead of t_matrix, compared with centring to a position
			# t_pre_index = np.squeeze(np.where(t_matrix_cen== t_matrix_cen.max()))
			# # if get two max values, make the first one
			# if t_pre_index.size == 2:
			# 	print 'only one preferred stimulus'
			# else:
			# 	t_pre_index = np.array([t_pre_index[0][0], t_pre_index[1][0]])
			# 	print 'more than one preferred stimulus', voxelIndex , t_pre_index[0].shape


			# t_pre_indices_bestVox[voxelii, :] = t_pre_index
#---------------------------------------------------------
	### a set of tunings for the specific leftout run. 

		betas_z_ori_leftOut = betas_z_ori_runs[run_nr_leftOut]
		betas_z_col_leftOut = betas_z_col_runs[run_nr_leftOut]
		# voxel_indices_bestVox 
		# t_pre_indices_bestVox
		
		if position_cen == 2:
			beta_oriBestVox = np.zeros((n_best, 9))
			beta_colBestVox = np.zeros((n_best, 9))

			for nrii, voxelIndex in enumerate(voxel_indices_bestVox):

				voxelIndex = int(voxelIndex)
				# t_matrix_leftOut = ts_leftOut[voxelIndex].reshape(8,8)
				beta_ori =  betas_z_ori_leftOut [voxelIndex]
				beta_col = betas_z_col_leftOut [voxelIndex]
				# t_pre_current_index = t_pre_indices_bestVox[nrii]		
				beta_pre_ori_current_index = beta_pre_ori_indices_bestVox[nrii] 
				beta_pre_col_current_index = beta_pre_col_indices_bestVox[nrii]

				# center --- always move the peaks to the position_cen, 'x axis will be relative positions'
				# so the labels of x-axis are not the actual values (ori/color), but the relative position on the axis.
				beta_ori_cen = np.roll(beta_ori, int(position_cen-beta_pre_ori_current_index[0]) )
				beta_col_cen = np.roll(beta_col, int(position_cen-beta_pre_col_current_index[0]) )				
				# t_matrix_leftOut_cenRow = np.roll(t_matrix_leftOut, int(position_cen-t_pre_current_index[0]), axis = 0)
				# t_matrix_leftOut_cen= np.roll(t_matrix_leftOut_cenRow, int(position_cen-t_pre_current_index[1]), axis = 1)

				# make it circlar
				beta_ori_cir = np.hstack((beta_ori_cen, beta_ori_cen[0]))
				beta_col_cir = np.hstack((beta_col_cen, beta_col_cen[0]))				
				# t_matrix_leftOut_add_column = np.hstack((t_matrix_leftOut_cen, t_matrix_leftOut_cen[:,0][:, np.newaxis]))
				# t_matrix_leftOut_cir = np.vstack ((t_matrix_leftOut_add_column, t_matrix_leftOut_add_column[0,:]))
				# t_ori = t_matrix_leftOut_cir[position_cen,:]
				# t_col = t_matrix_leftOut_cir[:,position_cen]

		
				beta_oriBestVox[nrii,:] = beta_ori_cir
				beta_colBestVox[nrii,:] = beta_col_cir

			beta_oriBestVox_mean = np.mean(beta_oriBestVox, axis = 0)
			beta_colBestVox_mean = np.mean(beta_colBestVox, axis = 0)

			beta_ori_mean_iterations [filepairii,:] = beta_oriBestVox_mean
			beta_col_mean_iterations [filepairii,:] = beta_colBestVox_mean

		elif position_cen == 'nan': 
			beta_z_oriBestVox_0 = []
			beta_z_colBestVox_0 = []

			beta_z_oriBestVox_1 = []
			beta_z_colBestVox_1 = []
			beta_z_oriBestVox_2 = []
			beta_z_colBestVox_2 = []
			beta_z_oriBestVox_3 = []
			beta_z_colBestVox_3 = []
			beta_z_oriBestVox_4 = []
			beta_z_colBestVox_4 = []
			beta_z_oriBestVox_5 = []
			beta_z_colBestVox_5 = []
			beta_z_oriBestVox_6 = []
			beta_z_colBestVox_6 = []
			beta_z_oriBestVox_7 = []
			beta_z_colBestVox_7 = []

			beta_z_ori_cate = [beta_z_oriBestVox_0, beta_z_oriBestVox_1, beta_z_oriBestVox_2, beta_z_oriBestVox_3, beta_z_oriBestVox_4, beta_z_oriBestVox_5, beta_z_oriBestVox_6, beta_z_oriBestVox_7]
			beta_z_col_cate = [beta_z_colBestVox_0, beta_z_colBestVox_1, beta_z_colBestVox_2, beta_z_colBestVox_3, beta_z_colBestVox_4, beta_z_colBestVox_5, beta_z_colBestVox_6, beta_z_colBestVox_7]

	
			for nrii, voxelIndex in enumerate(voxel_indices_bestVox):

				voxelIndex = int(voxelIndex)
				# t_matrix_leftOut = ts_leftOut[voxelIndex].reshape(8,8)
				beta_ori =  betas_z_ori_leftOut [voxelIndex]
				beta_col = betas_z_col_leftOut [voxelIndex]
				# # center --- exact values: make horizontal as the center of the x-axis, so the labels are still the extact positions.
				# # so exact positions for x-axis
				# # make the centers are green / horizontal. 
				# t_matrix_leftOut_cenRow = np.roll(t_matrix_leftOut, 5, axis = 0) # roll downwards by 5 steps, to make the green one as the 4th(0,1,2,3,4), then in the next step, the green one will be at the center.
				# t_matrix_leftOut_cen= np.roll(t_matrix_leftOut_cenRow, 1, axis = 1) # roll to right side by 1 step, to make horizontal be the 4th (be at center through next step)

				beta_pre_ori_current_index = beta_pre_ori_indices_bestVox[nrii] 
				beta_pre_col_current_index = beta_pre_col_indices_bestVox[nrii]

				# make it circlar
				beta_ori_cir = np.hstack((beta_ori, beta_ori[0]))
				beta_col_cir = np.hstack((beta_col, beta_col[0]))

				# t_matrix_leftOut_add_column = np.hstack((t_matrix_leftOut_cen, t_matrix_leftOut_cen[:,0][:, np.newaxis]))
				# t_matrix_leftOut_cir = np.vstack ((t_matrix_leftOut_add_column, t_matrix_leftOut_add_column[0,:]))

				# t_ori = t_matrix_leftOut_cir[t_pre_current_index[0],:]
				# t_col = t_matrix_leftOut_cir[:,t_pre_current_index[1]]
				
				for i in range(0,8):
					if beta_pre_ori_current_index == i:
						beta_z_ori_cate [i].append(beta_ori_cir)
					
					if beta_pre_col_current_index == i:
						beta_z_col_cate [i].append(beta_col_cir)

			
			beta_z_ori_cate_mean_leftOut = np.zeros((8,9)) #8 preference locations, 9 points 
			beta_z_col_cate_mean_leftOut = np.zeros((8,9)) 



			# t_ori_cate_mean_leftOut = np.mean(t_ori_cate, axis = 1) # for each i 

			for i in range(0,8):

				beta_z_ori_cate[i] = np.array(beta_z_ori_cate[i]) 
				beta_z_col_cate[i] = np.array(beta_z_col_cate[i])

				beta_z_ori_cate_mean_leftOut[i,:] = np.mean(beta_z_ori_cate[i], axis = 0)
				beta_z_col_cate_mean_leftOut[i,:] = np.mean(beta_z_col_cate[i], axis = 0)

				len(beta_z_ori_cate[i])



			beta_ori_cate_mean_iterations [filepairii, :, :] = beta_z_ori_cate_mean_leftOut
			beta_col_cate_mean_iterations [filepairii, :, :] = beta_z_col_cate_mean_leftOut


# plot figures! across iterations.

	print 'plot figures across iterations!'

	if position_cen == 2:

		beta_ori_mean = np.mean(beta_ori_mean_iterations, axis = 0)
		beta_col_mean = np.mean(beta_col_mean_iterations, axis = 0)


		sd = np.array([np.std(beta_ori_mean_iterations, axis = 0), np.std(beta_col_mean_iterations, axis = 0)])
		n = len(run_nr_all)
		yerr = (sd/np.sqrt(n))


		f2 = plt.figure(figsize = (8,6))
		s1 = f2.add_subplot(211)
		plt.plot(beta_ori_mean)
		plt.errorbar(range(0,9), beta_ori_mean, yerr= yerr[0])
		# s1.set_title('orientation', fontsize = 10)
		s1.set_xticklabels(['placeholder', -45, -22.5, 0, 22.5, 45, 67.5, 90, -67.5 ,-45])
		s1.set_xlabel('orientation - relative ')


		s2 = f2.add_subplot(212)
		plt.plot(beta_col_mean)
		plt.errorbar(range(0,9), beta_col_mean, yerr= yerr[1])
		# s2.set_title('color', fontsize = 10)
		s2.set_xticklabels(['placeholder', -2, -1, 0, 1, 2, 3, 4, -3, -2])
		s2.set_xlabel('color - relative')
		f2.savefig( '%s_%s_%s_%s_cen%s_tValues_%sVoxels.png'%(subname, ROI, data_type, regression, position_cen, n_best))



	elif position_cen == 'nan':
		beta_ori_cate_mean = np.mean(beta_ori_cate_mean_iterations, axis = 0)
		beta_col_cate_mean = np.mean(beta_col_cate_mean_iterations, axis = 0)

		beta_ori_cate_yerr = np.std(beta_ori_cate_mean_iterations, axis = 0)/ np.sqrt(len(run_nr_all)) 
		beta_col_cate_yerr = np.std(beta_col_cate_mean_iterations, axis = 0)/ np.sqrt(len(run_nr_all)) # standard error


		f3 = plt.figure(figsize = (12,10))

		colors1 = plt.cm.rainbow(np.linspace(0, 1, len(range(0,8))))

		s1 = f3.add_subplot(2,1,1)
				
		for colii, color in enumerate(colors1, start =0):

			if beta_ori_cate_mean[colii].size == 1:
				s1.set_title('no voxels fall into peak position: %s' %(str(colii)) , fontsize = 10)

			else:
				plt.plot(beta_ori_cate_mean[colii], color = color, label = 'position:%s' %(str(colii)))
				plt.errorbar(range(0,9), beta_ori_cate_mean[colii], color = color, yerr= beta_ori_cate_yerr[colii]) #
				plt.legend(loc='best')
		

		# for i,j in zip(x,y):
		# 	ax.annotate(str(j),xy=(i,j))
		s1.set_xticklabels(['placeholder', 'horizontal 0', 22.5, 45, 67.5, 'vertical 90', 112.5, 135, 157.5, 'horizontal 0'])
		s1.set_xlabel('orientation - absolute')
		# f3.savefig( '%s_%s_%s_%sOri_8pos_tValues_%sVoxels.png'%(subname, ROI, data_type, regression, n_best))

		
		# f4 = plt.figure(figsize = (12,10))
		s2 = f3.add_subplot(2,1,2)
		# s2 = f4.add_subplot(2,1,2)
		# use colors in the mapping experiment
		# Compute evenly-spaced steps in (L)ab-space

		color_theta = (np.pi*2)/8
		color_angle = color_theta * np.arange(0, 8,dtype=float)
		color_radius = 75

		color_a = color_radius * np.cos(color_angle)
		color_b = color_radius * np.sin(color_angle)

		colors2 = np.array([ct.lab2rgb((55, a, b)) for a,b in zip(color_a, color_b)])	
		colors2 = np.hstack((colors2/255, np.ones((8,1))))

		for colii, color in enumerate(colors2, start =0):
			if beta_col_cate_mean[colii].size == 1:
				s2.set_title('no voxels fall into peak position: %s' %(str(colii)) , fontsize = 10)

			else:
				plt.plot(beta_col_cate_mean[colii], color = color , label = 'position:%s' %(str(colii)))
				plt.errorbar(range(0,9), beta_col_cate_mean[colii], color = color, yerr= beta_col_cate_yerr[colii])
			plt.legend(loc='best')

		s2.set_xlabel('color - absolute')
		# f4.savefig( '%s_%s_%s_%sCol_8pos_tValues_%sVoxels.png'%(subname, ROI, data_type, regression, n_best))
		f3.savefig( '%s_%s_%s_%s_8pos_tValues_%sVoxels.png'%(subname, ROI, data_type, regression, n_best))





# # 			t_oriBestVox_mean_0 = []
# # 			t_oriBestVox_mean_1 = []
# # 			t_oriBestVox_mean_2 = []
# # 			t_oriBestVox_mean_3 = []
# # 			t_oriBestVox_mean_4 = []
# # 			t_oriBestVox_mean_5 = []
# # 			t_oriBestVox_mean_6 = []
# # 			t_oriBestVox_mean_7 = []

# # 			t_colBestVox_mean_0 = []
# # 			t_colBestVox_mean_1 = []
# # 			t_colBestVox_mean_2 = []
# # 			t_colBestVox_mean_3 = []
# # 			t_colBestVox_mean_4 = []
# # 			t_colBestVox_mean_5 = []
# # 			t_colBestVox_mean_6 = []
# # 			t_colBestVox_mean_7 = []

# # 			t_oriBestVox_yerr_0 = []
# # 			t_oriBestVox_yerr_1 = []
# # 			t_oriBestVox_yerr_2 = []
# # 			t_oriBestVox_yerr_3 = []
# # 			t_oriBestVox_yerr_4 = []
# # 			t_oriBestVox_yerr_5 = []
# # 			t_oriBestVox_yerr_6 = []
# # 			t_oriBestVox_yerr_7 = []

# # 			t_colBestVox_yerr_0 = []
# # 			t_colBestVox_yerr_1 = []
# # 			t_colBestVox_yerr_2 = []
# # 			t_colBestVox_yerr_3 = []
# # 			t_colBestVox_yerr_4 = []
# # 			t_colBestVox_yerr_5 = []
# # 			t_colBestVox_yerr_6 = []
# # 			t_colBestVox_yerr_7 = []

# # 			t_ori_cate_mean = [t_oriBestVox_mean_0, t_oriBestVox_mean_1, t_oriBestVox_mean_2, t_oriBestVox_mean_3, t_oriBestVox_mean_4, t_oriBestVox_mean_5, t_oriBestVox_mean_6, t_oriBestVox_mean_7]
# # 			t_col_cate_mean = [t_colBestVox_mean_0, t_colBestVox_mean_1, t_colBestVox_mean_2, t_colBestVox_mean_3, t_colBestVox_mean_4, t_colBestVox_mean_5, t_colBestVox_mean_6, t_colBestVox_mean_7]

# # 			t_ori_cate_yerr = [t_oriBestVox_yerr_0, t_oriBestVox_yerr_1, t_oriBestVox_yerr_2, t_oriBestVox_yerr_3, t_oriBestVox_yerr_4, t_oriBestVox_yerr_5, t_oriBestVox_yerr_6, t_oriBestVox_yerr_7]
# # 			t_col_cate_yerr = [t_colBestVox_yerr_0, t_colBestVox_yerr_1, t_colBestVox_yerr_2, t_colBestVox_yerr_3, t_colBestVox_yerr_4, t_colBestVox_yerr_5, t_colBestVox_yerr_6, t_colBestVox_yerr_7]

		
# # 			for i in range(0,8):

# # 				t_ori_cate[i] = np.array(t_ori_cate[i]) 
# # 				t_col_cate[i] = np.array(t_col_cate[i])

# # 				t_ori_cate_mean[i] = np.mean(t_ori_cate[i], axis = 0)
# # 				t_ori_cate_yerr[i] = np.std(t_ori_cate[i], axis = 0)/ np.sqrt(t_ori_cate[i].shape[0]) * 1.96

# # 				t_col_cate_mean[i] = np.mean(t_col_cate[i], axis = 0)
# # 				t_col_cate_yerr[i] = np.std(t_col_cate[i], axis = 0)/ np.sqrt(t_col_cate[i].shape[0]) * 1.96

# # # ### 16 subplots, 8 for each condition
# # # 			f3 = plt.figure(figsize = (32,22))
# # # 			for i in range(0,8):

# # # 				if t_ori_cate_mean[i].size == 1:
# # # 					s1 = f3.add_subplot(4,4,i+1)
# # # 					s1.set_title('no voxels fall into peak position:%s' %(str(i)) , fontsize = 10)
# # # 				else:
# # # 					s1 = f3.add_subplot(4,4,i+1)
# # # 					plt.plot(t_ori_cate_mean[i])
# # # 					plt.errorbar(range(0,9), t_ori_cate_mean[i], yerr= t_ori_cate_yerr[i])
# # # 					s1.set_xlabel('orientation')
# # # 					s1.set_title('n_voxels:%s, peak position:%s' % (str(t_col_cate[i].shape[0]), str(i)) , fontsize = 10)
					
# # # 				if t_col_cate_mean[i].size == 1:
# # # 					s2 = f3.add_subplot(4,4,i+9)
# # # 					s2.set_title('no voxels fall into peak position:%s' %(str(i)) , fontsize = 10)
# # # 				else:
# # # 					s2 = f3.add_subplot(4,4,i+9)
# # # 					plt.plot(t_col_cate_mean[i])
# # # 					plt.errorbar(range(0,9), t_col_cate_mean[i], yerr= t_col_cate_yerr[i])
# # # 					s2.set_xlabel('color')
# # # 					s2.set_title('n_voxels:%s, peak position:%s' % (str(t_col_cate[i].shape[0]), str(i)) , fontsize = 10)

# # # 			f3.savefig( '%s_%s_%s_%s_8pos_tValues_%sVoxels.png'%(subname, ROI, data_type, regression, n_best))

# # ### 2 subplots, with all different colors
# # # cmap= plt.cm.ocean


# # 			# f3 = plt.figure(figsize = (12,10))
			

# # 			f3 = plt.figure(figsize = (12,10))

# # 			colors1 = plt.cm.rainbow(np.linspace(0, 1, len(range(0,8))))

# # 			s1 = f3.add_subplot(2,1,1)
					
# # 			for colii, color in enumerate(colors1, start =0):
# # 				if t_ori_cate_mean[colii].size == 1:
# # 					s1.set_title('no voxels fall into peak position: %s' %(str(colii)) , fontsize = 10)

# # 				else:
# # 					plt.plot(t_ori_cate_mean[colii], color = color) #, label = 'position:%s' %(str(colii)))
# # 					plt.errorbar(range(0,9), t_ori_cate_mean[colii], color = color, yerr= t_ori_cate_yerr[colii])

# # 			# 		leg = s1.legend()

# # 			# for color,text in zip(colors,leg.get_texts()):
# # 	#  			text.set_color(color)

# # 				# plt.legend(loc='best')
# # 				# plt.close()
# # 			s1.set_xticklabels(['placeholder', 'vertical 90', -67.5, -45, -22.5, 'horizontal 0', 22.5, 45, 67.5, 'vertical 90'])
# # 			s1.set_xlabel('orientation')
# # 			f3.savefig( '%s_%s_%s_%sOri_8pos_tValues_%sVoxels.png'%(subname, ROI, data_type, regression, n_best))

			
# # 			# f4 = plt.figure(figsize = (12,10))
# # 			s2 = f3.add_subplot(2,1,2)
# # 			# colors2 = plt.cm.gist_rainbow(np.linspace(0, 1, len(range(0,8))))  # gist_rainbow not rainbow
# # 			# use colors in the mapping experiment
# # 		# Compute evenly-spaced steps in (L)ab-space

# # 			color_theta = (np.pi*2)/8
# # 			color_angle = color_theta * np.arange(0, 8,dtype=float)
# # 			color_radius = 75

# # 			color_a = color_radius * np.cos(color_angle)
# # 			color_b = color_radius * np.sin(color_angle)

# # 			colors2 = np.array([ct.lab2rgb((55, a, b)) for a,b in zip(color_a, color_b)])	

# # 			colors2 = np.hstack((colors2/255, np.ones((8,1))))

# # 			for colii, color in enumerate(colors2, start =0):
# # 				if t_col_cate_mean[colii].size == 1:
# # 					s2.set_title('no voxels fall into peak position: %s' %(str(colii)) , fontsize = 10)

# # 				else:
# # 					plt.plot(t_col_cate_mean[colii], color = color) #, label = 'position:%s' %(str(colii)))
# # 					plt.errorbar(range(0,9), t_col_cate_mean[colii], color = color, yerr= t_col_cate_yerr[colii])
				
# # 				# plt.legend(loc='best')

# # 			# s1.set_xticklabels(['pink', 'orange', 'yellow', 'green', 'light_green', 'blue', 'dark_blue', 67.5, 90])
# # 			s2.set_xlabel('color')
# # 			# f4.savefig( '%s_%s_%s_%sCol_8pos_tValues_%sVoxels.png'%(subname, ROI, data_type, regression, n_best))
# # 			f3.savefig( '%s_%s_%s_%s_8pos_tValues_%sVoxels.png'%(subname, ROI, data_type, regression, n_best))

# 		# plt.close()
# 	# f1 = plt.figure(figsize = (8,6))

# 	# plt.plot(t_20)
# 	# plt.close()

# 	# f1.savefig('%s_%s_%s_run%s_AVERAGE20_tValues.png'%(subname, data_type, ROI, str(run_nr) ))



