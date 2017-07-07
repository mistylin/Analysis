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
import seaborn as sn
from matplotlib.gridspec import GridSpec
import pandas as pd

from sklearn.linear_model import RidgeCV
from fractions import Fraction
import re
import sys

import ColorTools as ct
from Staircase import ThreeUpOneDownStaircase
from tools import two_gamma as hrf

import mri_load_data as ld
import mri_main_analysis as ma
import mri_plot_tunings as pt



#-------------------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------------------
 # without def 

sublist = [ ('sub-n001', False, False), ('sub-n003', False, False), ('sub-n005', False, False), ]#('sub-n001', False, False), 
data_dir_fmri = '/home/shared/2017/visual/OriColorMapper/preproc/'
data_dir_beh = '/home/shared/2017/visual/OriColorMapper/bids_converted/'

data_type = 'psc'#'tf' #'psc'
each_run = True #False #True #False
ROI = 'V1' # 'V4'
regression = 'RidgeCV' #'GLM' #'RidgeCV'
position_cen = 2 #2 4  #'nan'




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
				lh = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/dc/lh.V1_exvivo_vol_dil.nii.gz')).get_data(), dtype=bool) #'masks/dc/lh.V1_exvivo.thresh_vol_dil.nii.gz'
				rh = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/dc/rh.V1_exvivo_vol_dil.nii.gz')).get_data(), dtype=bool) #'masks/dc/rh.V1_exvivo.thresh_vol_dil.nii.gz'
			
			else:
				lh = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/dc/lh.V1_vol_dil.nii.gz')).get_data(), dtype=bool)
				rh = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/dc/rh.V1_vol_dil.nii.gz')).get_data(), dtype=bool)
		else:
			print 'ROIs not found. Redefine ROIs!'

	subject_dir_beh = os.path.join(data_dir_beh,subname)
	beh_files = glob.glob(subject_dir_beh +'/func'+ '/*.pickle')
	beh_files.sort()

	# subject_dir_fixation = os.path.join(data_dir_fixation, subname)
	# fixation_files = glob.glob(subject_dir_fixation +'/output' +'/*.pickle')
	# for new-standard data
	subject_dir_fixation = os.path.join(data_dir_beh, subname)
	fixation_files = glob.glob(subject_dir_fixation +'/func'+ '/*.pickle')
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
		if (os.path.split(beh_file)[1].split('_')[3]== 'trialinfo.pickle')*(os.path.split(beh_file)[1].split('_')[1]== target_condition):
			target_files_beh.append(beh_file)

	for moco_file in moco_files:
		if os.path.split(moco_file)[1].split('_')[1]== target_condition:
			target_files_moco.append(moco_file)
	# old data set
	# for fixation_file in fixation_files:
	# 	if os.path.split(fixation_file)[1].split('_')[1]== target_condition:
	# 		target_files_fixation.append(fixation_file)
	# new-standard data
	for fixation_file in fixation_files:
		if (os.path.split(fixation_file)[1].split('_')[3]== 'params.pickle')*(os.path.split(fixation_file)[1].split('_')[1]== target_condition):
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

	## check nans
	voxel_list = ld.nan_filter(target_files_fmri, lh, rh)

	## Load all types of data
	file_pairs_all = np.array(zip (target_files_fmri, target_files_beh, target_files_moco, target_files_fixation))

	
	t_runs = [] 
	r_squareds_runs_64 = []
	r_squareds_selection_runs_64 = [] 
	beta_runs_64 = []
	beta_selection_runs_64 = []

	r_squareds_runs_16 = []
	r_squareds_selection_runs_16 = [] 
	beta_runs_16 = []
	beta_selection_runs_16 = []
#0,1,2,
	for fileii, (filename_fmri, filename_beh, filename_moco, filename_fixation) in enumerate(file_pairs_all):		
		# file_pair = file_pairs_all[fileii]
		# filename_fmri = file_pair[0]
		# filename_beh = file_pair[1]
		# filename_moco = file_pair[2]
		# filename_fixation = file_pair[3]

		file_pair = file_pairs_all[fileii]
		filename_fmri = file_pair[0]
		filename_beh = file_pair[1]
		filename_moco = file_pair[2]
		filename_fixation = file_pair[3]

	## Load fmri data--run
		fmri_data = ld.load_fmri(filename_fmri, voxel_list, lh, rh) #
	## Load stimuli order (events)-run
		events_64 = ld.load_event_64channels (filename_beh, fmri_data, empty_start = 15, empty_end = 15, number_of_stimuli = 64)

		events_16 = ld.load_event_16channels (filename_beh, fmri_data, empty_start = 15, empty_end = 15, number_of_stimuli = 8)

	## Load motion correction parameters
		moco_params = pd.read_csv(filename_moco, delim_whitespace=True, header = None) # shape (286,6)
	# ## Load fixation task parameters
		key_press = ld.load_key_press_regressor (filename_fixation, fmri_data)
	### Load stimulus regressor
		stim_regressor = ld.load_stimuli_regressor (filename_beh, fmri_data, empty_start = 15, empty_end = 15)


	# convolve events with hrf, to get model_BOLD_timecourse
		TR = 0.945 #ms
		model_BOLD_timecourse_64 = fftconvolve(events_64, hrf(np.arange(0,30)* TR)[:,np.newaxis],'full')[:fmri_data.shape[1],:]
		design_matrix_64 = np.hstack([model_BOLD_timecourse_64, moco_params, key_press]) #np.ones((fmri_data.shape[1],1)), 
		# 64+6+1 = 71
		model_BOLD_timecourse_16 = fftconvolve(events_16, hrf(np.arange(0,30)* TR)[:,np.newaxis],'full')[:fmri_data.shape[1],:]
		design_matrix_16 = np.hstack([model_BOLD_timecourse_16, moco_params, key_press]) #np.ones((fmri_data.shape[1],1)), 
		#23:  16+6+1 = 23
		# for r_squareds selection
		model_BOLD_timecourse_selection = fftconvolve(stim_regressor, hrf(np.arange(0,30)* TR)[:,np.newaxis],'full')[:fmri_data.shape[1],:]
		design_matrix_selection = np.hstack([model_BOLD_timecourse_selection, moco_params, key_press]) # np.ones((fmri_data.shape[1],1)), 
		# ori_indices = (0:8)
		# col_indices = (8:16)


		shell()
		r_squareds_64, r_squareds_selection_64, betas_64, betas_selection_64, _sse_64, intercept_64, alphas_64 = ma.run_regression(fileii, design_matrix_64, design_matrix_selection, fmri_data, regression = 'RidgeCV')
		#here
		shell()
		r_squareds_16, r_squareds_selection_16, betas_16, betas_selection_16, _sse_16, intercept_16, alphas_16 = ma.run_regression(fileii, design_matrix_16, design_matrix_selection, fmri_data, regression = 'RidgeCV')


		# r_squareds_64, r_squareds_selection_64, betas_64, betas_selection_64, _sse_64, intercept_64, alphas_64 = run_regression(fileii, design_matrix_64, design_matrix_selection, fmri_data, regression = 'RidgeCV')
		# shell()
		# r_squareds_16, r_squareds_selection_16, betas_16, betas_selection_16, _sse_16, intercept_16, alphas_16 = run_regression(fileii, design_matrix_16, design_matrix_selection, fmri_data, regression = 'RidgeCV')

		# # np.argmax(r_squareds)
		# shell()

		shell()
		f1 = plt.figure(figsize = (16,16))
		s1 = f1.add_subplot(5,1,1)
		best_voxel = np.argsort(r_squareds_64)[-1]
		plt.plot(fmri_data[best_voxel])
		plt.plot(design_matrix_64.dot(betas_64[best_voxel] + intercept_64 [best_voxel]))
		s1.set_title('64ch_avgAlpha_%s_r2_%s'%( np.mean(alphas_64, axis =0), str(r_squareds_64[best_voxel])))
		# f1.savefig('%s_64_time_course_%s-[(1,400,400)]'  %(str(subname), str(best_voxel) ))

		#subn001--64-max1265--0.57
		s2 = f1.add_subplot(4,1,2)
		plt.hist(alphas_64, bins = 100)
		s2.set_title('alphas_64_channels')
		# f1.savefig('%s_64_alphas-[(1,500,500)]' ) %(subname )
		# f1.savefig('%s_64_time_course&alphas_%s-[(1,1000,1000)]'  %(str(subname), str(np.argmax(r_squareds_64)) ))

		s3 = f1.add_subplot(4,1,3)
		# plt.figure()
		plt.plot(fmri_data[np.argmax(r_squareds_16)])
		plt.plot(design_matrix_16.dot(betas_16[np.argmax(r_squareds_16)]+intercept_16 [np.argmax(r_squareds_16)]))
		s3.set_title('16ch_avgAlpha_%s_r2_%s'%( np.mean(alphas_16, axis =0), str(r_squareds_16[np.argmax(r_squareds_16)])))
		# savefig('%s_64_time_course_%s-[(1,500,500)]' ) %(str(subname), str(np.argmax(r_squareds_16)) )
		#subn001--16-max4339--0.57

		s4 = f1.add_subplot(4,1,4)
		plt.hist(alphas_16, bins = 100)
		s4.set_title('alphas_16_channels')		
		# savefig('%s_64_alphas-[(1,500,500)]' ) %(str(subname) )

		f1.savefig('%s_64vs16_time_course[(1,400,400)].png'  %(str(subname)  ))


		shell()
		r_squareds_runs_64.append(r_squareds_64) 
		# r_squareds_selection_runs_64.append(r_squareds_selection_64)
		beta_runs_64.append(betas_64[:, 0:64]) 
		# beta_selection_runs_64.append(betas_selection[:, 0:64])
		
		r_squareds_runs_16.append(r_squareds_16) 
		# r_squareds_selection_runs_16.append(r_squareds_selection_16)
		beta_runs_16.append(betas_16[:, 0:16]) 
		# beta_selection_runs_16.append(betas_selection[:, 0:16])


		# # compute contrasts ( t p values )
		# # if type_contrasts == 'full':
		# n_contrasts = 64

		# t, p = ma.calculate_t_p_values (betas, fmri_data, moco_params, key_press, design_matrix, _sse, n_contrasts = 64 )
		# t_runs.append(t)

###  beta values  ---------------------------------------------------------------
###  beta values  ---------------------------------------------------------------
###  beta values  ---------------------------------------------------------------
	shell()
	beta_runs_64 = np.array(beta_runs_64)
	r_squareds_runs_64 = np.array(r_squareds_runs_64)
	beta_runs_16 = np.array(beta_runs_16)
	r_squareds_runs_16 = np.array(r_squareds_runs_16)

	r_squareds_mean_64 = np.mean(r_squareds_runs_64, axis = 0)
	r_squareds_mean_16 = np.mean(r_squareds_runs_16, axis = 0)	

	t_64vs16_r2, p_64vs16_r2 = scipy.stats.ttest_rel(r_squareds_mean_64, r_squareds_mean_16, equal_var=False, nan_policy='omit')

	

	



	shell()
	print 'prepare preference, & a set of tunings for 3 leftin runs. '

	# get voxel_indices_reliVox (indices of reliable voxels)
	voxel_indices_reliVox, n_reli = ma.get_voxel_indices_reliVox( r_squareds_selection_runs, r_squareds_threshold = 0.05, select_100 = False ) 

	beta_runs = np.array(beta_runs)
	r_squareds_runs = np.array(r_squareds_runs)

	# # prepare preference. 
	beta_pre_indices_reliVox = ma.find_preference_matrix_allRuns ( beta_runs, n_reli, voxel_indices_reliVox)



	# a set of tunings for 3 leftin runs. 
	run_nr_all = np.arange(file_pairs_all.shape[0])
	beta_ori_mean_iterations = np.zeros((len(run_nr_all), 9)) 
	beta_col_mean_iterations = np.zeros((len(run_nr_all), 9))

	for filepairii in run_nr_all :
	
		run_nr_leftOut = filepairii
		run_nr_rest = run_nr_all[~(run_nr_all == run_nr_leftOut)]
		beta_mean = np.mean(beta_runs[run_nr_rest], axis = 0)

		beta_ori_reliVox_mean, beta_col_reliVox_mean  = ma.calculate_tunings_matrix (n_reli, voxel_indices_reliVox, beta_mean, beta_pre_indices_reliVox, position_cen)

		beta_ori_mean_iterations [filepairii,:] = beta_ori_reliVox_mean # shape: (4,9)
		beta_col_mean_iterations [filepairii,:] = beta_col_reliVox_mean


# plot figures across iterations!
	print 'plot figures across iterations!'

	pt.plot_tunings(run_nr_all, n_reli, beta_ori_mean_iterations, beta_col_mean_iterations, position_cen = 2)




	# # print t_64vs16_r2, p_64vs16_r2
	# sys.stdout = open('%s_t_64vs16_r2_.txt'%(subname), 'w')
	# # print 'r_squareds_runs_64_%2.f' %(np.mean(r_squareds_runs_64))
	# # print 'r_squareds_runs_16_%2.f' %(np.mean(r_squareds_runs_16))	
	# print 't_64vs16_r2: %.2f' %(t_64vs16_r2)
	# print 'p_64vs16_r2: %.2f' %(p_64vs16_r2)
	# sys.stdout.close()


# ###  t values  ---------------------------------------------------------------
# ###  t values  ---------------------------------------------------------------
# ###  t values  ---------------------------------------------------------------
# 	print 'prepare preference, & a set of tunings for 3 leftin runs. '
# # # prepare preference. 
# 	# get voxel_indices_reliVox (indices of reliable voxels)
# 	voxel_indices_reliVox = np.squeeze(np.where(np.mean(r_squareds_selection_runs, axis = 0) > 0.05)) #2403 voxels in total left, out of 5734
# 	n_reli = voxel_indices_reliVox.shape[0]

# 	t_runs = np.array(t_runs)
# 	r_squareds_runs = np.array(r_squareds_runs)

# 	t_mean_all = np.mean(t_runs, axis = 0)
# 	t_pre_indices_reliVox = np.zeros((n_reli, 2 ))

# 	for voxelii, voxelIndex in enumerate(voxel_indices_reliVox):					

# 		voxelIndex = int(voxelIndex)
# 		t_matrix_pre = t_mean_all [voxelIndex ].reshape(8,8)
# 		# plt.imshow(t_matrix, cmap= plt.cm.ocean)

# 		t_pre_index = np.squeeze(np.where(t_matrix_pre == t_matrix_pre.max()))
# 		# if get two max values, make the first one
# 		if t_pre_index.size != 2:
# 			t_pre_index = np.array([t_pre_index[0][0], t_pre_index[1][0]])

# 		t_pre_indices_reliVox[voxelii, :] = t_pre_index # 7,0 -- 
	

# # a set of tunings for 3 leftin runs. 
# 	run_nr_all = np.arange(file_pairs_all.shape[0])
# 	t_ori_mean_iterations = np.zeros((len(run_nr_all), 9)) 
# 	t_col_mean_iterations = np.zeros((len(run_nr_all), 9))

# 	for filepairii in run_nr_all :
	
# 		run_nr_leftOut = filepairii
# 		run_nr_rest = run_nr_all[~(run_nr_all == run_nr_leftOut)]
# 		t_mean = np.mean(t_runs[run_nr_rest], axis = 0)

# 		t_ori_reliVox = np.zeros((n_reli, 9))
# 		t_col_reliVox = np.zeros((n_reli, 9))
# 		for nrii, voxelIndex in enumerate(voxel_indices_reliVox):
# 			voxelIndex = int(voxelIndex)
# 			t_matrix = t_mean [voxelIndex ].reshape(8,8)
# 			t_pre_current_index = t_pre_indices_reliVox[nrii]

# 			t_matrix_leftOut_cenRow = np.roll(t_matrix, int(position_cen-t_pre_current_index[0]), axis = 0)
# 			t_matrix_leftOut_cen= np.roll(t_matrix_leftOut_cenRow, int(position_cen-t_pre_current_index[1]), axis = 1)

# 			# make it circlar
# 			t_matrix_leftOut_add_column = np.hstack((t_matrix_leftOut_cen, t_matrix_leftOut_cen[:,0][:, np.newaxis]))
# 			t_matrix_leftOut_cir = np.vstack ((t_matrix_leftOut_add_column, t_matrix_leftOut_add_column[0,:]))

# 			t_ori = t_matrix_leftOut_cir[position_cen,:]
# 			t_col = t_matrix_leftOut_cir[:,position_cen]

# 			t_ori_reliVox[nrii,:] = t_ori # shape(2403,9)
# 			t_col_reliVox[nrii,:] = t_col

# 		t_ori_reliVox_mean = np.mean(t_ori_reliVox, axis = 0)  #shape: (9,)
# 		t_col_reliVox_mean = np.mean(t_col_reliVox, axis = 0)

# 		t_ori_mean_iterations [filepairii,:] = t_ori_reliVox_mean # shape: (4,9)
# 		t_col_mean_iterations [filepairii,:] = t_col_reliVox_mean

# # plot figures across iterations!
# 	print 'plot figures across iterations!'

# 	# if position_cen == 2:
# 	t_ori_mean = np.mean(t_ori_mean_iterations, axis = 0)
# 	t_col_mean = np.mean(t_col_mean_iterations, axis = 0)
# 	# t_oriBestVox_mean = np.mean(t_oriBestVox, axis = 0)
# 	# t_colBestVox_mean = np.mean(t_colBestVox, axis = 0)

# 	sd = np.array([np.std(t_ori_mean_iterations, axis = 0), np.std(t_col_mean_iterations, axis = 0)])
# 	n = len(run_nr_all)
# 	yerr = (sd/np.sqrt(n)) #*1.96


# 	f2 = plt.figure(figsize = (8,6))
# 	s1 = f2.add_subplot(211)
# 	plt.plot(t_ori_mean)
# 	plt.errorbar(range(0,9), t_ori_mean, yerr= yerr[0])
# 	# s1.set_title('orientation', fontsize = 10)
# 	s1.set_xticklabels(['placeholder', -45, -22.5, 0, 22.5, 45, 67.5, 90, -67.5 ,-45])
# 	s1.set_xlabel('orientation - relative ')


# 	s2 = f2.add_subplot(212)
# 	plt.plot(t_col_mean)
# 	plt.errorbar(range(0,9), t_col_mean, yerr= yerr[1])
# 	# s2.set_title('color', fontsize = 10)
# 	s2.set_xticklabels(['placeholder', -2, -1, 0, 1, 2, 3, 4, -3, -2])
# 	s2.set_xlabel('color - relative')
# 	f2.savefig( '%s_%s_%s_%s_cen%s_tValues_%sVoxels.png'%(subname, ROI, data_type, regression, position_cen, n_reli))










# 	run_nr_all = np.arange(file_pairs_all.shape[0])

# 	t_ori_mean_iterations = np.zeros((len(run_nr_all), 9)) 
# 	t_col_mean_iterations = np.zeros((len(run_nr_all), 9)) 

# 	t_ori_cate_mean_iterations = np.zeros((len(run_nr_all), 8,9)) 
# 	t_col_cate_mean_iterations = np.zeros((len(run_nr_all), 8,9)) 



# 	for filepairii in run_nr_all :
	
# 		run_nr_leftOut = filepairii
# 		run_nr_rest = run_nr_all[~(run_nr_all == run_nr_leftOut)]
# 		# file_pairs = file_pairs_all[~(np.arange(file_pairs_all.shape[0]) == run_nr_leftOut)]

# 		t_mean = np.mean(t_runs[run_nr_rest], axis = 0) # average across rest runs 		# t_mean shape: (5638,64)		
# 		r_squareds_mean = np.mean(r_squareds_runs[run_nr_rest], axis = 0) 

# 		order = np.argsort(r_squareds_mean)
# 		voxels_all = sorted( zip(order, r_squareds_mean) , key = lambda tup: tup [1] )
# 		n_best = 100
# 		voxels = voxels_all[-n_best:]

# 		voxel_indices_bestVox = np.array(voxels)[:,0]
# 		t_pre_indices_bestVox = np.zeros((n_best, 2 ))

# 		# elif type_contrasts == 'full':
# 	### prepare t_pre_index
# 		if position_cen == 2: 

			
# 			for voxelii, voxelIndex in enumerate(voxel_indices_bestVox):					

# 				voxelIndex = int(voxelIndex)
# 				t_matrix = t_mean [voxelIndex ].reshape(8,8)

# 				t_pre_index = np.squeeze(np.where(t_matrix== t_matrix.max()))
# 				# if get two max values, make the first one
# 				if t_pre_index.size == 2:
# 					pass
# 				else:
# 					t_pre_index = np.array([t_pre_index[0][0], t_pre_index[1][0]])

# 				t_pre_indices_bestVox[voxelii, :] = t_pre_index

		
# 		elif position_cen == 'nan': 
# 			for voxelii, voxelIndex in enumerate(voxel_indices_bestVox):

# 				voxelIndex = int(voxelIndex)
# 				# r_squared_best = voxel[1]

# 				t_matrix = t_mean [voxelIndex ].reshape(8,8)
				
# 				# plt.imshow( t_matrix , cmap= plt.cm.ocean, interpolation = "None")

# 				# center --- exact values: make horizontal as the center of the x-axis, so the labels are still the extact positions.
# 				# so exact positions for x-axis
# 				# make the centers are green / horizontal. 
# 				t_matrix_cenRow = np.roll(t_matrix, 5, axis = 0) # roll downwards by 5 steps, to make the green one as the 4th(0,1,2,3,4), then in the next step, the green one will be at the center.
# 				t_matrix_cen= np.roll(t_matrix_cenRow, 1, axis = 1) # roll to right side by 1 step, to make horizontal be the 4th (be at center through next step)

# 				# note that t_matrix_cen instead of t_matrix, compared with centring to a position
# 				t_pre_index = np.squeeze(np.where(t_matrix_cen== t_matrix_cen.max()))
# 				# if get two max values, make the first one
# 				if t_pre_index.size == 2:
# 					print 'only one preferred stimulus'
# 				else:
# 					t_pre_index = np.array([t_pre_index[0][0], t_pre_index[1][0]])
# 					print 'more than one preferred stimulus', voxelIndex , t_pre_index[0].shape


# 				t_pre_indices_bestVox[voxelii, :] = t_pre_index
# #---------------------------------------------------------
# 	### a set of tunings for the specific leftout run. 

# 		ts_leftOut = t_runs[run_nr_leftOut]
# 		# voxel_indices_bestVox 
# 		# t_pre_indices_bestVox
		
# 		if position_cen == 2:
# 			t_oriBestVox = np.zeros((n_best, 9))
# 			t_colBestVox = np.zeros((n_best, 9))

# 			for nrii, voxelIndex in enumerate(voxel_indices_bestVox):

# 				voxelIndex = int(voxelIndex)
# 				t_matrix_leftOut = ts_leftOut[voxelIndex].reshape(8,8)
# 				t_pre_current_index = t_pre_indices_bestVox[nrii]		

# 				# center --- always move the peaks to the position_cen, 'x axis will be relative positions'
# 				# so the labels of x-axis are not the actual values (ori/color), but the relative position on the axis.
				
# 				t_matrix_leftOut_cenRow = np.roll(t_matrix_leftOut, int(position_cen-t_pre_current_index[0]), axis = 0)
# 				t_matrix_leftOut_cen= np.roll(t_matrix_leftOut_cenRow, int(position_cen-t_pre_current_index[1]), axis = 1)

# 				# make it circlar
# 				t_matrix_leftOut_add_column = np.hstack((t_matrix_leftOut_cen, t_matrix_leftOut_cen[:,0][:, np.newaxis]))
# 				t_matrix_leftOut_cir = np.vstack ((t_matrix_leftOut_add_column, t_matrix_leftOut_add_column[0,:]))


# 				t_ori = t_matrix_leftOut_cir[position_cen,:]
# 				t_col = t_matrix_leftOut_cir[:,position_cen]

		
# 				t_oriBestVox[nrii,:] = t_ori
# 				t_colBestVox[nrii,:] = t_col

# 			t_oriBestVox_mean = np.mean(t_oriBestVox, axis = 0)
# 			t_colBestVox_mean = np.mean(t_colBestVox, axis = 0)

# 			t_ori_mean_iterations [filepairii,:] = t_oriBestVox_mean
# 			t_col_mean_iterations [filepairii,:] = t_colBestVox_mean

# 		elif position_cen == 'nan': 
# 			t_oriBestVox_0 = []
# 			t_colBestVox_0 = []

# 			t_oriBestVox_1 = []
# 			t_colBestVox_1 = []
# 			t_oriBestVox_2 = []
# 			t_colBestVox_2 = []						
# 			t_oriBestVox_3 = []
# 			t_colBestVox_3 = []
# 			t_oriBestVox_4 = []
# 			t_colBestVox_4 = []
# 			t_oriBestVox_5 = []
# 			t_colBestVox_5 = []
# 			t_oriBestVox_6 = []
# 			t_colBestVox_6 = []
# 			t_oriBestVox_7 = []
# 			t_colBestVox_7 = []

# 			t_ori_cate = [t_oriBestVox_0, t_oriBestVox_1, t_oriBestVox_2, t_oriBestVox_3, t_oriBestVox_4, t_oriBestVox_5, t_oriBestVox_6, t_oriBestVox_7]
# 			t_col_cate = [t_colBestVox_0, t_colBestVox_1, t_colBestVox_2, t_colBestVox_3, t_colBestVox_4, t_colBestVox_5, t_colBestVox_6, t_colBestVox_7]

# 			for nrii, voxelIndex in enumerate(voxel_indices_bestVox):

# 				voxelIndex = int(voxelIndex)
# 				t_matrix_leftOut = ts_leftOut[voxelIndex].reshape(8,8)

# 				# center --- exact values: make horizontal as the center of the x-axis, so the labels are still the extact positions.
# 				# so exact positions for x-axis
# 				# make the centers are green / horizontal. 
# 				t_matrix_leftOut_cenRow = np.roll(t_matrix_leftOut, 5, axis = 0) # roll downwards by 5 steps, to make the green one as the 4th(0,1,2,3,4), then in the next step, the green one will be at the center.
# 				t_matrix_leftOut_cen= np.roll(t_matrix_leftOut_cenRow, 1, axis = 1) # roll to right side by 1 step, to make horizontal be the 4th (be at center through next step)

# 				t_pre_current_index = t_pre_indices_bestVox[nrii]

# 				# make it circlar
# 				t_matrix_leftOut_add_column = np.hstack((t_matrix_leftOut_cen, t_matrix_leftOut_cen[:,0][:, np.newaxis]))
# 				t_matrix_leftOut_cir = np.vstack ((t_matrix_leftOut_add_column, t_matrix_leftOut_add_column[0,:]))

# 				t_ori = t_matrix_leftOut_cir[t_pre_current_index[0],:]
# 				t_col = t_matrix_leftOut_cir[:,t_pre_current_index[1]]
				
# 				for i in range(0,8):
# 					if t_pre_current_index[1] == i:
# 						t_ori_cate[i].append(t_ori)
					
# 					if t_pre_current_index[0] == i:
# 						t_col_cate[i].append(t_col)

			
# 			t_ori_cate_mean_leftOut = np.zeros((8,9)) #8 preference locations, 9 points 
# 			t_col_cate_mean_leftOut = np.zeros((8,9)) 

# 			# t_ori_cate_mean_leftOut = np.mean(t_ori_cate, axis = 1) # for each i 

# 			for i in range(0,8):

# 				t_ori_cate[i] = np.array(t_ori_cate[i]) 
# 				t_col_cate[i] = np.array(t_col_cate[i])

# 				t_ori_cate_mean_leftOut[i,:] = np.mean(t_ori_cate[i], axis = 0)
# 				t_col_cate_mean_leftOut[i,:] = np.mean(t_col_cate[i], axis = 0)


# 			t_ori_cate_mean_iterations [filepairii, :, :] =  t_ori_cate_mean_leftOut
# 			t_col_cate_mean_iterations [filepairii, :, :] = t_col_cate_mean_leftOut


# # plot figures! across iterations.

# 	print 'plot figures across iterations!'

# 	if position_cen == 2:

# 		t_ori_mean = np.mean(t_ori_mean_iterations, axis = 0)
# 		t_col_mean = np.mean(t_col_mean_iterations, axis = 0)


# 		# t_oriBestVox_mean = np.mean(t_oriBestVox, axis = 0)
# 		# t_colBestVox_mean = np.mean(t_colBestVox, axis = 0)

# 		sd = np.array([np.std(t_ori_mean_iterations, axis = 0), np.std(t_col_mean_iterations, axis = 0)])
# 		n = len(run_nr_all)
# 		yerr = (sd/np.sqrt(n))*1.96


# 		f2 = plt.figure(figsize = (8,6))
# 		s1 = f2.add_subplot(211)
# 		plt.plot(t_ori_mean)
# 		plt.errorbar(range(0,9), t_ori_mean, yerr= yerr[0])
# 		# s1.set_title('orientation', fontsize = 10)
# 		s1.set_xticklabels(['placeholder', -45, -22.5, 0, 22.5, 45, 67.5, 90, -67.5 ,-45])
# 		s1.set_xlabel('orientation - relative ')


# 		s2 = f2.add_subplot(212)
# 		plt.plot(t_col_mean)
# 		plt.errorbar(range(0,9), t_col_mean, yerr= yerr[1])
# 		# s2.set_title('color', fontsize = 10)
# 		s2.set_xticklabels(['placeholder', -2, -1, 0, 1, 2, 3, 4, -3, -2])
# 		s2.set_xlabel('color - relative')
# 		f2.savefig( '%s_%s_%s_%s_cen%s_tValues_%sVoxels.png'%(subname, ROI, data_type, regression, position_cen, n_best))



# 	elif position_cen == 'nan':
# 		t_ori_cate_mean = np.mean(t_ori_cate_mean_iterations, axis = 0)
# 		t_col_cate_mean = np.mean(t_col_cate_mean_iterations, axis = 0)

# 		t_ori_cate_yerr = np.std(t_ori_cate_mean_iterations, axis = 0)/ np.sqrt(len(run_nr_all)) * 1.96
# 		t_col_cate_yerr = np.std(t_col_cate_mean_iterations, axis = 0)/ np.sqrt(len(run_nr_all)) * 1.96


# 		f3 = plt.figure(figsize = (12,10))

# 		colors1 = plt.cm.rainbow(np.linspace(0, 1, len(range(0,8))))

# 		s1 = f3.add_subplot(2,1,1)
				
# 		for colii, color in enumerate(colors1, start =0):
# 			if t_ori_cate_mean[colii].size == 1:
# 				s1.set_title('no voxels fall into peak position: %s' %(str(colii)) , fontsize = 10)

# 			else:
# 				plt.plot(t_ori_cate_mean[colii], color = color, label = 'position:%s' %(str(colii)))
# 				plt.errorbar(range(0,9), t_ori_cate_mean[colii], color = color, yerr= t_ori_cate_yerr[colii])
# 			plt.legend(loc='best')
		
# 		s1.set_xticklabels(['placeholder', 'vertical 90', -67.5, -45, -22.5, 'horizontal 0', 22.5, 45, 67.5, 'vertical 90'])
# 		s1.set_xlabel('orientation - absolute')
# 		# f3.savefig( '%s_%s_%s_%sOri_8pos_tValues_%sVoxels.png'%(subname, ROI, data_type, regression, n_best))

		
# 		# f4 = plt.figure(figsize = (12,10))
# 		s2 = f3.add_subplot(2,1,2)
# 		# use colors in the mapping experiment
# 		# Compute evenly-spaced steps in (L)ab-space

# 		color_theta = (np.pi*2)/8
# 		color_angle = color_theta * np.arange(0, 8,dtype=float)
# 		color_radius = 75
 
# 		color_a = color_radius * np.cos(color_angle)
# 		color_b = color_radius * np.sin(color_angle)

# 		colors2 = np.array([ct.lab2rgb((55, a, b)) for a,b in zip(color_a, color_b)])	
# 		colors2 = np.hstack((colors2/255, np.ones((8,1))))

# 		for colii, color in enumerate(colors2, start =0):
# 			if t_col_cate_mean[colii].size == 1:
# 				s2.set_title('no voxels fall into peak position: %s' %(str(colii)) , fontsize = 10)

# 			else:
# 				plt.plot(t_col_cate_mean[colii], color = color , label = 'position:%s' %(str(colii)))
# 				plt.errorbar(range(0,9), t_col_cate_mean[colii], color = color, yerr= t_col_cate_yerr[colii])
# 			plt.legend(loc='best')

# 		s2.set_xlabel('color - absolute')
# 		# f4.savefig( '%s_%s_%s_%sCol_8pos_tValues_%sVoxels.png'%(subname, ROI, data_type, regression, n_best))
# 		f3.savefig( '%s_%s_%s_%s_8pos_tValues_%sVoxels.png'%(subname, ROI, data_type, regression, n_best))





# 			t_oriBestVox_mean_0 = []
# 			t_oriBestVox_mean_1 = []
# 			t_oriBestVox_mean_2 = []
# 			t_oriBestVox_mean_3 = []
# 			t_oriBestVox_mean_4 = []
# 			t_oriBestVox_mean_5 = []
# 			t_oriBestVox_mean_6 = []
# 			t_oriBestVox_mean_7 = []

# 			t_colBestVox_mean_0 = []
# 			t_colBestVox_mean_1 = []
# 			t_colBestVox_mean_2 = []
# 			t_colBestVox_mean_3 = []
# 			t_colBestVox_mean_4 = []
# 			t_colBestVox_mean_5 = []
# 			t_colBestVox_mean_6 = []
# 			t_colBestVox_mean_7 = []

# 			t_oriBestVox_yerr_0 = []
# 			t_oriBestVox_yerr_1 = []
# 			t_oriBestVox_yerr_2 = []
# 			t_oriBestVox_yerr_3 = []
# 			t_oriBestVox_yerr_4 = []
# 			t_oriBestVox_yerr_5 = []
# 			t_oriBestVox_yerr_6 = []
# 			t_oriBestVox_yerr_7 = []

# 			t_colBestVox_yerr_0 = []
# 			t_colBestVox_yerr_1 = []
# 			t_colBestVox_yerr_2 = []
# 			t_colBestVox_yerr_3 = []
# 			t_colBestVox_yerr_4 = []
# 			t_colBestVox_yerr_5 = []
# 			t_colBestVox_yerr_6 = []
# 			t_colBestVox_yerr_7 = []

# 			t_ori_cate_mean = [t_oriBestVox_mean_0, t_oriBestVox_mean_1, t_oriBestVox_mean_2, t_oriBestVox_mean_3, t_oriBestVox_mean_4, t_oriBestVox_mean_5, t_oriBestVox_mean_6, t_oriBestVox_mean_7]
# 			t_col_cate_mean = [t_colBestVox_mean_0, t_colBestVox_mean_1, t_colBestVox_mean_2, t_colBestVox_mean_3, t_colBestVox_mean_4, t_colBestVox_mean_5, t_colBestVox_mean_6, t_colBestVox_mean_7]

# 			t_ori_cate_yerr = [t_oriBestVox_yerr_0, t_oriBestVox_yerr_1, t_oriBestVox_yerr_2, t_oriBestVox_yerr_3, t_oriBestVox_yerr_4, t_oriBestVox_yerr_5, t_oriBestVox_yerr_6, t_oriBestVox_yerr_7]
# 			t_col_cate_yerr = [t_colBestVox_yerr_0, t_colBestVox_yerr_1, t_colBestVox_yerr_2, t_colBestVox_yerr_3, t_colBestVox_yerr_4, t_colBestVox_yerr_5, t_colBestVox_yerr_6, t_colBestVox_yerr_7]

		
# 			for i in range(0,8):

# 				t_ori_cate[i] = np.array(t_ori_cate[i]) 
# 				t_col_cate[i] = np.array(t_col_cate[i])

# 				t_ori_cate_mean[i] = np.mean(t_ori_cate[i], axis = 0)
# 				t_ori_cate_yerr[i] = np.std(t_ori_cate[i], axis = 0)/ np.sqrt(t_ori_cate[i].shape[0]) * 1.96

# 				t_col_cate_mean[i] = np.mean(t_col_cate[i], axis = 0)
# 				t_col_cate_yerr[i] = np.std(t_col_cate[i], axis = 0)/ np.sqrt(t_col_cate[i].shape[0]) * 1.96

# # ### 16 subplots, 8 for each condition
# # 			f3 = plt.figure(figsize = (32,22))
# # 			for i in range(0,8):

# # 				if t_ori_cate_mean[i].size == 1:
# # 					s1 = f3.add_subplot(4,4,i+1)
# # 					s1.set_title('no voxels fall into peak position:%s' %(str(i)) , fontsize = 10)
# # 				else:
# # 					s1 = f3.add_subplot(4,4,i+1)
# # 					plt.plot(t_ori_cate_mean[i])
# # 					plt.errorbar(range(0,9), t_ori_cate_mean[i], yerr= t_ori_cate_yerr[i])
# # 					s1.set_xlabel('orientation')
# # 					s1.set_title('n_voxels:%s, peak position:%s' % (str(t_col_cate[i].shape[0]), str(i)) , fontsize = 10)
					
# # 				if t_col_cate_mean[i].size == 1:
# # 					s2 = f3.add_subplot(4,4,i+9)
# # 					s2.set_title('no voxels fall into peak position:%s' %(str(i)) , fontsize = 10)
# # 				else:
# # 					s2 = f3.add_subplot(4,4,i+9)
# # 					plt.plot(t_col_cate_mean[i])
# # 					plt.errorbar(range(0,9), t_col_cate_mean[i], yerr= t_col_cate_yerr[i])
# # 					s2.set_xlabel('color')
# # 					s2.set_title('n_voxels:%s, peak position:%s' % (str(t_col_cate[i].shape[0]), str(i)) , fontsize = 10)

# # 			f3.savefig( '%s_%s_%s_%s_8pos_tValues_%sVoxels.png'%(subname, ROI, data_type, regression, n_best))

# ### 2 subplots, with all different colors
# # cmap= plt.cm.ocean


# 			# f3 = plt.figure(figsize = (12,10))
			

# 			f3 = plt.figure(figsize = (12,10))

# 			colors1 = plt.cm.rainbow(np.linspace(0, 1, len(range(0,8))))

# 			s1 = f3.add_subplot(2,1,1)
					
# 			for colii, color in enumerate(colors1, start =0):
# 				if t_ori_cate_mean[colii].size == 1:
# 					s1.set_title('no voxels fall into peak position: %s' %(str(colii)) , fontsize = 10)

# 				else:
# 					plt.plot(t_ori_cate_mean[colii], color = color) #, label = 'position:%s' %(str(colii)))
# 					plt.errorbar(range(0,9), t_ori_cate_mean[colii], color = color, yerr= t_ori_cate_yerr[colii])

# 			# 		leg = s1.legend()

# 			# for color,text in zip(colors,leg.get_texts()):
# 	#  			text.set_color(color)

# 				# plt.legend(loc='best')
# 				# plt.close()
# 			s1.set_xticklabels(['placeholder', 'vertical 90', -67.5, -45, -22.5, 'horizontal 0', 22.5, 45, 67.5, 'vertical 90'])
# 			s1.set_xlabel('orientation')
# 			f3.savefig( '%s_%s_%s_%sOri_8pos_tValues_%sVoxels.png'%(subname, ROI, data_type, regression, n_best))

			
# 			# f4 = plt.figure(figsize = (12,10))
# 			s2 = f3.add_subplot(2,1,2)
# 			# colors2 = plt.cm.gist_rainbow(np.linspace(0, 1, len(range(0,8))))  # gist_rainbow not rainbow
# 			# use colors in the mapping experiment
# 		# Compute evenly-spaced steps in (L)ab-space

# 			color_theta = (np.pi*2)/8
# 			color_angle = color_theta * np.arange(0, 8,dtype=float)
# 			color_radius = 75

# 			color_a = color_radius * np.cos(color_angle)
# 			color_b = color_radius * np.sin(color_angle)

# 			colors2 = np.array([ct.lab2rgb((55, a, b)) for a,b in zip(color_a, color_b)])	

# 			colors2 = np.hstack((colors2/255, np.ones((8,1))))

# 			for colii, color in enumerate(colors2, start =0):
# 				if t_col_cate_mean[colii].size == 1:
# 					s2.set_title('no voxels fall into peak position: %s' %(str(colii)) , fontsize = 10)

# 				else:
# 					plt.plot(t_col_cate_mean[colii], color = color) #, label = 'position:%s' %(str(colii)))
# 					plt.errorbar(range(0,9), t_col_cate_mean[colii], color = color, yerr= t_col_cate_yerr[colii])
				
# 				# plt.legend(loc='best')

# 			# s1.set_xticklabels(['pink', 'orange', 'yellow', 'green', 'light_green', 'blue', 'dark_blue', 67.5, 90])
# 			s2.set_xlabel('color')
# 			# f4.savefig( '%s_%s_%s_%sCol_8pos_tValues_%sVoxels.png'%(subname, ROI, data_type, regression, n_best))
# 			f3.savefig( '%s_%s_%s_%s_8pos_tValues_%sVoxels.png'%(subname, ROI, data_type, regression, n_best))

		# plt.close()
	# f1 = plt.figure(figsize = (8,6))

	# plt.plot(t_20)
	# plt.close()

	# f1.savefig('%s_%s_%s_run%s_AVERAGE20_tValues.png'%(subname, data_type, ROI, str(run_nr) ))



