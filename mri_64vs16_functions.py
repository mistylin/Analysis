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
from tools import two_gamma as hrf_fun

import mri_load_data as ld
import mri_main_analysis as ma
import mri_plot_tunings as pt
import mri_statistical_analysis as sa

import numpy as np
import pyvttbl as pt
from collections import namedtuple
from voxel_lists import *

#-------------------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------------------
 # without def 


def roate_90_clockwise ( myarray ):

	x = np.arange(0, len(myarray) )
	y = myarray

	x_new = y
	y_new = len(myarray)-1 -x 

sublist = [ ('sub-n001', False, False), ('sub-n003', False, False), ('sub-n005', False, False) ]#('sub-n001', False, False), 
data_dir_fmri = '/home/shared/2017/visual/OriColorMapper/preproc/'
data_dir_beh = '/home/shared/2017/visual/OriColorMapper/bids_converted/'

data_type = 'psc'#'tf' #'psc'
each_run = True #False #True #False
ROI = 'V1' # 'V4'
regression = 'RidgeCV' #'GLM' #'RidgeCV'
position_cen = 2 #2 4  #'nan'


aov_values = []
t_64vs8ori = []
t_64vs8col = []
t_8oriVs8col = []

for subii, sub in enumerate(sublist):

	shell()

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

	shell()
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
	# voxel_list = ld.voxel_filter(target_files_fmri, lh, rh)
	
	# voxel_list = voxel_lists[subii]
	
	##subn001
	# voxel_list = [voxel_lists[subii][442] ]
	voxel_list = [voxel_lists[subii][442], voxel_lists[subii][5046], voxel_lists[subii][684], voxel_lists[subii][415], voxel_lists[subii][5059], voxel_lists[subii][2194], voxel_lists[subii][5169], voxel_lists[subii][387], voxel_lists[subii][3544], voxel_lists[subii][428]]
	##subn005
	# voxel_list = [voxel_lists[subii][2224], voxel_lists[subii][4687]]
# 	[442], [5046], [684], [415], [5059], [2194], [5169], [387], [3544], [428]
# a: 415"4" 428"10" 442"1"  2194"6" 5046 "2"
# s: 387"8" 684"3" 3544"9" 5059"5" 5169"7"
# a: 5-6, 1-1.5, 1-1.5(0.4)
# s:same

	## Load all types of data
	file_pairs_all = np.array(zip (target_files_fmri, target_files_beh, target_files_moco, target_files_fixation))

	
	# t_runs = [] 
	r_squareds_runs_64 = []
	r_squareds_selection_runs = [] 
	beta_runs_64 = []
	beta_selection_runs = []

	r_squareds_runs_8_ori = []
	beta_runs_8_ori = []

	r_squareds_runs_8_col = []
	beta_runs_8_col = []

	beta_runs_64_allRegressors = []
	beta_runs_8_ori_allRegressors = []
	beta_runs_8_col_allRegressors = []
	beta_selection_runs_allRegressors = []

	intercept_runs_64 = []
	intercept_runs_8_ori = []
	intercept_runs_8_col = []
	intercept_runs_selection = []

	alphas_runs_64 = []
	alphas_runs_8_ori = []
	alphas_runs_8_col = []
	alphas_runs_selection = []

	fmri_data_runs = []

	events_64_runs =[]
	events_8_ori_runs =[]
	events_8_col_runs =[]
	moco_params_runs =[]
	key_press_runs  =[]

	design_matrix_64_runs = []
	design_matrix_8_ori_runs = []
	design_matrix_8_col_runs = []
	shell()
	for fileii, (filename_fmri, filename_beh, filename_moco, filename_fixation) in enumerate(file_pairs_all):	
		# shell()
		#0,1,2,3
		file_pair = file_pairs_all[fileii]
		filename_fmri = file_pair[0]
		filename_beh = file_pair[1]
		filename_moco = file_pair[2]
		filename_fixation = file_pair[3]

	## Load fmri data--run
		fmri_data = ld.load_fmri(filename_fmri, voxel_list, lh, rh) #

		# psc = fmri_data.sum(axis=1)
		# voxel_list_new = np.argsort(psc)[-50:].flatten()
		# fmri_data = ld.load_fmri(filename_fmri, voxel_list_new, lh, rh)

		fmri_data_runs.append(fmri_data)

		# for i in range(fmri_data.shape[0]):
		# 	f = plt.figure(figsize = (12,4))
		# 	plt.plot(fmri_data[i, :])
		# 	# f.savefig('442.pdf')

		# 	f.savefig( 'fmri_data_voxel_#%i.pdf' % voxel_list_new[i]  )
		# 	plt.close()
		# 	print 'plotting fmri_data_voxel_#%i' % voxel_list_new[i]


	## Load stimuli order (events)-run
		events_64 = ld.load_event_64channels (filename_beh, fmri_data, empty_start = 15, empty_end = 15, number_of_stimuli = 64)
		
		events_8_ori, events_8_col = ld.load_event_16channels (filename_beh, fmri_data, empty_start = 15, empty_end = 15, number_of_stimuli = 8)


		# Only one event was created at the specific TR when a stimulus was presented.

	## Load motion correction parameters
		moco_params = pd.read_csv(filename_moco, delim_whitespace=True, header = None) # shape (286,6)
	# ## Load fixation task parameters
		key_press = ld.load_key_press_regressor (filename_fixation, fmri_data)

	### Load stimulus regressor
		stim_regressor = ld.load_stimuli_regressor (filename_beh, fmri_data, empty_start = 15, empty_end = 15)

		events_64_runs.append(events_64)
		events_8_ori_runs.append(events_8_ori)
		events_8_col_runs.append(events_8_col)
		moco_params_runs.append(moco_params)
		key_press_runs .append(key_press)

	# convolve events with hrf, to get model_BOLD_timecourse
		TR = 0.945 #ms
		hrf = hrf_fun(np.arange(0,30)* TR)#[:,np.newaxis]
		hrf_dt = np.r_[0,np.diff(hrf)]

		events_64_doubled = np.repeat(events_64, 2, axis = 1)
		model_BOLD_timecourse_64 = np.zeros(events_64_doubled.shape)
		model_BOLD_timecourse_64[:,::2] = fftconvolve(events_64_doubled[:,::2], hrf[:,np.newaxis],'full')[:fmri_data.shape[1],:]
		model_BOLD_timecourse_64[:,1::2] = fftconvolve(events_64_doubled[:,1::2], hrf_dt[:,np.newaxis],'full')[:fmri_data.shape[1],:]

		# model_BOLD_timecourse_64 = fftconvolve(events_64, hrf[:,np.newaxis],'full')[:fmri_data.shape[1],:]
		design_matrix_64 = np.hstack([model_BOLD_timecourse_64, moco_params, key_press]) #np.ones((fmri_data.shape[1],1)), 
		# 64*2+6+1 = 
		# [:, 0:128:2], [:, 0:16:2]
		# [:, -7:]
		design_matrix_64_runs.append(design_matrix_64)

		events_8_ori_doubled = np.repeat(events_8_ori, 2, axis = 1)
		model_BOLD_timecourse_8_ori = np.zeros(events_8_ori_doubled.shape)
		model_BOLD_timecourse_8_ori[:,::2] = fftconvolve(events_8_ori_doubled[:,::2], hrf[:,np.newaxis],'full')[:fmri_data.shape[1],:]
		model_BOLD_timecourse_8_ori[:,1::2] = fftconvolve(events_8_ori_doubled[:,1::2], hrf_dt[:,np.newaxis],'full')[:fmri_data.shape[1],:]
		# model_BOLD_timecourse_8_ori = fftconvolve(events_8_ori, hrf[:,np.newaxis],'full')[:fmri_data.shape[1],:]
		design_matrix_8_ori = np.hstack([model_BOLD_timecourse_8_ori, moco_params, key_press]) #np.ones((fmri_data.shape[1],1)), 

		design_matrix_8_ori_runs.append(design_matrix_8_ori)

		events_8_col_doubled = np.repeat(events_8_col, 2, axis = 1)
		model_BOLD_timecourse_8_col = np.zeros(events_8_col_doubled.shape)
		model_BOLD_timecourse_8_col[:,::2] = fftconvolve(events_8_col_doubled[:,::2], hrf[:,np.newaxis],'full')[:fmri_data.shape[1],:]
		model_BOLD_timecourse_8_col[:,1::2] = fftconvolve(events_8_col_doubled[:,1::2], hrf_dt[:,np.newaxis],'full')[:fmri_data.shape[1],:]
		# model_BOLD_timecourse_8_col = fftconvolve(events_8_col, hrf[:,np.newaxis],'full')[:fmri_data.shape[1],:]
		design_matrix_8_col = np.hstack([model_BOLD_timecourse_8_col, moco_params, key_press]) #np.ones((fmri_data.shape[1],1)), 

		design_matrix_8_col_runs.append(design_matrix_8_col )
		#:  8*2+6+1 = 23 (with time derivative)

		# # for r_squareds selection
		model_BOLD_timecourse_selection = fftconvolve(stim_regressor, hrf[:,np.newaxis],'full')[:fmri_data.shape[1],:]
		design_matrix_selection = np.hstack([model_BOLD_timecourse_selection, moco_params, key_press]) # np.ones((fmri_data.shape[1],1)), 

		r_squareds_64, betas_64, _sse_64, intercept_64, alphas_64 = ma.run_regression(fileii, design_matrix_64, fmri_data, regression = 'RidgeCV')
		r_squareds_8_ori, betas_8_ori, _sse_8_ori, intercept_8_ori, alphas_8_ori = ma.run_regression(fileii, design_matrix_8_ori, fmri_data, regression = 'RidgeCV')
		r_squareds_8_col, betas_8_col, _sse_8_col, intercept_8_col, alphas_8_col = ma.run_regression(fileii, design_matrix_8_col, fmri_data, regression = 'RidgeCV')

		r_squareds_selection, betas_selection, _sse_selection, intercept_selection, alphas_selection = ma.run_regression(fileii, design_matrix_selection, fmri_data, regression = 'RidgeCV')


		# r_squareds_64, r_squareds_selection_64, betas_64, betas_selection_64, _sse_64, intercept_64, alphas_64 = run_regression(fileii, design_matrix_64, design_matrix_selection, fmri_data, regression = 'RidgeCV')
		# r_squareds_16, r_squareds_selection_16, betas_16, betas_selection_16, _sse_16, intercept_16, alphas_16 = run_regression(fileii, design_matrix_16, design_matrix_selection, fmri_data, regression = 'RidgeCV')

		# pt.plot_3models_timeCourse_alphaHist(fmri_data, r_squareds_64, design_matrix_64, betas_64, intercept_64, alphas_64, r_squareds_8_ori, design_matrix_8_ori, betas_8_ori, intercept_8_ori, alphas_8_ori, r_squareds_8_col, design_matrix_8_col, betas_8_col, intercept_8_col, alphas_8_col, subname, fileii): 
		# plot_3models_timeCourse_alphaHist(fmri_data, r_squareds_64, design_matrix_64, betas_64, intercept_64, alphas_64, r_squareds_8_ori, design_matrix_8_ori, betas_8_ori, intercept_8_ori, alphas_8_ori, r_squareds_8_col, design_matrix_8_col, betas_8_col, intercept_8_col, alphas_8_col, subname, fileii)

		r_squareds_runs_64.append(r_squareds_64) 
		beta_runs_64.append(betas_64[:, 0:128:2]) 
		beta_runs_64_allRegressors.append(betas_64)
		intercept_runs_64.append(intercept_64)
		alphas_runs_64.append(alphas_64)

		r_squareds_runs_8_ori.append(r_squareds_8_ori) 
		beta_runs_8_ori.append(betas_8_ori[:, 0:16:2]) 
		beta_runs_8_ori_allRegressors.append(betas_8_ori)
		intercept_runs_8_ori.append(intercept_8_ori)
		alphas_runs_8_ori.append(alphas_8_ori)


		r_squareds_runs_8_col.append(r_squareds_8_col) 
		beta_runs_8_col.append(betas_8_col[:, 0:16:2]) 
		beta_runs_8_col_allRegressors.append(betas_8_col)
		intercept_runs_8_col.append(intercept_8_col)
		alphas_runs_8_col.append(alphas_8_col)

		r_squareds_selection_runs.append(r_squareds_selection)
		beta_selection_runs_allRegressors.append(betas_selection)
		intercept_runs_selection.append(intercept_selection)
		alphas_runs_selection.append(alphas_selection)

###  beta values  ---------------------------------------------------------------
###  beta values  ---------------------------------------------------------------
###  beta values  ---------------------------------------------------------------
	shell()
	beta_runs_64 = np.array(beta_runs_64)
	r_squareds_runs_64 = np.array(r_squareds_runs_64)
	beta_runs_8_ori = np.array(beta_runs_8_ori)
	beta_runs_8_col = np.array(beta_runs_8_col)
	r_squareds_runs_8_ori = np.array(r_squareds_runs_8_ori)
	r_squareds_runs_8_col = np.array(r_squareds_runs_8_col)

	r_squareds_mean_64 = np.mean(r_squareds_runs_64, axis = 0)
	r_squareds_mean_8_ori = np.mean(r_squareds_runs_8_ori, axis = 0)	
	r_squareds_mean_8_col = np.mean(r_squareds_runs_8_col, axis = 0)

##### stat test
	# repeated measure
	aov = sa.repeated_anova(fmri_data, r_squareds_mean_64, r_squareds_mean_8_ori, r_squareds_mean_8_col )


	# post-hoc
	t_64vs8ori_r2_rel, p_64vs8ori_r2_rel = scipy.stats.ttest_rel(r_squareds_mean_64, r_squareds_mean_8_ori)
	t_64vs8col_r2_rel, p_64vs8col_r2_rel = scipy.stats.ttest_rel(r_squareds_mean_64, r_squareds_mean_8_col)
	t_8oriVs8col_r2_rel, p_8oriVs8col_r2_rel = scipy.stats.ttest_rel(r_squareds_mean_8_ori, r_squareds_mean_8_col)

	print 't_64vs8ori_r2_rel: %.2f, p_64vs8ori_r2_rel: %.2f, correction: %f' %(t_64vs8ori_r2_rel, p_64vs8ori_r2_rel, p_64vs8ori_r2_rel*len(voxel_list)*3 )
	print 't_64vs8col_r2_rel: %.2f, p_64vs8col_r2_rel: %.2f, correction: %f ' %(t_64vs8col_r2_rel, p_64vs8col_r2_rel, p_64vs8ori_r2_rel*len(voxel_list)*3 )
	print 't_8oriVs8col_r2_rel: %.2f, p_8oriVs8col_r2_rel: %.2f, correction: %f' %(t_8oriVs8col_r2_rel, p_8oriVs8col_r2_rel, p_64vs8ori_r2_rel*len(voxel_list)*3 )

	# t_64vs8ori_r2_rel: 114.14, p_64vs8ori_r2_rel: 0.00
	# t_64vs8col_r2_rel: 112.13, p_64vs8col_r2_rel: 0.00
	# t_8oriVs8col_r2_rel: 30.36, p_8oriVs8col_r2_rel: 0.00

	aov_values.append(aov)
	t_64vs8ori.append((t_64vs8ori_r2_rel, p_64vs8ori_r2_rel))
	t_64vs8col.append((t_64vs8col_r2_rel, p_64vs8col_r2_rel))
	t_8oriVs8col.append((t_8oriVs8col_r2_rel, p_8oriVs8col_r2_rel))
	##(115.18453285570681, 0.0), (113.0876960551475, 0.0), (30.362206735333434, 6.6824459323599391e-188)

##### leave 3 in :  orientation and color tunings

	print 'prepare preference, & a set of tunings for 3 leftin runs. '

	# get voxel_indices_reliVox (indices of reliable voxels)
	voxel_indices_reliVox, n_reli = ma.get_voxel_indices_reliVox( r_squareds_selection_runs, r_squareds_threshold = 0.05, select_100 = True ) 
	# beta_runs = np.array(beta_runs)
	# r_squareds_runs = np.array(r_squareds_runs)


	# # prepare preference. 
	beta_pre_indices_reliVox = ma.find_preference_matrix_allRuns ( beta_runs_64, n_reli, voxel_indices_reliVox)


	# a set of tunings for 3 leftin runs. 
	run_nr_all = np.arange(file_pairs_all.shape[0])
	beta_ori_mean_iterations = np.zeros((len(run_nr_all), 9)) 
	beta_col_mean_iterations = np.zeros((len(run_nr_all), 9))

	for filepairii in run_nr_all :
	
		run_nr_leftOut = filepairii
		run_nr_rest = run_nr_all[~(run_nr_all == run_nr_leftOut)]
		beta_mean = np.mean(beta_runs_64[run_nr_rest], axis = 0)

		beta_ori_reliVox_mean, beta_col_reliVox_mean  = ma.calculate_tunings_matrix (n_reli, voxel_indices_reliVox, beta_mean, beta_pre_indices_reliVox, position_cen)

		beta_ori_mean_iterations [filepairii,:] = beta_ori_reliVox_mean # shape: (4,9)
		beta_col_mean_iterations [filepairii,:] = beta_col_reliVox_mean


	# plot figures across iterations!
	print 'plot figures across iterations!'

	# pt.plot_tunings(run_nr_all, n_reli, beta_ori_mean_iterations, beta_col_mean_iterations, position_cen = 2)

	position_cen == 2
	beta_ori_mean = np.mean(beta_ori_mean_iterations, axis = 0)
	beta_col_mean = np.mean(beta_col_mean_iterations, axis = 0)
	# beta_oriBestVox_mean = np.mean(beta_oriBestVox, axis = 0)
	# beta_colBestVox_mean = np.mean(beta_colBestVox, axis = 0)

	sd = np.array([np.std(beta_ori_mean_iterations, axis = 0), np.std(beta_col_mean_iterations, axis = 0)])
	n = len(run_nr_all)
	yerr = (sd/np.sqrt(n)) #*1.96


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
	f2.savefig( '%s_%s_%s_%s_cen%s_betaValues_%sVoxels.pdf'%(subname, ROI, data_type, regression, position_cen, n_reli))
	print 'plotting  %s_%s_%s_%s_cen%s_betaValues_%sVoxels.pdf'%(subname, ROI, data_type, regression, position_cen, n_reli)






#### circle or oval? the best voxel
	beta_64_across_runs = np.mean(beta_runs_64, axis = 0)

	fmri_data_runs = np.array(fmri_data_runs)
	beta_runs_64_allRegressors = np.array(beta_runs_64_allRegressors)
	intercept_runs_64  = np.array(intercept_runs_64 )
	alphas_runs_64 = np.array(alphas_runs_64)

	beta_runs_8_ori_allRegressors = np.array(beta_runs_8_ori_allRegressors)	
	intercept_runs_8_ori = np.array(intercept_runs_8_ori)
	alphas_runs_8_ori = np.array(alphas_runs_8_ori)

	beta_runs_8_col_allRegressors = np.array(beta_runs_8_col_allRegressors)	
	intercept_runs_8_col = np.array(intercept_runs_8_col)
	alphas_runs_8_col = np.array(alphas_runs_8_col)

	events_64_runs = np.array(events_64_runs)
	events_8_ori_runs = np.array(events_8_ori_runs)
	events_8_col_runs = np.array(events_8_col_runs)

	design_matrix_64_runs = np.array(design_matrix_64_runs)
	design_matrix_8_ori_runs = np.array(design_matrix_8_ori_runs)
	design_matrix_8_col_runs = np.array(design_matrix_8_col_runs)


	shell()
	for modelii in ['64','8ori','8col']:
		for volii in np.arange(1,31,1): #(1,31,1)
			# pt.plot_3ModelFit_beta_matrix(subname, volii, r_squareds_64, beta_64_across_runs, fmri_data, design_matrix_64, betas_64, intercept_64, alphas_64, r_squareds_8_ori, design_matrix_8_ori, betas_8_ori, intercept_8_ori, alphas_8_ori, r_squareds_8_col, design_matrix_8_col, betas_8_col, intercept_8_col, alphas_8_col) 
			# plot_3ModelFit_beta_matrix(subname, volii, r_squareds_64, beta_64_across_runs, fmri_data, design_matrix_64, betas_64, intercept_64, alphas_64, r_squareds_8_ori, design_matrix_8_ori, betas_8_ori, intercept_8_ori, alphas_8_ori, r_squareds_8_col, design_matrix_8_col, betas_8_col, intercept_8_col, alphas_8_col) 
			if modelii == '64':
				best_voxel_index = np.argsort(r_squareds_mean_64)[-volii]
			elif modelii == '8ori':
				best_voxel_index = np.argsort(r_squareds_mean_8_ori)[-volii]
			elif modelii == '8col':
				best_voxel_index = np.argsort(r_squareds_mean_8_col)[-volii]
			
			# beta_64_best_voxel = beta_64_across_runs [best_voxel_index]

			# modelii = '64'
			# best_voxel_index = 0


			# run_nr_all = np.arange(file_pairs_all.shape[0])			
			# # for x in range(voxel_list):
			# # 	best_voxel_index = x

			# 	f = plt.figure(figsize = (12,12))
			# 	for i in run_nr_all:
			# 		s1=f.add_subplot(4,1,i+1)

			# 		plt.plot(fmri_data_runs[i, best_voxel_index, :])
			# 		plt.plot(design_matrix_64_runs[i].dot(beta_runs_64_allRegressors[ i, best_voxel_index, :] + intercept_runs_64 [ i , best_voxel_index]), 'g')
			# 		s1.set_xlabel('64_run%i_alpha_%s_r2_[%.2f]'%( i, str(alphas_runs_8_ori[ i, best_voxel_index]), r_squareds_runs_8_ori[ i, best_voxel_index]), fontsize = 10)

			# 	f.savefig('%s-%s_%s-modelFit&Events.pdf'%(subname, '64', str(x) ))	
		
			# 	f = plt.figure(figsize = (12,12))
			# 	for i in run_nr_all:
			# 		s1=f.add_subplot(4,1,i+1)
			# 		plt.plot(fmri_data_runs[i, best_voxel_index, :])
			# 		plt.plot(design_matrix_8_ori_runs[i].dot(beta_runs_8_ori_allRegressors[ i, best_voxel_index, :] + intercept_runs_8_ori [ i, best_voxel_index]))
			# 		# plt.plot(events_8_ori_runs[i])
			# 		s1.set_xlabel('8ch_ori_run%i_alpha_%s_r2_[%.2f]'%( i, str(alphas_runs_8_ori[ i, best_voxel_index]), r_squareds_runs_8_ori[ i, best_voxel_index]), fontsize = 10)
			# 	f.savefig('%s-%s_%s-modelFit&Events.pdf'%(subname, '8chOri', str(x)  ))

			# 	f = plt.figure(figsize = (12,12))
			# 	for i in run_nr_all:
			# 		s1=f.add_subplot(4,1,i+1)
			# 		plt.plot(fmri_data_runs[i, best_voxel_index, :])
			# 		plt.plot(design_matrix_8_col_runs[i].dot(beta_runs_8_col_allRegressors[ i, best_voxel_index, :] + intercept_runs_8_col [ i, best_voxel_index]))
			# 		# plt.plot(events_8_col_runs[i])
			# 		s1.set_xlabel('8ch_col_run%i_alpha_%s_r2_[%.2f]'%( i, str(alphas_runs_8_col[ i, best_voxel_index]), r_squareds_runs_8_col[ i, best_voxel_index]), fontsize = 10)
			# 	# f.savefig('%s-%s_%s-modelFit&Events.pdf'%(subname, '8chCol', str(x)  ))


###----------------------------------------------------------------------------
			# run_nr_all = np.arange(file_pairs_all.shape[0])
			# f = plt.figure(figsize = (12*len(run_nr_all),12))
			# gs=GridSpec(3,len(run_nr_all)) # (2,3)2 rows, 3 columns
			# gs=GridSpec(3,1) # (2,3)2 rows, 3 columns			
			# 1. fmri data & model BOLD response
			# s1 = f.add_subplot(4,1,1)


			# names = ['442', '5046', '684', '415', '5059', '2194', '5169', '387', '3544', '428']
			# for y in range(len(names)):
			# 	best_voxel_index = y

			run_nr_all = np.arange(file_pairs_all.shape[0])
			f = plt.figure(figsize = (12*len(run_nr_all),12))
			gs=GridSpec(3,len(run_nr_all)) # (2,3)2 rows, 3 columns
				
			for i in run_nr_all:
						
				s1=f.add_subplot(gs[0,i]) # First row, first column
				plt.plot(fmri_data_runs[i, best_voxel_index, :])	
							
				# design_matrix_64_plot = np.hstack([ design_matrix_64[:, 0:128:2],  design_matrix_64[:, -7:] ])
				# beta_runs_64_allRegressors_plot = np.hstack( [ beta_runs_64_allRegressors[ i, best_voxel_index, 0:128:2],  beta_runs_64_allRegressors[ i, best_voxel_index, -7:]   ]  )
				plt.plot(design_matrix_64_runs[i].dot(beta_runs_64_allRegressors[ i, best_voxel_index, :] + intercept_runs_64 [ i , best_voxel_index]))
				s1.set_xlabel('64ch_time_course_alpha_%s_r2_[%.2f]'%( str(alphas_runs_64 [ i, best_voxel_index]), r_squareds_runs_64[ i, best_voxel_index]), fontsize = 10)

				s2=f.add_subplot(gs[1,i]) # First row, first column
				plt.plot(fmri_data_runs[i, best_voxel_index, :])
				# design_matrix_8_ori = np.hstack( [design_matrix_8_ori[:, 0:16:2], design_matrix_8_ori[:, -7:] ] )
				# beta_runs_8_ori_allRegressors = np.hstack ( [beta_runs_8_ori_allRegressors[ i, best_voxel_index,0:16:2 ],  beta_runs_8_ori_allRegressors[ i, best_voxel_index, -7: ] ])
				plt.plot(design_matrix_8_ori_runs[i].dot(beta_runs_8_ori_allRegressors[ i, best_voxel_index, :] + intercept_runs_8_ori [ i, best_voxel_index]))
				s2.set_xlabel('8ch_ori_time_course_alpha_%s_r2_[%.2f]'%( str(alphas_runs_8_ori[ i, best_voxel_index]), r_squareds_runs_8_ori[ i, best_voxel_index]), fontsize = 10)

				s3=f.add_subplot(gs[2,i]) # First row, first column
				plt.plot(fmri_data_runs[i, best_voxel_index, :])
				# design_matrix_8_col = np.hstack( [design_matrix_8_col[:, 0:16:2], design_matrix_8_col[:, -7:] ] )
				# beta_runs_8_col_allRegressors = np.hstack ( [beta_runs_8_col_allRegressors[ i, best_voxel_index,0:16:2 ],  beta_runs_8_col_allRegressors[ i, best_voxel_index, -7: ] ])
				plt.plot(design_matrix_8_col_runs[i].dot(beta_runs_8_col_allRegressors[ i, best_voxel_index, :] + intercept_runs_8_col [ i, best_voxel_index]))
				s3.set_xlabel('8ch_col_time_course_alpha_%s_r2_[%.2f]'%( str(alphas_runs_8_col[ i, best_voxel_index]), r_squareds_runs_8_col[ i, best_voxel_index]), fontsize = 10)

			# f.savefig('%s-3models_%s_%s.pdf'%(subname, '64-8ori-col', names[y] ))
			f.savefig('%s-3models_%s_best%s_%s.pdf'%(subname, modelii,volii, best_voxel_index))
			print 'plotting %s-3models_%s_best%s_%s.pdf'%(subname, modelii,volii, best_voxel_index)

			plt.close()
			# f.savefig('%s-3models_%s_best%s_%s-without_dt_fitting,manual.pdf'%(subname, '64','1', '442'))
			# f.savefig('%s-3models_%s_best%s_%s-without_dt_fitting,auto.pdf'%(subname, modelii,volii, best_voxel_index))
###----------------------------------------------------------------------------
			# ### without selecting dt
			# 	s1=f.add_subplot(gs[0,i]) # First row, first column
			# 	plt.plot(fmri_data_runs[i, best_voxel_index, :])	

			# 	design_matrix_64_plot = np.hstack([ design_matrix_64[:, 0:128:2],  design_matrix_64[:, -7:] ])
			# 	beta_runs_64_allRegressors_plot = np.hstack( [ beta_runs_64_allRegressors[ i, best_voxel_index, 0:128:2],  beta_runs_64_allRegressors[ i, best_voxel_index, -7:]   ]  )
			# 	plt.plot(design_matrix_64_plot.dot(beta_runs_64_allRegressors_plot + intercept_runs_64 [ i , best_voxel_index]))
			# 	s1.set_xlabel('64ch_time_course_alpha_%s_r2_[%.2f]'%( str(alphas_runs_64 [ i, best_voxel_index]), r_squareds_runs_64[ i, best_voxel_index]), fontsize = 10)

			# 	s2=f.add_subplot(gs[1,i]) # First row, first column
			# 	plt.plot(fmri_data_runs[i, best_voxel_index, :])
			# 	design_matrix_8_ori_plot = np.hstack( [design_matrix_8_ori[:, 0:16:2], design_matrix_8_ori[:, -7:] ] )
			# 	beta_runs_8_ori_allRegressors_plot = np.hstack ( [beta_runs_8_ori_allRegressors[ i, best_voxel_index,0:16:2 ],  beta_runs_8_ori_allRegressors[ i, best_voxel_index, -7: ] ])
			# 	plt.plot(design_matrix_8_ori_plot.dot(beta_runs_8_ori_allRegressors_plot + intercept_runs_8_ori [ i, best_voxel_index]))
			# 	s2.set_xlabel('8ch_ori_time_course_alpha_%s_r2_[%.2f]'%( str(alphas_runs_8_ori[ i, best_voxel_index]), r_squareds_runs_8_ori[ i, best_voxel_index]), fontsize = 10)

			# 	s3=f.add_subplot(gs[2,i]) # First row, first column
			# 	plt.plot(fmri_data_runs[i, best_voxel_index, :])
			# 	design_matrix_8_col_plot = np.hstack( [design_matrix_8_col[:, 0:16:2], design_matrix_8_col[:, -7:] ] )
			# 	beta_runs_8_col_allRegressors_plot = np.hstack ( [beta_runs_8_col_allRegressors[ i, best_voxel_index,0:16:2 ],  beta_runs_8_col_allRegressors[ i, best_voxel_index, -7: ] ])
			# 	plt.plot(design_matrix_8_col_plot.dot(beta_runs_8_col_allRegressors_plot + intercept_runs_8_col [ i, best_voxel_index]))
			# 	s3.set_xlabel('8ch_col_time_course_alpha_%s_r2_[%.2f]'%( str(alphas_runs_8_col[ i, best_voxel_index]), r_squareds_runs_8_col[ i, best_voxel_index]), fontsize = 10)

			# # f.savefig('%s-3models_%s_best%s_%s,manual.png'%(subname, '64','1', '442'))
			# f.savefig('%s-3models_%s_best%s_%s,-wrong_auto.pdf'%(subname, modelii,volii, best_voxel_index))
			# print 'finish print %s-3models_%s_best%i_%i'%(subname, modelii,volii, best_voxel_index)

##### for each regressor----------------------------------------------------------------------------
		## for each regressor
		# orientation
			# f = plt.figure(figsize = (12*len(run_nr_all),12))
			# for i in []
			# a = {
			# 'regressor_of_interest': [ model_BOLD_timecourse_64[:,::2], model_BOLD_timecourse_8_ori [:,::2], model_BOLD_timecourse_8_col [:,::2] ],
			# 'time_derivative': [ model_BOLD_timecourse_64[:,1::2], model_BOLD_timecourse_8_ori [:,1::2], model_BOLD_timecourse_8_col [:,1::2] ],
			# 'moco_params', 
			# 'key_press',}
			# 'events': [events_64_runs, events_8_ori_runs, events_8_col_runs],


			# run_nr_all = np.arange(file_pairs_all.shape[0])
			# best_voxel_index = 0


			# for j in range(design_matrix_8_ori_runs.shape[2]):
			# 	f = plt.figure(figsize = (12,12))
			# 	if j in range(model_BOLD_timecourse_8_ori.shape[1])[::2] :
			# 		regressor_type = 'regressor_of_interest_%i_start_from1'%( (j/2)+1)
			# 	elif j in range(model_BOLD_timecourse_8_ori.shape[1])[1::2] :
			# 		regressor_type = 'dt_%i_start_from1'%( (j+1)/2)
			# 	elif j in range(design_matrix_8_ori_runs.shape[2])[-7:-1]:
			# 		regressor_type = 'moco_parameter_%i' %( (j-model_BOLD_timecourse_8_ori.shape[1] +1 ) )
			# 	elif j == range(design_matrix_8_ori_runs.shape[2])[-1]:
			# 		regressor_type = 'key_press'

			# 	for i in run_nr_all:
					
			# 		s1=f.add_subplot(4,1,i+1)

			# 		# num_plots = 64

			# 		plt.plot(fmri_data_runs[i, best_voxel_index, :], 'k')
					
			# 		plt.plot(design_matrix_8_ori_runs[i].dot(beta_runs_8_ori_allRegressors[ i, best_voxel_index, :] + intercept_runs_8_ori [ i, best_voxel_index]), 'b')

			# 		# colormap = plt.cm.gist_ncar
			# 		# plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)]) 
			# 		plt.plot(design_matrix_8_ori_runs[i, :, j] *(beta_runs_8_ori_allRegressors[ i, best_voxel_index, j] +  intercept_runs_8_ori [ i, best_voxel_index] ) ,'g'  )

			# 		s1.set_xlabel('8ch_ori_%s_run%i'%( regressor_type, i))


			# 	f.savefig('%s-%s_%s-modelFit_%s.pdf'%(subname, '8ori', '442', regressor_type))
			# 	plt.close()
			# 	print 'plotting %s-%s_%s-modelFit_%s'%(subname, '8ori', '442', regressor_type)

			# ### color

			# run_nr_all = np.arange(file_pairs_all.shape[0])
			# best_voxel_index = 0


			# for j in range(design_matrix_8_col_runs.shape[2]):
			# 	f = plt.figure(figsize = (12,12))
			# 	if j in range(model_BOLD_timecourse_8_col.shape[1])[::2] :
			# 		regressor_type = 'regressor_of_interest_%i_start_from1'%( (j/2)+1)
			# 	elif j in range(model_BOLD_timecourse_8_col.shape[1])[1::2] :
			# 		regressor_type = 'dt_%i_start_from1'%( (j+1)/2)
			# 	elif j in range(design_matrix_8_col_runs.shape[2])[-7:-1]:
			# 		regressor_type = 'moco_parameter_%i' %( (j-model_BOLD_timecourse_8_col.shape[1] +1 ) )
			# 	elif j == range(design_matrix_8_col_runs.shape[2])[-1]:
			# 		regressor_type = 'key_press'

			# 	for i in run_nr_all:
					
			# 		s1=f.add_subplot(4,1,i+1)

			# 		# num_plots = 64

			# 		plt.plot(fmri_data_runs[i, best_voxel_index, :], 'k')
					
			# 		plt.plot(design_matrix_8_col_runs[i].dot(beta_runs_8_col_allRegressors[ i, best_voxel_index, :] + intercept_runs_8_col [ i, best_voxel_index]), 'b')

			# 		# colormap = plt.cm.gist_ncar
			# 		# plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)]) 
			# 		plt.plot(design_matrix_8_col_runs[i, :, j] *(beta_runs_8_col_allRegressors[ i, best_voxel_index, j] +  intercept_runs_8_col [ i, best_voxel_index] ) ,'g'  )

			# 		s1.set_xlabel('8ch_ori_%s_run%i'%( regressor_type, i))


			# 	f.savefig('%s-%s_%s-modelFit_%s.pdf'%(subname, '8col', '442', regressor_type))
			# 	plt.close()
			# 	print 'plotting %s-%s_%s-modelFit_%s'%(subname, '8col', '442', regressor_type)
###----------------------------------------------------------------------------
	shell()
# #-----------------------
	for modelii in ['64','8ori','8col']:
		for volii in np.arange(1,31,1):
			# pt.plot_3ModelFit_beta_matrix(subname, volii, r_squareds_64, beta_64_across_runs, fmri_data, design_matrix_64, betas_64, intercept_64, alphas_64, r_squareds_8_ori, design_matrix_8_ori, betas_8_ori, intercept_8_ori, alphas_8_ori, r_squareds_8_col, design_matrix_8_col, betas_8_col, intercept_8_col, alphas_8_col) 
			# plot_3ModelFit_beta_matrix(subname, volii, r_squareds_64, beta_64_across_runs, fmri_data, design_matrix_64, betas_64, intercept_64, alphas_64, r_squareds_8_ori, design_matrix_8_ori, betas_8_ori, intercept_8_ori, alphas_8_ori, r_squareds_8_col, design_matrix_8_col, betas_8_col, intercept_8_col, alphas_8_col) 
			if modelii == '64':
				best_voxel_index = np.argsort(r_squareds_mean_64)[-volii]
			elif modelii == '8ori':
				best_voxel_index = np.argsort(r_squareds_mean_8_ori)[-volii]
			elif modelii == '8col':
				best_voxel_index = np.argsort(r_squareds_mean_8_col)[-volii]
			
			# beta_64_best_voxel = beta_64_across_runs [best_voxel_index]
# #-----------------------

# 			f1 = plt.figure(figsize = (12,12))

# 			s4 = f1.add_subplot(1,1,1)
# 			beta_matrix = beta_64_best_voxel.reshape(8,8)
# 			# plt.plot()

# 			# make it circlar
# 			beta_matrix_add_column = np.hstack((beta_matrix[:,4:8],beta_matrix, beta_matrix[:,0:4]))
# 			beta_matrix_cir = np.vstack ((beta_matrix_add_column[4:8, :], beta_matrix_add_column, beta_matrix_add_column[0:4, :]))
# 			plt.imshow(beta_matrix_cir, cmap= plt.cm.viridis, interpolation = 'gaussian' ,  vmin= 0) #  vmin= -4.5, vmax= 4.5   #'bilinear' #"bicubic"
# 			# im = mainax.imshow(beta_image,clim=(0,np.abs(all_stuff_betas[voxelii,1:]).max()), cmap='viridis')
# 			plt.xticks( np.arange(16), (0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5))
# 			plt.yticks(  np.arange(16), (4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3 ))

# 			color_theta = (np.pi*2)/8
# 			color_angle = color_theta * np.arange(0, 8,dtype=float)
# 			color_radius = 75
# 			color_a = color_radius * np.cos(color_angle)
# 			color_b = color_radius * np.sin(color_angle)
# 			colors2 = np.array([ct.lab2rgb((55, a, b)) for a,b in zip(color_a, color_b)])	
# 			colors2 = np.hstack((colors2/255, np.ones((8,1))))
# 			colors = np.vstack((colors2[4:8, :], colors2, colors2[0:4, :]))

# 			for ytick, color in zip(s4.get_yticklabels(), colors):
# 				ytick.set_color(color)

# 			s4.grid(False)
# 			plt.colorbar()
# 			s4.set_xlabel('orientation')
# 			s4.set_ylabel('color')
# 			s4.set_xlabel('voxel_index: %s' % (best_voxel_index) )

# 			f1.savefig('%s-%smodel_64matrix-positive_best%i_%i.png'%(subname, modelii, volii, best_voxel_index))
# 			print 'finish plot %s-%smodel_64matrix-positive_best%i_%i'%(subname, modelii, volii, best_voxel_index)
# 			plt.close()

		#### beta_matrix = beta_runs_64_allRegressors[ i, best_voxel_index, 0:128:2]


		# beta_64_best_voxel = beta_64_across_runs [best_voxel_index]


		# names = ['442', '5046', '684', '415', '5059', '2194', '5169', '387', '3544', '428']
		# for y in range(len(names)):
		# 	best_voxel_index = y

		# 2. beta matrix- for each run


		# # best_voxel_index = 442
		# 	run_nr_all = np.arange(file_pairs_all.shape[0])
		# 	f = plt.figure(figsize = (12*len(run_nr_all),12))
		# 	# s4=f.add_subplot(gs[:,1]) # First row, second column

		# 	preference_64 = []
		# 	for i in run_nr_all:

		# 		s4 = f.add_subplot(1,4,i+1)#(3,1,2)
		# 		# beta_matrix = beta_64_best_voxel.reshape(8,8)
		# 		beta_matrix = beta_runs_64_allRegressors[ i, best_voxel_index, 0:128:2].reshape(8,8)


		# 		# beta_pre_index = np.squeeze(np.where(beta_matrix == beta_matrix.max()))
		# 		a = beta_runs_64_allRegressors[ i, best_voxel_index, 0:128:2]

		# 		beta_pre_index_64 = np.where(a == np.max(a) )
		# 		print beta_pre_index_64
		# 		preference_64.append(beta_pre_index_64)


		# 		# make it circlar
		# 		beta_matrix_add_column = np.hstack((beta_matrix[:,4:8],beta_matrix, beta_matrix[:,0:4]))
		# 		beta_matrix_cir = np.vstack ((beta_matrix_add_column[4:8, :], beta_matrix_add_column, beta_matrix_add_column[0:4, :]))
		# 		plt.imshow(beta_matrix_cir, cmap= plt.cm.viridis, interpolation = 'gaussian' ,  vmin= 0) #  vmin= -4.5, vmax= 4.5   #'bilinear' #"bicubic"
		# 		# im = mainax.imshow(beta_image,clim=(0,np.abs(all_stuff_betas[voxelii,1:]).max()), cmap='viridis')
		# 		plt.xticks( np.arange(16), (0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5))
		# 		plt.yticks(  np.arange(16), (4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3 ))

		# 		color_theta = (np.pi*2)/8
		# 		color_angle = color_theta * np.arange(0, 8,dtype=float)
		# 		color_radius = 75
		# 		color_a = color_radius * np.cos(color_angle)
		# 		color_b = color_radius * np.sin(color_angle)
		# 		colors2 = np.array([ct.lab2rgb((55, a, b)) for a,b in zip(color_a, color_b)])	
		# 		colors2 = np.hstack((colors2/255, np.ones((8,1))))
		# 		colors = np.vstack((colors2[4:8, :], colors2, colors2[0:4, :]))

		# 		for ytick, color in zip(s4.get_yticklabels(), colors):
		# 			ytick.set_color(color)

		# 		s4.grid(False)
		# 		plt.colorbar()
		# 		s4.set_xlabel('orientation')
		# 		s4.set_ylabel('color')
		# 		s4.set_xlabel('run%i;voxel_index: %s' % (i, str(442)) ) #(best_voxel_index) 

		# 	# f.savefig('%s-3models_64matrix-positive_%s.pdf'%(subname, names[y] ))
		# 	f.savefig('%s-3models_%s_64matrix-positive_best%i_%i.pdf'%(subname, modelii, volii, best_voxel_index))
		# 	print 'plotting %s-3models_%s_64matrix-positive_best%i_%i.pdf'%(subname, modelii, volii, best_voxel_index)
		# 	plt.close()


		# 2. beta matrix- for all runs
		names = ['442', '5046', '684', '415', '5059', '2194', '5169', '387', '3544', '428']
		for y in range(len(names)):
			best_voxel_index = y

			# best_voxel_index = 0
			# run_nr_all = np.arange(file_pairs_all.shape[0])

			f = plt.figure(figsize = (24,24))
			# s4=f.add_subplot(gs[:,1]) # First row, second column

			# preference_64 = []
			gs=GridSpec(6,6) # (2,3)2 rows, 3 columns



			s4=f.add_subplot(gs[3:,0:-2])	
			beta_matrix = np.mean(beta_runs_64_allRegressors[ :, best_voxel_index, 0:128:2], axis = 0).reshape(8,8)


			beta_pre_index = np.squeeze(np.where(beta_matrix== beta_matrix.max()))
			# if get two max values, make the first one
			if beta_pre_index.size == 2:
				print 'only one preferred stimulus'
			else:
				beta_pre_index = np.array([beta_pre_index[0][0], beta_pre_index[1][0]])
				print 'more than one preferred stimulus'

			ori = beta_matrix[beta_pre_index[0],:]
			col = beta_matrix[:,beta_pre_index[1]]

			# make it circlar
			beta_matrix_add_column = np.hstack((beta_matrix[:,4:8],beta_matrix, beta_matrix[:,0:4]))
			beta_matrix_cir = np.vstack ((beta_matrix_add_column[4:8, :], beta_matrix_add_column, beta_matrix_add_column[0:4, :]))
			plt.imshow(beta_matrix_cir, cmap= plt.cm.viridis, interpolation = 'gaussian' ,  vmin= 0) #  vmin= -4.5, vmax= 4.5   #'bilinear' #"bicubic"
			# im = mainax.imshow(beta_image,clim=(0,np.abs(all_stuff_betas[voxelii,1:]).max()), cmap='viridis')
			plt.xticks( np.arange(16), (0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5), fontsize = 15)
			plt.yticks(  np.arange(16), (4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3 ), fontsize = 20)

			color_theta = (np.pi*2)/8
			color_angle = color_theta * np.arange(0, 8,dtype=float)
			color_radius = 75
			color_a = color_radius * np.cos(color_angle)
			color_b = color_radius * np.sin(color_angle)
			colors2 = np.array([ct.lab2rgb((55, a, b)) for a,b in zip(color_a, color_b)])	
			colors2 = np.hstack((colors2/255, np.ones((8,1))))
			colors = np.vstack((colors2[4:8, :], colors2, colors2[0:4, :]))

			for ytick, color in zip(s4.get_yticklabels(), colors):
				ytick.set_color(color)

			s4.grid(False)
			plt.colorbar()
			s4.set_xlabel('orientation', fontsize = 20)
			s4.set_ylabel('color', fontsize = 20)
			# s4.set_xlabel('run%i;voxel_index: %s' % (i, str(442)) ) #(best_voxel_index) 


			# a = np.mean(beta_runs_64_allRegressors[ :, best_voxel_index, 0:128:2], axis = 0)

			# beta_pre_index_64 = np.where(a == np.max(a) )
			# print beta_pre_index_64

			# # make it circlar
			# t_matrix_add_column = np.hstack((t_matrix_cen, t_matrix_cen[:,0][:, np.newaxis]))
			# t_matrix_cir = np.vstack ((t_matrix_add_column, t_matrix_add_column[0,:]))




			# stimuli_64 = beta_pre_index_64[0][0]

			# if (stimuli_64 >= 0 ) * (stimuli_64 < (8*1))  :
			# 	stimuli_col = 0			
			# elif (stimuli_64 >= (8*1)) and (stimuli_64 < (8*2)) :
			# 	stimuli_col = 1
			# elif (stimuli_64 >= (8*2)) and (stimuli_64 < (8*3)) :
			# 	stimuli_col = 2
			# elif (stimuli_64 >= (8*3)) and (stimuli_64 < (8*4)) :
			# 	stimuli_col = 3
			# elif (stimuli_64 >= (8*4)) and (stimuli_64 < (8*5)) :
			# 	stimuli_col = 4
			# elif (stimuli_64 >= (8*5)) and (stimuli_64 < (8*6)) :
			# 	stimuli_col = 5							
			# elif (stimuli_64 >= (8*6)) and (stimuli_64 < (8*7)) :
			# 	stimuli_col = 6
			# elif (stimuli_64 >= (8*7)) and (stimuli_64 < (8*8)) :
			# 	stimuli_col = 7	

			# stimuli_ori = stimuli_64% 8

			##ori
			s3=f.add_subplot(gs[2,0:-2]) # First row, third column

			# ori = beta_matrix[stimuli_ori, :]
			ori_cir = np.hstack( (ori[4:8], ori, ori[0:4] ) )
			plt.xticks( np.arange(16), (0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5), fontsize = 20)
			plt.yticks( fontsize = 20)
			# s3.set_xlabel('orientation', fontsize = 20)

			# s3.tick_params(axis=u'both', which=u'both',length=0)

			plt.plot( ori_cir)


			s2 =f.add_subplot(gs[3:,-2])

			# col = beta_matrix[:, stimuli_col] 
			col_cir = np.hstack( (col[4:8], col, col[0:4] ) )
			# roate_90_clockwise( col_cir )
			# s2.tick_params(axis=u'both', which=u'both',length=0)


			x = np.arange(0, len(col_cir) )
			y = col_cir
			x_new = y
			y_new = len(col_cir)-1 -x 

			# ax.set_xticklabels([7,6,5,4,3,2,1,0])

			plt.plot(x_new, y_new)
 # (4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3 ))
			plt.yticks(  np.arange(16), (3,2,1,0,7,6,5,4,3,2,1,0,7,6,5,4 ), fontsize = 20)
			plt.xticks( fontsize = 20)

			color_theta = (np.pi*2)/8
			color_angle = color_theta * np.arange(0, 8,dtype=float)
			color_radius = 75
			color_a = color_radius * np.cos(color_angle)
			color_b = color_radius * np.sin(color_angle)
			colors2 = np.array([ct.lab2rgb((55, a, b)) for a,b in zip(color_a, color_b)])	
			colors2 = np.hstack((colors2/255, np.ones((8,1))))
			colors = np.vstack((colors2[0:4, :][::-1], colors2[::-1], colors2[4:8, :][::-1] ))

			for ytick, color in zip(s2.get_yticklabels(), colors):
				ytick.set_color(color)


			# s2.set_ylabel('color', fontsize = 20)

			# f.savefig('%s-3models_64matrix-positive_%s.pdf'%(subname, names[y] ))
			f.savefig('%s-%i-2.pdf'%(subname, best_voxel_index))
			print 'plotting %s-%i-2.pdf'%(subname, best_voxel_index)
			plt.close()











			preference_64 = np.array(preference_64)

			stimuli_col_runs=[]
			stimuli_ori_runs=[]
			stimuli_64_runs =[]

			for i in run_nr_all:

				stimuli_64 = preference_64[i]
				stimuli_64_runs.append(stimuli_64)

				if (preference_64[i] >= 0 ) * (preference_64[i] < (8*1))  :
					stimuli_col = 0			
				elif (preference_64[i] >= (8*1)) and (preference_64[i] < (8*2)) :
					stimuli_col = 1
				elif (preference_64[i] >= (8*2)) and (preference_64[i] < (8*3)) :
					stimuli_col = 2
				elif (preference_64[i] >= (8*3)) and (preference_64[i] < (8*4)) :
					stimuli_col = 3
				elif (preference_64[i] >= (8*4)) and (preference_64[i] < (8*5)) :
					stimuli_col = 4
				elif (preference_64[i] >= (8*5)) and (preference_64[i] < (8*6)) :
					stimuli_col = 5							
				elif (preference_64[i] >= (8*6)) and (preference_64[i] < (8*7)) :
					stimuli_col = 6
				elif (preference_64[i] >= (8*7)) and (preference_64[i] < (8*8)) :
					stimuli_col = 7	

				stimuli_ori = preference_64[i]% 8

				stimuli_col_runs.append(stimuli_col)
				stimuli_ori_runs.append(stimuli_ori)
			

			stimuli_64_runs = np.squeeze(np.array(stimuli_64_runs))
			stimuli_ori_runs = np.squeeze(np.array(stimuli_ori_runs))
			stimuli_col_runs = np.array(stimuli_col_runs)


			event_time_64_runs = np.zeros( (len(run_nr_all), 2, 13) )  # runs, repetitions, TRs
			event_time_ori_runs = np.zeros( (len(run_nr_all), 16, 13) )
			event_time_col_runs = np.zeros( (len(run_nr_all), 16, 13) )

			for i in run_nr_all:
				all_TRs_64 = events_64_runs[i, :, stimuli_64_runs[i]]
				target_TR_64 = np.squeeze(np.where(all_TRs_64 ==1) )

				all_TRs_ori = events_8_ori_runs[i, :, stimuli_ori_runs[i]]
				target_TR_ori = np.squeeze(np.where(all_TRs_ori ==1) )

				all_TRs_col = events_8_col_runs[i, :, stimuli_col_runs[i] ]
				target_TR_col = np.squeeze(np.where(all_TRs_col ==1) )

				for j in range(target_TR_64.shape[0]):
					# time_window = [target_TR[j]-4 : target_TR[j]+9 ]
					event_time_64_runs[i, j] = fmri_data_runs[i, best_voxel_index, target_TR_64[j]-4 : target_TR_64[j]+9]
				
				for j in range(target_TR_ori.shape[0]):	
					event_time_ori_runs[i, j] = fmri_data_runs[i, best_voxel_index, target_TR_ori[j]-4 : target_TR_ori[j]+9]

				for j in range(target_TR_col.shape[0]):	
					event_time_col_runs[i, j] = fmri_data_runs[i, best_voxel_index, target_TR_col[j]-4 : target_TR_col[j]+9]


			event_time_64_across_repetitions =  np.mean(event_time_64_runs, axis = 1)
			event_time_ori_across_repetitions =  np.mean(event_time_ori_runs, axis = 1)
			event_time_col_across_repetitions =  np.mean(event_time_col_runs, axis = 1)

			event_time_64_mean = np.mean(  event_time_64_across_repetitions, axis = 0)
			event_time_ori_mean = np.mean(  event_time_ori_across_repetitions, axis = 0)
			event_time_col_mean = np.mean(  event_time_col_across_repetitions , axis = 0)

			sd = np.array([np.std(event_time_64_mean, axis = 0), np.std(event_time_ori_mean, axis = 0), np.std(event_time_col_mean, axis = 0)])
			n = len(run_nr_all)
			yerr = (sd/np.sqrt(n)) #*1.96

			event_time_mean = [event_time_64_mean, event_time_ori_mean, event_time_col_mean]
			labels = ['time_course_of_preferred_stimuli', 'time_course_of_preferred_orientation', 'time_course_of_preferred_color' ]
			f = plt.figure(figsize = (12,12))
			for i in [0,1,2]:
				s1 = f.add_subplot(3,1,i+1)#(3,1,2)
				plt.plot(event_time_mean[i])
				plt.errorbar(range(0,13) , event_time_mean[i], yerr= yerr[i])
				plt.xticks( np.arange(13), (range(-5,9)))

				s1.set_xlabel(labels[i])
				# sn.despine(offset=10)

			# f.savefig('%s-3models_time_course_of_preferred_features_voxel%s.pdf'%(subname, names[y] ))
			f.savefig('%s-3models_%s_time_course_of_preferred_features_best%i_voxel%s.pdf'%(subname, modelii,volii, best_voxel_index))	
			plt.close()
			# print 'plot %s-3models_time_course_of_preferred_features_voxel%s.pdf'%(subname, names[y] )
			# f.savefig('%s-3models_%s_best%s_%s.pdf'%(subname, modelii,volii, best_voxel_index))
			print 'plottting %s-3models_%s_time_course_of_preferred_features_best%i_voxel%s.pdf'%(subname, modelii,volii, best_voxel_index)



t_64vs8ori_across_sub = np.mean(t_64vs8ori, axis = 0)
t_64vs8col_across_sub = np.mean(t_64vs8col, axis = 0)
t_8oriVs8col_across_sub = np.mean(t_8oriVs8col, axis = 0)
# print 't_64vs8ori_r2_rel_mean: %.2f, p_64vs8ori_r2_rel_mean: %.2f' % (t_64vs8ori_across_sub[0], t_64vs8ori_across_sub[1] )
# print 't_64vs8col_r2_rel_mean: %.2f, p_64vs8col_r2_rel_mean: %.2f' % (t_64vs8col_across_sub[0], t_64vs8col_across_sub[1])
# print 't_8oriVs8col_r2_rel_mean: %.2f, p_8oriVs8col_r2_rel_mean: %.2f' % (t_8oriVs8col_across_sub[0], t_8oriVs8col_across_sub[1])

# print 'sub-n001', 
# print aov_values[0]

# print 'sub-n003'
# print aov_values[1]

# print 'sub-n005'
# print aov_values[2]

# sys.stdout = open('%s_F_t_64v8v8_r2.txt'%(subname), 'w')
sys.stdout = open('ALL_F_t_64v8v8_r2.txt', 'w')
# print subname
# print aov

# print 't_64vs8ori_r2_rel: %f, p_64vs8ori_r2_rel: %f, correction: %f' %(t_64vs8ori_r2_rel, p_64vs8ori_r2_rel, p_64vs8ori_r2_rel*len(voxel_list)*3 )
# print 't_64vs8col_r2_rel: %f, p_64vs8col_r2_rel: %f, correction: %f ' %(t_64vs8col_r2_rel, p_64vs8col_r2_rel, p_64vs8ori_r2_rel*len(voxel_list)*3 )
# print 't_8oriVs8col_r2_rel: %f, p_8oriVs8col_r2_rel: %f, correction: %f' %(t_8oriVs8col_r2_rel, p_8oriVs8col_r2_rel, p_64vs8ori_r2_rel*len(voxel_list)*3 )

# print p_64vs8ori_r2_rel*len(voxel_list)*3
# print p_64vs8ori_r2_rel*len(voxel_list)*3
# print p_64vs8ori_r2_rel*len(voxel_list)*3


# print 'beta_runs_64'
# print beta_runs_64

# sys.stdout.close()


for i in [0,1,2]:
	print sublist[i]
	print aov_values[i]
	print 't_64vs8ori_r2_rel: %.2f, p_64vs8ori_r2_rel: %.2f' % (t_64vs8ori[i, 0], t_64vs8ori[i,1])

	print 't_64vs8ori_r2_rel: %f, p_64vs8ori_r2_rel: %f, correction: %f' %(t_64vs8ori[i,0], t_64vs8ori[i,1], t_64vs8ori[i,1]*len(voxel_lists[i])*3 )
	print 't_64vs8col_r2_rel: %f, p_64vs8col_r2_rel: %f, correction: %f ' %(t_64vs8col[i,0], t_64vs8col[i,1], t_64vs8col[i,1]*len(voxel_lists[i])*3 )
	print 't_8oriVs8col_r2_rel: %f, p_8oriVs8col_r2_rel: %f, correction: %f' %(t_8oriVs8col[i,0], t_8oriVs8col[i,1], t_8oriVs8col[i,1]*len(voxel_lists[i])*3 )

	print t_64vs8ori[i,1]*len(voxel_lists[i])*3
	print t_64vs8col[i,1]*len(voxel_lists[i])*3 
	print t_8oriVs8col[i,1]*len(voxel_lists[i])*3



	# aov_values.append(aov)
	# t_64vs8ori.append((t_64vs8ori_r2_rel, p_64vs8ori_r2_rel))
	# t_64vs8col.append((t_64vs8col_r2_rel, p_64vs8col_r2_rel))
	# t_8oriVs8col.append((t_8oriVs8col_r2_rel, p_8oriVs8col_r2_rel))





print 'MEAN across Subjects'
print 't_64vs8ori_r2_rel_mean: %.2f, p_64vs8ori_r2_rel_mean: %.2f' % (t_64vs8ori_across_sub[0], t_64vs8ori_across_sub[1] )
print 't_64vs8col_r2_rel_mean: %.2f, p_64vs8col_r2_rel_mean: %.2f' % (t_64vs8col_across_sub[0], t_64vs8col_across_sub[1])
print 't_8oriVs8col_r2_rel_mean: %.2f, p_8oriVs8col_r2_rel_mean: %.2f' % (t_8oriVs8col_across_sub[0], t_8oriVs8col_across_sub[1])

sys.stdout.close()

# sys.stdout = open('%s_voxel_list_.txt'%(subname), 'w')
# print voxel_list
# sys.stdout.close()


### bonferroni correction: 

# TESTS OF WITHIN SUBJECTS EFFECTS

# Measure: r_squareds
#      Source                              Type III    eps       df        MS          F       Sig.   et2_G   Obs.      SE       95% CI    lambda     Obs.  
#                                             SS                                                                                                      Power 
# =========================================================================================================================================================
# model_type          Sphericity Assumed    111.892       -          2    55.946   12921.075      0   1.037   5734   8.690e-04    0.002   12923.328       1 
#                     Greenhouse-Geisser    111.892   0.503      1.006   111.212   12921.075      0   1.037   5734   8.690e-04    0.002   12923.328       1 
#                     Huynh-Feldt           111.892   0.503      1.006   111.212   12921.075      0   1.037   5734   8.690e-04    0.002   12923.328       1 
#                     Box                   111.892   0.500          1   111.892   12921.075      0   1.037   5734   8.690e-04    0.002   12923.328       1 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# Error(model_type)   Sphericity Assumed     49.646       -      11466     0.004                                                                            
#                     Greenhouse-Geisser     49.646   0.503   5768.020     0.009                                                                            
#                     Huynh-Feldt            49.646   0.503   5768.020     0.009                                                                            
#                     Box                    49.646   0.500       5733     0.009                                                                            

# TABLES OF ESTIMATED MARGINAL MEANS

# Estimated Marginal Means for model_type
# model_type   Mean    Std. Error   95% Lower Bound   95% Upper Bound 
# ===================================================================
# 1            0.212        0.002             0.209             0.216 
# 2            0.044    3.883e-04             0.044             0.045 
# 3            0.039    3.377e-04             0.038             0.039 


# #### (400, with steps of 5)
# TESTS OF WITHIN SUBJECTS EFFECTS

# Measure: r_squareds
#      Source                              Type III    eps       df        MS          F       Sig.   et2_G   Obs.      SE       95% CI    lambda     Obs.  
#                                             SS                                                                                                      Power 
# =========================================================================================================================================================
# model_type          Sphericity Assumed    111.519       -          2    55.760   12696.634      0   1.021   5734   8.752e-04    0.002   12698.849       1 
#                     Greenhouse-Geisser    111.519   0.503      1.006   110.844   12696.634      0   1.021   5734   8.752e-04    0.002   12698.849       1 
#                     Huynh-Feldt           111.519   0.503      1.006   110.844   12696.634      0   1.021   5734   8.752e-04    0.002   12698.849       1 
#                     Box                   111.519   0.500          1   111.519   12696.634      0   1.021   5734   8.752e-04    0.002   12698.849       1 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# Error(model_type)   Sphericity Assumed     50.355       -      11466     0.004                                                                            
#                     Greenhouse-Geisser     50.355   0.503   5767.920     0.009                                                                            
#                     Huynh-Feldt            50.355   0.503   5767.920     0.009                                                                            
#                     Box                    50.355   0.500       5733     0.009                                                                            

# TABLES OF ESTIMATED MARGINAL MEANS

# Estimated Marginal Means for model_type
# model_type   Mean    Std. Error   95% Lower Bound   95% Upper Bound 
# ===================================================================
# 1            0.212        0.002             0.209             0.216 
# 2            0.044    3.882e-04             0.044             0.045 
# 3            0.039    3.377e-04             0.038             0.039 









# 	shell()
# 	print 'prepare preference, & a set of tunings for 3 leftin runs. '

# 	# get voxel_indices_reliVox (indices of reliable voxels)
# 	voxel_indices_reliVox_64, n_reli_64 = ma.get_voxel_indices_reliVox( r_squareds_selection_runs_64, r_squareds_threshold = 0.05, select_100 = False ) 

# 	# beta_runs_64 = np.array(beta_runs)
# 	# r_squareds_runs = np.array(r_squareds_runs)

# 	# # prepare preference. 
# 	beta_pre_indices_reliVox_64 = ma.find_preference_matrix_allRuns ( beta_runs_64, n_reli_64, voxel_indices_reliVox_64)



# 	# a set of tunings for 3 leftin runs. 
# 	run_nr_all = np.arange(file_pairs_all.shape[0])
# 	beta_ori_mean_iterations_64 = np.zeros((len(run_nr_all), 9)) 
# 	beta_col_mean_iterations_64 = np.zeros((len(run_nr_all), 9))

# 	beta_ori_mean_iterations_16 = np.zeros((len(run_nr_all), 9))
# 	beta_col_mean_iterations_16 = np.zeros((len(run_nr_all), 9))

# 	for filepairii in run_nr_all :
	
# 		run_nr_leftOut = filepairii
# 		run_nr_rest = run_nr_all[~(run_nr_all == run_nr_leftOut)]

# 		beta_mean_64 = np.mean(beta_runs_64[run_nr_rest], axis = 0)
# 		beta_ori_reliVox_mean_64, beta_col_reliVox_mean_64  = ma.calculate_tunings_matrix (n_reli_64, voxel_indices_reliVox_64, beta_mean_64, beta_pre_indices_reliVox_64, position_cen=2)
# 		beta_ori_mean_iterations_64 [filepairii,:] = beta_ori_reliVox_mean_64 # shape: (4,9)
# 		beta_col_mean_iterations_64 [filepairii,:] = beta_col_reliVox_mean_64


# 		# betas_z_ori_mean = np.mean(betas_z_ori_runs[run_nr_rest], axis = 0)
# 		# betas_z_col_mean = np.mean(betas_z_col_runs[run_nr_rest], axis = 0)




# # plot figures across iterations!
# 	print 'plot figures across iterations!'

# 	pt.plot_tunings(run_nr_all, n_reli_64, beta_ori_mean_iterations_64, beta_col_mean_iterations_64, subname, ROI, data_type, regression, position_cen = 2)




# ##################
# #######################
# #######################
# #########################


# 	# print t_64vs16_r2, p_64vs16_r2
# 	sys.stdout = open('%s_t_64vs16_r2_.txt'%(subname), 'w')
# 	# print 'r_squareds_runs_64_%2.f' %(np.mean(r_squareds_runs_64))
# 	# print 'r_squareds_runs_16_%2.f' %(np.mean(r_squareds_runs_16))	
# 	print 't_64vs16_r2: %.2f' %(t_64vs16_r2)
# 	print 'p_64vs16_r2: %.2f' %(p_64vs16_r2)
# 	sys.stdout.close()

