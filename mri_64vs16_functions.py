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
import mri_statistical_analysis as sa

import numpy as np
import pyvttbl as pt
from collections import namedtuple

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


aov_values = []
t_64vs8ori = []
t_64vs8col = []
t_8oriVs8col = []

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

	
	# t_runs = [] 
	r_squareds_runs_64 = []
	r_squareds_selection_runs_64 = [] 
	beta_runs_64 = []
	beta_selection_runs_64 = []

	r_squareds_runs_8_ori = []
	r_squareds_selection_runs_8_ori = [] 
	beta_runs_8_ori = []
	beta_selection_runs_8_ori = []

	r_squareds_runs_8_col = []
	r_squareds_selection_runs_8_col = [] 
	beta_runs_8_col = []
	beta_selection_runs_8_col = []	

	for fileii, (filename_fmri, filename_beh, filename_moco, filename_fixation) in enumerate(file_pairs_all):		
		#0,1,2,3
		# file_pair = file_pairs_all[fileii]
		# filename_fmri = file_pair[0]
		# filename_beh = file_pair[1]
		# filename_moco = file_pair[2]
		# filename_fixation = file_pair[3]

	## Load fmri data--run
		fmri_data = ld.load_fmri(filename_fmri, voxel_list, lh, rh) #
	## Load stimuli order (events)-run
		events_64 = ld.load_event_64channels (filename_beh, fmri_data, empty_start = 15, empty_end = 15, number_of_stimuli = 64)
		
		events_8_ori, events_8_col = ld.load_event_16channels (filename_beh, fmri_data, empty_start = 15, empty_end = 15, number_of_stimuli = 8)

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
		model_BOLD_timecourse_8_ori = fftconvolve(events_8_ori, hrf(np.arange(0,30)* TR)[:,np.newaxis],'full')[:fmri_data.shape[1],:]
		design_matrix_8_ori = np.hstack([model_BOLD_timecourse_8_ori, moco_params, key_press]) #np.ones((fmri_data.shape[1],1)), 
		model_BOLD_timecourse_8_col = fftconvolve(events_8_col, hrf(np.arange(0,30)* TR)[:,np.newaxis],'full')[:fmri_data.shape[1],:]
		design_matrix_8_col = np.hstack([model_BOLD_timecourse_8_col, moco_params, key_press]) #np.ones((fmri_data.shape[1],1)), 
		#:  8+6+1 = 15

		# for r_squareds selection
		model_BOLD_timecourse_selection = fftconvolve(stim_regressor, hrf(np.arange(0,30)* TR)[:,np.newaxis],'full')[:fmri_data.shape[1],:]
		design_matrix_selection = np.hstack([model_BOLD_timecourse_selection, moco_params, key_press]) # np.ones((fmri_data.shape[1],1)), 


		r_squareds_64, r_squareds_selection_64, betas_64, betas_selection_64, _sse_64, intercept_64, alphas_64 = ma.run_regression(fileii, design_matrix_64, design_matrix_selection, fmri_data, regression = 'RidgeCV')
		r_squareds_8_ori, r_squareds_selection_8_ori, betas_8_ori, betas_selection_8_ori, _sse_8_ori, intercept_8_ori, alphas_8_ori = ma.run_regression(fileii, design_matrix_8_ori, design_matrix_selection, fmri_data, regression = 'RidgeCV')
		r_squareds_8_col, r_squareds_selection_8_col, betas_8_col, betas_selection_8_col, _sse_8_col, intercept_8_col, alphas_8_col = ma.run_regression(fileii, design_matrix_8_col, design_matrix_selection, fmri_data, regression = 'RidgeCV')

		# r_squareds_64, r_squareds_selection_64, betas_64, betas_selection_64, _sse_64, intercept_64, alphas_64 = run_regression(fileii, design_matrix_64, design_matrix_selection, fmri_data, regression = 'RidgeCV')
		# r_squareds_16, r_squareds_selection_16, betas_16, betas_selection_16, _sse_16, intercept_16, alphas_16 = run_regression(fileii, design_matrix_16, design_matrix_selection, fmri_data, regression = 'RidgeCV')

		# pt.plot_3models_timeCourse_alphaHist(fmri_data, r_squareds_64, design_matrix_64, betas_64, intercept_64, alphas_64, r_squareds_8_ori, design_matrix_8_ori, betas_8_ori, intercept_8_ori, alphas_8_ori, r_squareds_8_col, design_matrix_8_col, betas_8_col, intercept_8_col, alphas_8_col, subname, fileii): 
		# plot_3models_timeCourse_alphaHist(fmri_data, r_squareds_64, design_matrix_64, betas_64, intercept_64, alphas_64, r_squareds_8_ori, design_matrix_8_ori, betas_8_ori, intercept_8_ori, alphas_8_ori, r_squareds_8_col, design_matrix_8_col, betas_8_col, intercept_8_col, alphas_8_col, subname, fileii)

		r_squareds_runs_64.append(r_squareds_64) 
		# r_squareds_selection_runs_64.append(r_squareds_selection_64)
		beta_runs_64.append(betas_64[:, 0:64]) 
		# beta_selection_runs_64.append(betas_selection_64[:, 0:64])
		
		r_squareds_runs_8_ori.append(r_squareds_8_ori) 
		# r_squareds_selection_runs_8_ori.append(r_squareds_selection_8_ori)
		beta_runs_8_ori.append(betas_8_ori[:, 0:8]) 
		# beta_selection_runs_8_ori.append(betas_selection_8_ori[:, 0:8])

		r_squareds_runs_8_col.append(r_squareds_8_col) 
		# r_squareds_selection_runs_8_col.append(r_squareds_selection_8_col)
		beta_runs_8_col.append(betas_8_col[:, 0:8]) 
		# beta_selection_runs_8_col.append(betas_selection_8_col[:, 0:8])

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

	# print 't_64vs8ori_r2_rel: %.2f, p_64vs8ori_r2_rel: %.2f' %(t_64vs8ori_r2_rel, p_64vs8ori_r2_rel)
	# print 't_64vs8col_r2_rel: %.2f, p_64vs8col_r2_rel: %.2f' %(t_64vs8col_r2_rel, p_64vs8col_r2_rel)
	# print 't_8oriVs8col_r2_rel: %.2f, p_8oriVs8col_r2_rel: %.2f' %(t_8oriVs8col_r2_rel, p_8oriVs8col_r2_rel)

	# t_64vs8ori_r2_rel: 114.14, p_64vs8ori_r2_rel: 0.00
	# t_64vs8col_r2_rel: 112.13, p_64vs8col_r2_rel: 0.00
	# t_8oriVs8col_r2_rel: 30.36, p_8oriVs8col_r2_rel: 0.00

	aov_values.append(aov)
	t_64vs8ori.append((t_64vs8ori_r2_rel, p_64vs8ori_r2_rel))
	t_64vs8col.append((t_64vs8col_r2_rel, p_64vs8col_r2_rel))
	t_8oriVs8col.append((t_8oriVs8col_r2_rel, p_8oriVs8col_r2_rel))
	##(115.18453285570681, 0.0), (113.0876960551475, 0.0), (30.362206735333434, 6.6824459323599391e-188)

	shell()	
#### circle or oval? the best voxel
	beta_64_across_runs = np.mean(beta_runs_64, axis = 0)

	for volii in np.arange(1,51,1):
		# pt.plot_3ModelFit_beta_matrix(subname, volii, r_squareds_64, beta_64_across_runs, fmri_data, design_matrix_64, betas_64, intercept_64, alphas_64, r_squareds_8_ori, design_matrix_8_ori, betas_8_ori, intercept_8_ori, alphas_8_ori, r_squareds_8_col, design_matrix_8_col, betas_8_col, intercept_8_col, alphas_8_col) 
		# plot_3ModelFit_beta_matrix(subname, volii, r_squareds_64, beta_64_across_runs, fmri_data, design_matrix_64, betas_64, intercept_64, alphas_64, r_squareds_8_ori, design_matrix_8_ori, betas_8_ori, intercept_8_ori, alphas_8_ori, r_squareds_8_col, design_matrix_8_col, betas_8_col, intercept_8_col, alphas_8_col) 

		best_voxel_index = np.argsort(r_squareds_64)[-volii]
		beta_64_best_voxel = beta_64_across_runs [best_voxel_index]

		f = plt.figure(figsize = (24,12))

		gs=GridSpec(3,2) # (2,3)2 rows, 3 columns
		
		# 1. fmri data & model BOLD response
		# s1 = f.add_subplot(4,1,1)
		s1=f.add_subplot(gs[0,0]) # First row, first column
		plt.plot(fmri_data[best_voxel_index])
		plt.plot(design_matrix_64.dot(betas_64[best_voxel_index] + intercept_64 [best_voxel_index]))
		s1.set_title('64ch_time_course_alpha_%s_r2_[%.2f]'%( str(alphas_64[best_voxel_index]), r_squareds_64[best_voxel_index]), fontsize = 10)

		s2=f.add_subplot(gs[1,0]) # First row, first column
		plt.plot(fmri_data[best_voxel_index])
		plt.plot(design_matrix_8_ori.dot(betas_8_ori[best_voxel_index] + intercept_8_ori [best_voxel_index]))
		s2.set_title('8ch_ori_time_course_alpha_%s_r2_[%.2f]'%( str(alphas_8_ori[best_voxel_index]), r_squareds_8_ori[best_voxel_index]), fontsize = 10)

		s3=f.add_subplot(gs[2,0]) # First row, first column
		plt.plot(fmri_data[best_voxel_index])
		plt.plot(design_matrix_8_col.dot(betas_8_col[best_voxel_index] + intercept_8_col [best_voxel_index]))
		s3.set_title('8ch_col_time_course_alpha_%s_r2_[%.2f]'%( str(alphas_8_col[best_voxel_index]), r_squareds_8_col[best_voxel_index]), fontsize = 10)

		# 2. beta matrix
		# f = plt.figure(figsize = (12,12))
		s4=f.add_subplot(gs[:,1]) # First row, second column
		# s2 = f.add_subplot(1,1,1)#(3,1,2)
		beta_matrix = beta_64_best_voxel.reshape(8,8)

		# make it circlar
		beta_matrix_add_column = np.hstack((beta_matrix[:,4:8],beta_matrix, beta_matrix[:,0:4]))
		beta_matrix_cir = np.vstack ((beta_matrix_add_column[4:8, :], beta_matrix_add_column, beta_matrix_add_column[0:4, :]))
		plt.imshow(beta_matrix_cir, cmap= plt.cm.viridis, interpolation = 'gaussian' ,  vmin= 0) #  vmin= -4.5, vmax= 4.5   #'bilinear' #"bicubic"
		# im = mainax.imshow(beta_image,clim=(0,np.abs(all_stuff_betas[voxelii,1:]).max()), cmap='viridis')
		plt.xticks( np.arange(16), (0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5))
		plt.yticks(  np.arange(16), (4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3 ))

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
		s4.set_xlabel('orientation')
		s4.set_ylabel('color')
		s4.set_title('voxel_index: %s' % (best_voxel_index) )

		f.savefig('%s-3models_64matrix-positive_best%i_%i.png'%(subname, volii, best_voxel_index))
		plt.close()
shell()

t_64vs8ori_across_sub = np.mean(t_64vs8ori, axis = 0)
t_64vs8col_across_sub = np.mean(t_64vs8col, axis = 0)
t_8oriVs8col_across_sub = np.mean(t_8oriVs8col, axis = 0)
print 't_64vs8ori_r2_rel: %.2f, p_64vs8ori_r2_rel: %.2f' % (t_64vs8ori_across_sub[0], t_64vs8ori_across_sub[1] )
print 't_64vs8col_r2_rel: %.2f, p_64vs8col_r2_rel: %.2f' % (t_64vs8col_across_sub[0], t_64vs8col_across_sub[1])
print 't_8oriVs8col_r2_rel: %.2f, p_8oriVs8col_r2_rel: %.2f' % (t_8oriVs8col_across_sub[0], t_8oriVs8col_across_sub[1])

print 'sub-n001', 
print aov_values[0]

print 'sub-n003'
print aov_values[1]

print 'sub-n005'
print aov_values[2]
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

