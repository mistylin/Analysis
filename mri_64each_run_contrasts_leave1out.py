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


sublist = [  ('sub-002', True, False), ('sub-004', True, False)]#[('sub-001', False, True) ]# , , [('sub-002', True, False)], [('sub-004', True, False)]
#sublist = ['sub-001','sub-002']


data_dir_fmri = '/home/shared/2017/visual/OriColorMapper/preproc/'
#	data_dir_fmri = '/home/shared/2017/visual/OriColorMapper/preproc/sub-002/psc/'
data_dir_beh = '/home/shared/2017/visual/OriColorMapper/bids_converted/'
#	data_dir_beh = '/home/shared/2017/visual/OriColorMapper/bids_converted/sub-002/func/'
#/Users/xiaomeng/subjects/XY_01052017/mri/brainmask.mgz  #or T1.mgz
data_dir_fixation = '/home/shared/2017/visual/OriColorMapper/raw/'

# get fullfield files
# sub-002_task-location_run-1_bold_brain_B0_volreg_sg.nii.gz
# sub-002_task-fullfield_run-2.pickle
# sub-002_task-fullfield_output_run-2.pickle ---output files

data_type = 'tf'#'tf' #'psc'
each_run = True #False #True #False
ROI = 'V4' # 'V4'
regression = 'RidgeCV' #'GLM' #'RidgeCV'
# type_contrasts = 'full' # 'ori', 'color', 'full'
position_cen = 'nan' #2 4  #'nan'



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
		if fmri_file.split('_')[1]== target_condition:
			target_files_fmri.append(fmri_file)

	for beh_file in beh_files:
		if beh_file.split('_')[2]== target_condition:
			target_files_beh.append(beh_file)

	for moco_file in moco_files:
		if moco_file.split('_')[2]== target_condition:
			target_files_moco.append(moco_file)

	for fixation_file in fixation_files:
		if fixation_file.split('_')[1]== target_condition:
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

		if data_type == 'tf':
			## name it with fmri_data, in fact it's for each run, namely(fmri_data_run)
			fmri_data = (fmri_data - np.nanmean(fmri_data, axis = 1)[:, np.newaxis]) / np.nanstd(fmri_data, axis = 1)[:, np.newaxis]
			#fmri_data shape: (5728, 286)
			'finish normalization fmri data!'
		elif data_type == 'psc':
			print 'psc data type, normalization not needed!'
		
		nan_voxels_run = np.unique(np.argwhere(np.isnan(fmri_data))[:,0])
		nan_voxels.extend(nan_voxels_run)

	voxel_list =[]

	for i in range(n_all_voxels):
		if i not in nan_voxels:
			voxel_list.append(i)
	print 'finish checking nan values'



	## Load all types of data
	file_pairs_all = np.array(zip (target_files_fmri, target_files_beh, target_files_moco, target_files_fixation))
	
	t_runs = [] 
	r_squareds_runs = []

	# for fileii, file_pair in enumerate(file_pairs):
		
	for fileii, file_pair in enumerate(file_pairs_all):
		# if fileii == run_nr_leftOut			

		filename_fmri = file_pair[0]
		filename_beh = file_pair[1]
		filename_moco = file_pair[2]
		filename_fixation = file_pair[3]
	
	## Load fmri data--run
		unmasked_fmri_data = nib.load(filename_fmri).get_data()
		fmri_data = np.vstack([unmasked_fmri_data[lh,:], unmasked_fmri_data[rh,:]])
		# another way to flatten ---- e.g. moco_params.reshape(-1, moco_params.shape[-1])

		# Z scored fmri_data, but with the same name
		if data_type == 'tf':
			#fmri_data = (fmri_data -fmri_data.mean()) / fmri_data.std()
			## name it with fmri_data, in fact it's for each run, namely(fmri_data_run)
			fmri_data = (fmri_data - np.nanmean(fmri_data, axis = 1)[:, np.newaxis]) / np.nanstd(fmri_data, axis = 1)[:, np.newaxis]
			#fmri_data shape: (5728, 286)


		fmri_data = fmri_data[voxel_list,:]
		# fmri_data = fmri_data[np.isnan(fmri_data).sum(axis=1)==0,:]



	## Load stimuli order (events)-run
		trial_order_run = pickle.load(open(filename_beh, 'rb'))[1]
		#create events with 1
		empty_start = 15
		empty_end = 15
		number_of_stimuli = 64
		tmp_trial_order_run  = np.zeros((fmri_data.shape[1],1))
		#15 + 256( 2* 128) +15 =286
		tmp_trial_order_run[empty_start:-empty_end:2] = trial_order_run[:]+1 # [:,np.newaxis]+1
		events = np.hstack([np.array(tmp_trial_order_run == stim, dtype=int) for stim in np.arange(1,number_of_stimuli+1)])


	## Load motion correction parameters

		moco_params = pd.read_csv(filename_moco, delim_whitespace=True, header = None)
		# nib.load(filename_moco).get_data()
		# shape (286,6)

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


	# convolve events with hrf, to get model_BOLD_timecourse
		TR = 0.945 #ms
		model_BOLD_timecourse = fftconvolve(events, hrf(np.arange(0,30)* TR)[:,np.newaxis],'full')[:fmri_data.shape[1],:]

		design_matrix = np.hstack([np.ones((fmri_data.shape[1],1)), model_BOLD_timecourse, moco_params, key_press])
		# shape: (286,71--1+64+6)

		n_voxels = fmri_data.shape[0]
		n_TRs = fmri_data.shape[1]
		n_regressors = design_matrix.shape[1]
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
			ridge_fit = RidgeCV(alphas = [0.5] , fit_intercept = False, normalize = True )

			results = np.zeros((n_voxels,3))
			r_squareds =  np.zeros((n_voxels, ))
			alphas =  np.zeros((n_voxels, 1))
			betas = np.zeros((n_voxels, n_regressors ))
			_sse = np.zeros((n_voxels, ))
			print 'start RidgeCV fitting'

			for x in range(n_voxels):
				
				ridge_fit.fit(design_matrix, fmri_data[x, :])
				# print x, ridge_fit.score(design_matrix, fmri_data[x,:]), ridge_fit.alpha_ #, ridge_fit.coef_.T

				# results[x] = [ridge_fit.score(design_matrix, fmri_data[x,:]), ridge_fit.alpha_, ridge_fit.coef_.T]	#ridge_fit.fit(design_matrix, fmri_data.T)

				r_squareds[x] = ridge_fit.score(design_matrix, fmri_data[x,:])
				alphas[x] = ridge_fit.alpha_
				betas[x] = ridge_fit.coef_.T

				_sse[x] = np.sqrt(np.sum((design_matrix.dot(betas[x]) - fmri_data[x,:])**2)/df)
			print 'finish RidgeCV'

		r_squareds_runs.append(r_squareds) 

		# compute contrasts
		# if type_contrasts == 'full':

		n_contrasts = 64
		t = np.zeros((n_voxels, n_contrasts))
		p = np.zeros((n_voxels, n_contrasts))

		for i in range(n_contrasts):

			c_moco = np.zeros(moco_params.shape[1])
			c_key_press = np.zeros(key_press.shape[1])
			a = np.ones(n_contrasts) *  -1/float(n_contrasts-1)  #-1/ 63.0
			# a = np.zeros(64)
			a[i] = 1

			c = np.r_[0, a, c_moco, c_key_press] #.reshape(8,8) moco_params, key_press
			design_var = c.dot(np.linalg.pinv(design_matrix.T.dot(design_matrix))).dot(c.T)
			SE_c = np.sqrt(_sse * design_var)

			t[:,i] = betas.dot(c) / SE_c  # SE_c (10508,)
			p[:,i] = scipy.stats.t.sf(np.abs(t[:,i]), df)*2		

		t_runs.append(t)

	print 'prepare preference, & a set of tunings for leftout runs. '
# # prepare preference, & a set of tunings for leftout runs. 
	t_runs = np.array(t_runs)
	r_squareds_runs = np.array(r_squareds_runs)

	run_nr_all = np.arange(file_pairs_all.shape[0])

	t_ori_mean_iterations = np.zeros((len(run_nr_all), 9)) 
	t_col_mean_iterations = np.zeros((len(run_nr_all), 9)) 

	t_ori_cate_mean_iterations = np.zeros((len(run_nr_all), 8,9)) 
	t_col_cate_mean_iterations = np.zeros((len(run_nr_all), 8,9)) 

	for filepairii in run_nr_all :
	
		run_nr_leftOut = filepairii
		run_nr_rest = run_nr_all[~(run_nr_all == run_nr_leftOut)]
		# file_pairs = file_pairs_all[~(np.arange(file_pairs_all.shape[0]) == run_nr_leftOut)]

		t_mean = np.mean(t_runs[run_nr_rest], axis = 0) # average across rest runs 		# t_mean shape: (5638,64)		
		r_squareds_mean = np.mean(r_squareds_runs[run_nr_rest], axis = 0) 

		order = np.argsort(r_squareds_mean)
		voxels_all = sorted( zip(order, r_squareds_mean) , key = lambda tup: tup [1] )
		n_best = 100
		voxels = voxels_all[-n_best:]

		voxel_indices_bestVox = np.array(voxels)[:,0]
		t_pre_indices_bestVox = np.zeros((n_best, 2 ))

		# elif type_contrasts == 'full':
	### prepare t_pre_index
		if position_cen == 2: 
			t_oriBestVox = np.zeros((n_best, 9))
			t_colBestVox = np.zeros((n_best, 9))
			
			for voxelii, voxelIndex in enumerate(voxel_indices_bestVox):					

				voxelIndex = int(voxelIndex)
				t_matrix = t_mean [voxelIndex ].reshape(8,8)

				t_pre_index = np.squeeze(np.where(t_matrix== t_matrix.max()))
				# if get two max values, make the first one
				if t_pre_index.size == 2:
					pass
				else:
					t_pre_index = np.array([t_pre_index[0][0], t_pre_index[1][0]])

				t_pre_indices_bestVox[voxelii, :] = t_pre_index

		
		elif position_cen == 'nan': 
			for voxelii, voxelIndex in enumerate(voxel_indices_bestVox):

				voxelIndex = int(voxelIndex)
				# r_squared_best = voxel[1]

				t_matrix = t_mean [voxelIndex ].reshape(8,8)
				
				# plt.imshow( t_matrix , cmap= plt.cm.ocean, interpolation = "None")

				# center --- exact values: make horizontal as the center of the x-axis, so the labels are still the extact positions.
				# so exact positions for x-axis
				# make the centers are green / horizontal. 
				t_matrix_cenRow = np.roll(t_matrix, 5, axis = 0) # roll downwards by 5 steps, to make the green one as the 4th(0,1,2,3,4), then in the next step, the green one will be at the center.
				t_matrix_cen= np.roll(t_matrix_cenRow, 1, axis = 1) # roll to right side by 1 step, to make horizontal be the 4th (be at center through next step)

				# note that t_matrix_cen instead of t_matrix, compared with centring to a position
				t_pre_index = np.squeeze(np.where(t_matrix_cen== t_matrix_cen.max()))
				# if get two max values, make the first one
				if t_pre_index.size == 2:
					print 'only one preferred stimulus'
				else:
					t_pre_index = np.array([t_pre_index[0][0], t_pre_index[1][0]])
					print 'more than one preferred stimulus', voxelIndex , t_pre_index[0].shape


				t_pre_indices_bestVox[voxelii, :] = t_pre_index
#---------------------------------------------------------
	### a set of tunings for the specific leftout run. 

		ts_leftOut = t_runs[run_nr_leftOut]
		# voxel_indices_bestVox 
		# t_pre_indices_bestVox
		
		if position_cen == 2:
			t_oriBestVox = np.zeros((n_best, 9))
			t_colBestVox = np.zeros((n_best, 9))

			for nrii, voxelIndex in enumerate(voxel_indices_bestVox):

				voxelIndex = int(voxelIndex)
				t_matrix_leftOut = ts_leftOut[voxelIndex].reshape(8,8)
				t_pre_current_index = t_pre_indices_bestVox[nrii]		

				# center --- always move the peaks to the position_cen, 'x axis will be relative positions'
				# so the labels of x-axis are not the actual values (ori/color), but the relative position on the axis.
				
				t_matrix_leftOut_cenRow = np.roll(t_matrix_leftOut, int(position_cen-t_pre_current_index[0]), axis = 0)
				t_matrix_leftOut_cen= np.roll(t_matrix_leftOut_cenRow, int(position_cen-t_pre_current_index[1]), axis = 1)

				# make it circlar
				t_matrix_leftOut_add_column = np.hstack((t_matrix_leftOut_cen, t_matrix_leftOut_cen[:,0][:, np.newaxis]))
				t_matrix_leftOut_cir = np.vstack ((t_matrix_leftOut_add_column, t_matrix_leftOut_add_column[0,:]))


				t_ori = t_matrix_leftOut_cir[position_cen,:]
				t_col = t_matrix_leftOut_cir[:,position_cen]

		
				t_oriBestVox[nrii,:] = t_ori
				t_colBestVox[nrii,:] = t_col

			t_oriBestVox_mean = np.mean(t_oriBestVox, axis = 0)
			t_colBestVox_mean = np.mean(t_colBestVox, axis = 0)

			t_ori_mean_iterations [filepairii,:] = t_oriBestVox_mean
			t_col_mean_iterations [filepairii,:] = t_colBestVox_mean

		elif position_cen == 'nan': 
			t_oriBestVox_0 = []
			t_colBestVox_0 = []

			t_oriBestVox_1 = []
			t_colBestVox_1 = []
			t_oriBestVox_2 = []
			t_colBestVox_2 = []						
			t_oriBestVox_3 = []
			t_colBestVox_3 = []
			t_oriBestVox_4 = []
			t_colBestVox_4 = []
			t_oriBestVox_5 = []
			t_colBestVox_5 = []
			t_oriBestVox_6 = []
			t_colBestVox_6 = []
			t_oriBestVox_7 = []
			t_colBestVox_7 = []

			t_ori_cate = [t_oriBestVox_0, t_oriBestVox_1, t_oriBestVox_2, t_oriBestVox_3, t_oriBestVox_4, t_oriBestVox_5, t_oriBestVox_6, t_oriBestVox_7]
			t_col_cate = [t_colBestVox_0, t_colBestVox_1, t_colBestVox_2, t_colBestVox_3, t_colBestVox_4, t_colBestVox_5, t_colBestVox_6, t_colBestVox_7]

			for nrii, voxelIndex in enumerate(voxel_indices_bestVox):

				voxelIndex = int(voxelIndex)
				t_matrix_leftOut = ts_leftOut[voxelIndex].reshape(8,8)

				# center --- exact values: make horizontal as the center of the x-axis, so the labels are still the extact positions.
				# so exact positions for x-axis
				# make the centers are green / horizontal. 
				t_matrix_leftOut_cenRow = np.roll(t_matrix_leftOut, 5, axis = 0) # roll downwards by 5 steps, to make the green one as the 4th(0,1,2,3,4), then in the next step, the green one will be at the center.
				t_matrix_leftOut_cen= np.roll(t_matrix_leftOut_cenRow, 1, axis = 1) # roll to right side by 1 step, to make horizontal be the 4th (be at center through next step)

				t_pre_current_index = t_pre_indices_bestVox[nrii]

				# make it circlar
				t_matrix_leftOut_add_column = np.hstack((t_matrix_leftOut_cen, t_matrix_leftOut_cen[:,0][:, np.newaxis]))
				t_matrix_leftOut_cir = np.vstack ((t_matrix_leftOut_add_column, t_matrix_leftOut_add_column[0,:]))

				t_ori = t_matrix_leftOut_cir[t_pre_current_index[0],:]
				t_col = t_matrix_leftOut_cir[:,t_pre_current_index[1]]
				
				for i in range(0,8):
					if t_pre_current_index[1] == i:
						t_ori_cate[i].append(t_ori)
					
					if t_pre_current_index[0] == i:
						t_col_cate[i].append(t_col)

			
			t_ori_cate_mean_leftOut = np.zeros((8,9)) #8 preference locations, 9 points 
			t_col_cate_mean_leftOut = np.zeros((8,9)) 

			# t_ori_cate_mean_leftOut = np.mean(t_ori_cate, axis = 1) # for each i 

			for i in range(0,8):

				t_ori_cate[i] = np.array(t_ori_cate[i]) 
				t_col_cate[i] = np.array(t_col_cate[i])

				t_ori_cate_mean_leftOut[i,:] = np.mean(t_ori_cate[i], axis = 0)
				t_col_cate_mean_leftOut[i,:] = np.mean(t_col_cate[i], axis = 0)


			t_ori_cate_mean_iterations [filepairii, :, :] =  t_ori_cate_mean_leftOut
			t_col_cate_mean_iterations [filepairii, :, :] = t_col_cate_mean_leftOut


# plot figures! across iterations.

	print 'plot figures across iterations!'

	if position_cen == 2:

		t_ori_mean = np.mean(t_ori_mean_iterations, axis = 0)
		t_col_mean = np.mean(t_col_mean_iterations, axis = 0)


		t_oriBestVox_mean = np.mean(t_oriBestVox, axis = 0)
		t_colBestVox_mean = np.mean(t_colBestVox, axis = 0)

		sd = np.array([np.std(t_ori_mean_iterations, axis = 0), np.std(t_col_mean_iterations, axis = 0)])
		n = len(run_nr_all)
		yerr = (sd/np.sqrt(n))*1.96


		f2 = plt.figure(figsize = (8,6))
		s1 = f2.add_subplot(211)
		plt.plot(t_ori_mean)
		plt.errorbar(range(0,9), t_ori_mean, yerr= yerr[0])
		# s1.set_title('orientation', fontsize = 10)
		s1.set_xticklabels(['placeholder', -45, -22.5, 0, 22.5, 45, 67.5, 90, -67.5 ,-45])
		s1.set_xlabel('orientation - relative ')


		s2 = f2.add_subplot(212)
		plt.plot(t_col_mean)
		plt.errorbar(range(0,9), t_col_mean, yerr= yerr[1])
		# s2.set_title('color', fontsize = 10)
		s2.set_xticklabels(['placeholder', -2, -1, 0, 1, 2, 3, 4, -3, -2])
		s2.set_xlabel('color - relative')
		f2.savefig( '%s_%s_%s_%s_cen%s_tValues_%sVoxels.png'%(subname, ROI, data_type, regression, position_cen, n_best))



	elif position_cen == 'nan':
		t_ori_cate_mean = np.mean(t_ori_cate_mean_iterations, axis = 0)
		t_col_cate_mean = np.mean(t_col_cate_mean_iterations, axis = 0)

		t_ori_cate_yerr = np.std(t_ori_cate_mean_iterations, axis = 0)/ np.sqrt(len(run_nr_all)) * 1.96
		t_col_cate_yerr = np.std(t_col_cate_mean_iterations, axis = 0)/ np.sqrt(len(run_nr_all)) * 1.96


		f3 = plt.figure(figsize = (12,10))

		colors1 = plt.cm.rainbow(np.linspace(0, 1, len(range(0,8))))

		s1 = f3.add_subplot(2,1,1)
				
		for colii, color in enumerate(colors1, start =0):
			if t_ori_cate_mean[colii].size == 1:
				s1.set_title('no voxels fall into peak position: %s' %(str(colii)) , fontsize = 10)

			else:
				plt.plot(t_ori_cate_mean[colii], color = color, label = 'position:%s' %(str(colii)))
				plt.errorbar(range(0,9), t_ori_cate_mean[colii], color = color, yerr= t_ori_cate_yerr[colii])
			plt.legend(loc='best')
		
		s1.set_xticklabels(['placeholder', 'vertical 90', -67.5, -45, -22.5, 'horizontal 0', 22.5, 45, 67.5, 'vertical 90'])
		s1.set_xlabel('orientation - absolute')
		# f3.savefig( '%s_%s_%s_%sOri_8pos_tValues_%sVoxels.png'%(subname, ROI, data_type, regression, n_best))

		
		# f4 = plt.figure(figsize = (12,10))
		s2 = f3.add_subplot(2,1,2)
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
			if t_col_cate_mean[colii].size == 1:
				s2.set_title('no voxels fall into peak position: %s' %(str(colii)) , fontsize = 10)

			else:
				plt.plot(t_col_cate_mean[colii], color = color , label = 'position:%s' %(str(colii)))
				plt.errorbar(range(0,9), t_col_cate_mean[colii], color = color, yerr= t_col_cate_yerr[colii])
			plt.legend(loc='best')

		s2.set_xlabel('color - absolute')
		# f4.savefig( '%s_%s_%s_%sCol_8pos_tValues_%sVoxels.png'%(subname, ROI, data_type, regression, n_best))
		f3.savefig( '%s_%s_%s_%s_8pos_tValues_%sVoxels.png'%(subname, ROI, data_type, regression, n_best))





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



