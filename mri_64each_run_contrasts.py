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


sublist = [('sub-002', True, False)]#[ ('sub-001', False, True) ]# , , [('sub-002', True, False)]
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
ROI = 'V1' # 'V4'

for subii, sub in enumerate(sublist):

	subname = sub[0]
	retinotopic = sub[1]
	exvivo = sub[2]
	
	print '[main] Running analysis for %s' % (str(subname))
	
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

	beta_lists = []

	file_pairs = zip (target_files_fmri, target_files_beh, target_files_moco, target_files_fixation)

	for fileii, file_pair in enumerate(file_pairs):
		
		run_nr = fileii

		filename_fmri = file_pair[0]
		filename_beh = file_pair[1]
		filename_moco = file_pair[2]
		filename_fixation = file_pair[3]

		#shell()
		#fixation_order_run = pickle.load(open(filename_fixation, 'rb'))[1]

		
	## Load fmri data--run
		unmasked_fmri_data = nib.load(filename_fmri).get_data()
		fmri_data = np.vstack([unmasked_fmri_data[lh,:], unmasked_fmri_data[rh,:]])
		# another way to flatten ---- e.g. moco_params.reshape(-1, moco_params.shape[-1])

		# Z scored fmri_data, but with the same name
		if data_type == 'tf':
			#fmri_data = (fmri_data -fmri_data.mean()) / fmri_data.std()
			## name it with fmri_data, in fact it's for each run, namely(fmri_data_run)
			fmri_data = (fmri_data - np.nanmean(fmri_data, axis = 1)[:, np.newaxis]) / np.nanstd(fmri_data, axis = 1)[:, np.newaxis]

		fmri_data = fmri_data[np.isnan(fmri_data).sum(axis=1)==0,:]
		

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
		shell()
		fixation_order_run = pickle.load(open(filename_fixation, 'rb'))
		eventArray = fixation_order_run['eventArray']  # a list of lists

		key_press = np.zeros((fmri_data.shape[1],1))

		for n_event, event in enumerate (eventArray):
			for txt in event:
				if 'key: y' in txt:
					key_press[n_event] = 1

				elif 'key: b' in txt:
					key_press[n_event] = 1


		# staircase_events = []
		# staircase_events_info =[]
		# staircase_values = []

		# # fixation_task = np.zeros((fmri_data.shape[1],1))

		# for n_trial, trial in enumerate (eventArray):
		# 	for txt in trial:
		# 		if 'staircase updated from' in txt:
		# 			staircase_events.append(n_trial)  # len: 37
		# 			staircase_events_info.append(trial)

		# 			staircase_values.append( float(re.findall(r"[-+]?\d*\.\d+|\d+", txt)[1]) )
		# 			# take the numbers out

		# staircase_values_log = abs(np.array(staircase_values)) - abs(staircase_values[-1]) #np.log10(abs(np.array(staircase_values)) - abs(staircase_values[-1]))

		# # create a column with mental effort (refined staircase values) inside
		# staircase_column = []
		# for i , staircase_event in enumerate(staircase_events):
		# 	if staircase_event == 1:
				
		# 		tmp_staircases =  [staircase_values_log[i]]* 2 # np.repeat( [staircase_values_log[i]], 2, axis = 0)
		# 		staircase_column.append(tmp_staircases) 
		# 		# copy the staircase_values_log twice
		# 	elif i == len(staircase_events):
		# 		tmp_staircases = [[staircase_values_log[i]]] * (trial_order_run[0] - staircase_events[i]) #np.repeat( [staircase_values_log[i]], staircase_events[i+1] - staircase_events[i], axis = 0)
		# 		staircase_column.append(tmp_staircases) 

		# 	else:
		# 		tmp_staircases = [[staircase_values_log[i]]] * (staircase_events[i+1] - staircase_events[i]) #np.repeat( [staircase_values_log[i]], staircase_events[i+1] - staircase_events[i], axis = 0)
		# 		staircase_column.append(tmp_staircases) 



	# convolve events with hrf, to get model_BOLD_timecourse
		TR = 0.945 #ms
		model_BOLD_timecourse = fftconvolve(events, hrf(np.arange(0,30)* TR)[:,np.newaxis],'full')[:fmri_data.shape[1],:]

		design_matrix = np.hstack([np.ones((fmri_data.shape[1],1)), model_BOLD_timecourse, moco_params, key_press])
		# shape: (286,71--1+64+6)

	# GLM to get betas
		print 'start GLM fitting'
		betas, _sse, _r, _svs = np.linalg.lstsq(design_matrix, fmri_data.T )
		# betas shape (65, 9728--number of voxels)

		
	# RidgeCV to get betas

		# ridge_fit = RidgeCV(alphas = np.linspace(1,50,50) , fit_intercept = False, normalize = True )
		# ridge_fit = RidgeCV(alphas = [3] , fit_intercept = False, normalize = True )

		# ridge_fit.fit(design_matrix, fmri_data.T)
		
		# alpha = ridge_fit.alpha_
		# print alpha
		# # range--1000: alpha = 607.0 (16:58 - 17:03 --5 min)
		# # range--800: alpha = 607.0 (19:55-19:56 -1 min
		# # range --610: alpha = 607.0
		print 'finish GLM'
		n_voxels = fmri_data.shape[0]
		n_TRs = fmri_data.shape[1]
		n_regressors = design_matrix.shape[1]
		df = (n_TRs-n_regressors)

		#results = np.zeros((n_voxels,3))
		# r_squareds =  np.zeros((n_voxels, 1))
		# alphas =  np.zeros((n_voxels, 1))
		# betas = np.zeros((n_voxels, 65))
		r_squareds = 1.0 - ((design_matrix.dot(betas).T -fmri_data)**2).sum(axis=1) / (fmri_data**2).sum(axis=1)

		t = np.zeros((n_voxels, 64))
		p = np.zeros((n_voxels, 64))

		# shell()

		# dm_ni = design_matrix[:,1:]

		# for x in range(n_voxels):
		# 	# ridge_fit.fit(design_matrix, fmri_data[x, :])
		# 	# print x, ridge_fit.score(design_matrix, fmri_data[x,:]), ridge_fit.alpha_ #, ridge_fit.coef_.T

		# 	#results[x] = [ridge_fit.score(design_matrix, fmri_data[x,:]), ridge_fit.alpha_, ridge_fit.coef_.T]	#ridge_fit.fit(design_matrix, fmri_data.T)

		# 	# r_squareds[x] = ridge_fit.score(design_matrix, fmri_data[x,:])
		# 	# alphas[x] = ridge_fit.alpha_
		# 	# betas[x] = ridge_fit.coef_.T

		# 	SE = _sse[x]
		# 	# np.sqrt(np.sum((design_matrix.dot(betas[x]) - fmri_data[x,:])**2)/df)

		# 	for i in range(8):
		# 		a = np.ones(8) * -1/7.0
		# 		a[i] = 1
		# 		c = np.repeat(a, 8, axis = 0) #.reshape(8,8)
				
		# 		design_var = c.dot(np.linalg.pinv(dm_ni.T.dot(dm_ni))).dot(c.T)
		# 		SE_c = np.sqrt(SE * design_var)

		# 		t_color[x,i] = betas[1:,x].dot(c) / SE_c
		# 		p_color[x,i] = scipy.stats.t.sf(np.abs(t_color[x,i]), df)*2

		for i in range(64):
			# a = np.ones(8) * -1/7.0
			a = np.ones(64) * -1/63.0
			c_moco = np.zeros(moco_params.shape[1])
			c_key_press = np.zeros(key_press.shape[1])
			# a = np.zeros(64)
			a[i] = 1
			# c = np.r_[0,np.repeat(a, 8, axis = 0).T] #.reshape(8,8)
			c = np.r_[0, a, c_moco, c_key_press] #.reshape(8,8) moco_params, key_press
			
			design_var = c.dot(np.linalg.pinv(design_matrix.T.dot(design_matrix))).dot(c.T)
			SE_c = np.sqrt(_sse * design_var)

			t[:,i] = betas.T.dot(c) / SE_c
			p[:,i] = scipy.stats.t.sf(np.abs(t[:,i]), df)*2



		# t_color_z = (t_color - np.nanmean(t_color, axis = 1)[:, np.newaxis]) / np.nanstd(t_color, axis =1 )[:, np.newaxis]
		# betas_z = (betas - np.nanmean(betas, axis = 1)[:, np.newaxis]) / np.nanstd(betas, axis = 1)[:, np.newaxis]


		shell()
		# betas = np.array(betas) # shape: (10655,96)
		# betas = betas.T
		# betas shape: (97, 10655); fmri_data.shape: (10655,858); fmri_data_run.shape (10655,286)
		# average across channels? or voxels? voxels. because want to compare between voxels. fmri_data: average across time points, keep voxels intact. becuase we want to compare different runs--times.

		#fmri_data_run = (fmri_data_run - np.nanmean(fmri_data_run, axis = 1)[:, np.newaxis]) / np.nanstd(fmri_data_run, axis = 1)[:, np.newaxis]
		betas_z = (betas - np.nanmean(betas, axis = 1)[:, np.newaxis]) / np.nanstd(betas, axis = 1)[:, np.newaxis]

		order = np.argsort(r_squareds)

		#voxels.sort -- best 20 the first argument

		voxels_all = sorted( zip(order, r_squareds) , key = lambda tup: tup [1] )

		voxels = voxels_all[-20:]

		#voxels = [(100, index_100), (75,index_75), (50, index_50),(25,index_25)] 


		# beta_run = betas.T[:, 1:65]

		# if run_nr == 0:
		# 	beta_lists = beta_run

		# else:
		# 	beta_lists = np.concatenate((beta_lists, beta_run), axis=1)


		# shell()
		# plot figures
		# t_to_be_avaraged = []

		for voxelii, voxel in enumerate(voxels):
			
			f = plt.figure(figsize = (12,12))

			gs=GridSpec(6,6) # (2,3)2 rows, 3 columns

			# 1. first plot -- time series
			s1=f.add_subplot(gs[0:2,:]) # First row, first column
			plt.plot(fmri_data[voxel[0], :])
			plt.plot(design_matrix.dot(betas[:, voxel[0]]))

			# plot(design_matrix.dot(betas).T[ra])


			# 2. t values matrix
			s2=f.add_subplot(gs[3:,0:-2]) # First row, second column
			t_matrix = t [voxel[0]].reshape(8,8)
			
			plt.imshow( t_matrix , cmap= plt.cm.ocean, interpolation = "None")
			plt.colorbar()


			# 3. tuning curves over color and orientation dimensions
			s3=f.add_subplot(gs[2,0:-2]) # First row, third column
			plt.plot(t_matrix .max(axis = 0))

	
			s4 =f.add_subplot(gs[3:,-2]) # Second row, span all columns
			roate_90_clockwise( t_matrix.max(axis = 1) )

			plt.close()

			f.savefig( '%s_%s_%s_run%s_best%s_#%s_tValues_%s.png'%(subname, data_type, ROI, str(run_nr), str(20-voxelii), str(voxel[0]), str(voxel[1])))

			print "plotting_run%s_best%s"%(str(run_nr), str(20-voxelii))

			# t_to_be_avaraged.append (t_color[ voxel[0] ])

		# t_to_be_avaraged = np.array(t_to_be_avaraged).reshape(20, 8)
		# t_20 = np.mean(t_to_be_avaraged, axis = 0)
		
		# f1 = plt.figure(figsize = (8,6))

		# plt.plot(t_20)
		# plt.close()
		# #shell()
		# f1.savefig('%s_%s_%s_run%s_AVERAGE20_tValues.png'%(subname, data_type, ROI, str(run_nr) ))


# In [71]: for ra in np.argsort(r_squareds)[-10:]:
#     ...:     f = figure()
#     ...:     s = f.add_subplot(211)
#     ...:     imshow(t_color[ra].reshape((8,8)), cmap = 'viridis')
#     ...:     colorbar()
#     ...:     s = f.add_subplot(212)
#     ...:     plot(design_matrix.dot(betas).T[ra])
#     ...:     plot(fmri_data[ra])




	# 		gs=GridSpec(6,6) # (2,3)2 rows, 3 columns
			
	# 		# 1. fmri data & model BOLD response
	# 		# s1 = f.add_subplot(4,1,1)
	# 		s1=f.add_subplot(gs[0:2,:]) # First row, first column

	# 		plt.plot(fmri_data[voxel[0], :])
	# 		plt.plot(design_matrix.dot(betas[:, voxel[0]]))
	# 		#s1.set_title('time_course', fontsize = 10)

	# 		# 2. beta matrix
	# 		# s2 = f.add_subplot(1,1,1)#(3,1,2)

	# 		s2=f.add_subplot(gs[3:,0:-2]) # First row, second column
			
	# 		#shell()
	# 		beta_matrix = betas[1:65, voxel[0]].reshape(8,8)
	# 		#plt.plot(beta_matrix) #, cmap= plt.cm.ocean)
	# 		plt.imshow(beta_matrix, cmap= plt.cm.ocean, interpolation = None)
	# 		plt.colorbar()
	# 			#plt.pcolor(betas[1:, voxel[0]],cmap=plt.cm.Reds)
	# 		#sn.despine(offset=10)
	# 		#s2.set_title('beta_matrix', fontsize = 10)
			

	# 		# 3. tuning curves over color and orientation dimensions
	# 		# s3 = f.add_subplot(4,1,3)
	# 		s3=f.add_subplot(gs[2,0:-2]) # First row, third column
	# 		plt.plot(beta_matrix.max(axis = 0))
	# 		#s3.set_title('dimention_1-color?', fontsize = 10)



	# 		# s4 = f.add_subplot(4,1,4)
	# 		s4 =f.add_subplot(gs[3:,-2]) # Second row, span all columns
	# 		# plt.plot(beta_matrix.max(axis = 1))

	# 		roate_90_clockwise( beta_matrix.max(axis = 1) )


	# 		#s4.set_title('dimention_2-orientation?', fontsize = 10)
	# 		#a = plt.plot(beta_matrix.max(axis = 1))
	# 		#rotated_a = ndimage.rotate(a, 90)
	# 		#plt.plot(roatated_a)


	# 		# plt.savefig( '%s_100_%s_GLM.jpg'%(subname,str(voxel[0]voxelii)))
			
	# 		f.savefig( '%s_%s_run%s_best%s_#%s_r2_%s_RidgeCV_moco.png'%(subname, data_type, str(run_nr), str(20-voxelii), str(voxel[0]), str(voxel[1])))

	# 		print "plotting_run%s_best%s"%(str(run_nr), str(20-voxelii))

	# 		plt.close()


	# # beta_lists shape: (9728,256)
	# # only for each run

	# rs =[]
	# print 'start plot figures'
	# for voxel in beta_lists:

		
	# ## sub002- have 4 runs in totoal
	# 	run1 = voxel[ :64]
	# 	run2 = voxel[ 64:128]
	# 	run3 = voxel[ 128:192]
	# 	run4 = voxel[ 192:256]

	# 	r1 = pearsonr(run1, run2)[0]
	# 	r2 = pearsonr(run1, run3)[0]
	# 	r3 = pearsonr(run1, run4)[0]
	# 	r4 = pearsonr(run2, run3)[0]
	# 	r5 = pearsonr(run2, run4)[0]
	# 	r6 = pearsonr(run3, run4)[0]

	# 	r = np.average([r1, r2, r3, r4, r5, r6])

		
	# # ## sub001- have 3 runs 
	# # 	run1 = voxel[ :64]
	# # 	run2 = voxel[ 64:128]
	# # 	run3 = voxel[ 128:192]
	# # 	# run4 = voxel[ 192:256]

	# # 	r1 = pearsonr(run1, run2)[0]
	# # 	r2 = pearsonr(run1, run3)[0]
	# # 	# r3 = pearsonr(run1, run4)[0]
	# # 	r4 = pearsonr(run2, run3)[0]
	# # 	# r5 = pearsonr(run2, run4)[0]
	# # 	# r6 = pearsonr(run3, run4)[0]

	# # 	r = np.average([r1, r2, r4])

	# # 	rs.append(r)


	# plt.hist(np.array(rs)[~np.isnan(rs)])
	# plt.savefig( '%s_%s_r_between_runs_hist_RidgeCV_moco.jpg'%(subname, data_type))


	# plt.close()



# #----------------------------------------------------------------------------------------------------------	

	#'/home/xiaomeng/Analysis/figures/'+ 


