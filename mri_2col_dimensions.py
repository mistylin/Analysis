# from __future__ import division
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

def roate_90_clockwise ( myarray ):

	x = np.arange(0, len(myarray) )
	y = myarray

	x_new = y
	y_new = len(myarray)-1 -x 

	# ax.set_xticklabels([7,6,5,4,3,2,1,0])

	plt.plot(x_new, y_new)

# def prepare_trials(self):
# 	"""docstring for prepare_trials(self):"""

# 	self.standard_parameters = standard_parameters


# 	self.orientations = np.linspace(self.standard_parameters['stimulus_ori_min'], self.standard_parameters['stimulus_ori_max'], self.standard_parameters['stimulus_ori_steps']+1)[:-1]

# 	# Compute evenly-spaced steps in (L)ab-space

# 	color_theta = (np.pi*2)/self.standard_parameters['stimulus_col_steps']
# 	color_angle = color_theta * np.arange(self.standard_parameters['stimulus_col_min'], self.standard_parameters['stimulus_col_max'],dtype=float)
# 	color_radius = self.standard_parameters['stimulus_col_rad']

# 	color_a = color_radius * np.cos(color_angle)
# 	color_b = color_radius * np.sin(color_angle)

# 	# self.colors = [ct.lab2psycho((self.standard_parameters['stimulus_col_baselum'], a, b)) for a,b in zip(color_a, color_b)]			 
# 	self.colors = [(self.standard_parameters['stimulus_col_baselum'], a, b) for a,b in zip(color_a, color_b)]			 

# 	#self.stimulus_positions = self.standard_parameters['stimulus_positions']
	
# 	self.trial_array = []

# 	# self.trial_array = np.array([[[o,c[0],c[1],c[2]] for o in self.orientations] for c in self.colors]).reshape((self.standard_parameters['stimulus_ori_steps']*self.standard_parameters['stimulus_col_steps'],4))
# 	self.trial_array = np.array([[[o,c[0],c[1]] for o in self.orientations] for c in self.colors]).reshape((self.standard_parameters['stimulus_ori_steps']*self.standard_parameters['stimulus_col_steps'],3))
	
# 	# dbstop()

##-------------------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------------------
 # without def 


# # Load V1 mask
# data_dir_masks = '/home/shared/2017/visual/OriColorMapper/preproc/sub-002/masks/dc/'
# lhV1 = np.array(nib.load(os.path.join(data_dir_masks, 'lh.V1.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)
# rhV1 = np.array(nib.load(os.path.join(data_dir_masks, 'rh.V1.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)


sublist = [ ('sub-001',True) ]# , , ('sub-002', False)
#sublist = ['sub-001','sub-002']


data_dir_fmri = '/home/shared/2017/visual/OriColorMapper/preproc/'
#	data_dir_fmri = '/home/shared/2017/visual/OriColorMapper/preproc/sub-002/psc/'
data_dir_beh = '/home/shared/2017/visual/OriColorMapper/bids_converted/'
#	data_dir_beh = '/home/shared/2017/visual/OriColorMapper/bids_converted/sub-002/func/'
#/Users/xiaomeng/subjects/XY_01052017/mri/brainmask.mgz  #or T1.mgz


# get fullfield files

data_type = 'psc'#'tf' #'psc'
each_run = False #True #False

for subii, sub in enumerate(sublist):

	subname = sub[0]
	exvivo = sub[1]
	
	#print '[main] Running analysis for %s' % (str(subname))
	
	subject_dir_fmri= os.path.join(data_dir_fmri,subname)
	fmri_files = glob.glob(subject_dir_fmri  + '/' + data_type + '/*.nii.gz')
	fmri_files.sort()

	moco_files = glob.glob(subject_dir_fmri + '/mcf/parameter_info' + '/*.1D')
	moco_files.sort()

	#respiration&heart rate

		# 'masks/dc/lh.V1_exvivo.thresh_vol_dil.nii.gz'
	if exvivo == True:
		lhV1 = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/dc/lh.cortex_vol_dil.nii.gz')).get_data(), dtype=bool)
		rhV1 = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/dc/rh.cortex_vol_dil.nii.gz')).get_data(), dtype=bool)
	
	else:
		lhV1 = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/dc/lh.V1.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)
		rhV1 = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/dc/rh.V1.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)


	subject_dir_beh = os.path.join(data_dir_beh,subname)
	beh_files = glob.glob(subject_dir_beh +'/func'+ '/*.pickle')
	beh_files.sort()



	target_files_fmri = []
	target_files_beh = []
	target_files_moco = []

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

	
# #----------------------------------------------------------------------------------------------------------		
# #----------------------------------------------------------------------------------------------------------
###
###
####       for all runs (contatenated ) data!!!!
####       orientation circle; color -- 2 dimensions
###
###
# #----------------------------------------------------------------------------------------------------------		
# #----------------------------------------------------------------------------------------------------------
	#elif each_run == False:

	## version 1 - Load fmri data
	fmri_data = []#np.array( [[None] * number_of_voxels]).T
	for fileii, filename_fmri in enumerate(target_files_fmri):
		unmasked_fmri_data = nib.load(filename_fmri).get_data()
		#fmri_data_run = unmasked_fmri_data 
		fmri_data_run = np.vstack([unmasked_fmri_data[lhV1,:], unmasked_fmri_data[rhV1,:]])
		
		##Z scored fmri_data, but with the same name
		if data_type == 'tf':
			#fmri_data_run = (fmri_data_run - fmri_data_run.mean()) / fmri_data_run.std()
		# Z scored fmri_data, but with the same name
			fmri_data_run = (fmri_data_run - np.nanmean(fmri_data_run, axis = 1)[:, np.newaxis]) / np.nanstd(fmri_data_run, axis = 1)[:, np.newaxis]
		
		if fileii == 0:
			fmri_data = fmri_data_run
		else:
			fmri_data = np.hstack([fmri_data, fmri_data_run])
			# fmri_data shape: (9728,1144)
	fmri_data = fmri_data[np.isnan(fmri_data).sum(axis=1)==0,:]


	## Load stimuli order (events)
	events = []
	events_col = []
	events_ori = []

	for fileii, filename_beh in enumerate(target_files_beh):
		#shell()
		trial_order_run = pickle.load(open(filename_beh, 'rb'))[1] #(128,1)

		#create events with 1
		empty_start = 15
		empty_end = 15
		number_of_stimuli = 8
		number_of_elements = 6

		# parsopf = open(self.output_file + '_trialinfo.pickle', 'wb')
		#output = [self.trial_array, self.trial_indices, self.trial_params, self.per_trial_parameters, self.per_trial_phase_durations, self.staircase]
		# pickle.dump(output,parsopf)
		# trial_array shape: (64, 4); trial_indices: numbers 0-63 & above; trial_params, 

		#  the first 8 values represent all orientations but only 1color
		## for trial_order_col------------------------------------
		trial_order_col = np.zeros((len(trial_order_run),))
		trial_order_col_a = np.zeros((len(trial_order_run),))
		trial_order_col_b = np.zeros((len(trial_order_run),))

		# color_a_values = [75, 53, 0.001, -53, -75, -0.001 ]
		# color_b_values = [-0.001, 53, 75, 0.001, -53, -75 ]
		color_a_values = [75, 53, 0.001, -0.001, -53, -75 ]
		color_b_values = [75, 53, 0.001, -0.001, -53, -75 ]

		color1 = ( 75, -0.001) # although it's 0 when printing out the value, code it as -0.001, to make it different from color 5's b value
		color2 = ( 53, 53)
		color3 = ( 0.001, 75)
		color4 = ( -53, 53)
		color5 = ( -75, 0.001)
		color6 = ( -53, -53)
		color7 = ( -0.001, -75)
		color8 = ( 53, -53)

		for ii, stim_nr in enumerate(trial_order_run) :
			if (stim_nr >= 0 ) * (stim_nr < (8*1))  :
				trial_order_col[ii] = 1
				trial_order_col_a[ii] = color_a_values.index(color1[0])+1
				trial_order_col_b[ii] = color_b_values.index(color1[1])+1		

			elif (stim_nr >= (8*1)) and (stim_nr < (8*2)) :
				trial_order_col[ii] = 2
				trial_order_col_a[ii] = color_a_values.index(color2[0])+1
				trial_order_col_b[ii] = color_b_values.index(color2[1])+1

			elif (stim_nr >= (8*2)) and (stim_nr < (8*3)) :
				trial_order_col[ii] = 3
				trial_order_col_a[ii] = color_a_values.index(color3[0])+1
				trial_order_col_b[ii] = color_b_values.index(color3[1])+1

			elif (stim_nr >= (8*3)) and (stim_nr < (8*4)) :
				trial_order_col[ii] = 4
				trial_order_col_a[ii] = color_a_values.index(color4[0])+1
				trial_order_col_b[ii] = color_b_values.index(color4[1])+1

			elif (stim_nr >= (8*4)) and (stim_nr < (8*5)) :
				trial_order_col[ii] = 5
				trial_order_col_a[ii] = color_a_values.index(color5[0])+1
				trial_order_col_b[ii] = color_b_values.index(color5[1])+1

			elif (stim_nr >= (8*5)) and (stim_nr < (8*6)) :
				trial_order_col[ii] = 6	
				trial_order_col_a[ii] = color_a_values.index(color6[0])+1
				trial_order_col_b[ii] = color_b_values.index(color6[1])+1

			elif (stim_nr >= (8*6)) and (stim_nr < (8*7)) :
				trial_order_col[ii] = 7
				trial_order_col_a[ii] = color_a_values.index(color7[0])+1
				trial_order_col_b[ii] = color_b_values.index(color7[1])+1

			elif (stim_nr >= (8*7)) and (stim_nr < (8*8)) :
				trial_order_col[ii] = 8	
				trial_order_col_a[ii] = color_a_values.index(color8[0])+1
				trial_order_col_b[ii] = color_b_values.index(color8[1])+1


		trial_order_col = trial_order_col[:, np.newaxis]
		trial_order_col_a = trial_order_col_a[:, np.newaxis]
		trial_order_col_b = trial_order_col_b[:, np.newaxis]

		tmp_trial_order_col  = np.zeros((fmri_data_run.shape[1],1))
		tmp_trial_order_col_a  = np.zeros((fmri_data_run.shape[1],1))
		tmp_trial_order_col_b  = np.zeros((fmri_data_run.shape[1],1))
		#15 + 256( 2* 128) +15 =286, (286,1)
		tmp_trial_order_col[empty_start:-empty_end:2] = trial_order_col[:]
		tmp_trial_order_col_a[empty_start:-empty_end:2] = trial_order_col_a[:]
		tmp_trial_order_col_b[empty_start:-empty_end:2] = trial_order_col_b[:]  


		# shell()
		events_col_run = np.hstack([np.array(tmp_trial_order_col == stim, dtype=int) for stim in np.arange(1,number_of_stimuli+1)])
		events_col_a_run = np.hstack([np.array(tmp_trial_order_col_a == stim, dtype=int) for stim in np.arange(1,number_of_elements+1)])
		events_col_b_run = np.hstack([np.array(tmp_trial_order_col_b == stim, dtype=int) for stim in np.arange(1,number_of_elements+1)])

		events_col_ab_run = np.hstack([events_col_a_run, events_col_b_run])
		# events_col_ab_run shape: (286,12)

		#shell()
		## for trial_order_ori------------------------------------
		trial_order_ori = np.zeros((len(trial_order_run),))

		for ii, stim_nr in enumerate(trial_order_run):
			if stim_nr in np.arange(0, 64, 8):
				trial_order_ori[ii] = 1
			elif stim_nr in np.arange(1, 64, 8):
				trial_order_ori[ii] = 2
			elif stim_nr in np.arange(2, 64, 8):
				trial_order_ori[ii] = 3
			elif stim_nr in np.arange(3, 64, 8):
				trial_order_ori[ii] = 4		
			elif stim_nr in np.arange(4, 64, 8):
				trial_order_ori[ii] = 5
			elif stim_nr in np.arange(5, 64, 8):
				trial_order_ori[ii] = 6
			elif stim_nr in np.arange(6, 64, 8):
				trial_order_ori[ii] = 7
			elif stim_nr in np.arange(7, 64, 8):
				trial_order_ori[ii] = 8
		trial_order_ori = trial_order_ori[:, np.newaxis]

		tmp_trial_order_ori  = np.zeros((fmri_data_run.shape[1],1))
		#15 + 256( 2* 128) +15 =286, (286,1)
		tmp_trial_order_ori[empty_start:-empty_end:2] = trial_order_ori[:]   
		events_ori_run = np.hstack([np.array(tmp_trial_order_ori == stim, dtype=int) for stim in np.arange(1,number_of_stimuli+1)])
		# events_ori_run shape: (286,8)

		events_run = np.hstack([events_col_ab_run, events_ori_run]) # shape: (286,20)

		if fileii == 0:
			events = events_run
			events_col_ab = events_col_ab_run
			events_ori = events_ori_run

		else:
			events = np.vstack([events, events_run])
			# shell()
			events_col_ab = np.vstack([events_col_ab, events_col_ab_run])
			events_ori = np.vstack([events_ori, events_ori_run])

			# events shape: 

	# moco_params = []
	# for fileii, filename_moco in enumerate(target_files_moco):
	# 	moco_params_run = pd.read_csv(filename_moco, delim_whitespace=True, header = None)
	# 	# moco_params_run shape: (286, 6)

	# 	if fileii == 0:
	# 		moco_params = moco_params_run
	# 	else:
	# 		moco_params = np.vstack([moco_params, moco_params_run])

	print "finish preparation"
#----------------------------------------------------------------------------------------------------------

	# convolve events with hrf, to get model_BOLD_timecourse
	TR = 0.945 #ms
	model_BOLD_timecourse = fftconvolve(events, hrf(np.arange(0,30)* TR)[:,np.newaxis],'full')[:fmri_data.shape[1],:]

	model_BOLD_timecourse_col_ab = fftconvolve(events_col_ab, hrf(np.arange(0,30)* TR)[:,np.newaxis],'full')[:fmri_data.shape[1],:]
	model_BOLD_timecourse_ori = fftconvolve(events_ori, hrf(np.arange(0,30)* TR)[:,np.newaxis],'full')[:fmri_data.shape[1],:]

	design_matrix = np.hstack([np.ones((fmri_data.shape[1],1)), model_BOLD_timecourse])#, moco_params])
	design_matrix_col = np.hstack([ np.ones((fmri_data.shape[1],1)), model_BOLD_timecourse_col_ab ])# np.ones((fmri_data.shape[1],1)), model_BOLD_timecourse_col_ab ])#, moco_params])
	design_matrix_ori = np.hstack([np.ones((fmri_data.shape[1],1)), model_BOLD_timecourse_ori ])#, moco_params])


	# # GLM to get betas
	# betas, _sse, _r, _svs = np.linalg.lstsq(design_matrix, fmri_data.T )
	# #betas shape (65, 9728--number of voxels)

	# RidgeCV to get betas
	print "start RidgeCV"

	#ridge_fit = RidgeCV(alphas = np.linspace(1,1000,1000) , fit_intercept = False, normalize = True )
	##ridge_fit = RidgeCV(alphas = np.linspace(1,500,500) , fit_intercept = False, normalize = True )
	# ridge_fit = RidgeCV(alphas = np.linspace(1,100,100) , fit_intercept = False, normalize = True )
	# ridge_fit = RidgeCV(alphas = np.linspace(1,70,70) , fit_intercept = False, normalize = True )
	# ridge_fit = RidgeCV(alphas = np.linspace(1,60,60) , fit_intercept = False, normalize = True )
	
	ridge_fit_col = RidgeCV(alphas = [394] , fit_intercept = False, normalize = True )
	ridge_fit_ori = RidgeCV(alphas = [123] , fit_intercept = False, normalize = True )
	# ridge_fit_col = RidgeCV(alphas = np.linspace(1,500,500)  , fit_intercept = False )#, normalize = True )
	# ridge_fit_ori = RidgeCV(alphas = np.linspace(1,500,500)  , fit_intercept = False )#, normalize = True )	

	##ridge_fit.fit(design_matrix, fmri_data.T)
	ridge_fit_col.fit(design_matrix_col, fmri_data.T)	
	ridge_fit_ori.fit(design_matrix_ori, fmri_data.T)
	# shell()


	#shell()
	##alpha = ridge_fit.alpha_
	alpha_col = ridge_fit_col.alpha_
	alpha_ori = ridge_fit_ori.alpha_	
	# 1000 -- alpha = 70 (19:28 - 19:48--20mins)
	print alpha_col
	print alpha_ori


	##betas = ridge_fit.coef_.T
	betas_col = ridge_fit_col.coef_.T
	betas_ori = ridge_fit_ori.coef_.T

	# calculate r_squared, to select the best voxel
	##r_squared = 1.0 - ((design_matrix.dot(betas).T -fmri_data)**2).sum(axis=1) / (fmri_data**2).sum(axis=1)
	r_squared_col = 1.0 - ((design_matrix_col.dot(betas_col).T -fmri_data)**2).sum(axis=1) / (fmri_data**2).sum(axis=1)
	r_squared_ori = 1.0 - ((design_matrix_ori.dot(betas_ori).T -fmri_data)**2).sum(axis=1) / (fmri_data**2).sum(axis=1)
	
	r_squared_aver = (np.array(r_squared_col) + np.array(r_squared_ori) ) / 2

	# print 'histgrams plotting'
	# f = plt.figure(figsize = (12,12))
	# s1=f.add_subplot(2,1,1)
	# plt.hist(r_squared_col[ ~np.isnan(r_squared_col) ] ) #   'max must be larger than min in range parameter.')

	# s2=f.add_subplot(2,1,2)
	# plt.hist(r_squared_ori)
	# plt.savefig( '%s_%s_r_suqareds_RidgeCV.jpg'%(subname, data_type))

	# plt.close()

	# 9728

		# r_squared[~np.isnan(r_squared)].max()
		# r_squared[~np.isnan(r_squared)].argmax()


	order = np.argsort(r_squared_aver)

	#oxels.sort -- best 20 the first argument

	voxels_all = sorted( zip(order, r_squared_aver) , key = lambda tup: tup [1] )

	voxels = voxels_all[-20:]

	#voxels = [(100, index_100), (75,index_75), (50, index_50),(25,index_25)] 


	# beta_run = betas.T[:, 1:]

	# if run_nr == 0:
	# 	beta_lists = beta_run

	# else:
	# 	beta_lists = np.concatenate((beta_lists, beta_run), axis=1)


	print 'tuning curves plotting'
	# plot figures
	for voxelii, voxel in enumerate(voxels):


		f2 = plt.figure(figsize = (12,12))
		
		# 1. fmri data & model BOLD response
		# s1 = f.add_subplot(4,1,1)
		s1=f2.add_subplot(4,1,1) # First row, first column

		plt.plot(fmri_data[voxel[0], :])
		plt.plot(design_matrix_col.dot(betas_col[:, voxel[0]]))
		s1.set_title('time_course_col', fontsize = 10)

		
		beta_to_plot_col = betas_col[1:, voxel[0]]
		beta_to_plot_ori = betas_ori[1:, voxel[0]]


		s2 = f2.add_subplot(4,1,2)
		#s3=f.add_subplot(gs[2,0:-2]) # First row, third column
		plt.plot(beta_to_plot_col[0:12]) #.max(axis = 0))
		s2.set_title('dimention_1-color', fontsize = 10)


		s3=f2.add_subplot(4,1,3) # First row, first column

		plt.plot(fmri_data[voxel[0], :])
		plt.plot(design_matrix_ori.dot(betas_ori[:, voxel[0]]))
		s3.set_title('time_course_ori', fontsize = 10)


		s4 = f2.add_subplot(4,1,4)
		#s4 =f.add_subplot(gs[3:,-2]) # Second row, span all columns
		plt.plot(beta_to_plot_ori[0:8]) #.max(axis = 1))
		#roate_90_clockwise( beta_matrix.max(axis = 1) )

		s4.set_title('dimention_2-orientation', fontsize = 10)
		#a = plt.plot(beta_matrix.max(axis = 1))
		#rotated_a = ndimage.rotate(a, 90)
		#plt.plot(roatated_a)


		# plt.savefig( '%s_100_%s_GLM.jpg'%(subname,str(voxel[0]voxelii)))
		
		f2.savefig( '%s_%s_2colDimen_best%s_#%s_r2_%s_RidgeCV.png'%(subname, data_type, str(20-voxelii), str(voxel[0]), str(voxel[1])))

		print "plotting figures"

		plt.close()


# #----------------------------------------------------------------------------------------------------------	

	#'/home/xiaomeng/Analysis/figures/'+ 


