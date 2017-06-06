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

from sklearn.linear_model import RidgeCV, Ridge



# In [31]: ridge_fit.score(design_matrix, fmri_data[x,:])
# Out[31]: 0.0044921326562293862

# In [32]: ridge_fit = RidgeCV(alphas = np.linspace(1,50,50) , fit_intercept = False, normalize
#     ...:  = True )
#     ...: n_voxels = fmri_data.shape[0]
#     ...: results = np.zeros((n_voxels,2))
#     ...: for x in n_voxels:
#     ...:     ridge_fit.fit(design_matrix, fmri_data[x, :])
#     ...:     print x, ridge_fit.score(design_matrix, fmri_data[x,:]), ridge_fit.alpha_
#     ...:     results[x] = [ridge_fit.score(design_matrix, fmri_data[x,:]), ridge_fit.alpha_]

# hist(results[:,0], bins = 200)
# scatter(results[:,0], results[:,1])





def roate_90_clockwise ( myarray ):

	x = np.arange(0, len(myarray) )
	y = myarray

	x_new = y
	y_new = len(myarray)-1 -x 

	# ax.set_xticklabels([7,6,5,4,3,2,1,0])

	plt.plot(x_new, y_new)


# def maxabs(a, axis=None):
#     """Return slice of a, keeping only those values that are furthest away
#     from 0 along axis"""
# 	maxa = a.max(axis=axis)
# 	mina = a.min(axis=axis)
# 	p = abs(maxa) >= abs(mina) # bool, or indices where +ve values win
# 	n = abs(mina) > abs(maxa) # bool, or indices where -ve values win
# 	if axis == None:
# 		if p: return maxa
# 		else: return mina
# 	shape = list(a.shape)
# 	shape.pop(axis)
# 	out = np.zeros(shape, dtype=a.dtype)
# 	out[p] = maxa[p]
# 	out[n] = mina[n]
# 	return out

##-------------------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------------------
 # without def 


# # Load V1 mask
# data_dir_masks = '/home/shared/2017/visual/OriColorMapper/preproc/sub-002/masks/dc/'
# lhV1 = np.array(nib.load(os.path.join(data_dir_masks, 'lh.V1.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)
# rhV1 = np.array(nib.load(os.path.join(data_dir_masks, 'rh.V1.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)


sublist = [('sub-001', False, True)]#[ ('sub-001', False, True) ]# , , [('sub-002', True, False)]
#sublist = ['sub-001','sub-002']
# name, retinotopic, exvivo

data_dir_fmri = '/home/shared/2017/visual/OriColorMapper/preproc/'
#	data_dir_fmri = '/home/shared/2017/visual/OriColorMapper/preproc/sub-002/psc/'
data_dir_beh = '/home/shared/2017/visual/OriColorMapper/bids_converted/'
#	data_dir_beh = '/home/shared/2017/visual/OriColorMapper/bids_converted/sub-002/func/'
#/Users/xiaomeng/subjects/XY_01052017/mri/brainmask.mgz  #or T1.mgz


# get fullfield files

data_type = 'tf'#'tf' #'psc'
each_run = False #True #False
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
###        orientation circle; color -- 2 dimensions; 6*8 = 48 channels
###
###
# #----------------------------------------------------------------------------------------------------------		
# #----------------------------------------------------------------------------------------------------------
	# elif each_run == False:

	## version 1 - Load fmri data
	fmri_data = []#np.array( [[None] * number_of_voxels]).T
	for fileii, filename_fmri in enumerate(target_files_fmri):
		unmasked_fmri_data = nib.load(filename_fmri).get_data()
		fmri_data_run = np.vstack([unmasked_fmri_data[lh,:], unmasked_fmri_data[rh,:]])

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
	# events_col = []
	# events_ori = []

	for fileii, filename_beh in enumerate(target_files_beh):
		#shell()
		trial_order_run = pickle.load(open(filename_beh, 'rb'))[1] #(128,1)

		#create events with 1
		empty_start = 15
		empty_end = 15
		# number_of_stimuli = 8?????
		number_of_elements = 48

		# parsopf = open(self.output_file + '_trialinfo.pickle', 'wb')
		#output = [self.trial_array, self.trial_indices, self.trial_params, self.per_trial_parameters, self.per_trial_phase_durations, self.staircase]
		# pickle.dump(output,parsopf)
		# trial_array shape: (64, 4); trial_indices: numbers 0-63 & above; trial_params, 

		#  the first 8 values represent all orientations but only 1color
		## for trial_order_col------------------------------------

		# trial_order_col = np.zeros((len(trial_order_run),))
		trial_order_a = np.zeros((len(trial_order_run),))
		trial_order_b = np.zeros((len(trial_order_run),))

		# color_a_values = [75, 53, 0.001, -53, -75, -0.001 ]
		# color_b_values = [-0.001, 53, 75, 0.001, -53, -75 ]
		color_a_values = [75, 53, 0.001, -0.001, -53, -75 ]
		color_b_values = [75, 53, 0.001, -0.001, -53, -75 ]

# 8 --> 6, therefore 2 repititions 53, -53
		color1 = ( 75, -0.001) # although it's 0 when printing out the value, code it as -0.001, to make it different from color 5's b value
		color2 = ( 53, 53)
		color3 = ( 0.001, 75)
		color4 = ( -53, 53)
		color5 = ( -75, 0.001)
		color6 = ( -53, -53)
		color7 = ( -0.001, -75)
		color8 = ( 53, -53)

		orientations = [0,1,2,3,4,5,6,7]

		for ii, stim_nr in enumerate(trial_order_run) :
			
			for ori in orientations: 
				if stim_nr in np.arange(ori, 64, 8): # loop over each orientation

					if (stim_nr >= 0 ) * (stim_nr < (8*1))  :  #the first color1
						# trial_order_col[ii] = 1
						trial_order_a[ii] = color_a_values.index(color1[0])+1 +(ori*6) # 1-6

						trial_order_b[ii] = color_b_values.index(color1[1])+1 +(ori*6)# 1-6	

					elif (stim_nr >= (8*1)) and (stim_nr < (8*2)) : # color 2
						# trial_order_col[ii] = 2
						trial_order_a[ii] = color_a_values.index(color2[0])+1+(ori*6)
						trial_order_b[ii] = color_b_values.index(color2[1])+1+(ori*6)

					elif (stim_nr >= (8*2)) and (stim_nr < (8*3)) :
						# trial_order_col[ii] = 3
						trial_order_a[ii] = color_a_values.index(color3[0])+1+(ori*6)
						trial_order_b[ii] = color_b_values.index(color3[1])+1+(ori*6)

					elif (stim_nr >= (8*3)) and (stim_nr < (8*4)) :
						# trial_order_col[ii] = 4
						trial_order_a[ii] = color_a_values.index(color4[0])+1+(ori*6)
						trial_order_b[ii] = color_b_values.index(color4[1])+1+(ori*6)

					elif (stim_nr >= (8*4)) and (stim_nr < (8*5)) :
						# trial_order_col[ii] = 5
						trial_order_a[ii] = color_a_values.index(color5[0])+1+(ori*6)
						trial_order_b[ii] = color_b_values.index(color5[1])+1+(ori*6)

					elif (stim_nr >= (8*5)) and (stim_nr < (8*6)) :
						# trial_order_col[ii] = 6	
						trial_order_a[ii] = color_a_values.index(color6[0])+1+(ori*6)
						trial_order_b[ii] = color_b_values.index(color6[1])+1+(ori*6)

					elif (stim_nr >= (8*6)) and (stim_nr < (8*7)) :
						# trial_order_col[ii] = 7
						trial_order_a[ii] = color_a_values.index(color7[0])+1+(ori*6)
						trial_order_b[ii] = color_b_values.index(color7[1])+1+(ori*6)

					elif (stim_nr >= (8*7)) and (stim_nr < (8*8)) :
						# trial_order_col[ii] = 8	
						trial_order_a[ii] = color_a_values.index(color8[0])+1+(ori*6)
						trial_order_b[ii] = color_b_values.index(color8[1])+1+(ori*6)

		#shell()

		# trial_order_col = trial_order_col[:, np.newaxis]
		trial_order_a = trial_order_a[:, np.newaxis] #(128,1)
		trial_order_b = trial_order_b[:, np.newaxis]

		# tmp_trial_order_col  = np.zeros((fmri_data_run.shape[1],1))
		tmp_trial_order_a  = np.zeros((fmri_data_run.shape[1],1))
		tmp_trial_order_b  = np.zeros((fmri_data_run.shape[1],1))
		#15 + 256( 2* 128) +15 =286, (286,1)
		# shell()
		# tmp_trial_order_col[empty_start:-empty_end:2] = trial_order_col[:]
		tmp_trial_order_a[empty_start:-empty_end:2] = trial_order_a[:]
		tmp_trial_order_b[empty_start:-empty_end:2] = trial_order_b[:]  


		# shell()
		# events_col_run = np.hstack([np.array(tmp_trial_order_col == stim, dtype=int) for stim in np.arange(1,number_of_stimuli+1)])
		events_a_run = np.hstack([np.array(tmp_trial_order_a == stim, dtype=int) for stim in np.arange(1,number_of_elements+1)])
		events_b_run = np.hstack([np.array(tmp_trial_order_b == stim, dtype=int) for stim in np.arange(1,number_of_elements+1)])
		# events_a_run shape: (286,48)

		events_run = np.hstack([events_a_run, events_b_run])
		# events_ab_run shape: (286,96)

		#shell()


		if fileii == 0:
			events = events_run

		else:
			events = np.vstack([events, events_run])
			# shell()
			# events shape: (858,96) -->3 runs



# # motion correction parameters
# 	moco_params = []
# 	for fileii, filename_moco in enumerate(target_files_moco):
# 		moco_params_run = pd.read_csv(filename_moco, delim_whitespace=True, header = None)
# 		# moco_params_run shape: (286, 6)

# 		if fileii == 0:
# 			moco_params = moco_params_run
# 		else:
# 			moco_params = np.vstack([moco_params, moco_params_run])

	print "finish preparation"
#----------------------------------------------------------------------------------------------------------

	# convolve events with hrf, to get model_BOLD_timecourse
	#shell()
	TR = 0.945 #ms
	model_BOLD_timecourse = fftconvolve(events, hrf(np.arange(0,30)* TR)[:,np.newaxis],'full')[:fmri_data.shape[1],:]

	# design_matrix = np.hstack([np.ones((fmri_data.shape[1],1)), model_BOLD_timecourse]) #, moco_params])
	design_matrix = model_BOLD_timecourse
	#design matrix shape: (858,97)


	# # GLM to get betas
	# betas, _sse, _r, _svs = np.linalg.lstsq(design_matrix, fmri_data.T )
	# #betas shape (65, 9728--number of voxels)

	# RidgeCV to get betas
	print "start RidgeCV"

	#ridge_fit = RidgeCV(alphas = np.linspace(1,1000,1000) , fit_intercept = False, normalize = True )
	# ridge_fit = RidgeCV(alphas = np.linspace(1,500,500) , fit_intercept = False, normalize = True )
	#ridge_fit = RidgeCV(alphas = np.linspace(1,100,100) , fit_intercept = False, normalize = True )
	# ridge_fit = RidgeCV(alphas = np.linspace(1,70,70) , fit_intercept = False, normalize = True )
	ridge_fit = RidgeCV(alphas = np.linspace(1,50,50) , fit_intercept = False, normalize = True )

	#ridge_fit = RidgeCV(alphas = [3] , fit_intercept = False, normalize = True ) # sub001 ingmar-freesurfer mask- 22
	# ridge_fit = RidgeCV(alphas = [55] , fit_intercept = False, normalize = True ) # sub002 bronagh-freesurfer mask- 55
	
	n_voxels = fmri_data.shape[0]
	#results = np.zeros((n_voxels,3))
	r_squareds = []
	alphas = []
	betas = []	


	#shell()

	for x in range(n_voxels):
		ridge_fit.fit(design_matrix, fmri_data[x, :])
		print x, ridge_fit.score(design_matrix, fmri_data[x,:]), ridge_fit.alpha_ #, ridge_fit.coef_.T

		#shell()
		#results[x] = [ridge_fit.score(design_matrix, fmri_data[x,:]), ridge_fit.alpha_, ridge_fit.coef_.T]	#ridge_fit.fit(design_matrix, fmri_data.T)

		r_squareds.append(ridge_fit.score(design_matrix, fmri_data[x,:]))
		alphas.append(ridge_fit.alpha_)
		betas.append(ridge_fit.coef_.T)		

	# # shell()
	# alpha = ridge_fit.alpha_
	# # 1000 -- alpha = 70 (19:28 - 19:48--20mins)
	# print alpha # sub001- alpha = 22.0


	#shell()
	betas = np.array(betas) # shape: (10655,96)
	betas = betas.T
	# betas shape: (97, 10655); fmri_data.shape: (10655,858); fmri_data_run.shape (10655,286)
	# average across channels? or voxels? voxels. because want to compare between voxels. fmri_data: average across time points, keep voxels intact. becuase we want to compare different runs--times.

	#fmri_data_run = (fmri_data_run - np.nanmean(fmri_data_run, axis = 1)[:, np.newaxis]) / np.nanstd(fmri_data_run, axis = 1)[:, np.newaxis]
	betas_z = (betas - np.nanmean(betas, axis = 1)[:, np.newaxis]) / np.nanstd(betas, axis = 1)[:, np.newaxis]




# In [32]: ridge_fit = RidgeCV(alphas = np.linspace(1,50,50) , fit_intercept = False, normalize
#     ...:  = True )
#     ...: n_voxels = fmri_data.shape[0]
#     ...: results = np.zeros((n_voxels,2))
#     ...: for x in n_voxels:
#     ...:     ridge_fit.fit(design_matrix, fmri_data[x, :])
#     ...:     print x, ridge_fit.score(design_matrix, fmri_data[x,:]), ridge_fit.alpha_
#     ...:     results[x] = [ridge_fit.score(design_matrix, fmri_data[x,:]), ridge_fit.alpha_]

# hist(results[:,0], bins = 200)
# scatter(results[:,0], results[:,1])

# predicted signal compare it with actual ones
# take out the intercept for both
	# calculate r_squared, to select the best voxel
	#r_squareds = 1.0 - ((design_matrix.dot(betas).T -fmri_data)**2).sum(axis=1) / (fmri_data**2).sum(axis=1)

	# 10655

		# r_squareds[~np.isnan(r_squareds)].max()
		# r_squareds[~np.isnan(r_squareds)].argmax()
	print 'histgrams plotting'
	f = plt.figure(figsize = (12,12))
	s1=f.add_subplot(1,1,1)
	plt.hist (r_squareds, bins = 20 )#(r_squareds[ ~np.isnan(r_squareds) ], bins = 20 )

	plt.savefig( '%s_96ch_%s_%s_r_suqareds_RidgeCV.jpg'%(subname, data_type, ROI))

	plt.close()

	order = np.argsort(r_squareds)


	#oxels.sort -- best 20 the first argument

	voxels_all = sorted( zip(order, r_squareds) , key = lambda tup: tup [1] )

	voxels = voxels_all[-20:]

	#voxels = [(100, index_100), (75,index_75), (50, index_50),(25,index_25)] 


	# beta_run = betas.T[:, 1:]

	# if run_nr == 0:
	# 	beta_lists = beta_run

	# else:
	# 	beta_lists = np.concatenate((beta_lists, beta_run), axis=1)


	print 'start plotting'
	# plot figures
	for voxelii, voxel in enumerate(voxels):


		f = plt.figure(figsize = (24,12))

		gs=GridSpec(6,10) #(6,12) # (2,3)2 rows, 3 columns
		
		# 1. fmri data & model BOLD response
		# s1 = f.add_subplot(4,1,1)
		s1=f.add_subplot(gs[0:2,:]) # First row, first column

		plt.plot(fmri_data[voxel[0], :])
		plt.plot(design_matrix.dot(betas_z[:, voxel[0]]))
		#s1.set_title('time_course', fontsize = 10)

		# 2. beta matrix
		# s2 = f.add_subplot(1,1,1)#(3,1,2)

		s2=f.add_subplot(gs[3:,0:-8]) # First row, second column
		
		beta_matrix_a = betas_z[0:48, voxel[0]].reshape(8,6)
		#plt.plot(beta_matrix_a) #, cmap= plt.cm.ocean)
		plt.imshow(beta_matrix_a, cmap= plt.cm.ocean, interpolation = "None")#"bicubic") #'bilinear' #"bicubic"
		plt.colorbar()
			#plt.pcolor(betas[1:, voxel[0]],cmap=plt.cm.Reds)
		#sn.despine(offset=10)
		#s2.set_title('beta_matrix_a', fontsize = 10)
		

		# 3. tuning curves over color and orientation dimensions
		# s3 = f.add_subplot(4,1,3)
		s3=f.add_subplot(gs[2,0:-8]) # First row, third column

# https://stackoverflow.com/questions/17794266/how-to-get-the-highest-element-in-absolute-value-in-a-numpy-matrix

		plt.plot(beta_matrix_a.max(axis = 0))
		#s3.set_title('dimention_1-color?', fontsize = 10)



		# s4 = f.add_subplot(4,1,4)
		s4 =f.add_subplot(gs[3:,-8]) # Second row, span all columns
		# plt.plot(beta_matrix_a.max(axis = 1))

		roate_90_clockwise( beta_matrix_a.max(axis = 1) )


		#s4.set_title('dimention_2-orientation?', fontsize = 10)
		#a = plt.plot(beta_matrix_a.max(axis = 1))
		#rotated_a = ndimage.rotate(a, 90)
		#plt.plot(roatated_a)

# another dimension
		# 5. beta matrix

		s5=f.add_subplot(gs[3:,6:-2]) # First row, second column
		
		beta_matrix_b = betas_z[48:96, voxel[0]].reshape(8,6)  # 8 rows-oientation, 6 columns.
		#plt.plot(beta_matrix) #, cmap= plt.cm.ocean)
		plt.imshow(beta_matrix_b, cmap= plt.cm.ocean, interpolation = "None")#"bicubic") #'bilinear' #"bicubic"
		plt.colorbar()
			#plt.pcolor(betas[1:, voxel[0]],cmap=plt.cm.Reds)
		#sn.despine(offset=10)
		#s2.set_title('beta_matrix', fontsize = 10)


		# 6. tuning curves over color and orientation dimensions

		s6=f.add_subplot(gs[2,6:-2]) # First row, third column
		plt.plot(beta_matrix_b.max(axis = 0))
		#s3.set_title('dimention_1-color?', fontsize = 10)



		# s4 = f.add_subplot(4,1,4)
		s4 =f.add_subplot(gs[3:,-2]) # Second row, span all columns
		# plt.plot(beta_matrix_a.max(axis = 1))

		roate_90_clockwise( beta_matrix_b.max(axis = 1) )


		#s4.set_title('dimention_2-orientation?', fontsize = 10)
		#a = plt.plot(beta_matrix_a.max(axis = 1))
		#rotated_a = ndimage.rotate(a, 90)
		#plt.plot(roatated_a)



		# plt.savefig( '%s_100_%s_GLM.jpg'%(subname,str(voxel[0]voxelii)))
		# shell()
		f.savefig( '%s_96ch_%s_%s_best%s_#%s_r2_%s_RidgeCV.png'%(subname, data_type, ROI, str(20-voxelii), str(voxel[0]), str(voxel[1])))

		print "plotting figures"

		plt.close()


# #----------------------------------------------------------------------------------------------------------	

	#'/home/xiaomeng/Analysis/figures/'+ 


