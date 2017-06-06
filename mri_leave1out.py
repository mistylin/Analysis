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
import sys

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
# lhV1 = np.array(nib.load(os.path.join(data_dir_masks, 'lh.V1.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)
# rhV1 = np.array(nib.load(os.path.join(data_dir_masks, 'rh.V1.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)


sublist = [ ('sub-001',True) ]# , ('sub-002', False) 
#sublist = ['sub-001','sub-002']


data_dir_fmri = '/home/shared/2017/visual/OriColorMapper/preproc/'
#	data_dir_fmri = '/home/shared/2017/visual/OriColorMapper/preproc/sub-002/psc/'
data_dir_beh = '/home/shared/2017/visual/OriColorMapper/bids_converted/'
#	data_dir_beh = '/home/shared/2017/visual/OriColorMapper/bids_converted/sub-002/func/'
#/Users/xiaomeng/subjects/XY_01052017/mri/brainmask.mgz  #or T1.mgz


# get fullfield files

data_type = 'psc'#'tf' #'psc'


for subii, sub in enumerate(sublist):

	subname = sub[0]
	exvivo = sub[1]
	
	print '[main] Running analysis for %s' % (str(subname))
	
	subject_dir_fmri= os.path.join(data_dir_fmri,subname)
	fmri_files = glob.glob(subject_dir_fmri  + '/' + data_type + '/*.nii.gz')
	fmri_files.sort()

	moco_files = glob.glob(subject_dir_fmri + '/mcf/parameter_info' + '/*.1D')
	moco_files.sort()

	#respiration&heart rate

		
	if exvivo == True:
		lhV1 = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/dc/lh.V1_exvivo.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)
		rhV1 = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/dc/rh.V1_exvivo.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)
	
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

	file_pairs_all = np.array(zip (target_files_fmri, target_files_beh, target_files_moco))

	
	r_squared_list = []
	fmri_data_list = []
	beta_list = []

	for filepairii in np.arange(file_pairs_all.shape[0]):

		file_pairs = file_pairs_all[~(np.arange(file_pairs_all.shape[0]) == filepairii)]

		print 'get into the appdend procedure'


# #----------------------------------------------------------------------------------------------------------		
# #----------------------------------------------------------------------------------------------------------
###
###
####       leave one run out !!!
###
###
# #----------------------------------------------------------------------------------------------------------		
# #----------------------------------------------------------------------------------------------------------
	# elif each_run == False:

	
		files_fmri = file_pairs[:,0]
		files_beh = file_pairs[:,1]
		files_moco = file_pairs[:,2]

		## version 1 - Load fmri data
		fmri_data = []#np.array( [[None] * number_of_voxels]).T
		for fileii, filename_fmri in enumerate(files_fmri):
			unmasked_fmri_data = nib.load(filename_fmri).get_data()
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
		fmri_data_list.append(fmri_data)

		print 'finish append fmri data'

		## Load stimuli order (events)
		events = []
		for fileii, filename_beh in enumerate(files_beh):
			trial_order_run = pickle.load(open(filename_beh, 'rb'))[1]

			#create events with 1
			empty_start = 15
			empty_end = 15
			number_of_stimuli = 64
			tmp_trial_order_run  = np.zeros((fmri_data_run.shape[1],1))
			#15 + 256( 2* 128) +15 =286
			tmp_trial_order_run[empty_start:-empty_end:2] = trial_order_run[:]+1 # [:,np.newaxis]+1
			events_run = np.hstack([np.array(tmp_trial_order_run == stim, dtype=int) for stim in np.arange(1,number_of_stimuli+1)])
			# events_run shape: (286, 64)	

			if fileii == 0:
				events = events_run
			else:
				events = np.vstack([events, events_run])
				# events shape: (1144,64)

		print 'finish append events'

		moco_params = []
		for fileii, filename_moco in enumerate(files_moco):
			moco_params_run = pd.read_csv(filename_moco, delim_whitespace=True, header = None)
			# moco_params_run shape: (286, 6)

			if fileii == 0:
				moco_params = moco_params_run
			else:
				moco_params = np.vstack([moco_params, moco_params_run])

		
		print 'finish preparation'
	#----------------------------------------------------------------------------------------------------------

		# convolve events with hrf, to get model_BOLD_timecourse
		TR = 0.945 #ms
		model_BOLD_timecourse = fftconvolve(events, hrf(np.arange(0,30)* TR)[:,np.newaxis],'full')[:fmri_data.shape[1],:]

		design_matrix = np.hstack([np.ones((fmri_data.shape[1],1)), model_BOLD_timecourse, moco_params])



		# # GLM to get betas
		# betas, _sse, _r, _svs = np.linalg.lstsq(design_matrix, fmri_data.T )
		# #betas shape (65, 9728--number of voxels)

		# RidgeCV to get betas
		#shell()
		print 'start RidgeCV'
		
		ridge_fit = RidgeCV(alphas = np.linspace(1,500,500) , fit_intercept = False, normalize = True )

		# ridge_fit = RidgeCV(alphas = np.linspace(1,160,160) , fit_intercept = False, normalize = True )

		ridge_fit.fit(design_matrix, fmri_data.T)
		betas = ridge_fit.coef_.T

		
		alpha = ridge_fit.alpha_
		print alpha 
		#sys.stdout = open('alpha_%s_%s.txt'%(subname, str(4-filepairii)), 'w')
		#print alpha 
		#sys.stdout.close()


		# alpha = 72.0 when (1,100,100)

		# alpha = xx when (1,1000,1000)




		# betas shape(9728, 71)


		# calculate r_squared, to select the best voxel
		r_squared = 1.0 - ((design_matrix.dot(betas).T -fmri_data)**2).sum(axis=1) / (fmri_data**2).sum(axis=1)
		# 9728
		
		#shell()
		r_squared_list.append(r_squared )

		beta_list.append(betas)

		print 'finish run'
		

			# r_squared[~np.isnan(r_squared)].max()
			# r_squared[~np.isnan(r_squared)].argmax()
	#rs 

	
	r_squared_mean = np.mean(np.array(r_squared_list), axis = 0)


	order = np.argsort(r_squared_mean)


	#oxels.sort -- best 20 the first argument

	voxels_all = sorted( zip(order, r_squared_mean) , key = lambda tup: tup [1] )

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


		f = plt.figure(figsize = (12,12))


		s1 = f.add_subplot(4,1,1)
		plt.plot(fmri_data_list[0][voxel[0], :]) 
			# shape(858)
		plt.plot(design_matrix.dot(beta_list[0][:, voxel[0]]))
			# shape also 858

		s2 = f.add_subplot(4,1,2)
		plt.plot(fmri_data_list[1][voxel[0], :]) 
		plt.plot(design_matrix.dot(beta_list[1][:, voxel[0]]))

		s3 = f.add_subplot(4,1,3)
		plt.plot(fmri_data_list[2][voxel[0], :]) 
		plt.plot(design_matrix.dot(beta_list[2][:, voxel[0]]))

		shell()
		# s4 = f.add_subplot(4,1,4)
		# plt.plot(fmri_data_list[3][voxel[0], :]) 
		# plt.plot(design_matrix.dot(beta_list[3][:, voxel[0]]))			


		#plt.show()
		f.savefig( '%s_%s_best%s_#%s_r2_%s_Ridge_leave1out_moco.png'%(subname, data_type, str(20-voxelii), str(voxel[0]), str(voxel[1])))

		plt.close()
		# # 2. beta matrix
		# # s2 = f.add_subplot(1,1,1)#(3,1,2)

		# s2=f.add_subplot(gs[3:,0:-2]) # First row, second column
		
		# beta_matrix = betas[1:65, voxel[0]].reshape(8,8)
		# #plt.plot(beta_matrix) #, cmap= plt.cm.ocean)
		# plt.imshow(beta_matrix, cmap= plt.cm.ocean, interpolation = "bicubic") #'bilinear' #"bicubic"
		# plt.colorbar()
		# 	#plt.pcolor(betas[1:, voxel[0]],cmap=plt.cm.Reds)
		# #sn.despine(offset=10)
		# #s2.set_title('beta_matrix', fontsize = 10)
		

		# # 3. tuning curves over color and orientation dimensions
		# # s3 = f.add_subplot(4,1,3)
		# s3=f.add_subplot(gs[2,0:-2]) # First row, third column
		# plt.plot(beta_matrix.max(axis = 0))
		# #s3.set_title('dimention_1-color?', fontsize = 10)



		# # s4 = f.add_subplot(4,1,4)
		# s4 =f.add_subplot(gs[3:,-2]) # Second row, span all columns
		# # plt.plot(beta_matrix.max(axis = 1))

		# roate_90_clockwise( beta_matrix.max(axis = 1) )


		# #s4.set_title('dimention_2-orientation?', fontsize = 10)
		# #a = plt.plot(beta_matrix.max(axis = 1))
		# #rotated_a = ndimage.rotate(a, 90)
		# #plt.plot(roatated_a)


		# # plt.savefig( '%s_100_%s_GLM.jpg'%(subname,str(voxel[0]voxelii)))
		





# #----------------------------------------------------------------------------------------------------------	

	#'/home/xiaomeng/Analysis/figures/'+ 


