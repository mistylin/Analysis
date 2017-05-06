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

def roate_90_clockwise ( myarray ):

	x = np.arange(0, len(myarray) )
	y = myarray

	x_new = y
	y_new = len(myarray)-1 -x 

	# ax.set_xticklabels([7,6,5,4,3,2,1,0])

	plt.plot(x_new, y_new)

def load_data_and_events():
	## version 1 - Load fmri data
	fmri_data = []#np.array( [[None] * number_of_voxels]).T
	for ii, filename_fmri in enumerate(target_files_fmri):
		unmasked_fmri_data = nib.load(filename_fmri).get_data()
		fmri_data_run = np.vstack([unmasked_fmri_data[lhV1,:], unmasked_fmri_data[rhV1,:]])

		if ii == 0:
			fmri_data = fmri_data_run
		else:
			fmri_data = np.hstack([fmri_data, fmri_data_run])
			# fmri_data shape: (9728,1144)

	# # version 2 - Load fmri data
	# # get number of voxels, for the column
	# filename_fmri_first = target_files_fmri[0]
	# unmasked_fmri_data = nib.load(filename_fmri_first).get_data()
	# fmri_data_run = np.vstack([unmasked_fmri_data[lhV1,:], unmasked_fmri_data[rhV1,:]])
	# 	#  # shape of fmri: (112,112,51,286) --> two dimensional matrix; fmri_data_run shape: (9728,286) 
	# number_of_voxels = fmri_data_run.shape[0] 

	# fmri_data = fmri_data_run
	# for filename_fmri in target_files_fmri[1:]:
	# 	unmasked_fmri_data = nib.load(filename_fmri).get_data()
	# 	fmri_data_run = np.vstack([unmasked_fmri_data[lhV1,:], unmasked_fmri_data[rhV1,:]])
	# 	fmri_data = np.hstack([fmri_data, fmri_data_run])


	## Load stimuli order (events)
	events = []
	for ii, filename_beh in enumerate(target_files_beh):
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

		if ii == 0:
			events = events_run
		else:
			events = np.vstack([events, events_run])
			# events shape: (1144,64)
	
	return fmri_data, events


def find_voxels ():
	fmri_data, events = load_data_and_events()

	# convolve events with hrf, to get model_BOLD_timecourse
	TR = 0.945 #ms
	model_BOLD_timecourse = fftconvolve(events, hrf(np.arange(0,30)* TR)[:,np.newaxis],'full')[:fmri_data.shape[1],:]

	design_matrix = np.hstack([np.ones((fmri_data.shape[1],1)), model_BOLD_timecourse])



	# GLM to get betas
	betas, _sse, _r, _svs = np.linalg.lstsq(design_matrix, fmri_data.T )
	#betas shape (65, 9728--number of voxels)


	# calculate r_squared, to select the best voxel
	r_squared = 1.0 - ((design_matrix.dot(betas).T -fmri_data)**2).sum(axis=1) / (fmri_data**2).sum(axis=1)

		# r_squared[~np.isnan(r_squared)].max()
		# r_squared[~np.isnan(r_squared)].argmax()

	voxel_100 = np.nanpercentile(r_squared, 100, interpolation='nearest') # no. 4296 voxel
	voxel_75 = np.nanpercentile(r_squared, 75, interpolation='nearest') # no. 232,8677
	voxel_50 = np.nanpercentile(r_squared, 50, interpolation='nearest')# no. 4642
	voxel_25 = np.nanpercentile(r_squared, 25, interpolation='nearest')# no. 4642

	index_100 = np.where(r_squared == voxel_100)[0]
	index_75 = np.where(r_squared == voxel_75)[0]
	index_50 = np.where(r_squared == voxel_50)[0]
	index_25 = np.where(r_squared == voxel_25)[0]

	voxels = [(100, index_100), (75,index_75), (50, index_50),(25,index_25)] 

	return fmri_data, design_matrix, betas, voxels


def plot_figures ():
	## plot figures

	fmri_data, design_matrix, betas, voxels = find_voxels()

	for voxel in voxels:

		for itemii, item in enumerate(voxel[1]): 

			f = plt.figure(figsize = (12,12))

			gs=GridSpec(6,6) # (2,3)2 rows, 3 columns
			
			# 1. fmri data & model BOLD response
			# s1 = f.add_subplot(4,1,1)
			s1=f.add_subplot(gs[0:2,:]) # First row, first column

			plt.plot(fmri_data[item, :])
			plt.plot(design_matrix.dot(betas[:, item]))
			#s1.set_title('time_course', fontsize = 10)

			# 2. beta matrix
			# s2 = f.add_subplot(1,1,1)#(3,1,2)

			s2=f.add_subplot(gs[3:,0:-2]) # First row, second column
			
			beta_matrix = betas[1:, item].reshape(8,8)
			#plt.plot(beta_matrix) #, cmap= plt.cm.ocean)
			plt.imshow(beta_matrix, cmap= plt.cm.ocean)
			plt.colorbar()
				#plt.pcolor(betas[1:, item],cmap=plt.cm.Reds)
			#sn.despine(offset=10)
			#s2.set_title('beta_matrix', fontsize = 10)
			

			# 3. tuning curves over color and orientation dimensions
			# s3 = f.add_subplot(4,1,3)
			s3=f.add_subplot(gs[2,0:-2]) # First row, third column
			plt.plot(beta_matrix.max(axis = 0))
			#s3.set_title('dimention_1-color?', fontsize = 10)



			# s4 = f.add_subplot(4,1,4)
			s4 =f.add_subplot(gs[3:,-2]) # Second row, span all columns
			# plt.plot(beta_matrix.max(axis = 1))

			roate_90_clockwise( beta_matrix.max(axis = 1) )


			#s4.set_title('dimention_2-orientation?', fontsize = 10)
			#a = plt.plot(beta_matrix.max(axis = 1))
			#rotated_a = ndimage.rotate(a, 90)
			#plt.plot(roatated_a)


			# plt.savefig( '%s_100_%s_GLM.jpg'%(subname,str(itemii)))
			
			f.savefig(  '%s_%spercentile_%s_#%s_GLM.png'%(subname,str(voxel[0]),str(itemii),str(item)))
		
	# figure_dir +



# # Load V1 mask
# data_dir_masks = '/home/shared/2017/visual/OriColorMapper/preproc/sub-002/masks/dc/'
# lhV1 = np.array(nib.load(os.path.join(data_dir_masks, 'lh.V1.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)
# rhV1 = np.array(nib.load(os.path.join(data_dir_masks, 'rh.V1.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)


#sublist = np.array([['sub-001','True'], ['sub-002','False'] ])#,
#exvivo = [True, False]
sublist = ['sub-001','sub-002']

data_dir_fmri = '/home/shared/2017/visual/OriColorMapper/preproc/'
#	data_dir_fmri = '/home/shared/2017/visual/OriColorMapper/preproc/sub-002/psc/'
data_dir_beh = '/home/shared/2017/visual/OriColorMapper/bids_converted/'
#	data_dir_beh = '/home/shared/2017/visual/OriColorMapper/bids_converted/sub-002/func/'
figure_dir = '/home/shared/2017/visual/OriColorMapper/'

# get fullfield files
for subii, subname in enumerate(sublist):

	print '[main] Running analysis for %s' % (str(subname))
	# 


	subject_dir_fmri= os.path.join(data_dir_fmri,subname)
	fmri_files = glob.glob(subject_dir_fmri+'/psc'+ '/*.nii.gz')
	fmri_files.sort()

	# for exvivo in sublist[subii][1]:
		
	# 	if exvivo== 'True':
	# 		lhV1 = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/dc/lh.V1_exvivo.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)
	# 		rhV1 = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/dc/rh.V1_exvivo.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)
		
	# 	else:
	# 		lhV1 = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/dc/lh.V1.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)
	# 		rhV1 = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/dc/rh.V1.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)

	lhV1 = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/dc/lh.V1_exvivo.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)
	rhV1 = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/dc/rh.V1_exvivo.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)

	# lhV1 = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/dc/lh.V1.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)
	# rhV1 = np.array(nib.load(os.path.join(subject_dir_fmri,'masks/dc/rh.V1.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)

	subject_dir_beh = os.path.join(data_dir_beh,subname)
	beh_files = glob.glob(subject_dir_beh +'/func'+ '/*.pickle')
	beh_files.sort()

	target_files_fmri = []
	target_files_beh = []
	target_condition = 'task-fullfield'
	for fmri_file in fmri_files:
		if fmri_file.split('_')[1]== target_condition:
			target_files_fmri.append(fmri_file)

	for beh_file in beh_files:
		if beh_file.split('_')[2]== target_condition:
			target_files_beh.append(beh_file)


	plot_figures()








