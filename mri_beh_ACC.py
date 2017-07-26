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


sublist = [ ('sub-n001', False, False), ('sub-n003', False, False), ('sub-n005', False, False) ]#('sub-n001', False, False), 
data_dir_fmri = '/home/shared/2017/visual/OriColorMapper/preproc/'
data_dir_beh = '/home/shared/2017/visual/OriColorMapper/bids_converted/'

data_type = 'psc'#'tf' #'psc'
each_run = True #False #True #False
ROI = 'V1' # 'V4'
regression = 'RidgeCV' #'GLM' #'RidgeCV'
position_cen = 2 #2 4  #'nan'


# aov_values = []
# t_64vs8ori = []
# t_64vs8col = []
# t_8oriVs8col = []


ACC_across_sub = []

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

# #----------------------------------------------------------------------------------------------------------		
###     load data for each run !!!
# #----------------------------------------------------------------------------------------------------------

	## Load all types of data
	file_pairs_all = np.array(zip (target_files_fmri, target_files_beh, target_files_moco, target_files_fixation))
	run_nr_all = np.arange(file_pairs_all.shape[0])
	
	n_response_all_runs = [] 
	n_response_correct_runs = [] 

	for fileii, (filename_fmri, filename_beh, filename_moco, filename_fixation) in enumerate(file_pairs_all):	
		# shell()
		#0,1,2,3
		# file_pair = file_pairs_all[fileii]
		# filename_fmri = file_pair[0]
		# filename_beh = file_pair[1]
		# filename_moco = file_pair[2]
		# filename_fixation = file_pair[3]
		# shell()
		# n_response_all, n_response_correct = ld.calculate_ACC(filename_fixation)

		fixation_order_run = pickle.load(open(filename_fixation, 'rb'))
		eventArray = fixation_order_run['eventArray']  # a list of lists

		n_response_all = 0
		n_response_correct = 0

		if subname == 'sub-n003':
			for event in eventArray:
				for txt in event:
					if 'after response' in txt:
						n_response_all += 1
					
					if 'after response 1' in txt:
						n_response_correct += 1
		else:
			for event in eventArray:
				if 'after response' in event:
					n_response_all += 1
				
				if 'after response 1' in event:
					n_response_correct += 1

		# shell()
		n_response_all_runs.append(n_response_all)
		n_response_correct_runs.append(n_response_correct)

	n_response_all_runs = np.array(n_response_all_runs)
	n_response_correct_runs = np.array(n_response_correct_runs)

	ACC_per_run = n_response_correct_runs / n_response_all_runs
	ACC_across_run_mean = np.mean(ACC_per_run)
	ACC_across_run_sd = np.std(ACC_per_run)
	yerr_run = ACC_across_run_sd/np.sqrt(len(run_nr_all))

	print sub
	print ACC_across_run_mean, ACC_across_run_sd, yerr_run

	ACC = np.sum(n_response_correct_runs)/ np.sum(n_response_all_runs)

	ACC_across_sub.append(ACC)
	print ACC

shell()
ACC_across_sub = np.array(ACC_across_sub)
ACC_across_sub_mean = np.mean(ACC_across_sub)
ACC_across_sub_sd = np.std(ACC_across_sub)
n_sub = len(sublist)

yerr = ACC_across_sub_sd/np.sqrt(n_sub)

print ACC_across_sub
print ACC_across_sub_mean, ACC_across_sub_sd, yerr 


