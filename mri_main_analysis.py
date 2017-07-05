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

import ColorTools as ct
from Staircase import ThreeUpOneDownStaircase
from tools import two_gamma as hrf

import mri_load_data as ld

	# results = np.zeros((n_voxels,3))
	# r_squareds =  np.zeros((n_voxels, )) 
	# alphas =  np.zeros((n_voxels, 1))
	# intercept =  np.zeros((n_voxels, 1))
	# betas = np.zeros((n_voxels, n_regressors ))  #shape (5734, 71)
	# _sse = np.zeros((n_voxels, ))

	# r_squareds_selection =  np.zeros((n_voxels, ))
	# betas_selection = np.zeros((n_voxels, n_regressors ))

def run_regression(fileii, design_matrix, design_matrix_selection, fmri_data, regression = 'RidgeCV'):

	global n_voxels, n_TRs, n_regressors, df #, results, r_squareds, alphas, intercept, betas, _sse 

	n_voxels = fmri_data.shape[0]
	n_TRs = fmri_data.shape[1]
	n_regressors = design_matrix.shape[1]
	df = (n_TRs-n_regressors)

	results = np.zeros((n_voxels,3))
	r_squareds =  np.zeros((n_voxels, )) 
	alphas =  np.zeros((n_voxels, 1))
	intercept =  np.zeros((n_voxels, 1))
	betas = np.zeros((n_voxels, n_regressors ))  #shape (5734, 71)
	_sse = np.zeros((n_voxels, ))

	r_squareds_selection =  np.zeros((n_voxels, ))
	betas_selection = np.zeros((n_voxels, n_regressors ))

	print 'n_voxels without nans', n_voxels
# # GLM to get betas

	if regression == 'GLM': #'RidgeCV'
		print 'start %s GLM fitting'%(str(fileii))
		betas, _sse, _r, _svs = np.linalg.lstsq(design_matrix, fmri_data.T )
	# betas shape (65, 9728--number of voxels)
	# _sse shape (10508)
		r_squareds = 1.0 - ((design_matrix.dot(betas).T -fmri_data)**2).sum(axis=1) / (fmri_data**2).sum(axis=1)
		betas = betas.T #(10508,72)

		# for selection
		# betas_selection, _sse_selection, _r_selection, _svs_selection = np.linalg.lstsq(design_matrix_selection, fmri_data.T )
		# r_squareds_selection = 1.0 - ((design_matrix_selection.dot(betas_selection).T -fmri_data)**2).sum(axis=1) / (fmri_data**2).sum(axis=1)
		print 'finish GLM'

	elif regression == 'RidgeCV':
		ridge_fit = RidgeCV(alphas = np.linspace(1,400,400) , fit_intercept = False, normalize = True )
		
		# alpha_range = [0.001, 1000]
		#[0.001,0.01,1,10,100,1000]
		#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
		# alpha_range = [0.5]
		# alpha_range = np.concatenate((np.array([0.001,0.01]), np.linspace(0.1,0.5,4, endpoint =False), np.linspace(0.5,10,19, endpoint =False), np.linspace(10,1000,100, endpoint =True)  ) )
		# alpha_range = np.concatenate((np.array([0.001,0.01,0.1]), np.linspace(0.5,10,19, endpoint =False), np.linspace(10,1000,100, endpoint =True)  ) )
		# alpha_range = np.concatenate((np.array([0.01,0.1]), np.linspace(0.5,10,19, endpoint =False), np.linspace(10,1000,100, endpoint =True)  ) )

		# ridge_fit = RidgeCV(alphas = alpha_range , fit_intercept = True, normalize = True)
		
		# ridge_fit_selection = RidgeCV(alphas = alpha_range , fit_intercept = True, normalize = True)

		
		print 'start %s RidgeCV fitting'%(str(fileii))
		
		for x in range(n_voxels):
			
			ridge_fit.fit(design_matrix, fmri_data[x, :])
			print x, ridge_fit.score(design_matrix, fmri_data[x,:]), ridge_fit.alpha_ #, ridge_fit.coef_.T
			# results[x] = [ridge_fit.score(design_matrix, fmri_data[x,:]), ridge_fit.alpha_, ridge_fit.coef_.T]	#ridge_fit.fit(design_matrix, fmri_data.T)
			r_squareds[x] = ridge_fit.score(design_matrix, fmri_data[x,:])
			alphas[x] = ridge_fit.alpha_
			betas[x] = ridge_fit.coef_.T
			intercept[x,:] = ridge_fit.intercept_
			_sse[x] = np.sqrt(np.sum((design_matrix.dot(betas[x]) - fmri_data[x,:])**2)/df)

			# # for selection
			# ridge_fit_selection.fit(design_matrix_selection, fmri_data[x,:])
			# r_squareds_selection[x] = ridge_fit_selection.score(design_matrix_selection, fmri_data[x,:])
			# betas_selection[x] = ridge_fit.coef_.T

		print 'finish RidgeCV'

	return r_squareds, r_squareds_selection, betas, betas_selection, _sse, intercept, alphas

def calculate_t_p_values (betas, fmri_data, moco_params, key_press, design_matrix, _sse, n_contrasts = 64 ):
	# compute contrasts
	# if type_contrasts == 'full':
	global n_voxels, n_TRs, n_regressors, df

	n_voxels = fmri_data.shape[0]
	n_TRs = fmri_data.shape[1]
	n_regressors = design_matrix.shape[1]
	df = (n_TRs-n_regressors)

	t = np.zeros((n_voxels, n_contrasts))
	p = np.zeros((n_voxels, n_contrasts))

	for i in range(n_contrasts):

		c_moco = np.zeros(moco_params.shape[1])
		c_key_press = np.zeros(key_press.shape[1])
		a = np.ones(n_contrasts) *  -1/float(n_contrasts-1)  #-1/ 63.0
		# a = np.zeros(64)
		a[i] = 1

		c = np.r_[ a, c_moco, c_key_press] #.reshape(8,8) moco_params, key_press
		design_var = c.dot(np.linalg.pinv(design_matrix.T.dot(design_matrix))).dot(c.T)
		SE_c = np.sqrt(_sse * design_var)

		t[:,i] = betas.dot(c) / SE_c  # SE_c (10508,)
		p[:,i] = scipy.stats.t.sf(np.abs(t[:,i]), df)*2		

	return t, p

def get_voxel_indices_reliVox(r_squareds_selection_runs, r_squareds_threshold = 0.05, select_100 = False ):
	# r_squareds_threshold = 0.05
	voxel_indices_reliVox = np.squeeze(np.where(np.mean(r_squareds_selection_runs, axis = 0) > r_squareds_threshold)) #2403 voxels in total left, out of 5734
	n_reli = voxel_indices_reliVox.shape[0]

	if select_100 == True:
		if n_reli >= 100:
			r_squareds_selection_mean = np.mean(r_squareds_selection_runs, axis = 0) 
			order = np.argsort(r_squareds_selection_mean)
			voxels_all = sorted( zip(order, r_squareds_selection_mean) , key = lambda tup: tup [1] )
			n_best = 100
			voxels = voxels_all[-n_best:]
			voxel_indices_reliVox = np.array(voxels)[:,0]
			n_reli = voxel_indices_reliVox.shape[0]
	return voxel_indices_reliVox, n_reli


def find_preference_matrix_allRuns( beta_runs, n_reli, voxel_indices_reliVox) :

	beta_mean_all = np.mean(beta_runs, axis = 0)
	beta_pre_indices_reliVox = np.zeros((n_reli, 2 ))

	for voxelii, voxelIndex in enumerate(voxel_indices_reliVox):					

		voxelIndex = int(voxelIndex)
		beta_matrix_pre = beta_mean_all [voxelIndex ].reshape(8,8)
		# plt.imshow(beta_matrix, cmap= plt.cm.ocean)

		beta_pre_index = np.squeeze(np.where(beta_matrix_pre == beta_matrix_pre.max()))
		# if get two max values, make the first one
		if beta_pre_index.size != 2:
			beta_pre_index = np.array([beta_pre_index[0][0], beta_pre_index[1][0]])

		beta_pre_indices_reliVox[voxelii, :] = beta_pre_index # 7,0 -- 
	return beta_pre_indices_reliVox



def calculate_tunings_matrix (n_reli, voxel_indices_reliVox, beta_mean, beta_pre_indices_reliVox, position_cen): 
	beta_ori_reliVox = np.zeros((n_reli, 9))
	beta_col_reliVox = np.zeros((n_reli, 9))
	for nrii, voxelIndex in enumerate(voxel_indices_reliVox):
		voxelIndex = int(voxelIndex)
		beta_matrix = beta_mean [voxelIndex ].reshape(8,8)
		beta_pre_current_index = beta_pre_indices_reliVox[nrii]

		beta_matrix_leftOut_cenRow = np.roll(beta_matrix, int(position_cen-beta_pre_current_index[0]), axis = 0)
		beta_matrix_leftOut_cen= np.roll(beta_matrix_leftOut_cenRow, int(position_cen-beta_pre_current_index[1]), axis = 1)

		# make it circlar
		beta_matrix_leftOut_add_column = np.hstack((beta_matrix_leftOut_cen, beta_matrix_leftOut_cen[:,0][:, np.newaxis]))
		beta_matrix_leftOut_cir = np.vstack ((beta_matrix_leftOut_add_column, beta_matrix_leftOut_add_column[0,:]))

		beta_ori = beta_matrix_leftOut_cir[position_cen,:]
		beta_col = beta_matrix_leftOut_cir[:,position_cen]

		beta_ori_reliVox[nrii,:] = beta_ori # shape(2403,9)
		beta_col_reliVox[nrii,:] = beta_col

	beta_ori_reliVox_mean = np.mean(beta_ori_reliVox, axis = 0)  #shape: (9,)
	beta_col_reliVox_mean = np.mean(beta_col_reliVox, axis = 0)

	return beta_ori_reliVox_mean, beta_col_reliVox_mean 
