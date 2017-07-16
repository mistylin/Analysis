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
import pandas as pd

import ColorTools as ct
import seaborn as sn
from matplotlib.gridspec import GridSpec

def roate_90_clockwise ( myarray ):

	x = np.arange(0, len(myarray) )
	y = myarray

	x_new = y
	y_new = len(myarray)-1 -x 

	# ax.set_xticklabels([7,6,5,4,3,2,1,0])

	plt.plot(x_new, y_new)

def plot_3models_timeCourse_alphaHist(subname, fileii, fmri_data, r_squareds_64, design_matrix_64, betas_64, intercept_64, alphas_64, r_squareds_8_ori, design_matrix_8_ori, betas_8_ori, intercept_8_ori, alphas_8_ori, r_squareds_8_col, design_matrix_8_col, betas_8_col, intercept_8_col, alphas_8_col): 
	f1 = plt.figure(figsize = (16,16))
	s1 = f1.add_subplot(6,1,1)
	best_voxel = np.argsort(r_squareds_64)[-1]
	plt.plot(fmri_data[best_voxel])
	plt.plot(design_matrix_64.dot(betas_64[best_voxel, :] + intercept_64 [best_voxel, :]))
	s1.set_title('64ch_Alpha_%s_r2_%s'%( alphas_64[best_voxel], str(r_squareds_64[best_voxel])))
	# f1.savefig('%s_64_time_course_%s-[(1,400,400)]'  %(str(subname), str(best_voxel) ))

	#subn001--64-max1265--0.57
	s2 = f1.add_subplot(6,1,2)
	plt.hist(alphas_64, bins = 100)
	s2.set_title('alphas_64_channels')
	# f1.savefig('%s_64_alphas-[(1,500,500)]' ) %(subname )
	# f1.savefig('%s_64_time_course&alphas_%s-[(1,1000,1000)]'  %(str(subname), str(np.argmax(r_squareds_64)) ))
	
	# f1 = plt.figure(figsize = (16,16))
	s3 = f1.add_subplot(6,1,3)
	# s3 = f1.add_subplot(4,1,1)		
	# plt.figure()
	plt.plot(fmri_data[np.argmax(r_squareds_8_ori)])
	plt.plot(design_matrix_8_ori.dot(betas_8_ori[np.argmax(r_squareds_8_ori)]+intercept_8_ori [np.argmax(r_squareds_8_ori)]))
	s3.set_title('8ch_ori_Alpha_%s_r2_%s'%( alphas_8_ori[np.argmax(r_squareds_8_ori)], str(r_squareds_8_ori[np.argmax(r_squareds_8_ori)])))
	# savefig('%s_64_time_course_%s-[(1,500,500)]' ) %(str(subname), str(np.argmax(r_squareds_16)) )
	#subn001--16-max4339--0.57

	s4 = f1.add_subplot(6,1, 4)
	# s4 = f1.add_subplot(4,1, 2)
	plt.hist(alphas_8_ori, bins = 100)
	s4.set_title('alphas_8_ori')		
	# savefig('%s_64_alphas-[(1,500,500)]' ) %(str(subname) )

	s5 = f1.add_subplot(6,1,5)
	# s5 = f1.add_subplot(4,1,3)
	# plt.figure()
	plt.plot(fmri_data[np.argmax(r_squareds_8_ori)])
	plt.plot(design_matrix_8_col.dot(betas_8_col[np.argmax(r_squareds_8_col)]+intercept_8_col [np.argmax(r_squareds_8_col)]))
	s5.set_title('8ch_col_Alpha_%s_r2_%s'%( alphas_8_col[np.argmax(r_squareds_8_ori)], str(r_squareds_8_col[np.argmax(r_squareds_8_col)])))
	# savefig('%s_64_time_course_%s-[(1,500,500)]' ) %(str(subname), str(np.argmax(r_squareds_16)) )
	#subn001--16-max4339--0.57

	s6 = f1.add_subplot(6,1, 6)
	# s6 = f1.add_subplot(4,1, 4)
	plt.hist(alphas_8_col, bins = 100)
	s6.set_title('alphas_8_col')		
	# savefig('%s_64_alphas-[(1,500,500)]' ) %(str(subname) )
	# f1.savefig('%s_run%s_8v8_time_course[(1,1000,1000)].png'  %(str(subname), str(fileii)  ))

	f1.savefig('%s_run%s_64v8v8_time_course[(1,400,400)].pdf'  %(str(subname), str(fileii)  ))


def plot_3ModelFit_beta_matrix(subname, volii, r_squareds_64, beta_64_across_runs, fmri_data, design_matrix_64, betas_64, intercept_64, alphas_64, r_squareds_8_ori, design_matrix_8_ori, betas_8_ori, intercept_8_ori, alphas_8_ori, r_squareds_8_col, design_matrix_8_col, betas_8_col, intercept_8_col, alphas_8_col): 
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

	f.savefig('%s-3models_64matrix-positive_best%i_%i.pdf'%(subname, volii, best_voxel_index))
	plt.close()
















def plot_tunings (run_nr_all, n_reli, beta_ori_mean_iterations, beta_col_mean_iterations, subname, ROI, data_type, regression, position_cen ):
	if position_cen == 2:
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


# def plot_tunings_16 (run_nr_all, n_reli, beta_ori_mean_iterations, beta_col_mean_iterations, position_cen = 2):
# 	if position_cen == 2:
# 		beta_ori_mean = np.mean(beta_ori_mean_iterations, axis = 0)
# 		beta_col_mean = np.mean(beta_col_mean_iterations, axis = 0)


# 		sd = np.array([np.std(beta_ori_mean_iterations, axis = 0), np.std(beta_col_mean_iterations, axis = 0)])
# 		n = len(run_nr_all)
# 		yerr = (sd/np.sqrt(n))


# 		f2 = plt.figure(figsize = (8,6))
# 		s1 = f2.add_subplot(211)
# 		plt.plot(beta_ori_mean)
# 		plt.errorbar(range(0,9), beta_ori_mean, yerr= yerr[0])
# 		# s1.set_title('orientation', fontsize = 10)
# 		s1.set_xticklabels(['placeholder', -45, -22.5, 0, 22.5, 45, 67.5, 90, -67.5 ,-45])
# 		s1.set_xlabel('orientation - relative ')

# 		s2 = f2.add_subplot(212)
# 		plt.plot(beta_col_mean)
# 		plt.errorbar(range(0,9), beta_col_mean, yerr= yerr[1])
# 		# s2.set_title('color', fontsize = 10)
# 		s2.set_xticklabels(['placeholder', -2, -1, 0, 1, 2, 3, 4, -3, -2])
# 		s2.set_xlabel('color - relative')
# 		f2.savefig( '%s_%s_%s_%s_cen%s_beta_z_%sVoxels.png'%(subname, ROI, data_type, regression, position_cen, n_reli))
