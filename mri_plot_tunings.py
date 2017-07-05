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



def roate_90_clockwise ( myarray ):

	x = np.arange(0, len(myarray) )
	y = myarray

	x_new = y
	y_new = len(myarray)-1 -x 

	# ax.set_xticklabels([7,6,5,4,3,2,1,0])

	plt.plot(x_new, y_new)



def plot_tunings (run_nr_all, n_reli, beta_ori_mean_iterations, beta_col_mean_iterations, position_cen = 2):
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
		f2.savefig( '%s_%s_%s_%s_cen%s_betaValues_%sVoxels.png'%(subname, ROI, data_type, regression, position_cen, n_reli))

