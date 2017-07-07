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

import numpy as np
import pyvttbl as pt
from collections import namedtuple

def repeated_anova(fmri_data, r_squareds_mean_64, r_squareds_mean_8_ori, r_squareds_mean_8_col ): 

	n_voxels = fmri_data.shape[0]
	N = n_voxels # number of voxels: n_voxels
	n_condition_iv1 = ['64ch','8ch_ori', '8ch_col'] #  64ch vs. 8ch vs. 8ch

	r_squareds = np.concatenate( (r_squareds_mean_64, r_squareds_mean_8_ori, r_squareds_mean_8_col)  )
	voxel_id = [i+1 for i in xrange(N)]*(len(n_condition_iv1)) # 1-20 1-20 ... so 120 in total. 
	model_type = np.concatenate([np.array([p]*N) for p in n_condition_iv1]).tolist() 
	 
	Sub = namedtuple('Sub', ['voxel_id', 'r_squareds','model_type'])               
	df = pt.DataFrame()
	for idx in xrange(len(voxel_id)):
		df.insert(Sub(voxel_id[idx],r_squareds[idx], model_type[idx])._asdict()) 

	# Two-way ANOVA
	aov = df.anova('r_squareds', sub='voxel_id', wfactors=['model_type'])
	print(aov)

	# box_plot
	df.box_plot('r_squareds', factors=['model_type'])

	return aov

