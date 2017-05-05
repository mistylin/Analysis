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

# Load behavioural data
data_dir_beh = '/home/shared/2017/visual/OriColorMapper/bids_converted/sub-002/func/'
filename_beh = data_dir_beh+ 'sub-002_task-fullfield_run-2.pickle' # 'iv.pickle'
trial_order = pickle.load(open(filename_beh, 'rb'))[1]
# shape of trial order: (128,1)


# Load fMRI data
data_dir_fmri = '/home/shared/2017/visual/OriColorMapper/preproc/sub-002/psc/'
filename_fmri = data_dir_fmri + 'sub-002_task-fullfield_run-2_bold_brain_B0_volreg_sg_psc.nii.gz' # 'iv.nii.gz'
unmasked_fmri_data = nib.load(filename_fmri).get_data()

# Load V1 mask
data_dir_masks = '/home/shared/2017/visual/OriColorMapper/preproc/sub-002/masks/dc/'
lhV1 = np.array(nib.load(os.path.join(data_dir_masks, 'lh.V1.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)
rhV1 = np.array(nib.load(os.path.join(data_dir_masks, 'rh.V1.thresh_vol_dil.nii.gz')).get_data(), dtype=bool)

fmri_data = np.vstack([unmasked_fmri_data[lhV1,:], unmasked_fmri_data[rhV1,:]])
# shape of fmri: (112,112,51,286) --> two dimensional matrix


#create events with 1
empty_start = 15
empty_end = 15
number_of_stimuli = 64
tmp_trial_order = np.zeros((fmri_data.shape[1],1))
#15 + 256( 2* 128) +15 =286
tmp_trial_order[empty_start:-empty_end:2] = trial_order[:]+1 # [:,np.newaxis]+1
events = np.hstack([np.array(tmp_trial_order == stim, dtype=int) for stim in np.arange(1,number_of_stimuli+1)])
# events shape: (286, 64)


# convolve events with hrf, to get model_BOLD_timecourse
TR = 0.945 #ms
model_BOLD_timecourse = fftconvolve(events, hrf(np.arange(0,30)* TR)[:,np.newaxis],'full')[:fmri_data.shape[1],:]

design_matrix = np.hstack([np.ones((fmri_data.shape[1],1)), model_BOLD_timecourse])

shell()

# GLM to get betas
betas, _sse, _r, _svs = np.linalg.lstsq(design_matrix, fmri_data.T )
#betas shape (65, 639744)


# calculate r_squared, to select the best voxel
r_squared = 1.0 - ((design_matrix.dot(betas).T -fmri_data)**2).sum(axis=1) / (fmri_data**2).sum(axis=1)



# plot the betas for the best voxel




